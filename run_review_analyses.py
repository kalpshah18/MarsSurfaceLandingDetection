"""Reviewer-response analyses on the 80 TEST tiles (no pipeline/hyperparameter changes).

Hazard methods (review_common.hazard_maps):
    grad, grad_shadow           -- exactly the Table 2 methods/config
    shadow_only                 -- ablation: hazard = retained shadow mask only
Selectors applied to each method's filtered safe mask:
    maxrect   -- the paper's maximal-area rectangle (Table 2 boxes)
    lsquare   -- largest inscribed square (ties -> lowest mean hazard)
    laware    -- for each L, the L x L square inside the mask with lowest mean hazard
Label-based upper bounds (oracles, NOT selectors):
    avail_mask    -- some L x L square inside the predicted safe mask has <= eps unsafe
    avail_terrain -- some L x L square anywhere on the tile has <= eps unsafe

Also: false-safe site rates, per-terrain and hazard-bearing (crater+rocky) results with
tile-bootstrap CIs, pooled and within-tile classification metrics (incl. shadow_only),
the B5 precision discrepancy, and the crater-bowl inversion diagnosis.

Outputs: results/review/per_tile_selectors.csv, results/review/analyses.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score

from footprint_metrics import EPSILON
from review_common import (HAZARD_TERRAINS, TERRAINS, any_safe_square, boot_ci, frac_in,
                           hazard_maps, iter_tiles, largest_inscribed_square, load_config,
                           min_hazard_square)
from run_evaluation import get_local_std

OUT_DIR = os.path.join('results', 'review')
LS = list(range(5, 61))
REPORT_L = [5, 9, 10, 15, 20, 25, 30, 35, 40, 50]
METHODS = ['grad', 'grad_shadow', 'shadow_only']
SELECTORS = ['maxrect', 'lsquare', 'laware']


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = load_config()
    rows = []          # one per (tile, method)
    avail_terrain = {}  # tile -> {L: bool}
    pooled = {k: [] for k in ['u', 'grad', 'grad_shadow', 'shadow_only', 'b1', 'b5_score', 'b5_pred',
                              'pred_grad', 'pred_grad_shadow', 'pred_shadow_only']}
    pooled_2d = {k: [] for k in ['u', 'grad', 'grad_shadow']}
    within = []        # per-tile AUROC
    b5_tiles = []      # per-tile flag rate and prevalence for the B5 decomposition
    crater_diag = []
    crater_pix = {'G_raw_lit_unsafe': [], 'G_raw_lit_safe': [], 'n_unsafe': 0, 'n_unsafe_shadow': 0}
    sub = slice(None, None, 10)

    for t in iter_tiles('TEST'):
        tid, terrain, ori, unsafe = t['tile_id'], t['terrain'], t['ori'], t['unsafe']
        maps = hazard_maps(ori, config, METHODS)
        avail_terrain[tid] = {L: any_safe_square(unsafe, L) for L in LS}

        for m in METHODS:
            H, filt, (x, y, w, h) = maps[m]
            sx, sy, side = largest_inscribed_square(filt, H)
            rec = {'tile_id': tid, 'terrain': terrain, 'method': m,
                   'maxrect_w': w, 'maxrect_h': h, 'maxrect_frac': frac_in(unsafe, x, y, w, h),
                   'lsquare_side': side, 'lsquare_frac': frac_in(unsafe, sx, sy, side, side)}
            for L in LS:
                sq = min_hazard_square(filt, H, L)
                rec[f'laware_frac_{L}'] = float('nan') if sq is None else frac_in(unsafe, sq[0], sq[1], L, L)
                rec[f'avail_mask_{L}'] = any_safe_square(unsafe, L, mask=filt)
            rows.append(rec)

        # --- classification, pooled (1-D stride 10, as in Table 1) and 2-D 10x10 grid
        u_flat = unsafe.flatten()[sub]
        pooled['u'].append(u_flat)
        for m in METHODS:
            H, filt, _ = maps[m]
            pooled[m].append(H.flatten()[sub])
        # predicted-unsafe masks as in compute_final_metrics (unfiltered threshold mask)
        gs_theta = config['V_grad_shadow']['theta']
        for m in METHODS:
            H = maps[m][0]
            pooled['pred_' + m].append(((1.0 - H) <= (config['V_grad']['theta'] if m == 'grad' else gs_theta))
                                       .flatten()[sub].astype(np.uint8))
        std = get_local_std(ori)
        pooled['b1'].append(std.flatten()[sub])
        seed_val = int(tid.split('_')[-1]) + 100
        np.random.seed(seed_val)
        h_b5 = np.random.rand(*ori.shape)
        rate_g = float(np.mean((1.0 - maps['grad'][0]) <= config['V_grad']['theta']))
        pooled['b5_score'].append(h_b5.flatten()[sub])
        pooled['b5_pred'].append((h_b5 > 1.0 - rate_g).flatten()[sub].astype(np.uint8))
        b5_tiles.append({'tile_id': tid, 'terrain': terrain, 'flag_rate': rate_g,
                         'prevalence': float(u_flat.mean()), 'n': int(u_flat.size)})
        pooled_2d['u'].append(unsafe[::10, ::10].ravel())
        for m in ('grad', 'grad_shadow'):
            pooled_2d[m].append(maps[m][0][::10, ::10].ravel())

        if terrain in HAZARD_TERRAINS:
            r = {'tile_id': tid, 'terrain': terrain}
            for m in METHODS:
                r[m] = float(roc_auc_score(u_flat, maps[m][0].flatten()[sub]))
            r['b1'] = float(roc_auc_score(u_flat, std.flatten()[sub]))
            within.append(r)

        # --- crater inversion diagnosis
        if terrain == 'crater':
            gs_cfg = config['V_grad_shadow']
            from pipeline import augment_shadow, compute_gradient_hazard
            G_raw = compute_gradient_hazard(ori)
            _, shadow = augment_shadow(ori, G_raw, tau=gs_cfg['tau'], a_min=gs_cfg['a_min'])
            sh = shadow == 255
            lit = ~sh
            crater_pix['G_raw_lit_unsafe'].append(G_raw[lit & unsafe][::10])
            crater_pix['G_raw_lit_safe'].append(G_raw[lit & ~unsafe][::10])
            crater_pix['n_unsafe'] += int(unsafe.sum())
            crater_pix['n_unsafe_shadow'] += int((unsafe & sh).sum())
            _, _, (x, y, w, h) = maps['grad_shadow']
            box = (slice(y, y + h), slice(x, x + w))
            crater_diag.append({
                'tile_id': tid,
                'box_frac_unsafe': frac_in(unsafe, x, y, w, h),
                'box_frac_shadow': float(sh[box].mean()),
                'box_mean_intensity': float(ori[box].mean()),
                'box_mean_slope_deg': float(t['slope'][box].mean()),
                'box_mean_G_raw': float(G_raw[box].mean()),
                'box_mean_depth_m': float(-t['dtm'][box].mean()),
                'tile_frac_unsafe_lit': float((unsafe & lit).sum() / max(1, unsafe.sum())),
                'tile_mean_G_raw_lit_unsafe': float(G_raw[lit & unsafe].mean()) if (lit & unsafe).any() else None,
                'tile_mean_G_raw_safe': float(G_raw[~unsafe].mean()),
            })
        print(tid, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'per_tile_selectors.csv'), index=False)

    # ------------------------------------------------------------------ site selection
    def outcomes(d, selector, L):
        """Per-tile (safe_landing, false_safe) booleans for a selector at side L."""
        if selector == 'maxrect':
            fit = np.minimum(d['maxrect_w'], d['maxrect_h']) >= L
            frac = d['maxrect_frac']
        elif selector == 'lsquare':
            fit = d['lsquare_side'] >= L
            frac = d['lsquare_frac']
        else:
            frac = d[f'laware_frac_{L}']
            fit = ~frac.isna()
        safe = fit & (frac <= EPSILON)
        false_safe = fit & (frac > EPSILON)
        return safe.to_numpy(), false_safe.to_numpy(), fit.to_numpy()

    def summarize(d, selector, L):
        safe, fs, fit = outcomes(d, selector, L)
        return {'n': int(len(d)), 'safe_landing': float(safe.mean()), 'safe_landing_ci': boot_ci(safe),
                'false_safe': float(fs.mean()), 'false_safe_ci': boot_ci(fs),
                'fit': float(fit.mean()),
                'false_safe_given_fit': float(fs.sum() / fit.sum()) if fit.sum() else None}

    groups = {'all': list(TERRAINS), 'hazard_bearing': list(HAZARD_TERRAINS),
              'hazard_free': ['smooth', 'dunes'], **{t: [t] for t in TERRAINS}}
    selection = {}
    for m in METHODS:
        selection[m] = {}
        dm = df[df['method'] == m]
        for sel in SELECTORS:
            selection[m][sel] = {}
            for g, ts in groups.items():
                d = dm[dm['terrain'].isin(ts)]
                selection[m][sel][g] = {str(L): summarize(d, sel, L) for L in REPORT_L}
                curve = [outcomes(d, sel, L)[0].mean() for L in LS]
                selection[m][sel][g]['curve_safe_landing'] = {str(L): float(c) for L, c in zip(LS, curve)}
                above = [L for L, c in zip(LS, curve) if c >= 0.5]
                selection[m][sel][g]['L_star'] = max(above) if above else None
                selection[m][sel][g]['max_safe_landing'] = float(max(curve))
        # oracle availability
        for g, ts in groups.items():
            d = dm[dm['terrain'].isin(ts)]
            selection[m].setdefault('avail_mask', {})[g] = {
                str(L): float(d[f'avail_mask_{L}'].mean()) for L in LS}
    terr = {}
    for g, ts in groups.items():
        ids = df[(df['method'] == 'grad') & df['terrain'].isin(ts)]['tile_id']
        terr[g] = {str(L): float(np.mean([avail_terrain[i][L] for i in ids])) for L in LS}

    # box geometry summary
    geometry = {}
    for m in METHODS:
        dm = df[df['method'] == m]
        geometry[m] = {
            'maxrect_max_short_side': int(np.minimum(dm['maxrect_w'], dm['maxrect_h']).max()),
            'lsquare_max_side': int(dm['lsquare_side'].max()),
            'lsquare_side_by_terrain_median': dm.groupby('terrain')['lsquare_side'].median().to_dict(),
            'lsquare_side_by_terrain_max': {k: int(v) for k, v in dm.groupby('terrain')['lsquare_side'].max().items()},
            'maxrect_mean_frac_unsafe': float(dm['maxrect_frac'].mean()),
        }

    # ------------------------------------------------------------------ classification
    u = np.concatenate(pooled['u'])
    cls = {'prevalence': float(u.mean()), 'n_pixels_evaluated': int(u.size)}
    for m in METHODS:
        h = np.concatenate(pooled[m])
        p = np.concatenate(pooled['pred_' + m])
        cls[m] = {'precision': float(precision_score(u, p, zero_division=0)),
                  'recall': float(recall_score(u, p, zero_division=0)),
                  'f1': float(f1_score(u, p, zero_division=0)),
                  'auroc': float(roc_auc_score(u, h)), 'prauc': float(average_precision_score(u, h))}
    b1 = np.concatenate(pooled['b1'])
    cls['b1'] = {'auroc': float(roc_auc_score(u, b1)), 'prauc': float(average_precision_score(u, b1))}
    b5p = np.concatenate(pooled['b5_pred'])
    b5s = np.concatenate(pooled['b5_score'])
    bt = pd.DataFrame(b5_tiles)
    expected_prec = float((bt['flag_rate'] * bt['n'] * bt['prevalence']).sum() / (bt['flag_rate'] * bt['n']).sum())
    corr = float(np.corrcoef(bt['flag_rate'], bt['prevalence'])[0, 1])
    cls['b5'] = {'precision': float(precision_score(u, b5p)), 'auroc': float(roc_auc_score(u, b5s)),
                 'prauc': float(average_precision_score(u, b5s)),
                 'expected_precision_from_per_tile_rates': expected_prec,
                 'corr_flag_rate_vs_prevalence': corr,
                 'mean_flag_rate_by_terrain': bt.groupby('terrain')['flag_rate'].mean().to_dict(),
                 'prevalence_by_terrain': bt.groupby('terrain')['prevalence'].mean().to_dict()}
    u2 = np.concatenate(pooled_2d['u'])
    cls['grid_10x10'] = {'n_pixels': int(u2.size), 'prevalence': float(u2.mean())}
    for m in ('grad', 'grad_shadow'):
        h2 = np.concatenate(pooled_2d[m])
        cls['grid_10x10'][m] = {'auroc': float(roc_auc_score(u2, h2)), 'prauc': float(average_precision_score(u2, h2))}
    wd = pd.DataFrame(within)
    cls['within_tile_auroc'] = {}
    for g, ts in {'hazard_bearing': list(HAZARD_TERRAINS), 'crater': ['crater'], 'rocky': ['rocky']}.items():
        d = wd[wd['terrain'].isin(ts)]
        cls['within_tile_auroc'][g] = {k: {'mean': float(d[k].mean()), 'ci': boot_ci(d[k].values)}
                                       for k in METHODS + ['b1']}

    # ------------------------------------------------------------------ crater diagnosis
    cd = pd.DataFrame(crater_diag)
    bad = cd[cd['box_frac_unsafe'] > EPSILON]
    lu = np.concatenate(crater_pix['G_raw_lit_unsafe'])
    ls_ = np.concatenate(crater_pix['G_raw_lit_safe'])
    lab = np.r_[np.ones(len(lu)), np.zeros(len(ls_))]
    crater = {
        'n_tiles': int(len(cd)),
        'n_boxes_over_eps': int(len(bad)),
        'bad_boxes_mean_frac_shadow': float(bad['box_frac_shadow'].mean()),
        'bad_boxes_max_frac_shadow': float(bad['box_frac_shadow'].max()),
        'bad_boxes_mean_slope_deg': float(bad['box_mean_slope_deg'].mean()),
        'bad_boxes_mean_intensity': float(bad['box_mean_intensity'].mean()),
        'bad_boxes_mean_G_raw': float(bad['box_mean_G_raw'].mean()),
        'bad_boxes_mean_depth_m': float(bad['box_mean_depth_m'].mean()),
        'unsafe_pixels_frac_in_shadow': crater_pix['n_unsafe_shadow'] / crater_pix['n_unsafe'],
        'G_raw_mean_lit_unsafe': float(lu.mean()),
        'G_raw_mean_lit_safe': float(ls_.mean()),
        'auroc_G_raw_lit_pixels_only': float(roc_auc_score(lab, np.r_[lu, ls_])),
        'per_tile': crater_diag,
    }

    out = {'epsilon': EPSILON, 'selection': selection, 'avail_terrain': terr,
           'geometry': geometry, 'classification': cls, 'crater_diagnosis': crater}
    with open(os.path.join(OUT_DIR, 'analyses.json'), 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('wrote', os.path.join(OUT_DIR, 'analyses.json'))


if __name__ == '__main__':
    main()
