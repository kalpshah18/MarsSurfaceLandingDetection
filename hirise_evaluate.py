"""Evaluate the frozen pipeline on real HiRISE tiles (protocol in DECISIONS.md).

Input: results/hirise/tiles_manifest.csv (hirise_make_tiles.py), tiles under $HIRISE_DATA_DIR.
Hyperparameters: results/best_config.json, unchanged. Nothing is tuned here.

Per split (DEV, TEST) and per site:
  * pooled unsafe-class precision / recall / F1 / AUROC / PR-AUC (1-D stride 10) for
    grad, grad_shadow, shadow_only, B1 (local std), B5 (random, per-tile rate matched to grad)
  * within-tile AUROC (tiles containing both classes)
  * site-selection outcomes for selectors maxrect (paper) and lsquare (largest inscribed square):
      correct_site      box fits L x L and <= 5% unsafe
      false_safe        box fits L x L and > 5% unsafe
      correct_reject    no fitting box, and no L x L square with <= 5% unsafe exists on the tile
      missed            no fitting box, but such a square exists
  * tile-bootstrap 95% CIs
Outputs: results/hirise/per_tile.csv, results/hirise/evaluation.json
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from footprint_metrics import EPSILON
from review_common import any_safe_square, boot_ci, frac_in, hazard_maps, largest_inscribed_square, load_config
from run_evaluation import compute_proxy_labels, get_local_std

DATA_DIR = os.environ.get('HIRISE_DATA_DIR', os.path.join('data', 'raw', 'hirise'))
OUT = os.path.join('results', 'hirise')
METHODS = ['grad', 'grad_shadow', 'shadow_only']
LS = [10, 20, 25, 30, 40, 50]
OUTCOMES = ['correct_site', 'false_safe', 'correct_reject', 'missed']


def outcome(fits, frac, avail):
    if fits:
        return 'correct_site' if frac <= EPSILON else 'false_safe'
    return 'missed' if avail else 'correct_reject'


def pooled_metrics(u, h, p=None):
    out = {'auroc': float(roc_auc_score(u, h)), 'prauc': float(average_precision_score(u, h))}
    if p is not None:
        out.update({'precision': float(precision_score(u, p, zero_division=0)),
                    'recall': float(recall_score(u, p, zero_division=0)),
                    'f1': float(f1_score(u, p, zero_division=0))})
    return out


def main():
    config = load_config()
    man = pd.read_csv(os.path.join(OUT, 'tiles_manifest.csv'))
    sub = slice(None, None, 10)
    rows = []
    pix = []  # (split, site, dict of arrays)
    for idx, t in man.iterrows():
        ori = cv2.imread(os.path.join(DATA_DIR, t['ori']), cv2.IMREAD_GRAYSCALE)
        dtm = np.load(os.path.join(DATA_DIR, t['dtm'])).astype(np.float64)
        # Baseline of 5 posts with slopes in true metres: divide elevation by the post spacing.
        unsafe, slope = compute_proxy_labels(dtm / t['map_scale_m'])
        maps = hazard_maps(ori, config, METHODS)
        rec = {'tile_id': t['tile_id'], 'site': t['site'], 'split': t['split'],
               'prevalence': float(unsafe.mean()), 'mean_slope_deg': float(slope.mean()),
               'dark_frac_tau': float((ori < config['V_grad_shadow']['tau']).mean()),
               'ori_mean': float(ori.mean()), 'ori_std': float(ori.std())}
        avail = {L: any_safe_square(unsafe, L) for L in LS}
        for L in LS:
            rec[f'avail_{L}'] = avail[L]
        u_flat = unsafe.flatten()[sub]
        arrays = {'u': u_flat}
        for m in METHODS:
            H, filt, (x, y, w, h) = maps[m]
            sx, sy, side = largest_inscribed_square(filt, H)
            mr_frac = frac_in(unsafe, x, y, w, h)
            ls_frac = frac_in(unsafe, sx, sy, side, side)
            rec[f'{m}_maxrect_w'], rec[f'{m}_maxrect_h'], rec[f'{m}_maxrect_frac'] = w, h, mr_frac
            rec[f'{m}_lsquare_side'], rec[f'{m}_lsquare_frac'] = side, ls_frac
            for L in LS:
                rec[f'{m}_maxrect_{L}'] = outcome(min(w, h) >= L, mr_frac, avail[L])
                rec[f'{m}_lsquare_{L}'] = outcome(side >= L, ls_frac, avail[L])
            arrays[m] = H.flatten()[sub]
            theta = config['V_grad']['theta'] if m == 'grad' else config['V_grad_shadow']['theta']
            arrays['pred_' + m] = ((1.0 - H) <= theta).flatten()[sub].astype(np.uint8)
            if 0 < u_flat.mean() < 1:
                rec[f'{m}_within_auroc'] = float(roc_auc_score(u_flat, arrays[m]))
        std = get_local_std(ori)
        arrays['b1'] = std.flatten()[sub]
        if 0 < u_flat.mean() < 1:
            rec['b1_within_auroc'] = float(roc_auc_score(u_flat, arrays['b1']))
        rate_g = float(arrays['pred_grad'].mean())
        rng = np.random.default_rng(1000 + idx)
        b5 = rng.random(ori.shape).flatten()[sub]
        arrays['b5'] = b5
        arrays['pred_b5'] = (b5 > 1.0 - rate_g).astype(np.uint8)
        rows.append(rec)
        pix.append((t['split'], t['site'], arrays))
        print(t['tile_id'], round(rec['prevalence'], 3), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'per_tile.csv'), index=False)

    def classification(sel):
        u = np.concatenate([a['u'] for _, _, a in sel])
        res = {'n_tiles': len(sel), 'n_pixels': int(u.size), 'prevalence': float(u.mean())}
        if u.min() == u.max():
            return res
        for m in METHODS:
            res[m] = pooled_metrics(u, np.concatenate([a[m] for _, _, a in sel]),
                                    np.concatenate([a['pred_' + m] for _, _, a in sel]))
        res['b1'] = pooled_metrics(u, np.concatenate([a['b1'] for _, _, a in sel]))
        res['b5'] = pooled_metrics(u, np.concatenate([a['b5'] for _, _, a in sel]),
                                   np.concatenate([a['pred_b5'] for _, _, a in sel]))
        return res

    def selection(d):
        res = {}
        for m in METHODS:
            for selr in ('maxrect', 'lsquare'):
                for L in LS:
                    col = d[f'{m}_{selr}_{L}']
                    res[f'{m}.{selr}.{L}'] = {o: float((col == o).mean()) for o in OUTCOMES}
                    res[f'{m}.{selr}.{L}']['false_safe_ci'] = boot_ci((col == 'false_safe').values)
                    res[f'{m}.{selr}.{L}']['correct_site_ci'] = boot_ci((col == 'correct_site').values)
        res['avail'] = {str(L): float(d[f'avail_{L}'].mean()) for L in LS}
        return res

    def within(d):
        out = {}
        for k in [f'{m}_within_auroc' for m in METHODS] + ['b1_within_auroc']:
            v = d[k].dropna().values if k in d else np.array([])
            out[k] = {'n': int(len(v)), 'mean': float(v.mean()) if len(v) else None, 'ci': boot_ci(v)}
        return out

    result = {'epsilon': EPSILON, 'config': config, 'splits': {}, 'sites': {}}
    for split in ('DEV', 'TEST'):
        sel = [p for p in pix if p[0] == split]
        d = df[df['split'] == split]
        if not len(d):
            continue
        result['splits'][split] = {'classification': classification(sel), 'selection': selection(d),
                                   'within_tile_auroc': within(d),
                                   'mean_prevalence_tiles': float(d['prevalence'].mean())}
    for site in df['site'].unique():
        sel = [p for p in pix if p[1] == site]
        d = df[df['site'] == site]
        result['sites'][site] = {'split': d['split'].iloc[0], 'classification': classification(sel),
                                 'selection': selection(d), 'within_tile_auroc': within(d),
                                 'mean_prevalence': float(d['prevalence'].mean()),
                                 'mean_dark_frac_tau': float(d['dark_frac_tau'].mean())}
    with open(os.path.join(OUT, 'evaluation.json'), 'w') as f:
        json.dump(result, f, indent=1)
    print('wrote', os.path.join(OUT, 'evaluation.json'))


if __name__ == '__main__':
    main()
