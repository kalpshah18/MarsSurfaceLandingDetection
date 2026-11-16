"""Evaluate the current pipeline (V_grad, V_grad+shadow, V_full) on real HiRISE DTM/ortho tiles.

Tiles: results/hirise/tiles_manifest.csv (hirise_make_tiles.py), 8 sites x 20 tiles,
4 DEV sites and 4 TEST sites. Labels: slope > 15 deg over 5 posts of the HiRISE DTM.
Sun geometry per observation: results/hirise/sun_geometry.json (from the PDS RDR labels).

Two protocols, each for the gradient normalizations p99 and fixed_b5 (run_normalization_study.py):
  transfer  parameters frozen from synthetic DEV tuning; the physical constants are recomputed per
            image from its median brightness and its real sun elevation (g_ref for fixed_b5, and
            kappa scaled by I_ref * cot(e) relative to the synthetic tiles). Nothing is tuned on HiRISE.
  hirise    parameters re-tuned on 5 tiles per HiRISE DEV site (20 tiles) with the same staged
            procedure, then evaluated once on the TEST sites.

Reported per split: pooled AUROC / PR-AUC, B1 (local std) and chance, mean within-tile AUROC, and
maximal-rectangle Safe Landing at L = 10, 20, 25, 50 next to the random-square null (probability that
a uniformly placed L x L square has <= 5% unsafe pixels) and site availability.

    HIRISE_DATA_DIR=C:/Users/shahk/hirise_data python hirise_eval_v2.py
Output: results/hirise/evaluation_v2.json
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from footprint_metrics import EPSILON
from pipeline import augment_shadow, compute_brightness_hazard, fuse_and_select
from review_common import any_safe_square, window_sums
from run_evaluation import compute_proxy_labels, get_local_std
from run_normalization_study import gradient_hazard, tune

DATA_DIR = os.environ.get('HIRISE_DATA_DIR', os.path.join('data', 'raw', 'hirise'))
OUT = os.path.join('results', 'hirise', 'evaluation_v2.json')
LS = [10, 20, 25, 50]
SYN_E = 30.0
SYN_IREF = 127.5  # level-ground brightness of the synthetic tiles: 255 * sin(30 deg)


def load_tiles():
    man = pd.read_csv(os.path.join('results', 'hirise', 'tiles_manifest.csv'))
    with open(os.path.join('results', 'hirise', 'sun_geometry.json')) as f:
        geo = json.load(f)
    tiles = []
    for _, t in man.iterrows():
        ori = cv2.imread(os.path.join(DATA_DIR, t['ori']), cv2.IMREAD_GRAYSCALE)
        dtm = np.load(os.path.join(DATA_DIR, t['dtm'])).astype(np.float64)
        unsafe, _ = compute_proxy_labels(dtm / t['map_scale_m'])
        tiles.append({'tile_id': t['tile_id'], 'site': t['site'], 'split': t['split'], 'ori': ori,
                      'unsafe': unsafe, 'elev': geo[t['site']]['sun_elevation_deg']})
    return tiles


def kappa_scale(ori, elev):
    """Physical rescaling of a synthetic-tuned kappa to this image's brightness and sun elevation."""
    return (float(np.median(ori)) / np.tan(np.radians(elev))) / (SYN_IREF / np.tan(np.radians(SYN_E)))


def maps_for(t, mode, cfg, scale_kappa):
    G = gradient_hazard(t['ori'], mode, t['elev'])
    G_aug, _ = augment_shadow(t['ori'], G, tau=cfg['tau'], a_min=cfg['a_min'])
    kappa = cfg['kappa'] * (kappa_scale(t['ori'], t['elev']) if scale_kappa else 1.0)
    H = np.maximum(G_aug, compute_brightness_hazard(t['ori'], kappa=kappa, ksize=cfg['ksize']))
    return {'grad': G, 'grad_shadow': G_aug, 'full': H}


def random_null(unsafe, L):
    if L > min(unsafe.shape):
        return 0.0
    return float((window_sums(unsafe, L) <= EPSILON * L * L + 1e-9).mean())


def evaluate(tiles, mode, cfg, scale_kappa):
    us, hs, b1 = [], {m: [] for m in ('grad', 'grad_shadow', 'full')}, []
    within = {m: [] for m in list(hs) + ['b1']}
    sel = {m: {L: [] for L in LS} for m in hs}
    null = {L: [] for L in LS}
    avail = {L: [] for L in LS}
    for t in tiles:
        u = t['unsafe']
        uf = u.ravel()[::10]
        us.append(uf)
        std = get_local_std(t['ori']).ravel()[::10]
        b1.append(std)
        both = 0 < uf.mean() < 1
        if both:
            within['b1'].append(roc_auc_score(uf, std))
        for L in LS:
            null[L].append(random_null(u, L))
            avail[L].append(any_safe_square(u, L))
        for m, H in maps_for(t, mode, cfg, scale_kappa).items():
            hf = H.ravel()[::10]
            hs[m].append(hf)
            if both:
                within[m].append(roc_auc_score(uf, hf))
            _, _, _, (x, y, w, h) = fuse_and_select(H, None, theta=cfg['theta'],
                                                    min_component_area=cfg['min_component_area'])
            fr = float(u[y:y + h, x:x + w].mean()) if w * h else 1.0
            for L in LS:
                sel[m][L].append(min(w, h) >= L and fr <= EPSILON)
    u = np.concatenate(us)
    res = {'n_tiles': len(tiles), 'prevalence': float(u.mean()),
           'b1': {'auroc': float(roc_auc_score(u, np.concatenate(b1))),
                  'prauc': float(average_precision_score(u, np.concatenate(b1))),
                  'within_auroc': float(np.mean(within['b1']))},
           'random_square_null': {L: float(np.mean(null[L])) for L in LS},
           'availability': {L: float(np.mean(avail[L])) for L in LS}}
    for m in hs:
        h = np.concatenate(hs[m])
        res[m] = {'auroc': float(roc_auc_score(u, h)), 'prauc': float(average_precision_score(u, h)),
                  'within_auroc': float(np.mean(within[m])),
                  'safe_landing': {L: float(np.mean(sel[m][L])) for L in LS}}
    return res


def main():
    tiles = load_tiles()
    with open(os.path.join('results', 'normalization_study.json')) as f:
        syn = json.load(f)
    out = {'sun_geometry': 'results/hirise/sun_geometry.json', 'protocols': {}}
    dev = [t for t in tiles if t['split'] == 'DEV']
    test = [t for t in tiles if t['split'] == 'TEST']
    with open(os.path.join('results', 'best_config.json')) as f:
        cfg_paper = json.load(f)['V_bright']
    out['protocols']['transfer_paper'] = {
        'config': cfg_paper,
        'DEV': evaluate(dev, 'p99', cfg_paper, True),
        'TEST': evaluate(test, 'p99', cfg_paper, True),
    }
    print('paper transfer done', flush=True)
    for mode in ('p99', 'fixed_b5'):
        cfg_syn = syn[mode]['config']
        out['protocols'][f'transfer_{mode}'] = {
            'config': cfg_syn,
            'DEV': evaluate(dev, mode, cfg_syn, True),
            'TEST': evaluate(test, mode, cfg_syn, True),
            'TEST_by_site': {s: evaluate([t for t in test if t['site'] == s], mode, cfg_syn, True)
                             for s in sorted({t['site'] for t in test})},
        }
        print(mode, 'transfer TEST', {k: (v['auroc'], v['prauc']) if isinstance(v, dict) and 'auroc' in v else None
                                      for k, v in out['protocols'][f'transfer_{mode}']['TEST'].items()}, flush=True)
        tune_tiles = []
        for s in sorted({t['site'] for t in dev}):
            tune_tiles += [t for t in dev if t['site'] == s][:5]
        # Same staged tuning as on synthetic data; kappa is tuned in this image's grey levels directly.
        cfg_h = tune(mode, [{'ori': t['ori'], 'unsafe': t['unsafe']} for t in tune_tiles],
                     elevation_deg=float(np.mean([t['elev'] for t in tune_tiles]))) if mode == 'p99' else \
            tune_with_elev(mode, tune_tiles)
        out['protocols'][f'hirise_{mode}'] = {
            'config': cfg_h,
            'TEST': evaluate(test, mode, cfg_h, False),
            'TEST_by_site': {s: evaluate([t for t in test if t['site'] == s], mode, cfg_h, False)
                             for s in sorted({t['site'] for t in test})},
        }
        print(mode, 'hirise-tuned', cfg_h, flush=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('wrote', OUT)


def tune_with_elev(mode, tiles):
    """tune() with each tile's own sun elevation for g_ref (fixed_b5 only)."""
    import run_normalization_study as rns
    orig = rns.gradient_hazard
    elev = {id(t['ori']): t['elev'] for t in tiles}
    rns.gradient_hazard = lambda img, m, e=30.0: orig(img, m, elev.get(id(img), e))
    try:
        return rns.tune(mode, [{'ori': t['ori'], 'unsafe': t['unsafe']} for t in tiles])
    finally:
        rns.gradient_hazard = orig


if __name__ == '__main__':
    main()
