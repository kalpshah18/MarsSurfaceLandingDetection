"""Evaluate the pipeline (V_grad, V_grad+shadow, V_full) on real HiRISE DTM/ortho tiles (paper Section 6).

Tiles: results/hirise/tiles_manifest.csv (hirise_make_tiles.py), 8 sites x 20 tiles: 4 DEV sites
(Gale, Elysium, Holden, Mawrth) and 4 TEST sites (Jezero, Eberswalde, Arcadia, Meridiani).
Labels: slope > 15 deg over 5 posts of the HiRISE DTM. Sun elevation of each observation:
results/hirise/sun_geometry.json (PDS RDR labels); both physical constants use it.

Protocols:
  transfer      configuration tuned on the synthetic DEV tiles (results/best_config.json), unchanged.
  transfer_p99  as transfer, with the per-tile p99 gradient normalization (ablation, config from
                results/normalization_study.json).
  hirise        configuration re-tuned on 5 tiles per HiRISE DEV site (20 tiles) with the same staged
                procedure and grids, then evaluated once on the TEST sites.

Reported: pooled AUROC / PR-AUC, mean within-tile AUROC, B1 (local std) and chance, maximal-rectangle
Safe Landing at L = 10, 20, 25, 50 next to the random-square null (probability that a uniformly placed
L x L square has <= 5% unsafe pixels) and site availability, per split and per TEST site.

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
from run_normalization_study import gradient_hazard, staged_tune

DATA_DIR = os.environ.get('HIRISE_DATA_DIR', os.path.join('data', 'raw', 'hirise'))
OUT = os.path.join('results', 'hirise', 'evaluation_v2.json')
LS = [10, 20, 25, 50]


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


def maps_for(t, mode, cfg):
    G = gradient_hazard(t['ori'], mode, t['elev'])
    G_aug, _ = augment_shadow(t['ori'], G, tau=cfg['tau'], a_min=cfg['a_min'])
    B = compute_brightness_hazard(t['ori'], phi_deg=cfg['phi_deg'], ksize=cfg['ksize'], sun_elevation_deg=t['elev'])
    return {'grad': G, 'grad_shadow': G_aug, 'full': np.maximum(G_aug, B)}


def random_null(unsafe, L):
    return float((window_sums(unsafe, L) <= EPSILON * L * L + 1e-9).mean())


def evaluate(tiles, mode, cfg):
    us, b1 = [], []
    hs = {m: [] for m in ('grad', 'grad_shadow', 'full')}
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
        for m, H in maps_for(t, mode, cfg).items():
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


def run(tiles, mode, cfg):
    dev = [t for t in tiles if t['split'] == 'DEV']
    test = [t for t in tiles if t['split'] == 'TEST']
    return {'config': cfg, 'gradient_normalization': mode,
            'DEV': evaluate(dev, mode, cfg), 'TEST': evaluate(test, mode, cfg),
            'TEST_by_site': {s: evaluate([t for t in test if t['site'] == s], mode, cfg)
                             for s in sorted({t['site'] for t in test})}}


def main():
    tiles = load_tiles()
    with open(os.path.join('results', 'best_config.json')) as f:
        cfg_syn = json.load(f)['V_bright']
    with open(os.path.join('results', 'normalization_study.json')) as f:
        cfg_p99 = json.load(f)['p99']['config']
    out = {'sun_geometry': 'results/hirise/sun_geometry.json', 'protocols': {}}
    out['protocols']['transfer'] = run(tiles, 'physical', cfg_syn)
    print('transfer', round(out['protocols']['transfer']['TEST']['full']['auroc'], 3), flush=True)
    out['protocols']['transfer_p99'] = run(tiles, 'p99', cfg_p99)
    print('transfer_p99', round(out['protocols']['transfer_p99']['TEST']['full']['auroc'], 3), flush=True)
    dev = [t for t in tiles if t['split'] == 'DEV']
    tune_tiles = []
    for s in sorted({t['site'] for t in dev}):
        tune_tiles += [t for t in dev if t['site'] == s][:5]
    cfg_h = staged_tune(tune_tiles, 'physical')
    out['protocols']['hirise'] = run(tiles, 'physical', cfg_h)
    print('hirise', cfg_h, round(out['protocols']['hirise']['TEST']['full']['auroc'], 3), flush=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
