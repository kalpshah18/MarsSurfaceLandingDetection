"""Gradient-normalization ablation for V_full (paper Section 5.2).

Variants of the gradient hazard G; the shadow and brightness cues are those of pipeline.py:
  p99        raw 3x3 Sobel, divided by the per-tile 99th percentile (earlier versions of the pipeline)
  fixed_raw  raw 3x3 Sobel, divided by the physical constant g_ref (pipeline.gradient_reference)
  p99_b5     Sobel of the 5x5 box-filtered image, divided by the per-tile 99th percentile
  physical   Sobel of the 5x5 box-filtered image, divided by g_ref (pipeline default)

Each variant is tuned with the staged DEV procedure of run_tuning.py (same 8 DEV tiles, same grids)
and evaluated once on TEST: pooled AUROC / PR-AUC and the maximal-rectangle Safe Landing rate.

    python run_normalization_study.py
Output: results/normalization_study.json
"""
import itertools
import json
import os

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from footprint_metrics import rates
from pipeline import (BASELINE_PX, augment_shadow, compute_brightness_hazard, compute_gradient_hazard,
                      fuse_and_select, gradient_reference)
from review_common import iter_tiles

THETAS = [0.05, 0.1, 0.2, 0.35, 0.5]
MIN_AREAS = [100, 500, 1500]
TAUS, AMINS = [20, 40, 60, 80], [100, 500]
PHIS, KSIZES = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0], [5, 9, 15]
LS = [5, 10, 15, 20, 25, 30, 40, 50, 100, 200]
OUT = os.path.join('results', 'normalization_study.json')


def gradient_hazard(img, mode, elev=30.0):
    if mode == 'physical':
        return compute_gradient_hazard(img, elev)
    if mode == 'p99':
        return compute_gradient_hazard(img, elev, normalization='p99')
    I = img.astype(np.float32)
    if mode == 'p99_b5':
        I = cv2.blur(I, (BASELINE_PX, BASELINE_PX), borderType=cv2.BORDER_REPLICATE)
    g = np.hypot(cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE),
                 cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE))
    ref = np.percentile(g[g > 0], 99) if mode == 'p99_b5' else gradient_reference(img, elev)
    return np.minimum(1.0, g / max(float(ref), 1e-8))


def staged_tune(tiles, mode):
    """Staged DEV tuning as in run_tuning.py; tiles are dicts with ori, unsafe and optionally elev."""
    data = [(t['ori'], t['unsafe'], t.get('elev', 30.0), gradient_hazard(t['ori'], mode, t.get('elev', 30.0)))
            for t in tiles]

    def f1_of(make_H, theta, mca):
        preds, labels = [], []
        for ori, unsafe, elev, G in data:
            _, _, safe, _ = fuse_and_select(make_H(ori, G, elev), None, theta=theta, min_component_area=mca)
            preds.append(safe.ravel()[::100])
            labels.append((~unsafe).ravel()[::100])
        return f1_score(np.concatenate(labels), np.concatenate(preds))

    best = max((f1_of(lambda o, G, e: G, th, mca), th, mca) for th, mca in itertools.product(THETAS, MIN_AREAS))
    cfg = {'theta': best[1], 'min_component_area': best[2]}
    best = max((f1_of(lambda o, G, e, ta=ta, am=am: augment_shadow(o, G, tau=ta, a_min=am)[0],
                      cfg['theta'], cfg['min_component_area']), ta, am)
               for ta, am in itertools.product(TAUS, AMINS))
    cfg.update(tau=best[1], a_min=best[2])
    best = max((f1_of(lambda o, G, e, ph=ph, ks=ks: np.maximum(
        augment_shadow(o, G, tau=cfg['tau'], a_min=cfg['a_min'])[0],
        compute_brightness_hazard(o, phi_deg=ph, ksize=ks, sun_elevation_deg=e)),
        cfg['theta'], cfg['min_component_area']), ph, ks)
               for ph, ks in itertools.product(PHIS, KSIZES))
    cfg.update(phi_deg=best[1], ksize=best[2])
    return cfg


def full_map(img, mode, cfg, elev=30.0):
    G = gradient_hazard(img, mode, elev)
    G_aug, _ = augment_shadow(img, G, tau=cfg['tau'], a_min=cfg['a_min'])
    return np.maximum(G_aug, compute_brightness_hazard(img, phi_deg=cfg['phi_deg'], ksize=cfg['ksize'],
                                                       sun_elevation_deg=elev))


def evaluate(tiles, mode, cfg):
    us, hs, boxes = [], [], []
    for t in tiles:
        H = full_map(t['ori'], mode, cfg)
        u = t['unsafe']
        us.append(u.ravel()[::10])
        hs.append(H.ravel()[::10])
        _, _, _, (x, y, w, h) = fuse_and_select(H, None, theta=cfg['theta'], min_component_area=cfg['min_component_area'])
        boxes.append({'w': w, 'h': h, 'frac_unsafe': float(u[y:y + h, x:x + w].mean()) if w * h else float('nan')})
    u, h = np.concatenate(us), np.concatenate(hs)
    return {'config': cfg, 'auroc': float(roc_auc_score(u, h)), 'prauc': float(average_precision_score(u, h)),
            'safe_landing': {L: rates(boxes, L)['safe_landing'] for L in LS},
            'max_min_side': int(max(min(b['w'], b['h']) for b in boxes))}


def main():
    with open('splits.json') as f:
        dev = json.load(f)['DEV']
    ids = []
    for site in ('crater_1', 'rocky_1', 'smooth_1', 'dunes_1'):
        ids += [x for x in dev if x.startswith(site + '_')][:2]
    dev_tiles = [t for t in iter_tiles('DEV') if t['tile_id'] in ids]
    test_tiles = list(iter_tiles('TEST'))
    res = {}
    for mode in ('p99', 'fixed_raw', 'p99_b5', 'physical'):
        cfg = staged_tune(dev_tiles, mode)
        res[mode] = evaluate(test_tiles, mode, cfg)
        print(mode, cfg, round(res[mode]['auroc'], 3), round(res[mode]['prauc'], 3),
              {L: round(v, 3) for L, v in res[mode]['safe_landing'].items()}, res[mode]['max_min_side'], flush=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
