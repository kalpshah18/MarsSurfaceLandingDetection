"""Gradient-normalization study (reviewer suggestion): per-tile p99 vs a fixed physical constant.

Variants of the gradient hazard G (everything else as in pipeline.py):
  p99         G = min(1, g / p99(g))                       (current paper)
  fixed       G = min(1, g / g_ref)                        (fixed constant, raw 3x3 Sobel)
  fixed_b5    G = min(1, g5 / g_ref), g5 = Sobel of a 5x5 box-filtered image
              (gradient measured at the 5 m label baseline, the scale g_ref is derived for)
  p99_b5      control: g5 as in fixed_b5, normalized per tile by its p99 (isolates the blur)

g_ref is the Sobel response (8 x intensity change per pixel) of a slope change equal to the label
threshold (15 deg) spread over the label baseline (5 px), for a Lambertian surface whose level-ground
brightness is I_ref = median(image) at sun elevation e:
    g_ref = 8 * I_ref * cot(e) * sin(15 deg) / 5
which is about 91 for the synthetic tiles (I_ref ~ 127, e = 30 deg).

Each variant is tuned with the same DEV procedure as run_tuning.py (8 DEV tiles, two per terrain,
safe-mask F1), with the theta grid widened to {0.05, 0.1, 0.2, 0.35, 0.5} for every variant, and
then evaluated once on TEST: pooled AUROC / PR-AUC, maximal-rectangle footprint sweep, per terrain.

    python run_normalization_study.py          # synthetic dataset
Output: results/normalization_study.json
"""
import itertools
import json
import os

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from footprint_metrics import rates
from pipeline import augment_shadow, compute_brightness_hazard, fuse_and_select
from review_common import TERRAINS, iter_tiles

SUN_ELEVATION = 30.0
SLOPE_DEG = 15.0
BASELINE_PX = 5
THETAS = [0.05, 0.1, 0.2, 0.35, 0.5]
MIN_AREAS = [100, 500, 1500]
TAUS, AMINS = [20, 40, 60, 80], [100, 500]
KAPPAS, KSIZES = [20, 30, 40, 50, 70], [5, 9, 15]
FINE_L = list(range(5, 61))
OUT = os.path.join('results', 'normalization_study.json')


def g_ref_for(image_gray, elevation_deg=SUN_ELEVATION):
    I_ref = float(np.median(image_gray))
    e = np.radians(elevation_deg)
    return 8.0 * I_ref / np.tan(e) * np.sin(np.radians(SLOPE_DEG)) / BASELINE_PX


def gradient_hazard(image_gray, mode, elevation_deg=SUN_ELEVATION):
    I = image_gray.astype(np.float32)
    if mode in ('fixed_b5', 'p99_b5'):
        I = cv2.blur(I, (BASELINE_PX, BASELINE_PX))
    Ix = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    g = np.sqrt(Ix ** 2 + Iy ** 2)
    if mode in ('p99', 'p99_b5'):
        nz = g[g > 0]
        ref = np.percentile(nz, 99) if len(nz) else 1.0
    else:
        ref = g_ref_for(image_gray, elevation_deg)
    return np.minimum(1.0, g / max(ref, 1e-8))


def hazards(ori, mode, cfg, elevation_deg=SUN_ELEVATION):
    """Return {'grad': G, 'grad_shadow': G_aug, 'full': H} for one image."""
    G = gradient_hazard(ori, mode, elevation_deg)
    G_aug, _ = augment_shadow(ori, G, tau=cfg['tau'], a_min=cfg['a_min'])
    H = np.maximum(G_aug, compute_brightness_hazard(ori, kappa=cfg['kappa'], ksize=cfg['ksize']))
    return {'grad': G, 'grad_shadow': G_aug, 'full': H}


def tune(mode, tiles, elevation_deg=SUN_ELEVATION):
    """Same staged procedure as run_tuning.py; returns the selected configuration."""
    data = [(t['ori'], t['unsafe'], gradient_hazard(t['ori'], mode, elevation_deg)) for t in tiles]

    def f1_of(make_H, theta, mca):
        preds, labels = [], []
        for ori, unsafe, G in data:
            _, _, safe, _ = fuse_and_select(make_H(ori, G), None, theta=theta, min_component_area=mca)
            preds.append(safe.ravel()[::100])
            labels.append((~unsafe).ravel()[::100])
        return f1_score(np.concatenate(labels), np.concatenate(preds))

    best = max(((f1_of(lambda o, G: G, th, mca), th, mca) for th, mca in itertools.product(THETAS, MIN_AREAS)))
    cfg = {'theta': best[1], 'min_component_area': best[2]}
    best = max(((f1_of(lambda o, G, ta=ta, am=am: augment_shadow(o, G, tau=ta, a_min=am)[0],
                       cfg['theta'], cfg['min_component_area']), ta, am)
                for ta, am in itertools.product(TAUS, AMINS)))
    cfg.update(tau=best[1], a_min=best[2])
    best = max(((f1_of(lambda o, G, ka=ka, ks=ks: np.maximum(
        augment_shadow(o, G, tau=cfg['tau'], a_min=cfg['a_min'])[0],
        compute_brightness_hazard(o, kappa=ka, ksize=ks)), cfg['theta'], cfg['min_component_area']), ka, ks)
                for ka, ks in itertools.product(KAPPAS, KSIZES)))
    cfg.update(kappa=best[1], ksize=best[2])
    return cfg


def evaluate(mode, cfg, tiles, elevation_deg=SUN_ELEVATION, terrains=TERRAINS, label_key='terrain'):
    us, hs = [], {m: [] for m in ('grad', 'grad_shadow', 'full')}
    boxes = {m: [] for m in hs}
    for t in tiles:
        maps = hazards(t['ori'], mode, cfg, elevation_deg)
        u = t['unsafe']
        us.append(u.ravel()[::10])
        for m, H in maps.items():
            hs[m].append(H.ravel()[::10])
            _, _, _, (x, y, w, h) = fuse_and_select(H, None, theta=cfg['theta'],
                                                    min_component_area=cfg['min_component_area'])
            fr = float(u[y:y + h, x:x + w].mean()) if w * h else float('nan')
            boxes[m].append({'group': t[label_key], 'w': w, 'h': h, 'frac_unsafe': fr})
    u = np.concatenate(us)
    out = {'config': cfg, 'methods': {}}
    for m in hs:
        h = np.concatenate(hs[m])
        b = boxes[m]
        fine = {L: rates(b, L) for L in FINE_L}
        lstar = max([L for L in FINE_L if fine[L]['safe_landing'] >= 0.5], default=None)
        groups = {}
        for g in terrains:
            bg = [x for x in b if x['group'] == g]
            if bg:
                groups[g] = {'n': len(bg), 'safe_landing': {L: rates(bg, L)['safe_landing'] for L in (5, 10, 20, 25)},
                             'max_min_side': max(min(x['w'], x['h']) for x in bg),
                             'mean_frac_unsafe': float(np.nanmean([x['frac_unsafe'] for x in bg]))}
        out['methods'][m] = {
            'auroc': float(roc_auc_score(u, h)), 'prauc': float(average_precision_score(u, h)),
            'safe_landing': {L: fine[L]['safe_landing'] for L in (5, 10, 15, 20, 25, 30, 40, 50)},
            'square_fit_25': fine[25]['square_fit'],
            'L_star': lstar,
            'first_zero_L': min([L for L in FINE_L if fine[L]['safe_landing'] == 0], default=None),
            'max_min_side': max(min(x['w'], x['h']) for x in b),
            'mean_frac_unsafe': float(np.nanmean([x['frac_unsafe'] for x in b])),
            'by_group': groups,
        }
    out['prevalence'] = float(u.mean())
    return out


def main():
    dev_ids = []
    with open('splits.json') as f:
        dev = json.load(f)['DEV']
    for site in ('crater_1', 'rocky_1', 'smooth_1', 'dunes_1'):
        dev_ids += [x for x in dev if x.startswith(site + '_')][:2]
    dev_tiles = [t for t in iter_tiles('DEV') if t['tile_id'] in dev_ids]
    test_tiles = list(iter_tiles('TEST'))
    res = {}
    import sys
    modes = sys.argv[1:] or ['p99', 'fixed', 'fixed_b5', 'p99_b5']
    if os.path.exists(OUT):
        with open(OUT) as f:
            res = json.load(f)
    for mode in modes:
        cfg = tune(mode, dev_tiles)
        print(mode, 'tuned', cfg, flush=True)
        res[mode] = evaluate(mode, cfg, test_tiles)
        for m, r in res[mode]['methods'].items():
            print(f"  {m}: AUROC {r['auroc']:.3f} PR-AUC {r['prauc']:.3f} SL {r['safe_landing']} "
                  f"L* {r['L_star']} max side {r['max_min_side']}", flush=True)
    os.makedirs('results', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
