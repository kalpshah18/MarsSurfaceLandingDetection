"""Photometric robustness ablation: how V_full degrades when the image model departs from the one the
cues assume (Lambertian, unit albedo, no skylight).

Each TEST tile is re-rendered from its DTM (sun azimuth 45, elevation 30, nadir view) under:
  baseline        Lambertian, unit albedo, black shadows (the dataset's own model)
  sky_a           ambient skylight equal to a fraction a of the direct illumination of level ground:
                  I = rho * (lit * R + a * R_level) / (1 + a), a in {0.1, 0.2, 0.3, 0.5, 1.0};
                  shadows then sit at about 255 sin(30 deg) a / (1 + a) grey levels (12 to 64)
  albedo_smooth_c smooth albedo field rho = 1 + c * N, N a unit-variance Gaussian random field
                  (correlation sigma 15 m), c in {0.1, 0.2, 0.3}
  albedo_patch_c  two-tone albedo units (thresholded field, sigma 40 m): rho = 1 +- c
  lunar_lambert_L Lunar-Lambert photometry (McEwen 1991), I = 2L cos_i / (cos_i + cos_e) + (1 - L) cos_i,
                  L in {0.5, 1.0}; cos_e from the surface normal and a nadir view
  combined        Lunar-Lambert L = 0.5, skylight a = 0.15, smooth albedo c = 0.2
Images are scaled so that a level, unit-albedo, lit surface keeps the brightness it has in the
baseline (255 sin 30 deg), then clipped to 8 bits.

Two protocols per variant:
  frozen   parameters as selected on the baseline DEV tiles (results/best_config.json)
  retuned  parameters re-selected on the 8 DEV tiles rendered under the same variant

Reported: pooled AUROC / PR-AUC for V_grad, V_grad+shadow, V_full and B1, the mean within-tile AUROC of
V_full on crater tiles, and V_full Safe Landing at L = 10, 25, 50, 100 m.

    python run_photometric_robustness.py
Output: results/photometric_robustness.json
"""
import json
import os

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from footprint_metrics import rates
from pipeline import augment_shadow, compute_brightness_hazard, compute_gradient_hazard, fuse_and_select
from render import cast_shadow, sun_vector
from review_common import iter_tiles, load_config
from run_evaluation import get_local_std
from run_normalization_study import staged_tune

AZ, EL = 45.0, 30.0
LS = [10, 25, 50, 100]
OUT = os.path.join('results', 'photometric_robustness.json')
LEVEL = np.sin(np.radians(EL))  # brightness of level, unit-albedo, lit ground in the baseline


def geometry(z):
    """cos(incidence), cos(emission) for a nadir view, and the cast-shadow mask."""
    gx, gy = np.gradient(z, axis=1), np.gradient(z, axis=0)
    lx, ly, lz = sun_vector(AZ, EL)
    norm = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    cos_i = (-gx * lx - gy * ly + lz) / norm
    cos_e = 1.0 / norm
    return cos_i, cos_e, cast_shadow(z, AZ, EL)


def albedo_field(shape, seed, kind, c):
    if c == 0:
        return np.ones(shape)
    rng = np.random.default_rng(seed)
    sigma = 15 if kind == 'smooth' else 40
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    n = (n - n.mean()) / (n.std() + 1e-9)
    if kind == 'smooth':
        return np.clip(1.0 + c * n, 0.2, 1.8)
    return np.where(n > 0, 1.0 + c, 1.0 - c)


def render_variant(geo, seed, v):
    cos_i, cos_e, shadow = geo
    ci = np.clip(cos_i, 0.0, 1.0)
    L = v.get('L')
    if L is None:
        refl = ci
        level = LEVEL
    else:
        refl = 2 * L * ci / (ci + cos_e + 1e-9) + (1 - L) * ci
        level = 2 * L * LEVEL / (LEVEL + 1.0) + (1 - L) * LEVEL
    lit = (~shadow).astype(np.float64)
    a = v.get('sky', 0.0)
    rho = albedo_field(cos_i.shape, seed, v.get('albedo_kind', 'smooth'), v.get('albedo_c', 0.0))
    I = rho * (lit * refl + a * level) / (1.0 + a)
    # keep level, lit, unit-albedo ground at the baseline brightness (255 sin 30 deg)
    scale = 255.0 * LEVEL / level
    return np.clip(I * scale, 0, 255).astype(np.uint8)


VARIANTS = {'baseline': {}}
for a in (0.1, 0.2, 0.3, 0.5, 1.0):
    VARIANTS[f'sky_{a}'] = {'sky': a}
for c in (0.1, 0.2, 0.3):
    VARIANTS[f'albedo_smooth_{c}'] = {'albedo_kind': 'smooth', 'albedo_c': c}
    VARIANTS[f'albedo_patch_{c}'] = {'albedo_kind': 'patch', 'albedo_c': c}
for L in (0.5, 1.0):
    VARIANTS[f'lunar_lambert_{L}'] = {'L': L}
VARIANTS['combined'] = {'L': 0.5, 'sky': 0.15, 'albedo_kind': 'smooth', 'albedo_c': 0.2}


def evaluate(tiles, images, cfg):
    us, hs, b1, crater_within = [], {m: [] for m in ('grad', 'grad_shadow', 'full')}, [], []
    boxes = []
    for t, img in zip(tiles, images):
        u = t['unsafe']
        uf = u.ravel()[::10]
        us.append(uf)
        b1.append(get_local_std(img).ravel()[::10])
        G = compute_gradient_hazard(img, EL)
        G_aug, _ = augment_shadow(img, G, tau=cfg['tau'], a_min=cfg['a_min'])
        H = np.maximum(G_aug, compute_brightness_hazard(img, phi_deg=cfg['phi_deg'], ksize=cfg['ksize'],
                                                        sun_elevation_deg=EL))
        for m, M in (('grad', G), ('grad_shadow', G_aug), ('full', H)):
            hs[m].append(M.ravel()[::10])
        if t['terrain'] == 'crater':
            crater_within.append(roc_auc_score(uf, H.ravel()[::10]))
        _, _, _, (x, y, w, h) = fuse_and_select(H, None, theta=cfg['theta'], min_component_area=cfg['min_component_area'])
        boxes.append({'w': w, 'h': h, 'frac_unsafe': float(u[y:y + h, x:x + w].mean()) if w * h else float('nan'),
                      'terrain': t['terrain']})
    u = np.concatenate(us)
    res = {m: {'auroc': float(roc_auc_score(u, np.concatenate(v))),
               'prauc': float(average_precision_score(u, np.concatenate(v)))} for m, v in hs.items()}
    res['b1'] = {'auroc': float(roc_auc_score(u, np.concatenate(b1))),
                 'prauc': float(average_precision_score(u, np.concatenate(b1)))}
    res['full']['crater_within_auroc'] = float(np.mean(crater_within))
    res['full']['safe_landing'] = {L: rates(boxes, L)['safe_landing'] for L in LS}
    res['full']['safe_landing_by_terrain_25'] = {
        ter: rates([b for b in boxes if b['terrain'] == ter], 25)['safe_landing']
        for ter in ('crater', 'rocky', 'smooth', 'dunes')}
    return res


def main():
    with open('splits.json') as f:
        dev = json.load(f)['DEV']
    ids = []
    for site in ('crater_1', 'rocky_1', 'smooth_1', 'dunes_1'):
        ids += [x for x in dev if x.startswith(site + '_')][:2]
    dev_tiles = [t for t in iter_tiles('DEV') if t['tile_id'] in ids]
    test_tiles = list(iter_tiles('TEST'))
    print('computing geometry', flush=True)
    dev_geo = [geometry(t['dtm']) for t in dev_tiles]
    test_geo = [geometry(t['dtm']) for t in test_tiles]
    # the baseline render must reproduce the stored images
    base = [render_variant(g, i, {}) for i, g in enumerate(test_geo)]
    diff = max(int(np.abs(b.astype(int) - t['ori'].astype(int)).max()) for b, t in zip(base, test_tiles))
    print('baseline max abs difference to stored images:', diff, flush=True)
    cfg_frozen = load_config()['V_bright']
    out = {'baseline_max_abs_diff': diff, 'frozen_config': cfg_frozen, 'variants': {}}
    for name, v in VARIANTS.items():
        test_imgs = [render_variant(g, 1000 + i, v) for i, g in enumerate(test_geo)]
        dev_imgs = [render_variant(g, 2000 + i, v) for i, g in enumerate(dev_geo)]
        cfg_re = staged_tune([{'ori': im, 'unsafe': t['unsafe']} for im, t in zip(dev_imgs, dev_tiles)], 'physical')
        r = {'spec': v,
             'shadow_grey_level': float(np.median([im[g[2]].mean() for im, g in zip(test_imgs, test_geo) if g[2].any()])),
             'frozen': evaluate(test_tiles, test_imgs, cfg_frozen),
             'retuned': {'config': cfg_re, **evaluate(test_tiles, test_imgs, cfg_re)}}
        out['variants'][name] = r
        f, rt = r['frozen'], r['retuned']
        print(f"{name:20s} shadow {r['shadow_grey_level']:5.1f} | frozen full {f['full']['auroc']:.3f}/{f['full']['prauc']:.3f} "
              f"crater {f['full']['crater_within_auroc']:.3f} SL {[round(x, 3) for x in f['full']['safe_landing'].values()]} "
              f"| gs {f['grad_shadow']['auroc']:.3f} g {f['grad']['auroc']:.3f} b1 {f['b1']['auroc']:.3f} "
              f"| retuned full {rt['full']['auroc']:.3f}/{rt['full']['prauc']:.3f} SL {[round(x, 3) for x in rt['full']['safe_landing'].values()]} "
              f"phi {cfg_re['phi_deg']} tau {cfg_re['tau']}", flush=True)
        with open(OUT, 'w') as fo:
            json.dump(out, fo, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
