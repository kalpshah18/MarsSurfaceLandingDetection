"""Sensitivity sweeps on TEST (reporting only; nothing here selects a configuration).

    python run_review_sweeps.py sens   # tau x theta (and A_min) sensitivity
    python run_review_sweeps.py sun    # sun elevation / azimuth re-render sweep
    python run_review_sweeps.py sunphys  # same, physical cast shadows, no painted shadow

The sun sweep re-renders each TEST tile from its stored DTM with the shading code of
generate_synthetic.py (Lambertian, unit albedo, plus the generator's sun-independent
"painted" shadow: crater pixels deeper than 10 m are clamped to intensity <= 30).
At azimuth 45 deg / elevation 30 deg the re-render must equal the stored images; the
script checks this before sweeping.

Outputs: results/review/sensitivity.json, results/review/sun_sweep.json
"""
import json
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from footprint_metrics import EPSILON
from pipeline import augment_shadow, compute_gradient_hazard, fuse_and_select
from review_common import (HAZARD_TERRAINS, boot_ci, frac_in, iter_tiles, largest_inscribed_square,
                           load_config, min_hazard_square)

OUT_DIR = os.path.join('results', 'review')
EVAL_L = [10, 20, 25]


def render(z, terrain, azimuth_deg, elevation_deg, gsd=1.0):
    """Shading model copied from generate_synthetic.py (which cannot be imported: it
    imports pipeline.run_v0, which no longer exists)."""
    dz_dx = np.gradient(z, axis=1) / gsd
    dz_dy = np.gradient(z, axis=0) / gsd
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    nx, ny, nz = -dz_dx, -dz_dy, np.ones_like(z)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx /= norm; ny /= norm; nz /= norm
    lx, ly, lz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    intensity = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    ori = (intensity * 255).astype(np.uint8)
    if terrain == 'crater':
        ori[z < -10] = np.minimum(ori[z < -10], 30)
    return ori


def render_physical(z, azimuth_deg, elevation_deg, gsd=1.0):
    """Lambertian shading (same model as render) WITHOUT the painted crater shadow, plus
    cast shadows from a height-field ray march toward the sun (shadowed pixels -> 0)."""
    dz_dx = np.gradient(z, axis=1) / gsd
    dz_dy = np.gradient(z, axis=0) / gsd
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    nx, ny, nz = -dz_dx, -dz_dy, np.ones_like(z)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    lx, ly, lz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    intensity = np.clip((nx * lx + ny * ly + nz * lz) / norm, 0, 1)
    H, W = z.shape
    shadow = np.zeros(z.shape, dtype=bool)
    tan_el = np.tan(el)
    t_max = int(np.ceil((z.max() - z.min()) / (tan_el * gsd))) + 1
    ys, xs = np.mgrid[0:H, 0:W]
    for t in range(1, t_max + 1):
        dx, dy = int(round(t * np.cos(az))), int(round(t * np.sin(az)))
        xx, yy = xs + dx, ys + dy
        inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        zz = np.full(z.shape, -np.inf)
        zz[inside] = z[yy[inside], xx[inside]]
        shadow |= zz > z + t * gsd * tan_el
    intensity[shadow] = 0.0
    return (intensity * 255).astype(np.uint8), shadow


def evaluate(tiles, theta, tau, a_min, mca, image_key='ori'):
    """Metrics for grad and grad_shadow under one configuration."""
    us, hs = [], {'grad': [], 'grad_shadow': []}
    per = {'grad': [], 'grad_shadow': []}
    sub = slice(None, None, 10)
    for t in tiles:
        ori, unsafe = t[image_key], t['unsafe']
        G = compute_gradient_hazard(ori)
        G_aug, _ = augment_shadow(ori, G, tau=tau, a_min=a_min)
        us.append(unsafe.flatten()[sub])
        for m, Gm in (('grad', G), ('grad_shadow', G_aug)):
            s, _, filt, (x, y, w, h) = fuse_and_select(Gm, None, theta=theta, min_component_area=mca)
            H = 1.0 - s
            hs[m].append(H.flatten()[sub])
            sx, sy, side = largest_inscribed_square(filt, H)
            r = {'terrain': t['terrain'], 'mr_short': min(w, h), 'mr_frac': frac_in(unsafe, x, y, w, h),
                 'ls_side': side, 'ls_frac': frac_in(unsafe, sx, sy, side, side)}
            for L in EVAL_L:
                sq = min_hazard_square(filt, H, L)
                r[f'la_{L}'] = float('nan') if sq is None else frac_in(unsafe, sq[0], sq[1], L, L)
            per[m].append(r)
    u = np.concatenate(us)
    out = {}
    for m in per:
        h = np.concatenate(hs[m])
        res = {'auroc': float(roc_auc_score(u, h)), 'prauc': float(average_precision_score(u, h))}
        for grp, keep in (('all', None), ('hazard_bearing', HAZARD_TERRAINS)):
            rows = [r for r in per[m] if keep is None or r['terrain'] in keep]
            g = {}
            for L in EVAL_L:
                for sel, fit, fr in (
                        ('maxrect', lambda r: r['mr_short'] >= L, lambda r: r['mr_frac']),
                        ('lsquare', lambda r: r['ls_side'] >= L, lambda r: r['ls_frac']),
                        ('laware', lambda r: not np.isnan(r[f'la_{L}']), lambda r: r[f'la_{L}'])):
                    safe = [fit(r) and fr(r) <= EPSILON for r in rows]
                    fs = [fit(r) and fr(r) > EPSILON for r in rows]
                    g[f'{sel}_safe_{L}'] = float(np.mean(safe))
                    g[f'{sel}_safe_{L}_ci'] = boot_ci(safe)
                    g[f'{sel}_falsesafe_{L}'] = float(np.mean(fs))
            res[grp] = g
        out[m] = res
    return out


def main(mode):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_config()['V_grad_shadow']
    tiles = list(iter_tiles('TEST'))
    if mode == 'sens':
        grid = [(theta, tau, 100) for tau in (20, 40, 60, 80, 100) for theta in (0.1, 0.2, 0.3, 0.4)]
        grid += [(0.2, 40, 500), (0.2, 60, 500)]
        res = []
        for theta, tau, a_min in grid:
            r = evaluate(tiles, theta, tau, a_min, cfg['min_component_area'])
            res.append({'theta': theta, 'tau': tau, 'a_min': a_min, **r})
            print(theta, tau, a_min, round(r['grad_shadow']['auroc'], 3),
                  r['grad_shadow']['hazard_bearing']['maxrect_safe_10'], flush=True)
        path = os.path.join(OUT_DIR, 'sensitivity.json')
        with open(path, 'w') as f:
            json.dump({'note': 'TEST-set sensitivity; reporting only, not used to select parameters',
                       'baseline': {'theta': cfg['theta'], 'tau': cfg['tau'], 'a_min': cfg['a_min']},
                       'runs': res}, f, indent=1)
    elif mode == 'sun':
        mism = 0
        for t in tiles:
            mism += int((render(t['dtm'], t['terrain'], 45, 30) != t['ori']).sum())
        if mism:
            sys.exit(f'Re-render at az 45 / el 30 differs from stored images in {mism} pixels; aborting.')
        print('Re-render check passed: az 45 / el 30 reproduces all stored TEST images exactly.')
        configs = [(45, el) for el in (15, 30, 45, 60)] + [(az, 30) for az in (135, 225, 315)]
        res = []
        for az, el in configs:
            for t in tiles:
                t['ori_r'] = render(t['dtm'], t['terrain'], az, el)
            r = evaluate(tiles, cfg['theta'], cfg['tau'], cfg['a_min'], cfg['min_component_area'], 'ori_r')
            res.append({'azimuth': az, 'elevation': el, **r})
            print(az, el, round(r['grad']['auroc'], 3), round(r['grad_shadow']['auroc'], 3), flush=True)
        path = os.path.join(OUT_DIR, 'sun_sweep.json')
        with open(path, 'w') as f:
            json.dump({'note': 'TEST tiles re-rendered from stored DTMs; painted crater shadow is '
                               'sun-independent as in generate_synthetic.py',
                       'config': cfg, 'runs': res}, f, indent=1)
    elif mode == 'sunphys':
        configs = [(45, el) for el in (15, 30, 45, 60)] + [(az, 30) for az in (135, 225, 315)]
        res = []
        for az, el in configs:
            cast = []
            for t in tiles:
                t['ori_r'], sh = render_physical(t['dtm'], az, el)
                if t['terrain'] == 'crater':
                    cast.append(float(sh.mean()))
            r = evaluate(tiles, cfg['theta'], cfg['tau'], cfg['a_min'], cfg['min_component_area'], 'ori_r')
            res.append({'azimuth': az, 'elevation': el, 'crater_mean_cast_shadow_frac': float(np.mean(cast)), **r})
            print(az, el, round(np.mean(cast), 4), round(r['grad']['auroc'], 3), round(r['grad_shadow']['auroc'], 3),
                  flush=True)
        path = os.path.join(OUT_DIR, 'sun_sweep_physical.json')
        with open(path, 'w') as f:
            json.dump({'note': 'TEST tiles re-rendered from stored DTMs with Lambertian shading and '
                               'ray-marched cast shadows; NO painted crater shadow',
                       'config': cfg, 'runs': res}, f, indent=1)
    else:
        sys.exit('usage: run_review_sweeps.py sens|sun|sunphys')
    print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '')
