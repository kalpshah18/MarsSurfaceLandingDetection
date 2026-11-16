"""Where do V_grad / V_grad+shadow fail on crater tiles? (paper Section 5.1)

For each crater TEST tile: where the V_grad+shadow box lies (unsafe fraction, slope,
intensity, fraction of box pixels on sun-facing slopes), within-tile AUROC of the raw
and augmented hazard maps, and how many unsafe crater pixels lie in the true cast
shadow and in the detected dark-region mask.

A pixel is "sun-facing" if its smoothed surface normal makes a smaller angle with the
sun than flat ground does by a margin: cos(incidence) > sin(elevation) + 0.1.

Output: results/review/crater_diagnosis.json
"""
import json
import os

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from generate_synthetic import SUN_AZIMUTH, SUN_ELEVATION
from pipeline import augment_shadow, compute_brightness_hazard, compute_gradient_hazard, fuse_and_select
from render import cast_shadow, sun_vector
from review_common import iter_tiles, load_config

OUT = os.path.join('results', 'review', 'crater_diagnosis.json')


def main():
    cfg = load_config()['V_grad_shadow']
    bcfg = load_config()['V_bright']
    lx, ly, lz = sun_vector(SUN_AZIMUTH, SUN_ELEVATION)
    tiles, g_sunfacing_unsafe, g_other_unsafe, g_safe = [], [], [], []
    n_unsafe = n_unsafe_cast = n_unsafe_mask = 0
    for t in iter_tiles('TEST'):
        if t['terrain'] != 'crater':
            continue
        ori, z, unsafe, slope = t['ori'], t['dtm'], t['unsafe'], t['slope']
        G = compute_gradient_hazard(ori)
        G_aug, mask = augment_shadow(ori, G, tau=cfg['tau'], a_min=cfg['a_min'])
        _, _, _, (x, y, w, h) = fuse_and_select(G_aug, None, theta=cfg['theta'],
                                                min_component_area=cfg['min_component_area'])
        zs = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 3)
        gx, gy = np.gradient(zs, axis=1), np.gradient(zs, axis=0)
        cos_i = (-gx * lx - gy * ly + lz) / np.sqrt(gx ** 2 + gy ** 2 + 1)
        sunfacing = cos_i > np.sin(np.radians(SUN_ELEVATION)) + 0.1
        cast = cast_shadow(z, SUN_AZIMUTH, SUN_ELEVATION)
        box = (slice(y, y + h), slice(x, x + w))
        H_br = np.maximum(G_aug, compute_brightness_hazard(ori, kappa=bcfg['kappa'], ksize=bcfg['ksize']))
        _, _, _, (bx, by, bw, bh) = fuse_and_select(H_br, None, theta=bcfg['theta'],
                                                    min_component_area=bcfg['min_component_area'])
        bbox = (slice(by, by + bh), slice(bx, bx + bw))
        sub = slice(None, None, 10)
        u = unsafe.ravel()[sub]
        tiles.append({
            'tile_id': t['tile_id'],
            'box_frac_unsafe': float(unsafe[box].mean()),
            'box_mean_slope_deg': float(slope[box].mean()),
            'box_mean_intensity': float(ori[box].mean()),
            'box_frac_sunfacing': float(sunfacing[box].mean()),
            'auroc_raw': float(roc_auc_score(u, G.ravel()[sub])),
            'auroc_aug': float(roc_auc_score(u, G_aug.ravel()[sub])),
            'auroc_bright': float(roc_auc_score(u, H_br.ravel()[sub])),
            'bright_box_frac_unsafe': float(unsafe[bbox].mean()),
            'bright_box_frac_sunfacing': float(sunfacing[bbox].mean()),
            'bright_box_mean_slope_deg': float(slope[bbox].mean()),
            'bright_box_min_side': int(min(bw, bh)),
            # |cos(slope aspect - sun azimuth)| averaged over the V_full box: 1 = slope along the
            # sun line (visible to the brightness cue), 0 = slope across it; 0.637 for random aspect.
            'bright_box_aspect_cos': float(np.mean(np.abs(gx[bbox] * np.cos(np.radians(SUN_AZIMUTH)) +
                                                          gy[bbox] * np.sin(np.radians(SUN_AZIMUTH))) /
                                                   (np.hypot(gx[bbox], gy[bbox]) + 1e-9))),
            'unsafe_sunfacing_frac_flagged_by_bright': float((H_br[unsafe & sunfacing] >= 1.0 - bcfg['theta']).mean())
            if (unsafe & sunfacing).any() else float('nan'),
        })
        g_sunfacing_unsafe.append(G[unsafe & sunfacing])
        g_other_unsafe.append(G[unsafe & ~sunfacing & ~cast])
        g_safe.append(G[~unsafe])
        n_unsafe += int(unsafe.sum())
        n_unsafe_cast += int((unsafe & cast).sum())
        n_unsafe_mask += int((unsafe & (mask == 255)).sum())
        print(t['tile_id'], {k: round(v, 3) for k, v in tiles[-1].items() if k != 'tile_id'}, flush=True)

    keys = [k for k in tiles[0] if k != 'tile_id']
    out = {
        'config': cfg,
        'n_tiles': len(tiles),
        'mean': {k: float(np.mean([r[k] for r in tiles])) for k in keys},
        'min': {k: float(np.min([r[k] for r in tiles])) for k in keys},
        'mean_G_raw': {'unsafe_sunfacing': float(np.concatenate(g_sunfacing_unsafe).mean()),
                       'unsafe_other_lit': float(np.concatenate(g_other_unsafe).mean()),
                       'safe': float(np.concatenate(g_safe).mean())},
        'bright_unsafe_boxes_mean_aspect_cos': float(np.mean([r['bright_box_aspect_cos'] for r in tiles
                                                              if r['bright_box_frac_unsafe'] > 0.5])),
        'frac_unsafe_in_cast_shadow': n_unsafe_cast / n_unsafe,
        'frac_unsafe_in_dark_mask': n_unsafe_mask / n_unsafe,
        'tiles': tiles,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'tiles'}, indent=1))


if __name__ == '__main__':
    main()
