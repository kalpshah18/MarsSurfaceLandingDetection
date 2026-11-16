"""Synthetic Martian DTM + orthoimage dataset (v2: physical rendering).

v2 changes (reviewer response, 2026-11-16):
* Craters are fresh simple craters: a paraboloidal bowl with a raised rim and an
  ejecta blanket, instead of a rimless cone of constant 26.6 deg slope.
* Images are rendered with Lambertian shading plus ray-marched cast shadows
  (render.py). The v1 sun-independent "painted" shadow (every crater pixel deeper
  than 10 m clamped to intensity <= 30) is removed.

Crater geometry (Pike 1977; Melosh 1989, simple craters):
  D            rim-crest diameter, 2R, R ~ U[100, 300) m
  d / D        rim-crest-to-floor depth ratio ~ U[0.10, 0.20]
               (0.2 = fresh; lower values = partly degraded)
  h_r / D      rim height above the surroundings = 0.04
  interior     z(r) = h_r - d * (1 - (r/R)^2)            for r <  R
  exterior     z(r) = h_r * (R / r)^3                     for r >= R
The interior wall is steepest at the rim, atan(4 d/D) = 21.8-38.7 deg, and flat at
the centre. At the 30 deg sun elevation a cast shadow therefore appears just inside
the sun-facing rim only when the rim slope exceeds 30 deg (d/D > 0.144).

Usage: python generate_synthetic.py   (overwrites data/synthetic/, MANIFEST.csv, splits.json)
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from render import render

SUN_AZIMUTH = 45.0
SUN_ELEVATION = 30.0
RIM_HEIGHT_RATIO = 0.04
DD_RANGE = (0.10, 0.20)


def crater_dtm(x, y, cx, cy, R, dd):
    D = 2.0 * R
    d = dd * D
    h_r = RIM_HEIGHT_RATIO * D
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.where(r < R, h_r - d * (1.0 - (r / R) ** 2), h_r * (R / np.maximum(r, R)) ** 3)


def generate_synthetic_site(site_name, seed, num_tiles=20, tile_size=768, gsd=1.0):
    rng = np.random.default_rng(seed)
    os.makedirs(f"data/synthetic/{site_name}", exist_ok=True)
    tiles = []
    x, y = np.meshgrid(np.arange(tile_size), np.arange(tile_size))

    for i in range(num_tiles):
        meta = {}
        if 'crater' in site_name:
            cx, cy = rng.integers(200, 568, 2)
            R = int(rng.integers(100, 300))
            dd = float(rng.uniform(*DD_RANGE))
            z = crater_dtm(x, y, cx, cy, R, dd)
            z = z + rng.standard_normal(z.shape) * 0.1
            meta = {'crater_cx': int(cx), 'crater_cy': int(cy), 'crater_D_m': 2 * R,
                    'crater_depth_m': round(dd * 2 * R, 2), 'crater_dd': round(dd, 4),
                    'crater_rim_height_m': round(RIM_HEIGHT_RATIO * 2 * R, 2)}
        elif 'rocky' in site_name:
            z = rng.standard_normal((tile_size, tile_size)) * 0.5
            for _ in range(50):
                rx, ry = rng.integers(0, tile_size, 2)
                rr = rng.integers(2, 10)
                rdist = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
                z[rdist < rr] += rng.random() * 2
        elif 'smooth' in site_name:
            z = 0.1 * x + 0.05 * y + rng.standard_normal((tile_size, tile_size)) * 0.05
        else:
            z = np.sin(x / 20.0) * 2.0 + rng.standard_normal((tile_size, tile_size)) * 0.05

        ori, shadow = render(z, SUN_AZIMUTH, SUN_ELEVATION, gsd)

        dtm_path = f"data/synthetic/{site_name}/dtm_{i}.npy"
        ori_path = f"data/synthetic/{site_name}/ori_{i}.png"
        np.save(dtm_path, z)
        cv2.imwrite(ori_path, ori)
        tiles.append({'site': site_name, 'dtm': dtm_path, 'ori': ori_path,
                      'tile_id': f"{site_name}_{i}", 'cast_shadow_frac': round(float(shadow.mean()), 5),
                      **meta})
        print(f"{site_name}_{i}", meta, f"shadow {shadow.mean():.4f}", flush=True)
    return tiles


def create_synthetic_dataset():
    sites = ['crater_1', 'crater_2', 'rocky_1', 'rocky_2', 'smooth_1', 'smooth_2', 'dunes_1', 'dunes_2']
    all_tiles = []
    for i, site in enumerate(sites):
        all_tiles.extend(generate_synthetic_site(site, seed=42 + i))

    dev_sites = ['crater_1', 'rocky_1', 'smooth_1', 'dunes_1']
    splits = {'DEV': [], 'TEST': []}
    for t in all_tiles:
        splits['DEV' if t['site'] in dev_sites else 'TEST'].append(t['tile_id'])
    with open('splits.json', 'w') as f:
        json.dump(splits, f, indent=4)
    pd.DataFrame(all_tiles).to_csv('data/MANIFEST.csv', index=False)


if __name__ == "__main__":
    create_synthetic_dataset()
    print("Synthetic dataset created.")
