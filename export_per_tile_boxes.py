"""Export the per-tile landing boxes used for Table 2 to results/per_tile_boxes.csv.

One row per (TEST tile, method), for methods ``grad`` (V_grad) and
``grad_shadow`` (V_grad+shadow). The boxes are produced by exactly the same
code path as ``compute_final_metrics.py`` (Table 2): the same pipeline
functions, the same configuration (``results/best_config.json``), the same
proxy labels (slope > 15 deg over a 5 m baseline at 1.0 m GSD) and the same
DEV/TEST split (``splits.json``). Nothing is re-tuned here.

The local-std baseline (B1) is not exported: Table 2 does not run the
maximal-rectangle step for B1 (its box is hard-coded to (0, 0, 0, 0)), so
there is no B1 box to export without inventing a new evaluation.

Usage:
    python export_per_tile_boxes.py
"""
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd

from pipeline import run_v_grad, run_v_grad_shadow
from run_evaluation import compute_proxy_labels

OUT_PATH = os.path.join('results', 'per_tile_boxes.csv')
FIELDS = ['tile_id', 'terrain', 'method', 'box_found', 'x', 'y', 'w', 'h',
          'area_px', 'frac_unsafe', 'n_unsafe_in_box']
TERRAINS = ('crater', 'rocky', 'smooth', 'dunes')


def terrain_from_site(site):
    """Map a site name such as 'crater_2' to its terrain type ('crater')."""
    terrain = site.rsplit('_', 1)[0]
    if terrain not in TERRAINS:
        raise ValueError(f'Unknown terrain for site {site!r}')
    return terrain


def box_row(tile_id, terrain, method, rect, unsafe):
    x, y, w, h = (int(v) for v in rect)
    area = w * h
    found = int(area > 0)
    if found:
        n_unsafe = int(np.sum(unsafe[y:y + h, x:x + w]))
        frac_unsafe = float(np.mean(unsafe[y:y + h, x:x + w]))
    else:
        n_unsafe, frac_unsafe = 0, float('nan')
    return {
        'tile_id': tile_id, 'terrain': terrain, 'method': method,
        'box_found': found, 'x': x, 'y': y, 'w': w, 'h': h,
        'area_px': area, 'frac_unsafe': frac_unsafe,
        'n_unsafe_in_box': n_unsafe,
    }


def iter_test_tiles():
    """Yield (tile_id, terrain, ori, unsafe) for every TEST tile, in Table 2 order."""
    manifest = pd.read_csv(os.path.join('data', 'MANIFEST.csv'))
    with open('splits.json') as f:
        splits = json.load(f)
    test_df = manifest[manifest['tile_id'].isin(splits['TEST'])]
    for _, row in test_df.iterrows():
        ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
        unsafe, _ = compute_proxy_labels(np.load(row['dtm']))
        yield row['tile_id'], terrain_from_site(row['site']), ori, unsafe


def load_config():
    with open(os.path.join('results', 'best_config.json')) as f:
        return json.load(f)


def main():
    config = load_config()
    rows = []
    for tile_id, terrain, ori, unsafe in iter_test_tiles():
        _, _, _, rect_g = run_v_grad(ori, **config['V_grad'])
        _, _, _, rect_gs = run_v_grad_shadow(ori, **config['V_grad_shadow'])
        rows.append(box_row(tile_id, terrain, 'grad', rect_g, unsafe))
        rows.append(box_row(tile_id, terrain, 'grad_shadow', rect_gs, unsafe))

    os.makedirs('results', exist_ok=True)
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            r = dict(r)
            r['frac_unsafe'] = '' if np.isnan(r['frac_unsafe']) else repr(r['frac_unsafe'])
            writer.writerow(r)
    print(f'Wrote {len(rows)} rows to {OUT_PATH}')


if __name__ == '__main__':
    main()
