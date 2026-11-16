"""Cut 768 x 768 ORI/DTM tile pairs from the HiRISE products (protocol in DECISIONS.md).

Reads $HIRISE_DATA_DIR/<site>/{DTEEC_*.IMG, *_RED_C_01_ORTHO.JP2}, writes
$HIRISE_DATA_DIR/tiles/<site>/{ori_<r>_<c>.png, dtm_<r>_<c>.npy} plus
results/hirise/tiles_manifest.csv (paths relative to $HIRISE_DATA_DIR) and
results/hirise/tiling_report.json (grids, value ranges, valid-tile counts).

The orthoimage is read on the DTM grid: if both products have the same shape and post spacing it
is read pixel-for-pixel (see the protocol amendment in DECISIONS.md); otherwise it is resampled
(nearest) onto the DTM grid with a WarpedVRT using the header georeferencing.
"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

DATA_DIR = os.environ.get('HIRISE_DATA_DIR', os.path.join('data', 'raw', 'hirise'))
OUT = os.path.join('results', 'hirise')
TILE = 768
PER_SITE = 20
MAX_ZERO_FRAC = 0.01


def to_uint8(ortho, valid):
    """Protocol: 8-bit products used as-is; otherwise one linear 0.1-99.9 percentile stretch."""
    if ortho.dtype == np.uint8:
        return ortho, None
    v = ortho[valid][::7]  # deterministic subsample for the percentiles (memory)
    lo, hi = np.percentile(v, [0.1, 99.9])
    out = np.empty(ortho.shape, dtype=np.uint8)
    for r0 in range(0, ortho.shape[0], 2048):  # chunked to bound memory
        blk = ortho[r0:r0 + 2048].astype(np.float32)
        out[r0:r0 + 2048] = np.clip((blk - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out, [float(lo), float(hi)]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open('hirise_sites.json') as f:
        sites = json.load(f)['sites']
    rows, report = [], {}
    for s in sites:
        sdir = os.path.join(DATA_DIR, s['site'])
        dtm_path = glob.glob(os.path.join(sdir, 'DTEEC_*.IMG'))[0]
        ortho_path = glob.glob(os.path.join(sdir, '*_ORTHO.JP2'))[0]
        with rasterio.open(dtm_path) as dsrc:
            dtm = dsrc.read(1, masked=True)
            dvalid = ~np.ma.getmaskarray(dtm) & np.isfinite(dtm.filled(np.nan)) & (dtm.filled(-1e38) > -1e30)
            dtm = dtm.filled(np.nan).astype(np.float32)
            map_scale = abs(dsrc.transform.a)
            with rasterio.open(ortho_path) as osrc:
                # Protocol amendment (DECISIONS.md): same shape and post spacing -> identical pixel
                # grids; the ~1.5 px header-origin difference is a format convention, not an offset.
                same = (osrc.shape == dsrc.shape and abs(abs(osrc.transform.a) - abs(dsrc.transform.a)) < 1e-6
                        and abs(abs(osrc.transform.e) - abs(dsrc.transform.e)) < 1e-6)
                header_offset_px = [(osrc.transform.c - dsrc.transform.c) / dsrc.transform.a,
                                    (osrc.transform.f - dsrc.transform.f) / dsrc.transform.e]
                if same:
                    ortho = osrc.read(1)
                else:
                    with WarpedVRT(osrc, crs=dsrc.crs, transform=dsrc.transform, width=dsrc.width,
                                   height=dsrc.height, resampling=Resampling.nearest) as vrt:
                        ortho = vrt.read(1)
                odtype = str(osrc.dtypes[0])
            info = {'dtm_shape': list(dsrc.shape), 'map_scale_m': map_scale, 'crs': str(dsrc.crs),
                    'ortho_dtype': odtype, 'ortho_on_dtm_grid_natively': bool(same),
                    'header_origin_offset_px': header_offset_px}
        ovalid = ortho != 0
        ori8, stretch = to_uint8(ortho, ovalid & dvalid)
        info['stretch'] = stretch
        H, W = dtm.shape
        cands = []
        for r in range(0, H - TILE + 1, TILE):
            for c in range(0, W - TILE + 1, TILE):
                if dvalid[r:r + TILE, c:c + TILE].all() and (~ovalid[r:r + TILE, c:c + TILE]).mean() <= MAX_ZERO_FRAC:
                    cands.append((r, c))
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(cands), size=min(PER_SITE, len(cands)), replace=False).tolist()) if cands else []
        info['n_valid_tiles'] = len(cands)
        info['n_selected'] = len(pick)
        tdir = os.path.join(DATA_DIR, 'tiles', s['site'])
        os.makedirs(tdir, exist_ok=True)
        for i in pick:
            r, c = cands[i]
            tid = f"{s['site']}_{r}_{c}"
            ori_rel = os.path.join('tiles', s['site'], f'ori_{r}_{c}.png')
            dtm_rel = os.path.join('tiles', s['site'], f'dtm_{r}_{c}.npy')
            cv2.imwrite(os.path.join(DATA_DIR, ori_rel), ori8[r:r + TILE, c:c + TILE])
            np.save(os.path.join(DATA_DIR, dtm_rel), dtm[r:r + TILE, c:c + TILE])
            rows.append({'tile_id': tid, 'site': s['site'], 'split': s['split'], 'row': r, 'col': c,
                         'map_scale_m': map_scale, 'ori': ori_rel.replace('\\', '/'), 'dtm': dtm_rel.replace('\\', '/')})
        report[s['site']] = info
        print(s['site'], info, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'tiles_manifest.csv'), index=False)
    with open(os.path.join(OUT, 'tiling_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print('tiles:', len(rows))


if __name__ == '__main__':
    main()
