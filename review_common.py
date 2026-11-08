"""Shared helpers for the reviewer-response analyses (run_review_*.py).

Nothing here changes the pipeline: hazard maps come from pipeline.py with the
Table 2 configuration (results/best_config.json). The extra pieces are
(a) alternative *selectors* applied to the same filtered safe mask and
(b) a shadow-only hazard score used as an ablation.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from footprint_metrics import EPSILON
from pipeline import augment_shadow, compute_gradient_hazard, fuse_and_select
from run_evaluation import compute_proxy_labels

TERRAINS = ('crater', 'rocky', 'smooth', 'dunes')
HAZARD_TERRAINS = ('crater', 'rocky')


def load_config():
    with open(os.path.join('results', 'best_config.json')) as f:
        return json.load(f)


def iter_tiles(split='TEST'):
    """Yield dicts with tile_id, terrain, ori (uint8), dtm, unsafe (bool), slope (deg)."""
    manifest = pd.read_csv(os.path.join('data', 'MANIFEST.csv'))
    with open('splits.json') as f:
        ids = json.load(f)[split]
    for _, row in manifest[manifest['tile_id'].isin(ids)].iterrows():
        dtm = np.load(row['dtm'])
        unsafe, slope = compute_proxy_labels(dtm)
        yield {
            'tile_id': row['tile_id'],
            'terrain': row['site'].rsplit('_', 1)[0],
            'ori': cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE),
            'dtm': dtm,
            'unsafe': unsafe,
            'slope': slope,
        }


def hazard_maps(ori, config, methods=('grad', 'grad_shadow', 'shadow_only')):
    """Return {method: (G, filtered_safe_mask, maxrect)} using pipeline.py functions.

    shadow_only: hazard = 1 inside the retained shadow regions, 0 elsewhere
    (same tau / a_min as V_grad+shadow), then the same theta / component filter /
    maximal rectangle as the other methods.
    """
    gs_cfg = config['V_grad_shadow']
    fuse_kw = {'theta': gs_cfg['theta'], 'min_component_area': gs_cfg['min_component_area']}
    out = {}
    G = compute_gradient_hazard(ori)
    if 'grad' in methods:
        g_cfg = config['V_grad']
        s, _, filt, rect = fuse_and_select(G, None, theta=g_cfg['theta'],
                                           min_component_area=g_cfg['min_component_area'])
        out['grad'] = (1.0 - s, filt, rect)
    G_aug, shadow = augment_shadow(ori, G, tau=gs_cfg['tau'], a_min=gs_cfg['a_min'])
    if 'grad_shadow' in methods:
        s, _, filt, rect = fuse_and_select(G_aug, None, **fuse_kw)
        out['grad_shadow'] = (1.0 - s, filt, rect)
    if 'shadow_only' in methods:
        G_sh = (shadow == 255).astype(np.float32)
        s, _, filt, rect = fuse_and_select(G_sh, None, **fuse_kw)
        out['shadow_only'] = (1.0 - s, filt, rect)
    return out


def window_sums(a, L):
    """Sum of `a` over every L x L window; result[y, x] covers a[y:y+L, x:x+L]."""
    ii = cv2.integral(np.ascontiguousarray(a, dtype=np.float64))
    return ii[L:, L:] - ii[:-L, L:] - ii[L:, :-L] + ii[:-L, :-L]


def largest_inscribed_square(mask, hazard):
    """Largest all-ones square in `mask` (binary search on side with integral images).

    Ties between equally large squares are broken by the lowest mean hazard score.
    Returns (x, y, side); side = 0 if the mask is empty.
    """
    bad = (mask == 0)
    lo, hi = 0, min(mask.shape)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if (window_sums(bad, mid) == 0).any():
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return 0, 0, 0
    feas = window_sums(bad, lo) == 0
    hs = window_sums(hazard, lo)
    hs[~feas] = np.inf
    y, x = np.unravel_index(np.argmin(hs), hs.shape)
    return int(x), int(y), int(lo)


def min_hazard_square(mask, hazard, L):
    """L x L square fully inside `mask` with the lowest mean hazard; None if none fits."""
    if L > min(mask.shape):
        return None
    feas = window_sums(mask == 0, L) == 0
    if not feas.any():
        return None
    hs = window_sums(hazard, L)
    hs[~feas] = np.inf
    y, x = np.unravel_index(np.argmin(hs), hs.shape)
    return int(x), int(y)


def frac_in(unsafe, x, y, w, h):
    if w <= 0 or h <= 0:
        return float('nan')
    return float(unsafe[y:y + h, x:x + w].mean())


def any_safe_square(unsafe, L, mask=None, eps=EPSILON):
    """Oracle (uses labels): does any L x L square (inside `mask` if given) have <= eps unsafe?"""
    if L > min(unsafe.shape):
        return False
    ok = window_sums(unsafe, L) <= eps * L * L + 1e-9
    if mask is not None:
        ok &= window_sums(mask == 0, L) == 0
    return bool(ok.any())


def boot_ci(values, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling tiles."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))]
