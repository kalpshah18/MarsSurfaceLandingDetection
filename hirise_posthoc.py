"""Post-hoc checks on the HiRISE evaluation (NOT pre-registered; reported as such).

1. Random-square null: for each tile and L, the probability that an L x L square placed uniformly
   at random on the tile has <= 5% proxy-unsafe pixels (exact, via integral images), and the
   tile-centre L x L square's outcome. Compared with the pipeline's correct-site rates.
2. Alignment sensitivity: pooled AUROC / PR-AUC with each site's ortho shifted by the best shift
   from results/hirise/alignment_check.json.
Output: results/hirise/posthoc.json
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from footprint_metrics import EPSILON
from review_common import hazard_maps, load_config, window_sums
from run_evaluation import compute_proxy_labels, get_local_std

DATA_DIR = os.environ.get('HIRISE_DATA_DIR', os.path.join('data', 'raw', 'hirise'))
OUT = os.path.join('results', 'hirise')
LS = [10, 20, 25, 30, 40, 50]


def shift(img, dy, dx):
    """out[y, x] = img[y + dy, x + dx], edge-replicated."""
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), img.shape[::-1],
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def main():
    config = load_config()
    man = pd.read_csv(os.path.join(OUT, 'tiles_manifest.csv'))
    align = json.load(open(os.path.join(OUT, 'alignment_check.json')))
    rows = []
    pooled = {sp: {k: [] for k in ('u', 'grad', 'grad_shadow', 'b1')} for sp in ('DEV', 'TEST')}
    sub = slice(None, None, 10)
    for _, t in man.iterrows():
        ori = cv2.imread(os.path.join(DATA_DIR, t['ori']), cv2.IMREAD_GRAYSCALE)
        dtm = np.load(os.path.join(DATA_DIR, t['dtm'])).astype(np.float64)
        unsafe, _ = compute_proxy_labels(dtm / t['map_scale_m'])
        r = {'tile_id': t['tile_id'], 'site': t['site'], 'split': t['split']}
        H, W = unsafe.shape
        for L in LS:
            ok = window_sums(unsafe, L) <= EPSILON * L * L + 1e-9
            r[f'random_safe_{L}'] = float(ok.mean())
            y0, x0 = (H - L) // 2, (W - L) // 2
            r[f'centre_safe_{L}'] = bool(unsafe[y0:y0 + L, x0:x0 + L].mean() <= EPSILON)
        rows.append(r)
        dy, dx = align[t['site']]['best_shift_rowcol']
        ori_s = shift(ori, dy, dx)
        maps = hazard_maps(ori_s, config, ('grad', 'grad_shadow'))
        p = pooled[t['split']]
        p['u'].append(unsafe.flatten()[sub])
        for m in ('grad', 'grad_shadow'):
            p[m].append(maps[m][0].flatten()[sub])
        p['b1'].append(get_local_std(ori_s).flatten()[sub])
        print(t['tile_id'], flush=True)
    df = pd.DataFrame(rows)
    out = {'note': 'post hoc, not pre-registered', 'random_square_null': {}, 'centre_square': {},
           'alignment_shifted': {}}
    for sp in ('DEV', 'TEST'):
        d = df[df['split'] == sp]
        out['random_square_null'][sp] = {str(L): float(d[f'random_safe_{L}'].mean()) for L in LS}
        out['centre_square'][sp] = {str(L): float(d[f'centre_safe_{L}'].mean()) for L in LS}
        u = np.concatenate(pooled[sp]['u'])
        out['alignment_shifted'][sp] = {m: {'auroc': float(roc_auc_score(u, np.concatenate(pooled[sp][m]))),
                                            'prauc': float(average_precision_score(u, np.concatenate(pooled[sp][m])))}
                                        for m in ('grad', 'grad_shadow', 'b1')}
    out['random_square_null_by_site'] = {s: {str(L): float(g[f'random_safe_{L}'].mean()) for L in LS}
                                         for s, g in df.groupby('site')}
    with open(os.path.join(OUT, 'posthoc.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'random_square_null_by_site'}, indent=1))


if __name__ == '__main__':
    main()
