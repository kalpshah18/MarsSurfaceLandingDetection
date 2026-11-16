"""Albedo-robust parameter selection for V_full (test of a fix for the albedo sensitivity found by
run_photometric_robustness.py).

Fix: the brightness cue averages brightness over a larger window, so that albedo patches finer than the
slopes of interest average out, and all parameters are selected once on DEV tiles rendered under mixed
albedo conditions instead of the clean renderer only:
  * k grid extended to {5, 9, 15, 31, 51}, phi grid to {5 ... 30} degrees;
  * DEV selection set: the same 8 DEV tiles, each rendered with no albedo variation, smooth albedo
    (c = 0.1 and 0.3) and patchy albedo (c = 0.2), i.e. 32 images; albedo fields use seeds disjoint
    from those of the TEST renders.
The selected configuration is then frozen and evaluated on every variant of run_photometric_robustness.py.

Success criterion (fixed before the run): V_full >= V_grad+shadow in AUROC on every albedo variant,
baseline AUROC within 0.02 of the current paper (0.866), and the baseline Safe Landing envelope kept.

    python run_albedo_fix.py
Output: results/albedo_fix.json
"""
import json
import os

import numpy as np

import run_normalization_study as rns
from review_common import iter_tiles
from run_photometric_robustness import VARIANTS, evaluate, geometry, render_variant

OUT = os.path.join('results', 'albedo_fix.json')
DEV_CONDITIONS = [{}, {'albedo_kind': 'smooth', 'albedo_c': 0.1}, {'albedo_kind': 'smooth', 'albedo_c': 0.3},
                  {'albedo_kind': 'patch', 'albedo_c': 0.2}]


def main():
    rns.PHIS = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
    rns.KSIZES = [5, 9, 15, 31, 51]
    with open('splits.json') as f:
        dev = json.load(f)['DEV']
    ids = []
    for site in ('crater_1', 'rocky_1', 'smooth_1', 'dunes_1'):
        ids += [x for x in dev if x.startswith(site + '_')][:2]
    dev_tiles = [t for t in iter_tiles('DEV') if t['tile_id'] in ids]
    test_tiles = list(iter_tiles('TEST'))
    print('geometry', flush=True)
    dev_geo = [geometry(t['dtm']) for t in dev_tiles]
    test_geo = [geometry(t['dtm']) for t in test_tiles]
    sel = []
    for ci, cond in enumerate(DEV_CONDITIONS):
        for i, (g, t) in enumerate(zip(dev_geo, dev_tiles)):
            sel.append({'ori': render_variant(g, 5000 + 100 * ci + i, cond), 'unsafe': t['unsafe']})
    cfg = rns.staged_tune(sel, 'physical')
    print('selected', cfg, flush=True)
    out = {'config': cfg, 'dev_conditions': DEV_CONDITIONS, 'variants': {}}
    for name, v in VARIANTS.items():
        imgs = [render_variant(g, 1000 + i, v) for i, g in enumerate(test_geo)]
        r = evaluate(test_tiles, imgs, cfg)
        out['variants'][name] = r
        print(f"{name:20s} full {r['full']['auroc']:.3f}/{r['full']['prauc']:.3f} crater {r['full']['crater_within_auroc']:.3f} "
              f"SL {[round(x, 3) for x in r['full']['safe_landing'].values()]} | gs {r['grad_shadow']['auroc']:.3f}/"
              f"{r['grad_shadow']['prauc']:.3f} g {r['grad']['auroc']:.3f}", flush=True)
        with open(OUT, 'w') as f:
            json.dump(out, f, indent=1)
    albedo = [n for n in VARIANTS if n.startswith('albedo_') or n == 'combined']
    base = out['variants']['baseline']['full']
    crit = {
        'full_ge_grad_shadow_on_albedo': all(out['variants'][n]['full']['auroc'] >= out['variants'][n]['grad_shadow']['auroc']
                                             for n in albedo),
        'baseline_auroc_within_0.02': base['auroc'] >= 0.866 - 0.02,
        'baseline_envelope_kept': all(abs(x - 0.75) < 1e-9 for x in base['safe_landing'].values()),
    }
    out['criterion'] = crit
    out['success'] = all(crit.values())
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('criterion', crit, 'success', out['success'])


if __name__ == '__main__':
    main()
