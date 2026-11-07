"""Footprint-requirement sweep (Tables 3 and 4, Figure fig_footprint_envelope).

Reads results/per_tile_boxes.csv (written by export_per_tile_boxes.py; it is
generated first if missing) and evaluates Footprint OK, Square Fit and Safe
Landing (footprint_metrics.py) over the 80 TEST tiles as the required
footprint side L is varied. This is post-hoc evaluation of the same boxes used
for Table 2; no pipeline parameter is changed.

Before anything is written, the script checks that L = 50 m reproduces Table 2
exactly and aborts otherwise.

Outputs:
    results/footprint_sweep.json
    results/footprint_sweep_by_terrain.json
    LaTeX-ready rows for Tables 3 and 4 on stdout.

Usage:
    python run_footprint_sweep.py
"""
import csv
import json
import math
import os
import sys

from footprint_metrics import EPSILON, rates

BOXES_PATH = os.path.join('results', 'per_tile_boxes.csv')
SWEEP_PATH = os.path.join('results', 'footprint_sweep.json')
TERRAIN_PATH = os.path.join('results', 'footprint_sweep_by_terrain.json')

TABLE_L = [20, 25, 30, 35, 40, 50]
FINE_L = list(range(5, 61))
LSTAR_RANGE = range(1, 61)
TERRAIN_L = 25
TERRAINS = ['crater', 'rocky', 'smooth', 'dunes']
METHODS = ['grad_shadow', 'grad']
N_TEST = 80

# Table 2 values (paper, Section 4.2) that L = 50 must reproduce.
TABLE2 = {
    'grad_shadow': {'footprint_ok': 0.000, 'mean_area': 744, 'max_area': 1612, 'frac_unsafe': 0.186},
    'grad': {'footprint_ok': 0.250, 'mean_area': 17028, 'frac_unsafe': 0.287},
}


def load_boxes(path=BOXES_PATH):
    if not os.path.exists(path):
        import export_per_tile_boxes
        export_per_tile_boxes.main()
    by_method = {m: [] for m in METHODS}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['method'] not in by_method:
                continue
            fu = r['frac_unsafe']
            by_method[r['method']].append({
                'tile_id': r['tile_id'],
                'terrain': r['terrain'],
                'box_found': int(r['box_found']),
                'w': int(r['w']),
                'h': int(r['h']),
                'area_px': int(r['area_px']),
                'frac_unsafe': float(fu) if fu != '' else float('nan'),
            })
    return by_method


def mean_frac_unsafe(boxes):
    """Mean box frac_unsafe over tiles with a box (Table 2 convention: NaN rows dropped)."""
    vals = [b['frac_unsafe'] for b in boxes if b['box_found'] and not math.isnan(b['frac_unsafe'])]
    return sum(vals) / len(vals) if vals else float('nan')


def box_summary(boxes):
    areas = [b['area_px'] for b in boxes]
    widths = [min(b['w'], b['h']) for b in boxes if b['box_found']]
    return {
        'n_tiles': len(boxes),
        'box_found': sum(b['box_found'] for b in boxes) / len(boxes),
        'mean_area': sum(areas) / len(areas),
        'max_area': max(areas),
        'max_min_side': max(widths) if widths else 0,
        'mean_frac_unsafe': mean_frac_unsafe(boxes),
    }


def sanity_check(by_method):
    problems = []
    for m in METHODS:
        if len(by_method[m]) != N_TEST:
            problems.append(f'{m}: expected {N_TEST} tiles, got {len(by_method[m])}')
    for m, ref in TABLE2.items():
        s = box_summary(by_method[m])
        r50 = rates(by_method[m], 50)
        got = {
            'footprint_ok': round(r50['footprint_ok'], 3),
            'mean_area': round(s['mean_area']),
            'max_area': s['max_area'],
            'frac_unsafe': round(s['mean_frac_unsafe'], 3),
        }
        for k, v in ref.items():
            if got[k] != v:
                problems.append(f'{m}.{k}: Table 2 = {v}, reproduced = {got[k]}')
    if problems:
        print('SANITY CHECK FAILED: L = 50 does not reproduce Table 2.')
        for p in problems:
            print('  ' + p)
        sys.exit(1)
    print('Sanity check passed: L = 50 reproduces Table 2 exactly.')


def largest_L(boxes, threshold=0.5):
    best = None
    for L in LSTAR_RANGE:
        if rates(boxes, L)['safe_landing'] >= threshold:
            best = L
    return best


def main():
    by_method = load_boxes()
    sanity_check(by_method)

    sweep = {
        'epsilon': EPSILON,
        'n_tiles': N_TEST,
        'denominator': 'all 80 TEST tiles (tiles with no box count as failures)',
        'source': BOXES_PATH,
        'box_summary': {m: box_summary(by_method[m]) for m in METHODS},
        'table': {},
        'fine': {m: {} for m in METHODS},
        'L_star': {},
    }
    for L in TABLE_L:
        sweep['table'][str(L)] = {'A_req_px': L * L, **{m: rates(by_method[m], L) for m in METHODS}}
    for m in METHODS:
        for L in FINE_L:
            sweep['fine'][m][str(L)] = rates(by_method[m], L)
        sweep['L_star'][m] = largest_L(by_method[m])
    gs_lstar = sweep['L_star']['grad_shadow']
    if gs_lstar is None:
        sweep['L_star_note'] = 'No L in 1..60 reaches Safe Landing >= 0.5 for grad_shadow; see fine curve.'

    terrain = {'L': TERRAIN_L, 'epsilon': EPSILON, 'by_method': {}}
    for m in METHODS:
        terrain['by_method'][m] = {}
        for t in TERRAINS:
            rows = [b for b in by_method[m] if b['terrain'] == t]
            r = rates(rows, TERRAIN_L)
            terrain['by_method'][m][t] = {
                'n_tiles': len(rows),
                'mean_area': sum(b['area_px'] for b in rows) / len(rows),
                'max_area': max(b['area_px'] for b in rows),
                'mean_frac_unsafe': mean_frac_unsafe(rows),
                'square_fit': r['square_fit'],
                'safe_landing': r['safe_landing'],
                'footprint_ok': r['footprint_ok'],
                'fine_safe_landing': {str(L): rates(rows, L)['safe_landing'] for L in FINE_L},
            }
        hz = [b for b in by_method[m] if b['terrain'] in ('crater', 'rocky')]
        terrain['by_method'][m]['crater+rocky'] = {
            'n_tiles': len(hz),
            'mean_frac_unsafe': mean_frac_unsafe(hz),
            **rates(hz, TERRAIN_L),
        }

    with open(SWEEP_PATH, 'w') as f:
        json.dump(sweep, f, indent=2)
    with open(TERRAIN_PATH, 'w') as f:
        json.dump(terrain, f, indent=2)
    print(f'Wrote {SWEEP_PATH} and {TERRAIN_PATH}')

    print('\n% Table 3 rows: L & A_req & GS FootprintOK & GS SquareFit & GS SafeLanding & G FootprintOK & G SafeLanding')
    for L in TABLE_L:
        gs, g = sweep['table'][str(L)]['grad_shadow'], sweep['table'][str(L)]['grad']
        print(f"    {L} & {L*L:,} & {gs['footprint_ok']:.3f} & {gs['square_fit']:.3f} & "
              f"{gs['safe_landing']:.3f} & {g['footprint_ok']:.3f} & {g['safe_landing']:.3f} \\\\")

    print(f'\n% Table 4 rows (grad_shadow, L = {TERRAIN_L} m): Terrain & Mean area & Frac Unsafe & Square Fit & Safe Landing')
    for t in TERRAINS:
        d = terrain['by_method']['grad_shadow'][t]
        print(f"    {t.capitalize()} & {d['mean_area']:,.0f} & {d['mean_frac_unsafe']:.3f} & "
              f"{d['square_fit']:.3f} & {d['safe_landing']:.3f} \\\\")

    print(f"\n% L* (largest integer L in 1..60 with grad_shadow Safe Landing >= 0.5): {gs_lstar}")
    print(f"% L* for grad: {sweep['L_star']['grad']}")
    print('\n% Fine sweep, grad_shadow (L: FO / SQ / SL):')
    for L in FINE_L:
        r = sweep['fine']['grad_shadow'][str(L)]
        print(f"%   {L:2d}: {r['footprint_ok']:.3f} / {r['square_fit']:.3f} / {r['safe_landing']:.3f}")


if __name__ == '__main__':
    main()
