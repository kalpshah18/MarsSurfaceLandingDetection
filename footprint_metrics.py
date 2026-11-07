"""Footprint criteria used in Section 3.4 / Table 3 of the paper.

All criteria are evaluated post hoc on the maximal-rectangle box B = (x, y, w, h)
already selected by the pipeline. At 1.0 m GSD, 1 px = 1 m, so the required
footprint side L (m) equals L pixels.

  Footprint OK(L)  = w*h >= L^2                  (area criterion, as in Table 2)
  Square Fit(L)    = min(w, h) >= L              (box can contain an L x L square)
  Safe Landing(L)  = Square Fit(L) and f_unsafe <= EPSILON

EPSILON = 0.05 is fixed a priori (stated in the paper before TEST results were
inspected); it is not tuned.
"""
import math

EPSILON = 0.05


def footprint_ok(w, h, L):
    return w * h >= L * L


def square_fit(w, h, L):
    return min(w, h) >= L


def safe_landing(w, h, frac_unsafe, L, eps=EPSILON):
    if not square_fit(w, h, L):
        return False
    if frac_unsafe is None or (isinstance(frac_unsafe, float) and math.isnan(frac_unsafe)):
        return False
    return frac_unsafe <= eps


def rates(boxes, L, eps=EPSILON):
    """Fraction of boxes (one per tile) meeting each criterion at side L.

    ``boxes`` is a sequence of dicts with keys w, h, frac_unsafe. The
    denominator is the number of tiles (including tiles with no box).
    """
    n = len(boxes)
    if n == 0:
        raise ValueError('rates() needs at least one tile')
    fo = sum(footprint_ok(b['w'], b['h'], L) for b in boxes)
    sq = sum(square_fit(b['w'], b['h'], L) for b in boxes)
    sl = sum(safe_landing(b['w'], b['h'], b['frac_unsafe'], L, eps) for b in boxes)
    return {'footprint_ok': fo / n, 'square_fit': sq / n, 'safe_landing': sl / n}
