# Claims

Every number in the paper (`Mars_Landing_Paper_ARXIV_Style_updated/templateArxiv.tex`) and the
script and output that produce it. All values are on the 80 TEST tiles.

## Footprint envelope (Section 4.3, Tables 3-4, Figure 2, Abstract, Discussion, Conclusion)

Produced by `python run_footprint_sweep.py` from `results/per_tile_boxes.csv`
(`export_per_tile_boxes.py`, same boxes as Table 2). The sweep aborts unless L = 50 reproduces
Table 2.

| Claim | Value | Key | Output file |
| :--- | :--- | :--- | :--- |
| Footprint OK at 25 m (V_grad+shadow) | 0.688 (68.8%) | `table.25.grad_shadow.footprint_ok` | `results/footprint_sweep.json` |
| Safe Landing at 25 m (V_grad+shadow) | 0.075 (7.5%) | `table.25.grad_shadow.safe_landing` | `results/footprint_sweep.json` |
| Square Fit at 25 m equals Safe Landing (gap is elongation) | 0.075 | `table.25.grad_shadow.square_fit` | `results/footprint_sweep.json` |
| Table 3, all rows | see table | `table.<L>.<method>.*` | `results/footprint_sweep.json` |
| Safe Landing 21.2% at 20 m, 1.3% at 30 m | 0.2125, 0.0125 | `table.20/30.grad_shadow.safe_landing` | `results/footprint_sweep.json` |
| L* = 9 m (largest L with Safe Landing >= 50%) | 9 | `L_star.grad_shadow` | `results/footprint_sweep.json` |
| Safe Landing never exceeds 60% (value at L = 5) | 0.600 | `fine.grad_shadow.5.safe_landing` | `results/footprint_sweep.json` |
| No box fits a square above 31 m; Square Fit zero from 32 m | 31 | `box_summary.grad_shadow.max_min_side` | `results/footprint_sweep.json` |
| No area requirement with L >= 41 m met (max area 1,612 px) | 1612 | `box_summary.grad_shadow.max_area` | `results/footprint_sweep.json` |
| 1,612 px box measures 26 x 62 | w=26, h=62 | row `crater_2_12, grad_shadow` | `results/per_tile_boxes.csv` |
| V_grad Safe Landing zero from 28 m | 28 | first zero in `fine.grad.*.safe_landing` | `results/footprint_sweep.json` |
| Table 4, all rows | see table | `by_method.grad_shadow.<terrain>` | `results/footprint_sweep_by_terrain.json` |
| Crater+rocky mean Frac Unsafe 37.1% | 0.371 | `by_method.grad_shadow.crater+rocky.mean_frac_unsafe` | `results/footprint_sweep_by_terrain.json` |
| All rocky boxes 11-19% unsafe; rocky Safe Landing zero at every L | 0.111-0.190 | `frac_unsafe` of rocky `grad_shadow` rows; `fine_safe_landing` | `results/per_tile_boxes.csv`, `results/footprint_sweep_by_terrain.json` |
| 8 of 20 crater boxes clean, 12 at least 89% unsafe | 8 x 0.0, min other 0.891 | `frac_unsafe` of crater `grad_shadow` rows | `results/per_tile_boxes.csv` |
| Smooth largest short side 21 m | 21 | max `min(w,h)` of smooth `grad_shadow` rows | `results/per_tile_boxes.csv` |
| V_grad crater boxes 99.96% unsafe | 0.9996 | `by_method.grad.crater.mean_frac_unsafe` | `results/footprint_sweep_by_terrain.json` |

## Tables 1 and 2 (unchanged in this revision)

Produced by `python compute_final_metrics.py` (printed to stdout; it writes no file).
Configuration `results/best_config.json` (V_grad+shadow: tau = 60, a_min = 100, as now stated in
Section 4.2 of the paper; Figure 1 is generated with the same configuration).

| Claim | Value | Source |
| :--- | :--- | :--- |
| V_grad AUROC / PR-AUC | 0.350 / 0.088 | `compute_final_metrics.py` stdout |
| V_grad+shadow Prec / Rec / F1 / AUROC / PR-AUC | 0.587 / 0.547 / 0.566 / 0.796 / 0.559 | `compute_final_metrics.py` stdout |
| Test prevalence | 0.095 | `compute_final_metrics.py` stdout |
| Table 2 box metrics (L = 50 m) | see Table 2 | `compute_final_metrics.py` stdout; re-checked by `run_footprint_sweep.py` |
| Frac Unsafe averaged over tiles with a box (all 80 here) | 0.186 / 0.287 | `compute_final_metrics.py` (`dropna().mean()`) |

| Evaluated pixels 4,718,640 (every 10th pixel per tile, raster order) | 4718640 | `classification.n_pixels_evaluated` | `results/review/analyses.json` |
| B5 precision 0.087 explained by per-tile rate copied from V_grad | expected 0.0870 | `classification.b5.expected_precision_from_per_tile_rates` | `results/review/analyses.json` |
| Unsafe crater boxes contain no shadow pixels, mean slope 26.3 deg | 0.0, 26.3 | `crater_diagnosis.bad_boxes_mean_frac_shadow`, `.bad_boxes_mean_slope_deg` | `results/review/analyses.json` |
| No 20 m safe square exists on any rocky tile | 0.0 for L >= 20 | `avail_terrain.rocky` | `results/review/analyses.json` |
| Table 4 CIs; 25 m Safe Landing CI 2.5-13.8% | see table | `by_method.grad_shadow.<terrain>.ci95`; `ci95.grad_shadow.25` | `results/footprint_sweep_by_terrain.json`, `results/footprint_sweep.json` |

## Runtime and memory (Table 5)

`python benchmark_stages.py` -> `results/review/benchmark_stages.json`: all 80 TEST tiles, 5 timed
runs each after warm-up, single-threaded OpenCV, i5-12450H, Windows 11, Python 3.12.4, OpenCV
4.9.0. Front-end 14.1 ms = sum of Sobel (9.0), shadow (0.9) and threshold + filter (4.2) medians at
768 x 768; peak RSS increase 28.1-29.9 MB; Python heap (tracemalloc) 6.9 / 15.6 / 27.8 MB. The
earlier Table 5 values (74.0 ms, under 16 MB) had no generating script and are superseded.

## Superseded claims

The following claims from earlier drafts are no longer in the paper and must not be reused:
proxy-label F1 = 0.025 (safe class, macro-averaged; `results/summary.json`), random baseline
F1 = 0.024, B1 F1 = 0.025, A1 F1 = 0.025, 135.2 ms at 1024 x 1024, unsafe recall 0.935, and
the framing that 2D optical sensing is fundamentally insufficient because Footprint OK = 0% at a
fixed 50 m x 50 m requirement. The 50 m result still holds (0.000) but is now one point on the
footprint sweep, and the envelope result is stated in Tables 3-4.
