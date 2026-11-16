# Report

## Completed Phases
- **Phase 0 (Audit & Setup)**: Completed. `pipeline.py` implemented strictly according to equations in the paper. `test_pipeline.py` created and passed. `AUDIT.md` mapped parameters. 
- **Phase 1 & 2 (Data and Labels)**: A synthetic dataset generator was used due to the massive size (>1GB each) and network constraints of downloading actual 1m/px HiRISE DTMs. The synthetic dataset models craters, rocky plains, dunes, and smooth planes with illumination mimicking descent imagery. Real URLs were documented in `DECISIONS.md`. Proxy labels created with slope > 15 degrees.
- **Phase 3 (Evaluation)**: Ran the evaluation harness on the synthetic tiles (split into DEV and TEST). Evaluated metrics (Precision, Recall, F1, IoU, ROC-AUC) and box-level constraints.
- **Phase 4 (Ablations)**: Implemented ablated versions of the pipeline.
- **Phase 5 (Complexity)**: Measured execution time.
- **Phase 6 & 7 (Figures & Paper Revision)**: Not fully integrated yet.
- **Phase 8 & 9**: Reproducibility package structured.

## Not Obtained / Deviations
- **Actual HiRISE DTM Data**: "NOT OBTAINED". Attempting to download multiple 500MB+ images would likely fail/timeout. Used synthetic equivalent mapped to the identical spatial scale for proof-of-concept testing, as permitted in an independent experiment.
- **Hardware Benchmarks for Flight**: Left out as unverified. Only local execution time reported.

## Action Items for Authors
- Verify licensing requirements of HiRISE data and workshop template if a specific target workshop is identified.
- Fill in missing citations manually if not fully resolved by standard bibtex extraction.

## Contradictory Results
- The original paper implied the pipeline successfully identified safe landing zones on Mars. The quantitative testing shows the F1 score is basically equal to a random baseline (0.025) because the deterministic FFT thresholds flag almost all realistic Martian texture as hazardous, making the safe zone fraction tiny. This result directly contradicts the implied performance claims of the original text, which have now been completely reframed.

## Author Verification Required
- Citations: The citations have been verified to compile and resolve, but manual verification of their academic appropriateness remains for the authors.
- Workshop Template & Policy: Confirm if double-blind is needed for the specific target venue.
- Archival status and HiRISE data licensing: Confirm data usage constraints.

## Footprint-Envelope Revision (branch `footprint-envelope`, 2026-11-08)

### Sanity check
`run_footprint_sweep.py` reproduces Table 2 exactly at L = 50 m before writing anything:
V_grad+shadow Footprint OK 0.000, mean area 744 px, max area 1,612 px, Frac Unsafe 0.186;
V_grad Footprint OK 0.250, mean area 17,028 px, Frac Unsafe 0.287.

### How Table 2 `Frac Unsafe` is averaged
`compute_final_metrics.py` sets `frac_unsafe = NaN` for tiles without a box and takes
`dropna().mean()`, i.e. the mean over tiles that have a box. Both V_grad and V_grad+shadow
return a box on all 80 TEST tiles, so the average is over all 80 tiles. The 40 smooth and dune
tiles have zero proxy-unsafe pixels, so they contribute 0; over the 40 crater and rocky tiles
alone, V_grad+shadow boxes are 37.1% unsafe. The paper text now says so explicitly.

### Results that weaken the "positive at 25 m" story
- Safe Landing at 25 m (V_grad+shadow) is 7.5% (6/80 tiles). Footprint OK (area only) is 68.8%.
- L* (largest L with Safe Landing >= 50%) is 9 m.
- Safe Landing is capped at 60% at every L: all 20 rocky boxes (11-19% unsafe) and 12 of 20
  crater boxes (>= 89% unsafe) fail epsilon = 0.05 regardless of size.
- Boxes are elongated. The largest short side min(w, h) is 31 px, so Square Fit and Safe Landing
  are zero from L = 32 m. The previous "envelope ends at about 40 m" was an area equivalence only
  (the 1,612 px box is 26 x 62).
- Smooth tiles are hazard-free but never fit a 25 m square (largest short side 21 m).
- The earlier sentence that V_grad+shadow is "conservative rather than dangerous" is not
  supported on crater or rocky terrain and was rewritten.

### Pre-existing inconsistency found (not changed; author decision needed)
The paper (Section 3.2, Figure 1 caption) states the shadow step uses tau = 40, A_min = 500.
Tables 1 and 2 and the runtime table were produced with `results/best_config.json`, which has
V_grad+shadow tau = 60, a_min = 100 (the DEV optimum in `results/tuning.csv`).
- Landing boxes: identical on all 80 TEST tiles under either setting, so Tables 2-4 and the
  footprint figure do not depend on this.
- Safe masks differ on 20 TEST tiles. With tau = 40, A_min = 500, the Table 1 V_grad+shadow row would be
  Prec 0.585, Rec 0.540, F1 0.562, AUROC 0.793, PR-AUC 0.554 (vs 0.587 / 0.547 / 0.566 / 0.796 /
  0.559 in the paper).
Either the text should say tau = 60, A_min = 100, or Table 1 should be regenerated at tau = 40,
A_min = 500. No hyperparameter was changed in this revision.

### Repository state found
- `generate_synthetic.py`, `evaluate.py`, `run_ablations.py`, `run_experiments.py`,
  `run_diagnosis.py` and `measure_complexity.py` import `run_v0` or `compute_fft_texture`,
  which no longer exist in `pipeline.py`; they do not run. `run_all.sh` called three of them
  and has been rewritten to call the scripts that produce the current paper numbers.
- `data/synthetic/` is git-ignored and cannot currently be regenerated (see above), so the
  results are reproducible only from an existing local copy of the dataset.
- `update_paper_numbers.py`, `rewrite_paper.py`, `revise_paper.py`, `clean_banned_terms.py`
  rewrote earlier drafts with the old framing; they now exit immediately.

## Reviewer-response analyses (2026-11-08)

Scripts: `run_review_analyses.py`, `run_review_sweeps.py {sens,sun,sunphys}`,
`benchmark_stages.py` (helpers in `review_common.py`). Outputs in `results/review/`.
No pipeline code or hyperparameter was changed; all sweeps are reporting only.

### Finding that invalidates the paper's central claims
`generate_synthetic.py` clamps every crater pixel deeper than 10 m to intensity <= 30
(`ori[z < -10] = min(ori, 30)`), independent of the sun. Crater walls have constant slope 0.5
(26.6 deg), below the 30 deg sun elevation, so physically the bowl casts almost no shadow (2-4% of a
crater tile, ray-marched) and the sun-facing deep wall is bright (mean ~109 vs 28 when painted).
87.6% of proxy-unsafe crater pixels lie inside the painted region, so the shadow step is reading a
painted depth label. Re-rendering TEST from the stored DTMs with the same Lambertian model but no
painting and ray-marched cast shadows (`sun_sweep_physical.json`):

| sun (az 45) | grad AUROC | grad+shadow AUROC | grad+shadow PR-AUC | hazard-tile Safe Landing |
|---|---|---|---|---|
| 15 deg | 0.370 | 0.514 | 0.099 | 0.00 |
| 30 deg (paper) | 0.500 | 0.614 | 0.222 | 0.00 |
| 45 deg | 0.495 | 0.495 | 0.103 | 0.00 |
| 60 deg | 0.489 | 0.489 | 0.100 | 0.00 |

Both the "inverted response" (AUROC 0.350) and its "correction" (0.796) depend on the painted
region. Without it, raw gradient is at chance and shadow augmentation adds 0.11 AUROC at 30 deg and
nothing at 45-60 deg. Consistent with this, tau = 20 fails (AUROC 0.400) while tau = 40-80 are
equivalent (0.793-0.800): any tau above the painted value 30 works.

### Other review points, as computed
- Hazard-bearing tiles (point 1): V_grad+shadow max-rectangle Safe Landing is at most 20% (8/40) at
  any L; false-safe site rate 57% at L = 10 m, 78% at 5 m. Rocky: 0% at every L.
- Rocky tiles have no L x L square with <= 5% unsafe pixels anywhere for L >= 20 m (label-based
  check, `avail_terrain`), so "no site" is the correct answer there and Safe Landing cannot credit it.
- Square selector (point 2): largest inscribed square on the same mask raises 25 m Safe Landing from
  0.075 to 0.35 overall and crater from 0.15 to 0.90; largest square side 36 m (vs 31 m short side).
  Min-hazard L x L square picks lit crater floor for L <= 15 m (crater Safe Landing 0).
- Inversion mechanism (point 3): the 12 unsafe crater boxes contain 0% shadow pixels, mean slope
  26.3 deg, mean intensity 180, mean depth 4.9 m (above the 10 m painting line). Among lit crater
  pixels, raw G ranks unsafe below safe (AUROC 0.450).
- Shadow-only (point 4): pooled AUROC 0.746 / PR-AUC 0.537 / F1 0.658 (vs 0.796 / 0.559 / 0.566);
  within-tile crater AUROC 0.923 vs 0.914. It finds a safe site on 100% of crater, smooth and dune
  tiles at every L up to 60 m and a false-safe site on 100% of rocky tiles.
- theta dominates the envelope: lsquare Safe Landing at 25 m is 0.75 / 0.35 / 0.01 / 0.00 for
  theta = 0.1 / 0.2 / 0.3 / 0.4.
- Painted-shadow sun sweep: azimuth has no effect (symmetric bowls, sun-independent painting);
  elevation 15 deg drops AUROC to 0.633.
- B5 precision 0.087 < prevalence 0.095 is explained: B5's per-tile flag rate copies V_grad's, which
  is lowest on crater tiles (2.8%) where prevalence is highest (corr -0.27); the weighted expectation is
  0.0870, observed 0.0872. Not a sampling bug.
- Evaluated pixel count is 4,718,640 (ceil(589,824/10) x 80, 1-D stride 10), not 4,718,592.
  A 10 x 10 grid gives AUROC 0.795 / PR-AUC 0.544 for grad+shadow (vs 0.796 / 0.559).
- Within-tile AUROC on hazard-bearing tiles: grad+shadow 0.756 [0.706, 0.807], grad 0.380,
  shadow-only 0.712, B1 0.336. Rocky: grad+shadow 0.597, shadow-only 0.500.
- Table 5's per-stage numbers have no generating script in the repo; they match benchmarks on
  random-noise images (`results/complexity.json`). `benchmark_stages.py` re-measures on TEST tiles.
- B1 has no Table 2 box because `compute_final_metrics.py` hard-codes its box to (0,0,0,0).

### Related work checked
- Huertas, Cheng & Madison, "Passive imaging based multi-cue hazard detection for spacecraft safe
  landing", IEEE Aerospace Conf. 2006 (NTRS 20060044143). The review's "Matthies" is incorrect.
- Golombek, Huertas, Marlow & McGrane, JGR 113, E00A09 (2008), doi:10.1029/2007JE003065.
- Li, Jiang & Tao, "Guidance Summary and Assessment of the Chang'e-3 Powered Descent and Landing",
  J. Spacecraft & Rockets 53(2):258-277 (2016), doi:10.2514/1.A33208.
- Chandrayaan-3 LHDAC: no technical publication found, only news coverage. Not citable yet.
- Runtime on TEST tiles (`results/review/benchmark_stages.json`; i5-12450H laptop CPU, Windows 11,
  Python 3.12.4, OpenCV 4.9.0 single-threaded, median [IQR] ms over 80 tiles x 5): at 768 x 768,
  Sobel 9.0 [8.0-10.0], shadow 0.9 [0.7-12.4], threshold+components 4.2 [0.7-7.8] (by subtraction,
  noisy), rectangle 193.0 [189.2-196.5], total 210.1 [205.0-230.1]. Peak RSS increase about 29 MB at
  every size, against a tracemalloc peak of 15.6 MB at 768. The paper's "under 16 MB" is the
  tracemalloc figure and omits native OpenCV buffers. No flight-processor scaling was measured.

### Decisions needed from the authors (not taken here)
1. Given the painted-shadow artifact, whether to (a) re-run everything on physically rendered data
   (`run_review_sweeps.py sunphys` is a starting point), (b) move to real HiRISE ORI + DTM pairs, or
   (c) reframe the paper as a negative/methodological result about synthetic evaluation.
2. Whether to adopt the largest-inscribed-square selector as the headline selector (contradicts the
   earlier instruction not to change the headline method).
3. Whether the outcome taxonomy should credit correct rejection (rocky tiles have no safe 25 m site).

## Physical-rendering revision (branch `physical-rendering`, 2026-11-16)

Trigger: reviewer pointed out that at 30 deg sun elevation a simple crater (d/D ~ 0.2) shows
a crescent shadow inside the sun-side rim with the far wall and most of the floor lit, so the
nearly all-dark crater of Fig. 1(a) is unphysical, and asked for the assumed D and depth.

### What was wrong (v1)
- `generate_synthetic.py` clamped every crater pixel deeper than 10 m to intensity <= 30,
  independent of the sun ("painted shadow"); it darkened 86% of each crater interior.
- Craters were rimless cones of constant 26.6 deg slope (D 200-600 m, depth 50-150 m,
  d/D 0.25). Below the 30 deg sun elevation, so physically they cast no shadow at all.
- The paper's inversion (AUROC 0.350) and correction (0.796) both came from the painted disc.

### What changed
- `render.py` (new): Lambertian shading + ray-marched cast shadows (bilinear, 0.5 px steps),
  shared by the generator and the sun sweeps. Checked against the analytic shadow length of
  a paraboloid: 27.3% of D vs 27.8% analytic at d/D 0.2, none below d/D 0.144.
- `generate_synthetic.py`: paraboloidal bowl, d/D ~ U[0.10, 0.20], rim height 0.04 D,
  ejecta (R/r)^3; crater geometry and cast-shadow fraction written to MANIFEST.csv.
  v1 data kept locally in `data/synthetic_v1_painted/` and `results_v1_painted/` (git-ignored).
- `run_tuning.py`: 8 DEV tiles are now two per terrain (v1: `dev_tiles[:8]`, all crater_1);
  tau grid {20, 40, 60, 80} x A_min {100, 500} (v1 grid {20, 40} could not produce the
  tau = 60 in best_config.json). New config: theta 0.2, mca 100, tau 60, A_min 500.
- `compute_final_metrics.py` writes `results/final_metrics.json`; `run_footprint_sweep.py`
  checks against it instead of hard-coded v1 values.
- New: `run_crater_diagnosis.py`, `generate_crater_model_figure.py`; Fig. 1 regenerated
  on crater_2_8 with sun arrow, rim, cast-shadow outline and crater D/depth annotated.

### Headline results (v2, TEST)
- Pooled AUROC / PR-AUC: V_grad 0.529 / 0.098; V_grad+shadow 0.598 / 0.194; B1 0.503 / 0.124;
  chance PR-AUC 0.081.
- Crater tiles: within-tile AUROC raw 0.435 -> augmented 0.575. Unsafe crater pixels:
  4.9% in true cast shadow, 25.7% in the tau = 60 dark mask.
- Boxes identical for both methods on 78/80 tiles. Every crater box is on the sun-facing wall
  (99.9% sun-facing pixels, mean intensity 196, 89-100% unsafe), although every crater tile
  has a safe 50 m square on the plain. Sun-facing walls show little texture contrast
  (mean raw G 0.254 vs 0.353 on safe terrain).
- Safe Landing: 0.50 at 5 m, 0.40 at 10 m, 0.20 at 20 m, 0.088 at 25 m, 0 from 31 m; all
  successes on smooth/dune tiles. L* = 5 m (v1: 9 m).

### Brightness (photometric slope) cue, V_full (added 2026-11-16 at the authors' request)
- `pipeline.compute_brightness_hazard`: B = min(1, |box_blur_k(I) - median| / kappa); V_full hazard
  H = max(shadow-augmented G, B). Tuned on the same 8 DEV tiles with tau/A_min/theta fixed:
  kappa in {20,30,40,50,70}, k in {5,9,15} -> kappa 50, k 15. Evaluated once on TEST.
- TEST: AUROC 0.750, PR-AUC 0.359 (V_grad+shadow 0.598 / 0.194). Crater within-tile AUROC 0.875.
- Safe Landing: 57.5% at 5 m, 47.5% at 10 m, 26.3% at 20 m, 12.5% at 25 m, 0 from 31 m; L* = 9 m.
  Crater safe sites on 30% of tiles up to 15 m (0% for V_grad / V_grad+shadow).
- Remaining unsafe crater boxes (14/20 > 5%) lie on walls sloping across the sun line:
  mean |cos(aspect - sun az)| 0.40 vs 0.64 for random aspect.
- Sun sweep (params tuned at 30 deg): V_full best at every elevation (AUROC 0.630 / 0.750 / 0.724 /
  0.669 at 15 / 30 / 45 / 60 deg); shadow cue matters at low sun, brightness at high sun.
- An exploratory DEV-only run with kappa = 30, k = 9 was done before the tuning grid was added;
  it informed the decision to add the cue but not the selected parameters.

## Reviewer suggestions: gradient normalization and real HiRISE data (2026-11-16, not yet in paper)

### Fixed gradient normalization (`run_normalization_study.py` -> results/normalization_study.json)
g_ref = 8 * median(I) * cot(e) * sin(15 deg) / 5 (Sobel response of a 15 deg slope change over the
5 px label baseline). Each variant tuned on the same 8 DEV tiles, theta grid widened to
{0.05, 0.1, 0.2, 0.35, 0.5} for all. TEST, V_full:
| variant | AUROC | PR-AUC | SL 10/25/50 m | max box side |
|---|---|---|---|---|
| p99 (paper config) | 0.750 | 0.359 | 0.475/0.125/0 | 33 |
| p99, widened theta | 0.755 | 0.373 | 0.625/0.362/0 | 51 |
| fixed, raw Sobel | 0.801 | 0.191 | 0.375/0.037/0 | 34 |
| p99 on 5x5-blurred image | 0.701 | 0.346 | 0.713/0.375/0.20 | 294 |
| fixed, 5x5-blurred (fixed_b5) | 0.864 | 0.315 | 0.75/0.75/0.75 | 768 |
fixed_b5 reaches the ceiling (rocky tiles have no safe 20 m square). Raw-Sobel fixed normalization
fragments more: pixel-scale texture already exceeds g_ref on smooth/dune tiles.

### Real HiRISE (`hirise_eval_v2.py` -> results/hirise/evaluation_v2.json)
Same 8 sites / 160 tiles as the 2026-11-08 run; sun elevations from PDS RDR labels
(results/hirise/sun_geometry.json, 26-57 deg). TEST (4 sites, 80 tiles, prevalence 0.086):
| protocol | V_full AUROC / PR-AUC | within-tile | SL 20 m | random null 20 m |
|---|---|---|---|---|
| transfer, paper config | 0.629 / 0.134 | 0.684 | 0.72 | 0.78 |
| transfer, fixed_b5 | 0.654 / 0.134 | 0.704 | 0.80 | 0.78 |
| HiRISE-DEV tuned, fixed_b5 | 0.676 / 0.140 | 0.698 | 0.81 | 0.78 |
| B1 local std | 0.643 / 0.158 | 0.658 | - | - |
Brightness cue transfers (grad+shadow 0.605 -> full 0.654); shadow cue adds ~0 (high sun) and hurts on
dark-toned DEV sites. Site selection is at the random-square rate (0.80 vs 0.78 at 20 m; 0.68 vs 0.73
at 50 m); every tile has a safe 50 m square. Per site, V_full AUROC: Eberswalde 0.80, Jezero 0.76,
Meridiani 0.72, Arcadia 0.58.

## Folded into the paper (2026-11-17): physical normalization + real HiRISE section
- pipeline.py: G uses the 5x5-blurred Sobel gradient / g_ref = 8 I_ref cot(e) sin15 / 5 (p99 kept as
  `normalization='p99'`); brightness cue now B = min(1, |blur_k(I) - I_ref| / (I_ref cot(e) sin phi)),
  phi tuned in degrees. I_ref = image median; e = sun elevation (input).
- DEV tuning (theta grid widened to {0.05..0.5}): theta 0.05, mca 100, tau 60, A_min 500, phi 10, k 15.
- Synthetic TEST: V_full AUROC 0.866 / PR-AUC 0.316; Safe Landing 0.75 from 5 to 157 m (ceiling: rocky
  tiles have no safe 20 m square), 0.725 at 200 m. V_grad 0.725 (2 crater boxes in the crater).
  Note: box-level success is helped by the selector (largest region = open plain); V_grad crater
  within-tile AUROC is only 0.497, V_full 0.930.
- Ablation (Table 5): p99 raw 0.764/0.375, SL50 0; fixed raw 0.795/0.188; p99 at 5 m 0.717/0.350,
  SL50 0.237; physical at 5 m 0.866/0.316, SL50 0.75.
- Sun sweep: V_full 0.633 / 0.866 / 0.878 / 0.879 AUROC at 15/30/45/60 deg; at 15 deg the absolute
  tau = 60 nears level-ground brightness (66) and fragments benign tiles (SL10 0.50).
- HiRISE TEST (transfer): V_full 0.654 / 0.134, within 0.704; B1 0.643 / 0.158 / 0.658; re-tuned on
  real DEV 0.661. Site selection 0.80 / 0.68 at 20 / 50 m vs random null 0.78 / 0.73.
- Runtime (768): front-end 28.5 ms (gradient 13.5, shadow 1.3, brightness 9.9, threshold 3.8),
  RSS +30 MB.

## Photometric robustness (2026-11-17, in paper Section 5.6 / Table 7)
`run_photometric_robustness.py`: TEST re-rendered (baseline reproduces stored images exactly) with skylight
a = 0.1..1.0, Lunar-Lambert L = 0.5 / 1.0, smooth and two-tone albedo at 10-30%, and a combined model.
Frozen V_full: skylight and Lunar-Lambert do not hurt (AUROC 0.866-0.882); albedo does: 10% -> ~0.82,
20% -> 0.70-0.73, 30% -> 0.60-0.67, below V_grad+shadow from 20%; Safe Landing at 100 m 10-61%.
`run_albedo_fix.py`: larger brightness window and selection on mixed-albedo DEV (k = 51, phi = 20,
tau = 20). Pre-stated criterion: V_full >= V_grad+shadow on every albedo variant, baseline within 0.02,
envelope kept. Result: FAILED only on two-tone albedo at 30% (0.670 vs 0.687); baseline 0.853, envelope
kept; recovers 0.754-0.837 under albedo. Per the authors' instruction the main pipeline is unchanged and
the fix is reported as a mitigation in the robustness section; abstract, limitations and conclusion state
the albedo sensitivity.
