# Decisions

- **Code Repository Absence**: The prompt stated the workspace contained the code repository, but it was not present. I initialized a new git repository in the workspace and implemented `v0` (the original pipeline) strictly following the equations and parameters provided in the LaTeX source.
- **Python Version**: Decided to use Python 3.10 with standard scientific libraries (numpy, scipy, OpenCV).
- **Sobel Image Input**: Assumed input image to Sobel is a float array scaled to [0, 1] or [0, 255]. Decided on float32 [0, 255] for mathematical precision without uint8 overflow, matching standard practice.
- **Sobel Border Handling**: Used `BORDER_REPLICATE` (or `cv2.BORDER_DEFAULT`) which is a conservative standard.
- **FFT Augmented Image**: The paper mentions "augmented image" for the FFT step, which I interpret as the image possibly after some preprocessing, but conservatively, it just means the input image, as no augmentation is explicitly defined. I'll use the original grayscale image.
- **Maximal Rectangle**: Used a standard maximal rectangle algorithm based on histograms.

## Footprint-envelope revision (2026-11-08)

- **Footprint as a parameter.** The required lander footprint side L is treated as an operating
  parameter swept over {20, 25, 30, 35, 40, 50} m (Table 3) and 5..60 m in 1 m steps (Figure 2),
  instead of the single fixed 50 m x 50 m requirement. At 1.0 m GSD, L m = L px.
- **Square-fit criterion.** The area criterion w*h >= L^2 accepts elongated boxes that cannot
  contain an L x L touchdown zone (e.g. 10 x 100 passes at L = 25). Square Fit,
  min(w, h) >= L, is the physically meaningful test and is required for Safe Landing. On TEST this
  matters: Footprint OK at 25 m is 68.8% but Square Fit is 7.5%.
- **epsilon = 0.05, fixed a priori.** The contamination tolerance was set in the revised paper text
  before TEST results were computed and was not adjusted afterwards. frac_unsafe exactly equal
  to epsilon passes (<=).
- **No hyperparameter changes.** The sweep is post-hoc evaluation of the boxes that produced
  Table 2 (`results/best_config.json`, `splits.json`, proxy labels at 15 deg / 5 m). Nothing was
  retuned. Denominator is always all 80 TEST tiles.
- **L\*** is the largest integer L in 1..60 with V_grad+shadow Safe Landing >= 0.5 (= 9 m).
- **B1 (local std) not in the sweep.** Table 2 never runs maximal-rectangle selection for B1 (its
  box is hard-coded to empty), so there is no B1 box to evaluate.
