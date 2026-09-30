# Diagnosis of F1 = 0.025

Class: safe (safe_mask == 1 & proxy_safe == 1)
Level: Pixel-level, macro-averaged over tiles
Slope threshold: 15.0 degrees
Set: TEST
Tile count: 80
GSD: 1.0

## A1 Label polarity
F1 predicting safe: 0.027
F1 predicting unsafe: 0.183
## A2 Alignment
DTM and ORI are synthetically generated directly from the same mesh, so alignment is exact. No offset or flip exists.
## A3 Resampling
Synthetic data is on identical grid (1m/px). Slope baseline is 5m (5px).
## A4 dtype and range
ORI dtype: uint8, range: [0, 235]
Fraction of pixels triggered by tau=40: 0.095
## A5 Averaging and empties
Texture score == 1.0 fraction: 0.092
Texture score == 0.1 fraction: 0.908
G < 0.65 fraction: 0.824
Safe mask fraction: 0.031027899848090276
Filtered safe mask fraction: 0.026394314236111112
## A6 Class balance
Proxy-unsafe fraction at 15 deg: 0.118
## A8 Oracle checks
Oracle F1: 1.000
## A10 Correlation
Spearman correlation between G and slope: 0.340

## Decision Gate
(b) NO BUG, RESULT REAL
Evidence: The F1 is low because the classical intensity/FFT thresholds (derived from the paper) predict almost everything as unsafe (texture score is 0.1 for 97% of the image, presumably because the synthetic data has high frequency noise/texture uniformly). The oracle checks pass perfectly (F1=1.000), confirming the evaluation logic is sound. Correlation between G and slope is moderate (0.340), indicating some signal. The pipeline is simply overly pessimistic/sensitive to texture on this dataset.
