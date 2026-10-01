# Final Evaluation Summary

## 1. Audit Findings on Original Metrics
The original reporting (macro F1 0.897 vs full pipeline 0.025) suffered from severe metric inconsistencies. The baseline and gradient-only ablations were computed using `macro` averaging, artificially inflating the score due to the overwhelming class imbalance (the average proxy-safe fraction across the entire dataset is exactly **90.2%**). The full pipeline F1 was inexplicably computed on the unsafe class, while the random baseline was evaluated as if predicting the heavily imbalanced class. These metrics were completely uncomparable. 

## 2. Headline Results (Safe-Class Metrics, 95% CIs)
*V_grad outperforms the LocalStd baseline (B1) slightly in Precision but is worse in AUROC due to the heavily skewed nature of the gradients. Shadow augmentation significantly recovers performance.*

| Method | Precision | Recall | F1 | AUROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V_grad** | 0.907 [0.880, 0.931] | 0.960 [0.959, 0.962] | 0.929 [0.913, 0.942] | 0.380 [0.311, 0.448] | 0.753 [0.698, 0.801] |
| **V_grad_shadow** | 0.952 [0.938, 0.965] | 0.960 [0.959, 0.962] | 0.955 [0.947, 0.962] | 0.752 [0.703, 0.802] | 0.911 [0.897, 0.926] |
| **V_fft_ablation** | 0.115 [0.073, 0.160] | 0.007 [0.004, 0.011] | 0.013 [0.008, 0.020] | 0.724 [0.683, 0.767] | 0.878 [0.869, 0.887] |
| **B1_LocalStd** | 0.906 [0.880, 0.930] | 0.970 [0.959, 0.980] | 0.933 [0.916, 0.949] | 0.336 [0.271, 0.401] | 0.740 [0.686, 0.786] |
| **B5_Random** | 0.905 [0.878, 0.929] | 0.958 [0.956, 0.961] | 0.926 [0.911, 0.940] | 0.499 [0.498, 0.500] | 0.809 [0.778, 0.836] |

## 3. Box-Level Results
*The maximal rectangle algorithm acts as a strict spatial constraint. V_grad found boxes on 100% of tiles, but 50% contained unsafe terrain.*

| Method | Box Found | Unsafe Tol (5%) | Rel Area | Footprint OK (>2500px) |
| :--- | :--- | :--- | :--- | :--- |
| **V_grad** | 1.000 [1.000, 1.000] | 0.500 [0.388, 0.613] | 0.119 [0.073, 0.179] | 0.250 [0.163, 0.350] |
| **V_grad_shadow** | 1.000 [1.000, 1.000] | 0.600 [0.487, 0.700] | 0.031 [0.021, 0.042] | 0.000 [0.000, 0.000] |
| **B1_LocalStd** | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## 4. Runtime
*Runtime of V_grad on a $1024 \times 1024$ image (Intel CPU, single-threaded NumPy/OpenCV)*
- Sobel Gradient: ~2-5 ms
- Percentile Norm: ~5 ms
- Maximal Rectangle: ~0.5s - 1.5s
- Total Median Runtime: ~1.0s

## 5. Paper Edits
- **Abstract & Title**: Reframed to "Characterizing Single-Frame Safe-Zone Estimation", acknowledging limitations. Removed "proof of concept" framing.
- **Introduction**: Shifted claims away from physical landing safety toward characterizing DTM geometry proxy performance.
- **Method**: Removed Section 3.3 (FFT Texture) completely as the core algorithm, maintaining it only for ablation. Updated to state $s = 1 - G$.
- **Dataset**: Added dataset description, reporting the 90.2% proxy-safe fraction and the 80-tile splits.
- **Results**: Completely rebuilt the results tables pulling directly from `results/aggregate_metrics.csv` without manual typing.
- **Discussion/Limitations**: Acknowledged the Lambertian rendering dependence, nadir-tilt blindspot, and proxy-label limitations explicitly.

## 6. Claim Framing Conclusion
The results **DO NOT support the positive claim** that single-frame intensity gradients are sufficient. `V_grad` achieves a PR-AUC of 0.753, which is nominally worse than the Random baseline (0.809) at predicting the heavily skewed safe class (90.2%), and its AUROC is worse than random chance (0.380) because bright crater rims (high gradient) align with slopes, but uniform tilted walls (high slope) yield near-zero gradient, creating massive false negatives.

Therefore, the paper has been explicitly reframed as a **characterization of the limits of single-frame intensity-based hazard scoring**, as requested by the prompt.
