# Claims

| Claim Sentence | Value | Artifact Key | Source File |
| :--- | :--- | :--- | :--- |
| Evaluated on a synthetic dataset utilizing DTM-derived proxy labels (safe = slope $\le$ 15 degrees), the system achieves a proxy-label F1 score of 0.025 | F1=0.025 | `TEST.f1.mean` | `results/summary.json` |
| which is marginally above a random baseline (0.024) | B5 F1=0.024 | `B5_f1.mean` | `results/baselines.csv` |
| The pipeline's measured execution time is 135.2 ms on a 1024x1024 image | 135.2 | `1024.Total.median` | `results/complexity.json` |
| The full pipeline (v0) achieves a pixel-level macro-averaged F1 score of 0.025 | F1=0.025 | `TEST.f1.mean` | `results/summary.json` |
| unsafe recall of 0.935 | Unsafe recall=0.935 | `TEST.unsafe_recall.mean` | `results/summary.json` |
| a local standard deviation baseline (B1) achieves F1=0.025 | B1 F1=0.025 | `B1_f1.mean` | `results/baselines.csv` |
| Gradient-only ablation (A1) yields F1=0.025 | A1 F1=0.025 | `A1_f1.mean` | `results/ablations.csv` |
| The Spearman correlation between the gradient hazard $G$ and true DTM slope is 0.340 | Spearman=0.340 | Printed via `run_diagnosis.py` | `DIAGNOSIS.md` |
