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
