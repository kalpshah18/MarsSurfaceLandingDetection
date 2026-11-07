#!/bin/bash
# Reproduces the numbers and the footprint figure in the paper from the
# existing synthetic tiles in data/synthetic/ and the split in splits.json.
#
# Note: generate_synthetic.py, evaluate.py, run_ablations.py, run_experiments.py
# and run_diagnosis.py import functions (run_v0, compute_fft_texture) that are no
# longer in pipeline.py, so they are not run here. The dataset is not
# regenerated; data/synthetic/ must already exist.
set -e

echo "Unit tests"
python -m pytest -q test_pipeline.py

echo "Tables 1 and 2 (pooled unsafe-class metrics, landing boxes at L = 50 m)"
python compute_final_metrics.py

echo "Per-tile landing boxes (same boxes as Table 2)"
python export_per_tile_boxes.py

echo "Footprint sweep (Tables 3 and 4, L*); aborts if L = 50 does not reproduce Table 2"
python run_footprint_sweep.py

echo "Footprint envelope figure"
python generate_figures.py footprint

echo "Done."
