#!/bin/bash
# Reproduces every number and figure in the paper from scratch (dataset v2:
# paraboloidal craters, Lambertian shading + ray-marched cast shadows, no painted shadow).
#
# Order matters: the dataset is generated first, hyperparameters are tuned on DEV,
# and every later script reads results/best_config.json.
# Runtime: about 2-3 hours on a laptop CPU (the review analyses and sun sweep dominate).
set -e

echo "Dataset (overwrites data/synthetic/, data/MANIFEST.csv, splits.json)"
python generate_synthetic.py

echo "Unit tests"
python -m pytest -q test_pipeline.py

echo "Tuning on 8 DEV tiles (two per terrain) -> results/best_config.json"
python run_tuning.py

echo "Tables 1 and 2 -> results/final_metrics.json"
python compute_final_metrics.py

echo "Per-tile landing boxes"
python export_per_tile_boxes.py

echo "Footprint sweep (Tables 3 and 4); aborts if L = 50 does not reproduce Table 2"
python run_footprint_sweep.py

echo "Crater diagnosis (Section 5.1)"
python run_crater_diagnosis.py

echo "Gradient-normalization ablation (Table 5)"
python run_normalization_study.py

echo "Review analyses and sun-angle sweep (Table 6)"
python run_review_analyses.py
python run_review_sweeps.py sunphys

echo "Real HiRISE evaluation (Tables 8-9); needs the tiles from hirise_download.py and"
echo "hirise_make_tiles.py under HIRISE_DATA_DIR (about 4.4 GB, outside the repo)"
if [ -n "$HIRISE_DATA_DIR" ]; then python hirise_eval_v2.py; else echo "HIRISE_DATA_DIR not set; skipped"; fi

echo "Runtime and memory (Table 5); run on an otherwise idle machine"
python benchmark_stages.py

echo "Figures"
python -c "import generate_figures as g; g.generate_footprint_envelope_figure(out_stem='Mars_Landing_Paper_ARXIV_Style_updated/media/fig_footprint_envelope')"
python generate_qualitative_figure.py crater_2_16
python generate_crater_model_figure.py

echo "Done."
