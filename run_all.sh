#!/bin/bash
set -e

echo "Running Mars Landing Site Identification Pipeline"
echo "Phase 0: Environment already set up"

echo "Phase 1 & 2: Generating synthetic dataset (Proxy for actual download)"
python generate_synthetic.py

echo "Phase 3: Running evaluation harness"
python evaluate.py

echo "Phase 4: Running ablations (stub)"
python run_ablations.py

echo "Pipeline execution complete."
