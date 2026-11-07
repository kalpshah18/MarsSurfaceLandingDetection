# Characterizing the Operational Bounds of Single-Frame 2D Hazard Detection for Autonomous Planetary Landers

Code, results and arXiv paper (`Mars_Landing_Paper_ARXIV_Style_updated/`) for a deterministic,
single-frame hazard detector for planetary landing: a Sobel gradient hazard score, morphological
shadow augmentation, and maximal-rectangle landing-box selection. It is evaluated against
slope-derived proxy labels (unsafe: slope > 15 deg over 5 m) on 80 held-out synthetic Martian
tiles from sites geographically disjoint from development.

Shadow augmentation corrects an inverted gradient response in shadowed craters (AUROC 0.350 ->
0.796). Rather than testing one fixed lander footprint, the paper treats the required footprint
side L as an operating parameter. The selected regions are small and elongated. At L = 25 m,
68.8% of tiles meet the area requirement, but only 7.5% yield a region that contains a 25 m
square with at most 5% unsafe terrain. That safe landing rate reaches 50% only up to L = 9 m,
never exceeds 60% (rocky-terrain boxes and most crater boxes exceed the 5% tolerance), and is
zero above 31 m, including the 50 m large-lander case. Single-frame 2D sensing is therefore usable
alone only for footprints of order 10 m; beyond that it is a low-cost cueing or fallback stage
for 3D sensors.

## Reproduce

Requires the synthetic tiles in `data/synthetic/` (git-ignored) and `pip install -r requirements.txt`.

```bash
bash run_all.sh
```

This runs the unit tests and `compute_final_metrics.py` (Tables 1-2). It then runs the footprint sweep:

```bash
python export_per_tile_boxes.py      # results/per_tile_boxes.csv (same boxes as Table 2)
python run_footprint_sweep.py        # results/footprint_sweep*.json, Tables 3-4 rows, L*
python generate_figures.py footprint # media/fig_footprint_envelope.pdf
```

`run_footprint_sweep.py` aborts unless L = 50 m reproduces Table 2 exactly. See `CLAIMS.md` for
the source of every number in the paper, and `DECISIONS.md` / `REPORT.md` for methodology
decisions and known issues.

The paper is built with `pdflatex -> bibtex -> pdflatex -> pdflatex` on `templateArxiv.tex`.
Its LaTeX style is the PRIME arXiv template by Moulay Akhloufi, adapted from George Kour's
arxiv-style (MIT License).
