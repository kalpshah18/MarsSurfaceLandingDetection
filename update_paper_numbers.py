"""DISABLED (footprint-envelope revision).

This script rewrote an earlier version of the paper (old title, F1-based
abstract, "fundamentally insufficient" framing). Running it against the
current paper in Mars_Landing_Paper_ARXIV_Style_updated/ would overwrite the
revised text with obsolete wording, so it now exits immediately. It is kept
only as a record of how earlier drafts were produced. Paper numbers are now
filled by hand from results/*.json; see CLAIMS.md for the source of each.
"""
import sys
sys.exit("update_paper_numbers.py is disabled: it would overwrite the revised paper with obsolete text. See its docstring.")

import json
import re

with open('results/summary.json', 'r') as f:
    summary = json.load(f)
with open('results/complexity.json', 'r') as f:
    complexity = json.load(f)

f1 = summary['TEST']['f1']['mean']
iou = summary['TEST']['iou']['mean']
runtime = complexity['1024']['Total']['median']

with open('templateArxiv_revised.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Update abstract
new_abstract = f"""\\begin{{abstract}}
Safe autonomous landing on planetary surfaces requires quick, accurate decisions under tight computational limits. This paper presents a proof-of-concept classical, deterministic computer vision pipeline for identifying safe landing zones on Mars from high-resolution single-frame nadir descent imagery. The pipeline fuses gradient-based intensity edge evaluation, contour-based shadow masking, and FFT-based image texture classification into a unified hazard score. A maximal-rectangle algorithm then extracts the largest contiguous safe landing zone. Evaluated on a dataset utilizing DTM-derived proxy labels, the system achieves a proxy-label F1 score of {f1:.3f} and IoU of {iou:.3f}. The pipeline's deterministic nature and low computational requirements ({runtime:.1f} ms on a 1024x1024 image on standard host hardware) make it a potential lightweight alternative for future deployment on flight processors, though it currently struggles with hazard proximity without explicit distance penalties.
\\end{{abstract}}"""

# Use lambda to avoid re.sub processing backslashes in replacement string
tex = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda _: new_abstract, tex, flags=re.DOTALL)

with open('templateArxiv_revised.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

# Re-run figure generation
import generate_figures
generate_figures.generate_figures()
