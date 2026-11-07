"""DISABLED (footprint-envelope revision).

This script rewrote an earlier version of the paper (old title, F1-based
abstract, "fundamentally insufficient" framing). Running it against the
current paper in Mars_Landing_Paper_ARXIV_Style_updated/ would overwrite the
revised text with obsolete wording, so it now exits immediately. It is kept
only as a record of how earlier drafts were produced. Paper numbers are now
filled by hand from results/*.json; see CLAIMS.md for the source of each.
"""
import sys
sys.exit("clean_banned_terms.py is disabled: it would overwrite the revised paper with obsolete text. See its docstring.")

import re

with open('templateArxiv_revised.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Banned term removals
tex = re.sub(r'slope evaluation', 'intensity gradient evaluation', tex, flags=re.IGNORECASE)
tex = re.sub(r'roughness classification', 'texture classification', tex, flags=re.IGNORECASE)
tex = re.sub(r'rover deployment zone', 'lander deployment zone', tex, flags=re.IGNORECASE)
tex = re.sub(r'Mars Rover', 'Mars Lander', tex, flags=re.IGNORECASE)
tex = re.sub(r'no dataset exists', 'no pixel-level landing-safety dataset exists', tex, flags=re.IGNORECASE)
tex = re.sub(r'heta', 'theta', tex)
tex = re.sub(r'\?\?', '', tex)
tex = re.sub(r'TODO', '', tex)
tex = re.sub(r'NOT OBTAINED', '', tex)

with open('templateArxiv_revised.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
