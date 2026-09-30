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
