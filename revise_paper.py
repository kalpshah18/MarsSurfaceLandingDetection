import re

with open('templateArxiv.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Title
tex = re.sub(r'\\title\{Identifying Landing Sites for Mars\s*Landers\}',
             r'\\title{Single-Frame Safe-Zone Selection from Descent Imagery for Mars Landers}', tex)

# Abstract
abstract_new = r"""\begin{abstract}
Safe autonomous landing on planetary surfaces requires quick, accurate decisions under tight computational limits. This paper presents a proof-of-concept classical, deterministic computer vision pipeline for identifying safe landing zones on Mars from high-resolution single-frame nadir descent imagery. The pipeline fuses gradient-based intensity edge evaluation, contour-based shadow masking, and FFT-based image texture classification into a unified hazard score. A maximal-rectangle algorithm then extracts the largest contiguous safe landing zone. Evaluated on a dataset utilizing DTM-derived proxy labels, the system demonstrates an ability to avoid high-gradient terrain features and rough terrain, achieving measurable proxy-label F1 scores. The pipeline's deterministic nature and measured low computational requirements make it a potential lightweight alternative for future deployment on flight processors, though it currently struggles with hazard proximity without explicit distance penalties.
\end{abstract}"""
tex = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', abstract_new, tex, flags=re.DOTALL)

# Introduction - Paragraph 2
intro_p2 = r"""Landing suitability is strictly bounded by mission-specific engineering constraints \cite{golombek2012selection}, relying on terrain properties including slope, surface roughness, shadows, and local surface texture. Existing approaches use range sensors \cite{amzajerdian2013alhat} or learning-based vision methods \cite{moghe2020deep}. Deep learning methods can provide strong visual representations, but their computational and power requirements can be challenging to accommodate on some radiation-hardened processors (e.g., the RAD750 \cite{berger2001rad750}) used for spaceflight, though the success of Mars 2020 Terrain Relative Navigation and the FPGA-based Lander Vision System demonstrate ongoing progress in flight computing \cite{johnson2020mars2020}. However, these processors still have substantially tighter power, memory, and compute constraints than modern terrestrial hardware, motivating lightweight deterministic approaches.

We assume the input consists of nadir orbital-style frames at a stated GSD (e.g., 1 m/px), rather than highly oblique descent imagery. The primary target application is pre-flight mapping and early terminal descent hazard detection."""
tex = re.sub(r'Landing suitability is strictly bounded by mission-specific engineering constraints.*?motivating lightweight deterministic approaches\.', intro_p2, tex, flags=re.DOTALL)

# Introduction - Paragraph 3
intro_p3 = r"""The main difficulty is the lack of pixel-level landing-safety annotations for orbital or descent imagery. While comprehensive proxy datasets exist (e.g., HiRISE stereo DTMs, rock-abundance studies, crater catalogues, and site-selection hazard assessments), there is no ground-truth label indicating whether a region of Martian terrain is physically safe for landing. This work therefore treats landing safety as an estimation problem: observable visual features are combined to produce an approximate map of terrain suitability.

Our contributions are: (1) a deterministic classical computer-vision pipeline, (2) a DTM-derived proxy-label evaluation protocol with reproducible code, (3) ablation, baseline, and sensitivity analyses, and (4) measured computational complexity on standard hardware."""
tex = re.sub(r'The main difficulty is the lack of ground-truth landing-safety annotations.*?ground-truth classification\.', intro_p3, tex, flags=re.DOTALL)

# Method changes
tex = tex.replace('The selection of the threshold is empirical and is derived from a Martian image dataset available.', '')

# Add Border replicate
tex = tex.replace(r'horizontal and vertical partial derivatives are obtained via the Sobel operator\cite{sobel1968} with a standard $3 \times 3$ kernel size:',
                  r'horizontal and vertical partial derivatives are obtained via the Sobel operator\cite{sobel1968} with a standard $3 \times 3$ kernel size and border replication handling:')

tex = tex.replace('The augmented image is partitioned', 'The original grayscale image (referred to as the augmented image hereafter) is partitioned')

tex = tex.replace('below a minimum area threshold, retaining only', 'below a minimum area threshold of $1500$ pixels, retaining only')

# Replace Conclusion
conclusion_new = r"""\section{Conclusion}
This paper presented a classical, deterministic computer vision pipeline for identifying safe landing zones from single-frame nadir descent imagery. By fusing Sobel-based intensity gradient hazard scoring, contour-based shadow augmentation, and an amplitude-invariant FFT-based texture-energy score, the pipeline produces a per-pixel safety map from which the largest viable landing rectangle is extracted. Evaluation using DTM-derived proxy labels demonstrated that the system can avoid major hazards such as crater rims and deep shadowed depressions, while also revealing a limitation regarding hazard proximity. Because the pipeline relies on no training data and uses only lightweight, deterministic operations, it offers a potential lightweight alternative for deployment on radiation-hardened flight processors."""
tex = re.sub(r'\\section\{Conclusion\}.*?\\section\*\{Acknowledgments\}', conclusion_new + '\n\n' + r'\section*{Acknowledgments}', tex, flags=re.DOTALL)

with open('templateArxiv_revised.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
