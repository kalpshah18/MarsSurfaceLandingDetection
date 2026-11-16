"""Crater-model figure: radial profile along the sun line with the grazing sun ray,
for the shallowest and deepest craters in the dataset (d/D = 0.10 and 0.20), plus
their rendered images. Uses the same crater_dtm() and render() as the generator.

    python generate_crater_model_figure.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from generate_synthetic import DD_RANGE, SUN_AZIMUTH, SUN_ELEVATION, crater_dtm
from render import render

OUT_DIR = 'Mars_Landing_Paper_ARXIV_Style_updated/media'
D = 400.0
N = 640


def main():
    R = D / 2
    c = N / 2
    x, y = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float))
    ux, uy = np.cos(np.radians(SUN_AZIMUTH)), np.sin(np.radians(SUN_AZIMUTH))
    tan_el = np.tan(np.radians(SUN_ELEVATION))

    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.0, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1], wspace=0.08)
    axp = fig.add_subplot(gs[0])
    colors = {DD_RANGE[0]: '#2a78d6', DD_RANGE[1]: '#eb6834'}
    for k, dd in enumerate(DD_RANGE):
        z = crater_dtm(x, y, c, c, R, dd)
        img, sh = render(z, SUN_AZIMUTH, SUN_ELEVATION)
        s = np.arange(-1.6 * R, 1.6 * R, 0.5)  # distance along the sun line (+ = toward the sun)
        zi = crater_dtm(c + s * ux, c + s * uy, c, c, R, dd)
        axp.plot(s, zi, color=colors[dd], lw=1.5,
                 label=f'd/D {dd:.2f}: depth {dd * D:.0f} m, wall ≤ {np.degrees(np.arctan(4 * dd)):.0f}°')
        # Grazing ray over the sun-side rim crest, drawn until it meets the terrain
        h_r = zi.max()
        ray = h_r - (R - s) * tan_el
        above = (ray >= zi - 1e-6) & (s <= R)
        s_hit = R
        for si, ok in zip(s[::-1], above[::-1]):
            if si > R:
                continue
            if not ok:
                break
            s_hit = si
        seg = (s >= s_hit) & (s <= 1.45 * R)
        axp.plot(s[seg], ray[seg], color=colors[dd], lw=0.8, ls='--')
        sh_in = (s >= s_hit) & (s <= R) & (ray > zi)
        if sh_in.any():
            axp.fill_between(s, zi, ray, where=sh_in, color=colors[dd], alpha=0.25, lw=0)
            axp.text(0.5 * (s_hit + R), zi[sh_in].min() - 0.03 * D, 'shadow', color=colors[dd],
                     fontsize=6.5, ha='center', va='top')
        frac = 100 * sh.sum() / (np.pi * R * R)
        ax_img = fig.add_subplot(gs[1 + k])
        ax_img.imshow(img[int(c - 1.3 * R):int(c + 1.3 * R), int(c - 1.3 * R):int(c + 1.3 * R)],
                      cmap='gray', vmin=0, vmax=255)
        ax_img.set_title(f'd/D = {dd:.2f}\n' + ('no cast shadow' if frac < 0.5 else
                                                 f'shadow: {frac:.0f}% of crater area'), fontsize=7)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        for sp in ax_img.spines.values():
            sp.set_edgecolor(colors[dd])
    z0 = 0.10 * D
    axp.annotate('', xy=(1.05 * R, z0), xytext=(1.5 * R, z0 + 0.45 * R * tan_el),
                 arrowprops=dict(arrowstyle='-|>', color='#3d3d3a', lw=1.2))
    axp.text(1.52 * R, z0 + 0.45 * R * tan_el, f'sun\n{SUN_ELEVATION:.0f}°', fontsize=6.5,
             va='center', ha='left')
    axp.set_ylim(-0.24 * D, 0.30 * D)
    axp.set_xlabel('Distance along sun line from crater centre (m; + toward sun)')
    axp.set_ylabel('Elevation (m)')
    axp.set_title(f'Profile along the sun line, D = {D:.0f} m (vertically exaggerated)',
                  fontsize=7.5)
    axp.legend(fontsize=6.3, loc='upper left', frameon=False)
    axp.grid(True, color='#e6e6e1', lw=0.6)
    for side in ('top', 'right'):
        axp.spines[side].set_visible(False)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'fig_crater_model.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('Saved', os.path.join(OUT_DIR, 'fig_crater_model.{pdf,png}'))


if __name__ == '__main__':
    main()
