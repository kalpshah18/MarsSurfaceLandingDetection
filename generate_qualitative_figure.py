"""Figure 1: qualitative progression on one crater TEST tile.

Panel (a) is annotated with the sun direction, the rim, the ray-marched cast shadow
and the crater geometry from data/MANIFEST.csv. All numbers printed on the figure are
computed here, not hard-coded.

    python generate_qualitative_figure.py [tile_id]
"""
import json
import os
import sys

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_synthetic import SUN_AZIMUTH, SUN_ELEVATION
from pipeline import (augment_shadow, compute_brightness_hazard, compute_gradient_hazard, run_v_bright, run_v_grad,
                      run_v_grad_shadow)
from render import cast_shadow
from run_evaluation import compute_proxy_labels

OUT_DIR = 'Mars_Landing_Paper_ARXIV_Style_updated/media'
LABEL_BOX = dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)


def pct_unsafe(unsafe, rect):
    x, y, w, h = rect
    return 100.0 * unsafe[y:y + h, x:x + w].mean() if w > 0 and h > 0 else float('nan')


def generate_figure(tile_id='crater_2_0'):
    row = pd.read_csv('data/MANIFEST.csv').set_index('tile_id').loc[tile_id]
    ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
    dtm = np.load(row['dtm'])
    unsafe, slope_deg = compute_proxy_labels(dtm)
    shadow_true = cast_shadow(dtm, SUN_AZIMUTH, SUN_ELEVATION)

    with open('results/best_config.json') as f:
        cfg = json.load(f)
    gs_cfg, g_cfg = cfg['V_grad_shadow'], cfg['V_grad']
    G = compute_gradient_hazard(ori)
    G_shadow, shadow_mask = augment_shadow(ori, G.copy(), tau=gs_cfg['tau'], a_min=gs_cfg['a_min'])
    _, _, _, rect_g = run_v_grad(ori, **g_cfg)
    _, _, _, rect_gs = run_v_grad_shadow(ori, **gs_cfg)
    pu_g, pu_gs = pct_unsafe(unsafe, rect_g), pct_unsafe(unsafe, rect_gs)
    br_cfg = cfg['V_bright']
    B = compute_brightness_hazard(ori, phi_deg=br_cfg['phi_deg'], ksize=br_cfg['ksize'])
    H_full = np.maximum(G_shadow, B)
    _, _, _, rect_br = run_v_bright(ori, **br_cfg)
    pu_br = pct_unsafe(unsafe, rect_br)

    cx, cy, D = float(row['crater_cx']), float(row['crater_cy']), float(row['crater_D_m'])
    R = D / 2.0
    yy, xx = np.mgrid[0:ori.shape[0], 0:ori.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < R ** 2
    shadow_in = 100.0 * shadow_true[inside].mean()
    print(f'{tile_id}: D {D:.0f} m, depth {row["crater_depth_m"]:.1f} m, d/D {row["crater_dd"]:.3f}, '
          f'cast shadow {shadow_in:.1f}% of crater interior; V_grad box {pu_g:.1f}% unsafe, '
          f'V_grad+shadow box {pu_gs:.1f}% unsafe, V_full box {pu_br:.1f}% unsafe')

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), dpi=300)
    plt.subplots_adjust(wspace=0.14, hspace=0.22, left=0.03, right=0.97, top=0.94, bottom=0.04)

    # (a) input image, annotated
    ax = axes[0, 0]
    ax.imshow(ori, cmap='gray', vmin=0, vmax=255)
    ax.add_patch(patches.Circle((cx, cy), R, fill=False, ec='#4fc3f7', lw=1.0, ls=':'))
    main_shadow = np.zeros_like(inside)
    if shadow_true[inside].any():
        # Outline the main cast shadow only (the DTM noise adds a few isolated shadowed pixels).
        n, lab, stats, _ = cv2.connectedComponentsWithStats((shadow_true & inside).astype(np.uint8), connectivity=8)
        main_shadow = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ax.contour(main_shadow, levels=[0.5], colors=['#ff9800'], linewidths=1.2)
        sy, sx = np.argwhere(main_shadow).mean(axis=0)
        ax.annotate('cast shadow', xy=(sx, sy), xytext=(ori.shape[1] * 0.97, max(sy - 0.45 * R, ori.shape[0] * 0.45)),
                    color='#ff9800', fontsize=8, ha='right',
                    arrowprops=dict(arrowstyle='->', color='#ff9800', lw=1.0))
    ux, uy = np.cos(np.radians(SUN_AZIMUTH)), np.sin(np.radians(SUN_AZIMUTH))
    ax.annotate('sunlit far wall', xy=(cx - 0.6 * R * ux, cy - 0.6 * R * uy),
                xytext=(max(cx - 1.3 * R, 90), max(cy - 1.5 * R, 40)), color='#ffeb3b', fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='#ffeb3b', lw=1.0))
    H_img, W_img = ori.shape
    ax.annotate('', xy=(W_img * 0.82, H_img * 0.08), xytext=(W_img * 0.96, H_img * 0.22),
                arrowprops=dict(arrowstyle='-|>', color='#ffeb3b', lw=2.0))
    ax.text(W_img * 0.97, H_img * 0.25, f'sunlight from\nlower right\naz {SUN_AZIMUTH:.0f}°, el {SUN_ELEVATION:.0f}°',
            color='#ffeb3b', fontsize=7.5, ha='right', va='top')
    ax.set_title('(a) Input Monocular Image', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, f'D = {D:.0f} m, depth = {row["crater_depth_m"]:.0f} m (d/D = {row["crater_dd"]:.2f})\n'
                        f'Cast shadow: {shadow_in:.0f}% of crater area',
            transform=ax.transAxes, color='white', fontsize=8, bbox=LABEL_BOX)
    ax.axis('off')

    # (b) DTM slope
    ax = axes[0, 1]
    im = ax.imshow(slope_deg, cmap='inferno', vmin=0, vmax=40)
    ax.contour(unsafe, levels=[0.5], colors=['cyan'], linewidths=1.2, linestyles='--')
    ax.set_title('(b) DTM Slope & True Hazard', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, 'Cyan: Slope > 15°\n(True Hazardous Region)', transform=ax.transAxes,
            color='cyan', fontsize=8.5, bbox=LABEL_BOX)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Slope (deg)', fontsize=8.5)
    ax.axis('off')

    # (c) raw gradient hazard
    ax = axes[0, 2]
    im = ax.imshow(G, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('(c) Raw Gradient Hazard $G(x,y)$', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, 'Smooth shading on the bowl\nand zero inside the shadow:\nlow $G$ on steep walls',
            transform=ax.transAxes, color='yellow', fontsize=8.5, bbox=LABEL_BOX)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Gradient Hazard Score', fontsize=8.5)
    ax.axis('off')

    # (d) shadow mask
    ax = axes[0, 3]
    ax.imshow(shadow_mask, cmap='gray')
    ax.set_title(r'(d) Detected Shadow Mask ($\tau=%d$)' % gs_cfg['tau'], fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, 'Morphological Contours\n' + r'Area Filter ($A_{min} \geq %d$ px)' % gs_cfg['a_min'],
            transform=ax.transAxes, color='lime', fontsize=8.5, bbox=LABEL_BOX)
    ax.axis('off')

    # (e) augmented hazard
    ax = axes[1, 0]
    im = ax.imshow(G_shadow, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('(e) Augmented Hazard Map', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, 'Shadow Overridden:\n' + r'$G(x,y) \leftarrow 1.0$', transform=ax.transAxes,
            color='white', fontsize=8.5, bbox=LABEL_BOX)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Augmented Hazard Score', fontsize=8.5)
    ax.axis('off')

    # (f) brightness cue
    ax = axes[1, 1]
    im = ax.imshow(B, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(r'(f) Brightness Cue $B(x,y)$', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, 'Deviation from level-ground\nbrightness: flags the\nsun-facing wall',
            transform=ax.transAxes, color='white', fontsize=8.5, bbox=LABEL_BOX)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Brightness Hazard', fontsize=8.5)
    ax.axis('off')

    # (g) full hazard map
    ax = axes[1, 2]
    im = ax.imshow(H_full, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(r'(g) Full Hazard Map $\max(G, B)$', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, 'Whole crater wall flagged;\nflat floor and plain stay safe',
            transform=ax.transAxes, color='white', fontsize=8.5, bbox=LABEL_BOX)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Hazard Score', fontsize=8.5)
    ax.axis('off')

    # (h) landing boxes
    ax = axes[1, 3]
    ax.imshow(ori, cmap='gray', vmin=0, vmax=255)
    for rect, ec, ls in ((rect_g, '#FF3333', '-'), (rect_gs, '#FFB300', ':'), (rect_br, '#00FF66', '--')):
        x, y, w, h = rect
        if w > 0 and h > 0:
            ax.add_patch(patches.Rectangle((x, y), w, h, lw=2.2, ec=ec, fc='none', ls=ls))
    x, y, w, h = rect_br
    if 0 < max(w, h) < 120:
        ins = ax.inset_axes([0.02, 0.56, 0.42, 0.42])
        ins.imshow(ori, cmap='gray', vmin=0, vmax=255)
        half = max(w, h)
        icx = min(max(x + w / 2, half), W_img - half)
        icy = min(max(y + h / 2, half), H_img - half)
        ins.set_xlim(icx - half, icx + half)
        ins.set_ylim(icy + half, icy - half)
        ins.add_patch(patches.Rectangle((x, y), w, h, lw=2.0, ec='#00FF66', fc='none', ls='--'))
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_edgecolor('#00FF66'); sp.set_linewidth(1.5)
        ax.indicate_inset_zoom(ins, edgecolor='#00FF66', linewidth=1.0, alpha=1.0)
    ax.set_title('(h) Landing Box Selection', fontsize=11, fontweight='bold', pad=6)
    ax.text(0.04, 0.06, f'Red: $V_{{grad}}$ ({pu_g:.0f}% unsafe)\n'
                        f'Amber: $V_{{grad+shadow}}$ ({pu_gs:.0f}% unsafe)\n'
                        f'Green: $V_{{full}}$ ({pu_br:.0f}% unsafe)',
            transform=ax.transAxes, color='white', fontsize=8.5, bbox=LABEL_BOX)
    ax.axis('off')

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(OUT_DIR, f'fig_qualitative_crater.{ext}'), bbox_inches='tight')
    plt.close()
    print('Saved', os.path.join(OUT_DIR, 'fig_qualitative_crater.{pdf,png}'))


if __name__ == '__main__':
    generate_figure(sys.argv[1] if len(sys.argv) > 1 else 'crater_2_0')
