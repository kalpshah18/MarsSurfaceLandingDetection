import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from run_evaluation import compute_proxy_labels
from pipeline import compute_gradient_hazard, augment_shadow, run_v_grad, run_v_grad_shadow

def generate_figure():
    tile_id = 'crater_2_0'
    ori_path = 'data/synthetic/crater_2/ori_0.png'
    dtm_path = 'data/synthetic/crater_2/dtm_0.npy'
    
    ori = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)
    dtm = np.load(dtm_path)
    
    unsafe, slope_deg = compute_proxy_labels(dtm)
    
    # 1. Gradient hazard
    G = compute_gradient_hazard(ori)
    
    # 2. Shadow augmentation
    # Same configuration as the evaluated pipeline (results/best_config.json).
    with open('results/best_config.json') as f:
        cfg = json.load(f)
    gs_cfg, g_cfg = cfg['V_grad_shadow'], cfg['V_grad']
    G_shadow, shadow_mask = augment_shadow(ori, G.copy(), tau=gs_cfg['tau'], a_min=gs_cfg['a_min'])

    # 3. Pipelines & Rectangles
    s_g, mask_g, _, rect_g = run_v_grad(ori, **g_cfg)
    s_gs, mask_gs, _, rect_gs = run_v_grad_shadow(ori, **gs_cfg)
    
    # Setup Figure (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), dpi=300)
    plt.subplots_adjust(wspace=0.12, hspace=0.22, left=0.03, right=0.97, top=0.94, bottom=0.04)
    
    # (a) Input Image
    axes[0, 0].imshow(ori, cmap='gray')
    axes[0, 0].set_title('(a) Input Monocular Image', fontsize=11, fontweight='bold', pad=6)
    axes[0, 0].text(0.04, 0.06, 'Lambertian Render\nCrater floor >10 m deep rendered dark', transform=axes[0, 0].transAxes,
                    color='white', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[0, 0].axis('off')
    
    # (b) DTM Slope (Ground Truth)
    im_slope = axes[0, 1].imshow(slope_deg, cmap='inferno', vmin=0, vmax=35)
    axes[0, 1].contour(unsafe, levels=[0.5], colors=['cyan'], linewidths=2.4, linestyles='--')
    axes[0, 1].set_title('(b) DTM Slope & True Hazard', fontsize=11, fontweight='bold', pad=6)
    axes[0, 1].text(0.04, 0.06, 'Cyan: Slope > 15°\n(True Hazardous Region)', transform=axes[0, 1].transAxes,
                    color='cyan', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    cbar1 = fig.colorbar(im_slope, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label('Slope (deg)', fontsize=8.5)
    axes[0, 1].axis('off')
    
    # (c) Gradient Hazard G(x,y) - Failure Mode
    im_grad = axes[0, 2].imshow(G, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('(c) Raw Gradient Hazard $G(x,y)$', fontsize=11, fontweight='bold', pad=6)
    axes[0, 2].text(0.04, 0.06, 'Critical Failure:\nZero gradient inside\ndark crater bowl!', transform=axes[0, 2].transAxes,
                    color='yellow', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    cbar2 = fig.colorbar(im_grad, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label('Gradient Hazard Score', fontsize=8.5)
    axes[0, 2].axis('off')
    
    # (d) Shadow Mask
    axes[1, 0].imshow(shadow_mask, cmap='gray')
    axes[1, 0].set_title(r'(d) Detected Shadow Mask ($\tau=%d$)' % gs_cfg['tau'], fontsize=11, fontweight='bold', pad=6)
    axes[1, 0].text(0.04, 0.06, 'Morphological Contours\n' + r'Area Filter ($A_{min} \geq %d$ px)' % gs_cfg['a_min'], transform=axes[1, 0].transAxes,
                    color='lime', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[1, 0].axis('off')
    
    # (e) Final Augmented Hazard
    im_aug = axes[1, 1].imshow(G_shadow, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('(e) Augmented Hazard Map', fontsize=11, fontweight='bold', pad=6)
    axes[1, 1].text(0.04, 0.06, 'Shadow Overridden:\n' + r'$G(x,y) \leftarrow 1.0$', transform=axes[1, 1].transAxes,
                    color='white', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    cbar3 = fig.colorbar(im_aug, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_label('Augmented Hazard Score', fontsize=8.5)
    axes[1, 1].axis('off')
    
    # (f) Landing Box Comparison
    axes[1, 2].imshow(ori, cmap='gray')
    # Draw V_grad box (Catastrophic failure in red)
    xg, yg, wg, hg = rect_g
    rect_box_g = patches.Rectangle((xg, yg), wg, hg, linewidth=2.5, edgecolor='#FF3333', facecolor='none', linestyle='-')
    axes[1, 2].add_patch(rect_box_g)
    
    # Draw V_grad_shadow box (Safe relocation in green)
    xgs, ygs, wgs, hgs = rect_gs
    if wgs > 0 and hgs > 0:
        rect_box_gs = patches.Rectangle((xgs, ygs), wgs, hgs, linewidth=2.5, edgecolor='#00FF66', facecolor='none', linestyle='--')
        axes[1, 2].add_patch(rect_box_gs)
        
    axes[1, 2].set_title('(f) Landing Box Selection', fontsize=11, fontweight='bold', pad=6)
    # Zoomed inset so the small V_grad+shadow box is visible at print size.
    if wgs > 0 and hgs > 0:
        # Matplotlib's native inset API (axes_grid1's inset_locator misplaced the colorbar
        # images in the PDF output).
        ins = axes[1, 2].inset_axes([0.02, 0.56, 0.42, 0.42])
        ins.imshow(ori, cmap='gray')
        half = max(wgs, hgs)
        H_img, W_img = ori.shape
        # Keep the zoom window inside the image so no blank margin appears.
        cx = min(max(xgs + wgs / 2, half), W_img - half)
        cy = min(max(ygs + hgs / 2, half), H_img - half)
        ins.set_xlim(cx - half, cx + half)
        ins.set_ylim(cy + half, cy - half)
        ins.add_patch(patches.Rectangle((xgs, ygs), wgs, hgs, linewidth=2.0, edgecolor='#00FF66',
                                        facecolor='none', linestyle='--'))
        ins.set_xticks([])
        ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_edgecolor('#00FF66')
            sp.set_linewidth(1.5)
        axes[1, 2].indicate_inset_zoom(ins, edgecolor='#00FF66', linewidth=1.0, alpha=1.0)
    axes[1, 2].text(0.04, 0.06, 'Red: $V_{grad}$ (100% unsafe)\nGreen: $V_{grad+shadow}$ (0% unsafe)',
                    transform=axes[1, 2].transAxes, color='white', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[1, 2].axis('off')
    
    # Save outputs
    out_dir = 'Mars_Landing_Paper_ARXIV_Style_updated/media'
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'fig_qualitative_crater.pdf')
    png_path = os.path.join(out_dir, 'fig_qualitative_crater.png')
    
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative figure to {pdf_path} and {png_path}")

if __name__ == '__main__':
    generate_figure()
