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
    G_shadow, shadow_mask = augment_shadow(ori, G.copy(), tau=40, a_min=500)
    
    # 3. Pipelines & Rectangles
    s_g, mask_g, _, rect_g = run_v_grad(ori, theta=0.2, min_component_area=100)
    s_gs, mask_gs, _, rect_gs = run_v_grad_shadow(ori, tau=40, a_min=500, theta=0.2, min_component_area=100)
    
    # Setup Figure (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.8), dpi=300)
    plt.subplots_adjust(wspace=0.12, hspace=0.22, left=0.03, right=0.97, top=0.94, bottom=0.04)
    
    # (a) Input Image
    axes[0, 0].imshow(ori, cmap='gray')
    axes[0, 0].set_title('(a) Input Monocular Image', fontsize=11, fontweight='bold', pad=6)
    axes[0, 0].text(0.04, 0.06, 'Lambertian Render\nCrater + Deep Shadow', transform=axes[0, 0].transAxes,
                    color='white', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[0, 0].axis('off')
    
    # (b) DTM Slope (Ground Truth)
    im_slope = axes[0, 1].imshow(slope_deg, cmap='inferno', vmin=0, vmax=35)
    axes[0, 1].contour(unsafe, levels=[0.5], colors=['cyan'], linewidths=1.2, linestyles='--')
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
    axes[1, 0].set_title(r'(d) Detected Shadow Mask ($\tau=40$)', fontsize=11, fontweight='bold', pad=6)
    axes[1, 0].text(0.04, 0.06, 'Morphological Contours\nArea Filter ($A_{min} \geq 500$ px)', transform=axes[1, 0].transAxes,
                    color='lime', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    axes[1, 0].axis('off')
    
    # (e) Final Augmented Hazard
    im_aug = axes[1, 1].imshow(G_shadow, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('(e) Augmented Hazard Map', fontsize=11, fontweight='bold', pad=6)
    axes[1, 1].text(0.04, 0.06, 'Shadow Overridden:\n$G(x,y) \leftarrow 1.0$', transform=axes[1, 1].transAxes,
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
    axes[1, 2].text(0.04, 0.06, 'Red Solid: $V_{grad}$ (100% unsafe!)\nGreen Dashed: $V_{grad+shadow}$ (0% unsafe)',
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
