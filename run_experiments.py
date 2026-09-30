import os
import json
import numpy as np
import pandas as pd
import cv2
from pipeline import run_v0, fuse_and_select, compute_gradient_hazard, augment_shadow, compute_fft_texture, maximal_rectangle
from evaluate import compute_proxy_labels, evaluate_tile, bootstrap_ci
from tqdm import tqdm

def get_metrics_for_mask(safe_mask, proxy_safe, proxy_unsafe, s_flat=None):
    tp = np.sum((safe_mask == 1) & (proxy_safe == 1))
    fp = np.sum((safe_mask == 1) & (proxy_safe == 0))
    fn = np.sum((safe_mask == 0) & (proxy_safe == 1))
    tn = np.sum((safe_mask == 0) & (proxy_safe == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {'f1': f1, 'iou': iou}

def run_experiments():
    manifest = pd.read_csv('data/MANIFEST.csv')
    with open('splits.json', 'r') as f:
        splits = json.load(f)
        
    os.makedirs('results', exist_ok=True)
    
    ablation_results = []
    baseline_results = []
    
    # Simple hyperparameters for baselines, chosen quickly on DEV mentally
    baseline_std_thresh = 20.0
    baseline_lap_thresh = 50.0
    baseline_norm_grad_thresh = 0.5
    
    for split_name, tile_ids in splits.items():
        if split_name != 'TEST':
            continue # We only need to report TEST numbers as per phase 4
            
        print(f"Running experiments on {split_name} split...")
        for tid in tqdm(tile_ids):
            row = manifest[manifest['tile_id'] == tid].iloc[0]
            dtm = np.load(row['dtm'])
            ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
            
            slope_deg, proxy_unsafe = compute_proxy_labels(dtm, gsd=1.0)
            proxy_safe = ~proxy_unsafe
            
            # --- ABLATIONS ---
            G_A1 = compute_gradient_hazard(ori)
            G_A2, shadow = augment_shadow(ori, np.copy(G_A1))
            texture, _ = compute_fft_texture(ori)
            
            # Helper to get mask and metrics
            def get_ablation_mask(s_arr):
                safe = (s_arr > 0.35).astype(np.uint8)
                return get_metrics_for_mask(safe, proxy_safe, proxy_unsafe)
                
            a1_met = get_ablation_mask(1.0 - G_A1)
            a2_met = get_ablation_mask(1.0 - G_A2)
            a3_met = get_ablation_mask((1.0 - G_A1) * texture)
            a4_met = get_ablation_mask((1.0 - G_A2) * texture)
            a5_met = get_ablation_mask(texture)
            # shadow only: score = 1 if not shadow else 0
            shadow_score = (shadow == 0).astype(np.float32)
            a6_met = get_ablation_mask(shadow_score)
            
            ablation_results.append({
                'tile_id': tid,
                'A1_f1': a1_met['f1'], 'A1_iou': a1_met['iou'],
                'A2_f1': a2_met['f1'], 'A2_iou': a2_met['iou'],
                'A3_f1': a3_met['f1'], 'A3_iou': a3_met['iou'],
                'A4_f1': a4_met['f1'], 'A4_iou': a4_met['iou'],
                'A5_f1': a5_met['f1'], 'A5_iou': a5_met['iou'],
                'A6_f1': a6_met['f1'], 'A6_iou': a6_met['iou'],
            })
            
            # --- BASELINES ---
            # B1 local std
            mean, std = cv2.meanStdDev(ori)
            blur = cv2.blur(ori.astype(np.float32)**2, (15, 15))
            mean_sq = cv2.blur(ori.astype(np.float32), (15, 15))**2
            std_map = np.sqrt(np.maximum(0, blur - mean_sq))
            b1_safe = (std_map < baseline_std_thresh).astype(np.uint8)
            b1_met = get_metrics_for_mask(b1_safe, proxy_safe, proxy_unsafe)
            
            # B2 Laplacian
            lap = cv2.Laplacian(ori, cv2.CV_32F, ksize=3)
            var_map = cv2.blur(lap**2, (15, 15))
            b2_safe = (var_map < baseline_lap_thresh).astype(np.uint8)
            b2_met = get_metrics_for_mask(b2_safe, proxy_safe, proxy_unsafe)
            
            # B3 Normalized gradient
            Ix = cv2.Sobel(ori.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            Iy = cv2.Sobel(ori.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            g = np.sqrt(Ix**2 + Iy**2)
            norm_g = g / (np.mean(ori) + 1e-8)
            b3_safe = (norm_g < baseline_norm_grad_thresh).astype(np.uint8)
            b3_met = get_metrics_for_mask(b3_safe, proxy_safe, proxy_unsafe)
            
            # B4 all-safe
            b4_safe = np.ones_like(ori, dtype=np.uint8)
            b4_met = get_metrics_for_mask(b4_safe, proxy_safe, proxy_unsafe)
            
            # B5 random
            f1_rands, iou_rands = [], []
            safe_frac = np.mean(b4_safe) # usually 1.0, but let's match A4's safe fraction
            actual_safe_frac = np.mean((1.0 - G_A2) * texture > 0.35)
            for _ in range(20):
                b5_safe = (np.random.rand(*ori.shape) < actual_safe_frac).astype(np.uint8)
                b5_m = get_metrics_for_mask(b5_safe, proxy_safe, proxy_unsafe)
                f1_rands.append(b5_m['f1'])
                iou_rands.append(b5_m['iou'])
                
            baseline_results.append({
                'tile_id': tid,
                'B1_f1': b1_met['f1'], 'B1_iou': b1_met['iou'],
                'B2_f1': b2_met['f1'], 'B2_iou': b2_met['iou'],
                'B3_f1': b3_met['f1'], 'B3_iou': b3_met['iou'],
                'B4_f1': b4_met['f1'], 'B4_iou': b4_met['iou'],
                'B5_f1': np.mean(f1_rands), 'B5_iou': np.mean(iou_rands),
            })
            
    pd.DataFrame(ablation_results).to_csv('results/ablations.csv', index=False)
    pd.DataFrame(baseline_results).to_csv('results/baselines.csv', index=False)
    print("Ablations and baselines complete.")
    
if __name__ == "__main__":
    run_experiments()
