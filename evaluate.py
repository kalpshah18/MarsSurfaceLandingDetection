import os
import json
import numpy as np
import pandas as pd
import cv2
from pipeline import run_v0, fuse_and_select
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_proxy_labels(dtm, gsd=1.0, baseline_m=5.0):
    baseline_px = max(1, int(baseline_m / gsd))
    dz_dx = (dtm[:, baseline_px:] - dtm[:, :-baseline_px]) / baseline_px
    dz_dy = (dtm[baseline_px:, :] - dtm[:-baseline_px, :]) / baseline_px
    
    slope_rad = np.zeros_like(dtm)
    slope_rad[:-baseline_px, :-baseline_px] = np.arctan(np.sqrt(dz_dx[:-baseline_px, :]**2 + dz_dy[:, :-baseline_px]**2))
    slope_deg = np.degrees(slope_rad)
    
    unsafe = slope_deg > 15.0
    return slope_deg, unsafe

def bootstrap_ci(data, n_resamples=10000):
    np.random.seed(42)
    resamples = np.random.choice(data, size=(n_resamples, len(data)), replace=True)
    means = np.mean(resamples, axis=1)
    return np.percentile(means, [2.5, 97.5])

def evaluate_tile(dtm_path, ori_path, gsd=1.0):
    dtm = np.load(dtm_path)
    ori = cv2.imread(ori_path, cv2.IMREAD_GRAYSCALE)
    
    slope_deg, proxy_unsafe = compute_proxy_labels(dtm, gsd)
    proxy_safe = ~proxy_unsafe
    
    G, shadow_mask, R_map, texture_score, s, safe_mask, filtered_safe, rect = run_v0(ori)
    
    x, y, w, h = rect
    box_unsafe_fraction = 0.0
    box_max_slope = 0.0
    box_p95_slope = 0.0
    if w > 0 and h > 0:
        box_unsafe_mask = proxy_unsafe[y:y+h, x:x+w]
        box_slope = slope_deg[y:y+h, x:x+w]
        box_unsafe_fraction = np.mean(box_unsafe_mask)
        if box_slope.size > 0:
            box_max_slope = np.max(box_slope)
            box_p95_slope = np.percentile(box_slope, 95)
            
    # Pixel metrics
    tp = np.sum((safe_mask == 1) & (proxy_safe == 1))
    fp = np.sum((safe_mask == 1) & (proxy_safe == 0))
    fn = np.sum((safe_mask == 0) & (proxy_safe == 1))
    tn = np.sum((safe_mask == 0) & (proxy_safe == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    unsafe_recall = tn / (tn + fp + 1e-8)
    
    # Continuous metrics
    # s is safety, we want s to predict proxy_safe, or (1-s) to predict proxy_unsafe
    s_flat = s.flatten()
    unsafe_flat = proxy_unsafe.flatten().astype(int)
    
    if len(np.unique(unsafe_flat)) > 1:
        roc_auc = roc_auc_score(unsafe_flat, 1.0 - s_flat)
        pr_auc = average_precision_score(unsafe_flat, 1.0 - s_flat)
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        
    spearman, _ = spearmanr(G.flatten(), slope_deg.flatten())
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'unsafe_recall': unsafe_recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'spearman': spearman,
        'box_unsafe_fraction': box_unsafe_fraction,
        'box_max_slope': box_max_slope,
        'box_p95_slope': box_p95_slope,
        'box_success': float(box_unsafe_fraction < 0.01),
        'box_w_m': w * gsd,
        'box_h_m': h * gsd
    }

def run_evaluation():
    manifest = pd.read_csv('data/MANIFEST.csv')
    with open('splits.json', 'r') as f:
        splits = json.load(f)
        
    os.makedirs('results', exist_ok=True)
    
    results = []
    for split_name, tile_ids in splits.items():
        print(f"Evaluating {split_name} split...")
        for tid in tqdm(tile_ids):
            row = manifest[manifest['tile_id'] == tid].iloc[0]
            metrics = evaluate_tile(row['dtm'], row['ori'])
            metrics['tile_id'] = tid
            metrics['split'] = split_name
            results.append(metrics)
            
    df = pd.DataFrame(results)
    df.to_csv('results/evaluation_metrics.csv', index=False)
    
    # Compute aggregates
    summary = {}
    for split in ['DEV', 'TEST']:
        split_df = df[df['split'] == split].dropna()
        summary[split] = {}
        for col in ['f1', 'iou', 'unsafe_recall', 'roc_auc', 'pr_auc', 'box_unsafe_fraction', 'box_success']:
            data = split_df[col].values
            mean = np.mean(data)
            if len(data) > 0:
                ci = bootstrap_ci(data)
                summary[split][col] = {'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1]}
                
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
if __name__ == "__main__":
    run_evaluation()
