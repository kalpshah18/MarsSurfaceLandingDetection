import json
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
import time

from pipeline import run_v_grad, run_v_grad_shadow, maximal_rectangle

def compute_proxy_labels(dtm, gsd=1.0, baseline_m=5.0):
    baseline_px = max(1, int(baseline_m / gsd))
    dz_dx = (dtm[:, baseline_px:] - dtm[:, :-baseline_px]) / baseline_px
    dz_dy = (dtm[baseline_px:, :] - dtm[:-baseline_px, :]) / baseline_px
    slope_rad = np.zeros_like(dtm)
    slope_rad[:-baseline_px, :-baseline_px] = np.arctan(np.sqrt(dz_dx[:-baseline_px, :]**2 + dz_dy[:, :-baseline_px]**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg > 15.0, slope_deg

def get_local_std(img, ksize=9):
    img_f = img.astype(np.float32) / 255.0
    mean = cv2.blur(img_f, (ksize, ksize))
    sq_mean = cv2.blur(img_f**2, (ksize, ksize))
    var = sq_mean - mean**2
    var[var < 0] = 0
    std = np.sqrt(var)
    max_val = np.max(std)
    if max_val > 0:
        return np.clip(std / max_val, 0, 1)
    return std

def bootstrap_ci(metric_list, n_resamples=2000):
    np.random.seed(42)
    data = np.array(metric_list)
    resamples = np.random.choice(data, size=(n_resamples, len(data)), replace=True)
    means = np.mean(resamples, axis=1)
    return np.percentile(means, [2.5, 97.5])

def main():
    manifest = pd.read_csv('data/MANIFEST.csv')
    with open('splits.json', 'r') as f:
        splits = json.load(f)
    with open('results/best_config.json', 'r') as f:
        config = json.load(f)
        
    test_tiles = splits['TEST']
    test_df = manifest[manifest['tile_id'].isin(test_tiles)]
    
    tile_metrics = []
    
    for _, row in test_df.iterrows():
        ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
        dtm = np.load(row['dtm'])
        
        unsafe, slope_deg = compute_proxy_labels(dtm)
        safe = ~unsafe
        
        # True max rect on safe
        safe_uint8 = safe.astype(np.uint8)
        true_rect = maximal_rectangle(safe_uint8)
        true_area = true_rect[2] * true_rect[3]
        
        methods = {}
        
        # V_grad
        t0 = time.time()
        s_g, mask_g, _, rect_g = run_v_grad(ori, **config['V_grad'])
        t_g = time.time() - t0
        h_g = 1.0 - s_g
        u_mask_g = (1 - mask_g).astype(np.uint8)
        methods['V_grad'] = {'h': h_g, 'u_mask': u_mask_g, 'rect': rect_g, 'time': t_g}
        
        # V_grad_shadow
        t0 = time.time()
        s_gs, mask_gs, _, rect_gs = run_v_grad_shadow(ori, **config['V_grad_shadow'])
        t_gs = time.time() - t0
        h_gs = 1.0 - s_gs
        u_mask_gs = (1 - mask_gs).astype(np.uint8)
        methods['V_grad_shadow'] = {'h': h_gs, 'u_mask': u_mask_gs, 'rect': rect_gs, 'time': t_gs}
        
        # B1: Local Std
        std = get_local_std(ori)
        h_b1 = std
        s_b1 = 1.0 - std
        mask_b1 = (s_b1 > config['V_grad']['theta']).astype(np.uint8)
        u_mask_b1 = (1 - mask_b1).astype(np.uint8)
        methods['B1_LocalStd'] = {'h': h_b1, 'u_mask': u_mask_b1, 'rect': (0,0,0,0), 'time': 0}
        
        # B5: Random
        seed_val = int(row['tile_id'].split('_')[-1]) + 100
        np.random.seed(seed_val)
        h_b5 = np.random.rand(*ori.shape)
        frac_unsafe_g = np.mean(u_mask_g)
        u_mask_b5 = (h_b5 > (1.0 - frac_unsafe_g)).astype(np.uint8)
        methods['B5_Random'] = {'h': h_b5, 'u_mask': u_mask_b5, 'rect': (0,0,0,0), 'time': 0}
        
        # Subsample for AUROC / correlation
        sub = slice(None, None, 10)
        u_sub = unsafe.flatten()[sub]
        slope_sub = slope_deg.flatten()[sub]
        
        for m_name, m_data in methods.items():
            h_flat = m_data['h'].flatten()[sub]
            u_pred = m_data['u_mask'].flatten()[sub]
            
            # Unsafe class evaluation
            prec = precision_score(u_sub, u_pred, zero_division=0)
            rec = recall_score(u_sub, u_pred, zero_division=0)
            f1 = f1_score(u_sub, u_pred, zero_division=0)
            
            if len(np.unique(u_sub)) > 1:
                auroc = roc_auc_score(u_sub, h_flat)
                prauc = average_precision_score(u_sub, h_flat)
            else:
                auroc, prauc = np.nan, np.nan
                
            sp, _ = spearmanr(h_flat, slope_sub)
            pe, _ = pearsonr(h_flat, slope_sub)
            
            # Box metrics
            x, y, w, h = m_data['rect']
            box_area = w * h
            box_found = int(box_area > 0)
            
            if box_found:
                box_unsafe = unsafe[y:y+h, x:x+w]
                frac_unsafe = np.mean(box_unsafe)
                tol_0 = int(frac_unsafe <= 0.0)
                tol_1 = int(frac_unsafe <= 0.01)
                tol_5 = int(frac_unsafe <= 0.05)
            else:
                frac_unsafe, tol_0, tol_1, tol_5 = np.nan, 0, 0, 0
                
            rel_area = box_area / true_area if true_area > 0 else 0
            footprint_ok = int(box_area >= 2500)
            
            tile_metrics.append({
                'tile_id': row['tile_id'],
                'method': m_name,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auroc': auroc,
                'prauc': prauc,
                'spearman': sp,
                'pearson': pe,
                'box_found': box_found,
                'frac_unsafe': frac_unsafe,
                'tol_0': tol_0,
                'tol_1': tol_1,
                'tol_5': tol_5,
                'rel_area': rel_area,
                'footprint_ok': footprint_ok,
                'time': m_data['time']
            })
            
    df = pd.DataFrame(tile_metrics)
    df.to_csv('results/heldout_metrics.csv', index=False)
    
    # Aggregate and Bootstrap
    agg_results = []
    methods = ['V_grad', 'V_grad_shadow', 'B1_LocalStd', 'B5_Random']
    
    for m in methods:
        m_df = df[df['method'] == m]
        row = {'method': m}
        for col in ['precision', 'recall', 'f1', 'auroc', 'prauc', 'spearman', 'pearson',
                   'box_found', 'frac_unsafe', 'tol_0', 'tol_1', 'tol_5', 'rel_area', 'footprint_ok']:
            vals = m_df[col].dropna().values
            if len(vals) > 0:
                mean_v = np.mean(vals)
                ci = bootstrap_ci(vals)
                row[col] = f"{mean_v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
            else:
                row[col] = "NaN"
        agg_results.append(row)
        
    agg_df = pd.DataFrame(agg_results)
    agg_df.to_csv('results/aggregate_metrics.csv', index=False)
    print("Aggregate metrics (Unsafe Primary Class):")
    for _, r in agg_df.iterrows():
        print(f"{r['method']}: Prec={r['precision']}, Rec={r['recall']}, F1={r['f1']}, AUROC={r['auroc']}, PR-AUC={r['prauc']}")

if __name__ == '__main__':
    main()
