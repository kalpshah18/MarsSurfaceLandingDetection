import json
import numpy as np
import pandas as pd
import cv2
import itertools
from sklearn.metrics import f1_score
from pipeline import compute_gradient_hazard, augment_shadow, fuse_and_select
import os

def compute_proxy_labels(dtm, gsd=1.0, baseline_m=5.0):
    baseline_px = max(1, int(baseline_m / gsd))
    dz_dx = (dtm[:, baseline_px:] - dtm[:, :-baseline_px]) / baseline_px
    dz_dy = (dtm[baseline_px:, :] - dtm[:-baseline_px, :]) / baseline_px
    slope_rad = np.zeros_like(dtm)
    slope_rad[:-baseline_px, :-baseline_px] = np.arctan(np.sqrt(dz_dx[:-baseline_px, :]**2 + dz_dy[:, :-baseline_px]**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg > 15.0

def main():
    manifest = pd.read_csv('data/MANIFEST.csv')
    with open('splits.json', 'r') as f:
        splits = json.load(f)
        
    dev_tiles = splits['DEV']
    np.random.seed(42)
    
    val_tiles = dev_tiles[:8] # just a subset for fast tuning
    
    val_df = manifest[manifest['tile_id'].isin(val_tiles)]
    
    print("Loading val data...")
    val_data = []
    for _, row in val_df.iterrows():
        ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
        dtm = np.load(row['dtm'])
        unsafe = compute_proxy_labels(dtm)
        G = compute_gradient_hazard(ori)
        val_data.append((ori, unsafe, G))
        
    thetas = [0.2, 0.35, 0.5]
    min_areas = [100, 500, 1500]
    
    best_v_grad_f1 = -1
    best_v_grad_params = {}
    
    results = []
    print("Tuning V_grad...")
    for theta, min_area in itertools.product(thetas, min_areas):
        preds = []
        labels = []
        for ori, unsafe, G in val_data:
            _, _, safe_mask, _ = fuse_and_select(G, texture_score=None, theta=theta, min_component_area=min_area)
            preds.extend(safe_mask.flatten()[::100])
            labels.extend((~unsafe).flatten()[::100])
        f1 = f1_score(labels, preds)
        results.append({'model': 'V_grad', 'theta': theta, 'min_area': min_area, 'f1': f1})
        if f1 > best_v_grad_f1:
            best_v_grad_f1 = f1
            best_v_grad_params = {'theta': theta, 'min_component_area': min_area}
            
    print(f"Best V_grad F1: {best_v_grad_f1:.4f} with {best_v_grad_params}")
    
    taus = [20, 40]
    a_mins = [500]
    
    best_v_grad_shadow_f1 = -1
    best_v_grad_shadow_params = {}
    
    print("Tuning V_grad_shadow...")
    for tau, a_min in itertools.product(taus, a_mins):
        preds = []
        labels = []
        for ori, unsafe, G in val_data:
            G_aug, _ = augment_shadow(ori, G, tau=tau, a_min=a_min)
            _, _, safe_mask, _ = fuse_and_select(G_aug, texture_score=None, **best_v_grad_params)
            preds.extend(safe_mask.flatten()[::100])
            labels.extend((~unsafe).flatten()[::100])
        f1 = f1_score(labels, preds)
        results.append({'model': 'V_grad_shadow', 'tau': tau, 'a_min': a_min, 'f1': f1})
        if f1 > best_v_grad_shadow_f1:
            best_v_grad_shadow_f1 = f1
            best_v_grad_shadow_params = {'tau': tau, 'a_min': a_min, **best_v_grad_params}
            
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/tuning.csv', index=False)
    
    config = {
        'V_grad': best_v_grad_params,
        'V_grad_shadow': best_v_grad_shadow_params,
        'V_fft_ablation': {'theta': best_v_grad_params['theta'], 'min_component_area': best_v_grad_params['min_component_area'], 'tau': 40, 'a_min': 500}
    }
    with open('results/best_config.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    main()
