import json
import numpy as np
import pandas as pd
import cv2
import itertools
from sklearn.metrics import f1_score
from pipeline import compute_gradient_hazard, augment_shadow, fuse_and_select, compute_brightness_hazard
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
    
    # 8 DEV tiles, two per terrain type (v1 used dev_tiles[:8], which are all crater_1).
    val_tiles = [t for site in ('crater_1', 'rocky_1', 'smooth_1', 'dunes_1')
                 for t in [x for x in dev_tiles if x.startswith(site + '_')][:2]]
    print('Tuning tiles:', val_tiles)
    
    val_df = manifest[manifest['tile_id'].isin(val_tiles)]
    
    print("Loading val data...")
    val_data = []
    for _, row in val_df.iterrows():
        ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
        dtm = np.load(row['dtm'])
        unsafe = compute_proxy_labels(dtm)
        G = compute_gradient_hazard(ori)
        val_data.append((ori, unsafe, G))
        
    thetas = [0.05, 0.1, 0.2, 0.35, 0.5]
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
    
    taus = [20, 40, 60, 80]
    a_mins = [100, 500]
    
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
            
    # Brightness cue: tau, a_min, theta and min area fixed from V_grad_shadow; tune the saturation tilt
    # phi (degrees) and the blur size.
    best_v_bright_f1 = -1
    best_v_bright_params = {}
    print("Tuning V_bright...")
    for phi, ksize in itertools.product([5.0, 7.5, 10.0, 12.5, 15.0, 20.0], [5, 9, 15]):
        preds = []
        labels = []
        for ori, unsafe, G in val_data:
            G_aug, _ = augment_shadow(ori, G, tau=best_v_grad_shadow_params['tau'], a_min=best_v_grad_shadow_params['a_min'])
            H = np.maximum(G_aug, compute_brightness_hazard(ori, phi_deg=phi, ksize=ksize))
            _, _, safe_mask, _ = fuse_and_select(H, texture_score=None, **best_v_grad_params)
            preds.extend(safe_mask.flatten()[::100])
            labels.extend((~unsafe).flatten()[::100])
        f1 = f1_score(labels, preds)
        results.append({'model': 'V_bright', 'phi_deg': phi, 'ksize': ksize, 'f1': f1})
        if f1 > best_v_bright_f1:
            best_v_bright_f1 = f1
            best_v_bright_params = {**best_v_grad_shadow_params, 'phi_deg': phi, 'ksize': ksize}
    print(f"Best V_bright F1: {best_v_bright_f1:.4f} with {best_v_bright_params}")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/tuning.csv', index=False)
    
    config = {
        'V_grad': best_v_grad_params,
        'V_grad_shadow': best_v_grad_shadow_params,
        'V_bright': best_v_bright_params,
        'V_fft_ablation': {'theta': best_v_grad_params['theta'], 'min_component_area': best_v_grad_params['min_component_area'], 'tau': 40, 'a_min': 500}
    }
    with open('results/best_config.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == '__main__':
    main()
