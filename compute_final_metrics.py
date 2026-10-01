import numpy as np
import pandas as pd
import cv2
import json
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from run_evaluation import compute_proxy_labels, get_local_std, bootstrap_ci
from pipeline import run_v_grad, run_v_grad_shadow, maximal_rectangle

def main():
    manifest = pd.read_csv('data/MANIFEST.csv')
    with open('splits.json') as f:
        splits = json.load(f)
    with open('results/best_config.json') as f:
        config = json.load(f)

    test_tiles = splits['TEST']
    test_df = manifest[manifest['tile_id'].isin(test_tiles)]

    all_u_true = []
    all_u_g = []
    all_h_g = []
    all_u_gs = []
    all_h_gs = []
    all_u_b1 = []
    all_h_b1 = []
    all_u_b5 = []
    all_h_b5 = []

    box_metrics = {'V_grad': [], 'V_grad_shadow': [], 'B1_LocalStd': []}

    for _, row in test_df.iterrows():
        ori = cv2.imread(row['ori'], cv2.IMREAD_GRAYSCALE)
        dtm = np.load(row['dtm'])
        unsafe, _ = compute_proxy_labels(dtm)
        safe = ~unsafe
        
        # Ground truth maximal rectangle
        true_rect = maximal_rectangle(safe.astype(np.uint8))
        true_area = true_rect[2] * true_rect[3]
        tile_area = ori.shape[0] * ori.shape[1] # 589824
        
        # V_grad
        s_g, mask_g, _, rect_g = run_v_grad(ori, **config['V_grad'])
        h_g = 1.0 - s_g
        u_mask_g = (1 - mask_g).astype(np.uint8)
        
        # V_grad_shadow
        s_gs, mask_gs, _, rect_gs = run_v_grad_shadow(ori, **config['V_grad_shadow'])
        h_gs = 1.0 - s_gs
        u_mask_gs = (1 - mask_gs).astype(np.uint8)
        
        # B1 LocalStd
        std = get_local_std(ori)
        h_b1 = std
        s_b1 = 1.0 - std
        mask_b1 = (s_b1 > config['V_grad']['theta']).astype(np.uint8)
        u_mask_b1 = (1 - mask_b1).astype(np.uint8)
        
        # B5 Random
        seed_val = int(row['tile_id'].split('_')[-1]) + 100
        np.random.seed(seed_val)
        h_b5 = np.random.rand(*ori.shape)
        frac_unsafe_g = np.mean(u_mask_g)
        u_mask_b5 = (h_b5 > (1.0 - frac_unsafe_g)).astype(np.uint8)
        
        sub = slice(None, None, 10)
        all_u_true.extend(unsafe.flatten()[sub])
        all_u_g.extend(u_mask_g.flatten()[sub])
        all_h_g.extend(h_g.flatten()[sub])
        all_u_gs.extend(u_mask_gs.flatten()[sub])
        all_h_gs.extend(h_gs.flatten()[sub])
        all_u_b1.extend(u_mask_b1.flatten()[sub])
        all_h_b1.extend(h_b1.flatten()[sub])
        all_u_b5.extend(u_mask_b5.flatten()[sub])
        all_h_b5.extend(h_b5.flatten()[sub])
        
        # Process box metrics
        for m, rect in [('V_grad', rect_g), ('V_grad_shadow', rect_gs), ('B1_LocalStd', (0,0,0,0))]:
            x, y, w, h = rect
            area = w * h
            found = int(area > 0)
            frac_un = np.mean(unsafe[y:y+h, x:x+w]) if found else np.nan
            ok = int(area >= 2500)
            rel_tile = area / tile_area
            rel_true = area / true_area if true_area > 0 else 0
            box_metrics[m].append({
                'box_found': found,
                'area_px': area,
                'rel_tile': rel_tile,
                'rel_true': rel_true,
                'frac_unsafe': frac_un,
                'footprint_ok': ok
            })

    all_u_true = np.array(all_u_true)
    methods_data = {
        'V_grad': (np.array(all_u_g), np.array(all_h_g)),
        'V_grad_shadow': (np.array(all_u_gs), np.array(all_h_gs)),
        'B1_LocalStd': (np.array(all_u_b1), np.array(all_h_b1)),
        'B5_Random': (np.array(all_u_b5), np.array(all_h_b5))
    }

    print('=== POOLED TEST METRICS (Prevalence = {:.3f}) ==='.format(np.mean(all_u_true)))
    for m, (u_pred, h_score) in methods_data.items():
        p = precision_score(all_u_true, u_pred, zero_division=0)
        r = recall_score(all_u_true, u_pred, zero_division=0)
        f = f1_score(all_u_true, u_pred, zero_division=0)
        auc = roc_auc_score(all_u_true, h_score)
        prauc = average_precision_score(all_u_true, h_score)
        print(f'{m}: Prec={p:.3f}, Rec={r:.3f}, F1={f:.3f}, AUROC={auc:.3f}, PR-AUC={prauc:.3f}')

    print('\n=== BOX METRICS ===')
    for m in ['V_grad', 'V_grad_shadow', 'B1_LocalStd']:
        m_df = pd.DataFrame(box_metrics[m])
        print(f'=== {m} ===')
        print('  Box Found:', m_df['box_found'].mean())
        print('  Mean Area:', m_df['area_px'].mean())
        print('  Median Area:', m_df['area_px'].median())
        print('  Max Area:', m_df['area_px'].max())
        print('  Rel to Tile:', m_df['rel_tile'].mean())
        print('  Rel to True Max Box:', m_df['rel_true'].mean())
        print('  Frac Unsafe:', m_df['frac_unsafe'].dropna().mean())
        print('  Footprint OK (>= 2500 px):', m_df['footprint_ok'].mean())

if __name__ == '__main__':
    main()
