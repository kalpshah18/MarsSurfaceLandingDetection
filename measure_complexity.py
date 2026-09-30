import time
import numpy as np
import cv2
import json
import tracemalloc
from pipeline import compute_gradient_hazard, augment_shadow, compute_fft_texture, fuse_and_select, maximal_rectangle

def measure_module(func, *args, n_iter=50):
    # warmup
    for _ in range(5):
        func(*args)
    
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - start) * 1000.0) # in ms
        
    return np.median(times), np.percentile(times, 75) - np.percentile(times, 25)

def run():
    img_512 = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    img_1024 = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    
    res = {}
    for name, img in [('512', img_512), ('1024', img_1024)]:
        G = compute_gradient_hazard(img)
        G2, _ = augment_shadow(img, G.copy())
        t_score, _ = compute_fft_texture(img)
        
        grad_t, grad_iqr = measure_module(compute_gradient_hazard, img)
        shad_t, shad_iqr = measure_module(augment_shadow, img, G.copy())
        fft_t, fft_iqr = measure_module(compute_fft_texture, img)
        fuse_t, fuse_iqr = measure_module(fuse_and_select, G2, t_score)
        
        # total 
        def full(i):
            G = compute_gradient_hazard(i)
            G, _ = augment_shadow(i, G)
            ts, _ = compute_fft_texture(i)
            fuse_and_select(G, ts)
            
        tot_t, tot_iqr = measure_module(full, img)
        
        res[name] = {
            'Gradient': {'median': grad_t, 'iqr': grad_iqr},
            'Shadow': {'median': shad_t, 'iqr': shad_iqr},
            'FFT': {'median': fft_t, 'iqr': fft_iqr},
            'Fusion': {'median': fuse_t, 'iqr': fuse_iqr},
            'Total': {'median': tot_t, 'iqr': tot_iqr}
        }
        
    with open('results/complexity.json', 'w') as f:
        json.dump(res, f, indent=4)
        
    print("Complexity measurement complete.")

if __name__ == "__main__":
    run()
