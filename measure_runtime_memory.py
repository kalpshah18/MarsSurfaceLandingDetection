import time
import tracemalloc
import numpy as np
import cv2
import json
from pipeline import run_v_grad, run_v_grad_shadow, compute_gradient_hazard, augment_shadow, maximal_rectangle

def benchmark():
    sizes = [512, 768, 1024]
    results = {}
    
    for s in sizes:
        img = np.random.randint(0, 256, (s, s), dtype=np.uint8)
        
        # Warmup
        run_v_grad(img, theta=0.2, min_component_area=100)
        run_v_grad_shadow(img, tau=60, a_min=100, theta=0.2, min_component_area=100)
        
        # Benchmark V_grad
        tracemalloc.start()
        times_g = []
        for _ in range(25):
            t0 = time.perf_counter()
            run_v_grad(img, theta=0.2, min_component_area=100)
            times_g.append((time.perf_counter() - t0) * 1000.0)
        _, peak_g = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Benchmark V_grad_shadow
        tracemalloc.start()
        times_gs = []
        for _ in range(25):
            t0 = time.perf_counter()
            run_v_grad_shadow(img, tau=60, a_min=100, theta=0.2, min_component_area=100)
            times_gs.append((time.perf_counter() - t0) * 1000.0)
        _, peak_gs = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Breakdown components on 768x768 (native dataset size)
        results[s] = {
            'v_grad_median_ms': float(np.median(times_g)),
            'v_grad_iqr_ms': float(np.percentile(times_g, 75) - np.percentile(times_g, 25)),
            'v_grad_ram_mb': float(peak_g / (1024.0 * 1024.0)),
            'v_grad_shadow_median_ms': float(np.median(times_gs)),
            'v_grad_shadow_iqr_ms': float(np.percentile(times_gs, 75) - np.percentile(times_gs, 25)),
            'v_grad_shadow_ram_mb': float(peak_gs / (1024.0 * 1024.0))
        }
        print(f"Dimension {s}x{s}:")
        print(f"  V_grad: {results[s]['v_grad_median_ms']:.2f} ms (IQR: {results[s]['v_grad_iqr_ms']:.2f} ms), Peak RAM: {results[s]['v_grad_ram_mb']:.2f} MB")
        print(f"  V_grad_shadow: {results[s]['v_grad_shadow_median_ms']:.2f} ms (IQR: {results[s]['v_grad_shadow_iqr_ms']:.2f} ms), Peak RAM: {results[s]['v_grad_shadow_ram_mb']:.2f} MB")
        
    with open('results/runtime_memory.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    benchmark()
