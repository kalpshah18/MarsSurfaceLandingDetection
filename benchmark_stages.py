"""Per-stage runtime and peak memory of V_grad+shadow on real TEST tiles (Table 5).

Replaces the earlier benchmarks on random-noise images (results/complexity.json,
results/runtime_memory.json), which are a worst case for contour extraction and the
maximal-rectangle search and are not representative of the evaluated imagery.

Stages (config from results/best_config.json):
    sobel      compute_gradient_hazard
    shadow     augment_shadow
    fuse       threshold + connected-component filter  (= fuse_and_select - rectangle)
    rectangle  maximal_rectangle on the filtered mask
Timing: every TEST tile, `REPEATS` timed runs after one warm-up, OpenCV restricted to
one thread. Native 768x768 tiles, plus the same tiles resized to 512 and 1024 for scaling.
Memory: peak resident set (Windows peak working set / Linux ru_maxrss) of a fresh
subprocess, minus its resident set just before the pipeline call; tracemalloc's Python-heap
peak is reported alongside for comparison (it does not see OpenCV's native buffers).

Output: results/review/benchmark_stages.json
"""
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc

import cv2
import numpy as np

from pipeline import augment_shadow, compute_gradient_hazard, fuse_and_select, maximal_rectangle
from review_common import iter_tiles, load_config

REPEATS = 5
SIZES = [512, 768, 1024]
OUT = os.path.join('results', 'review', 'benchmark_stages.json')


def stage_times(img, cfg):
    t0 = time.perf_counter()
    G = compute_gradient_hazard(img)
    t1 = time.perf_counter()
    G_aug, _ = augment_shadow(img, G, tau=cfg['tau'], a_min=cfg['a_min'])
    t2 = time.perf_counter()
    _, _, filt, _ = fuse_and_select(G_aug, None, theta=cfg['theta'], min_component_area=cfg['min_component_area'])
    t3 = time.perf_counter()
    maximal_rectangle(filt)
    t4 = time.perf_counter()
    fuse_total = t3 - t2
    rect = t4 - t3
    # fuse_and_select also runs maximal_rectangle internally; subtract one rectangle call.
    return {'sobel': t1 - t0, 'shadow': t2 - t1, 'fuse': max(0.0, fuse_total - rect), 'rectangle': rect}


def peak_rss_mb():
    import psutil
    mi = psutil.Process().memory_info()
    return getattr(mi, 'peak_wset', None) or getattr(mi, 'rss')


def memory_child(size):
    """Run in a fresh process: peak RSS increase and tracemalloc peak for one tile."""
    import psutil
    cfg = load_config()['V_grad_shadow']
    tile = next(iter_tiles('TEST'))
    img = tile['ori'] if size == 768 else cv2.resize(tile['ori'], (size, size), interpolation=cv2.INTER_LINEAR)
    del tile
    base = psutil.Process().memory_info().rss
    tracemalloc.start()
    G = compute_gradient_hazard(img)
    G_aug, _ = augment_shadow(img, G, tau=cfg['tau'], a_min=cfg['a_min'])
    fuse_and_select(G_aug, None, theta=cfg['theta'], min_component_area=cfg['min_component_area'])
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(json.dumps({'peak_rss_increase_mb': (peak_rss_mb() - base) / 2 ** 20,
                      'tracemalloc_peak_mb': tm_peak / 2 ** 20}))


def main():
    cv2.setNumThreads(1)
    cfg = load_config()['V_grad_shadow']
    tiles = [t['ori'] for t in iter_tiles('TEST')]
    res = {}
    for size in SIZES:
        imgs = tiles if size == 768 else [cv2.resize(t, (size, size), interpolation=cv2.INTER_LINEAR) for t in tiles]
        stage_times(imgs[0], cfg)  # warm-up
        samples = {k: [] for k in ('sobel', 'shadow', 'fuse', 'rectangle', 'total')}
        for img in imgs:
            for _ in range(REPEATS):
                st = stage_times(img, cfg)
                for k, v in st.items():
                    samples[k].append(v * 1000)
                samples['total'].append(sum(st.values()) * 1000)
        stats = {k: {'median_ms': float(np.median(v)), 'q25_ms': float(np.percentile(v, 25)),
                     'q75_ms': float(np.percentile(v, 75)), 'n': len(v)} for k, v in samples.items()}
        mem = json.loads(subprocess.run([sys.executable, __file__, '--mem', str(size)], capture_output=True,
                                        text=True, check=True).stdout.strip().splitlines()[-1])
        res[str(size)] = {'stages': stats, 'memory': mem}
        print(size, {k: round(v['median_ms'], 1) for k, v in stats.items()}, mem, flush=True)
    import psutil
    try:
        cpu = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'],
                             capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        cpu = platform.processor()
    host = {'cpu': cpu, 'logical_cpus': os.cpu_count(), 'ram_gb': psutil.virtual_memory().total / 2 ** 30,
            'os': platform.platform(), 'python': platform.python_version(), 'opencv': cv2.__version__,
            'numpy': np.__version__, 'opencv_threads': 1}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'host': host, 'repeats_per_tile': REPEATS, 'n_tiles': len(tiles), 'results': res}, f, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--mem':
        memory_child(int(sys.argv[2]))
    else:
        main()
