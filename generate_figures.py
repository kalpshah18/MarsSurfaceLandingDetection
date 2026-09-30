import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_figures():
    os.makedirs("media", exist_ok=True)
    
    with open('results/summary.json', 'r') as f:
        summary = json.load(f)
        
    # Example: F4 ablation bar chart
    fig, ax = plt.subplots()
    metrics = ['f1', 'iou', 'roc_auc']
    test_means = [summary['TEST'][m]['mean'] for m in metrics]
    test_errs = [summary['TEST'][m]['ci_upper'] - summary['TEST'][m]['mean'] for m in metrics]
    
    ax.bar(metrics, test_means, yerr=test_errs, capsize=5)
    ax.set_ylabel('Score')
    ax.set_title('TEST Set Metrics (Full Pipeline)')
    plt.savefig('media/f4_metrics.pdf')
    plt.close()
    
    # F8 runtime bar chart
    fig, ax = plt.subplots()
    labels = ['Gradient', 'Shadow', 'FFT', 'Fusion']
    times = [10.5, 2.1, 45.2, 5.0] # mocked for now, to be populated by measure_complexity
    ax.bar(labels, times)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Pipeline Runtime per Module (1024x1024)')
    plt.savefig('media/f8_runtime.pdf')
    plt.close()
    
    print("Figures generated in media/")

if __name__ == "__main__":
    generate_figures()
