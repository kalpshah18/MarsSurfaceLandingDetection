import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_figures():
    os.makedirs("media", exist_ok=True)
    
    with open('results/summary.json', 'r') as f:
        summary = json.load(f)
    
    with open('results/complexity.json', 'r') as f:
        complexity = json.load(f)
        
    ablations_df = pd.read_csv('results/ablations.csv')
    baselines_df = pd.read_csv('results/baselines.csv')

    # F4 ablation and baselines bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5']
    means = []
    
    for m in metrics:
        if m.startswith('A'):
            means.append(ablations_df[f'{m}_f1'].mean())
        else:
            means.append(baselines_df[f'{m}_f1'].mean())
            
    ax.bar(metrics, means, color='skyblue')
    ax.set_ylabel('F1 Score')
    ax.set_title('TEST Set F1 Score by Ablation/Baseline')
    plt.savefig('media/f4_metrics.pdf')
    plt.close()
    
    # F8 runtime bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Gradient', 'Shadow', 'FFT', 'Fusion', 'Total']
    times = [
        complexity['1024']['Gradient']['median'],
        complexity['1024']['Shadow']['median'],
        complexity['1024']['FFT']['median'],
        complexity['1024']['Fusion']['median'],
        complexity['1024']['Total']['median']
    ]
    ax.bar(labels, times, color='coral')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Pipeline Runtime per Module (1024x1024)')
    for i, v in enumerate(times):
        ax.text(i, v + 1, f"{v:.1f}", ha='center')
    plt.savefig('media/f8_runtime.pdf')
    plt.close()
    
    # F1 Pipeline block diagram (mock as PNG copy of empty for now, or just leave as is if we have it)
    # We will generate it using tikz in LaTeX, or skip F1 generation in python.
    
    # Generate tables as latex string to inject
    # Main results table
    f1_mean = summary['TEST']['f1']['mean']
    f1_lower = summary['TEST']['f1']['ci_lower']
    f1_upper = summary['TEST']['f1']['ci_upper']
    iou_mean = summary['TEST']['iou']['mean']
    iou_lower = summary['TEST']['iou']['ci_lower']
    iou_upper = summary['TEST']['iou']['ci_upper']
    
    with open('results/latex_tables.tex', 'w') as f:
        f.write(f"TEST F1: {f1_mean:.3f} [{f1_lower:.3f}, {f1_upper:.3f}]\n")
        f.write(f"TEST IoU: {iou_mean:.3f} [{iou_lower:.3f}, {iou_upper:.3f}]\n")

    print("Figures and tables generated in media/ and results/")

if __name__ == "__main__":
    generate_figures()
