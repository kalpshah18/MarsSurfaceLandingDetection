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

def generate_footprint_envelope_figure(sweep_path='results/footprint_sweep.json',
                                      out_stem='media/fig_footprint_envelope'):
    """Operating-envelope figure (paper Section 4.3) from run_footprint_sweep.py output.

    Fraction of the 80 TEST tiles meeting Footprint OK, Square Fit and Safe
    Landing as a function of the required footprint side L. Colours are a
    colour-blind-checked categorical set; every line also has its own marker
    and line style so the figure survives greyscale printing.
    """
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    with open(sweep_path) as f:
        sweep = json.load(f)
    fine = sweep['fine']
    Ls = sorted(int(L) for L in fine['grad_shadow'])
    main = 'bright' if 'bright' in fine else 'grad_shadow'
    max_side = sweep['box_summary'][main]['max_min_side']

    def series(method, key):
        return [fine[method][str(L)][key] for L in Ls]

    lab = r'$V_{\mathrm{full}}$' if main == 'bright' else r'$V_{\mathrm{grad+shadow}}$'
    lines = [
        (main, 'footprint_ok', lab + ' Footprint OK', '#2a78d6', '-', 'o'),
        (main, 'square_fit', lab + ' Square Fit', '#eb6834', '--', 's'),
        (main, 'safe_landing', lab + ' Safe Landing', '#1baf7a', '-', '^'),
    ]
    if main == 'bright':
        lines.append(('grad_shadow', 'safe_landing', r'$V_{\mathrm{grad+shadow}}$ Safe Landing', '#6b6b66', ':', 'D'))
    else:
        lines.append(('grad', 'safe_landing', r'$V_{\mathrm{grad}}$ Safe Landing', '#6b6b66', ':', 'D'))

    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.1))

    if max_side < Ls[-1]:
        ax.axvspan(max_side + 0.5, Ls[-1] + 0.5, color='#d9d9d4', alpha=0.6, lw=0, zorder=0)
        ax.text(max_side + 1.2, 0.97, f'no box fits an\n$L\\times L$ square\n($L>{max_side}$ m)',
                va='top', ha='left', fontsize=6.5, color='#3d3d3a')
    ax.axvline(50, color='#3d3d3a', ls='--', lw=1.0, zorder=1)
    ax.text(50.8, 0.55, 'large-lander\nrequirement\n(50 m)', va='center', ha='left',
            fontsize=6.5, color='#3d3d3a')

    for method, key, label, color, ls, marker in lines:
        ax.plot(Ls, series(method, key), color=color, ls=ls, lw=1.6, marker=marker,
                markersize=3.2, markevery=5, label=label, zorder=3)

    ax.set_xlim(Ls[0], Ls[-1])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Required footprint side $L$ (m)')
    ax.set_ylabel('Fraction of TEST tiles (n = 80)')
    ax.set_xticks(range(10, Ls[-1] + 1, 10))
    ax.grid(True, color='#e6e6e1', lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              handlelength=2.4, columnspacing=1.0, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_stem + '.pdf', bbox_inches='tight')
    fig.savefig(out_stem + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Wrote {out_stem}.pdf and {out_stem}.png')


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'footprint':
        generate_footprint_envelope_figure()
    else:
        generate_figures()
        generate_footprint_envelope_figure()
