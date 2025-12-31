#!/usr/bin/env python3
"""
Generate publication-quality activation drift comparison plots.
Designed for academic papers with clean aesthetics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# Use academic-friendly style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data from your analysis
DATA = {
    'layers': ['Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC2'],
    'layer_full': ['features.3', 'features.6', 'features.8', 'features.10', 'classifier.4'],
    'nude': {
        'l2': [0.88, 0.94, 1.09, 0.75, 1.17],
        'cosine': [0.76, 0.50, 0.34, 0.66, 0.17],
        'l2_overall': 0.9673,
        'cosine_overall': 0.4878,
        'acc': 14,
    },
    'srqat': {
        'l2': [0.48, 0.65, 0.65, 0.69, 0.86],
        'cosine': [0.88, 0.78, 0.79, 0.83, 0.52],
        'l2_overall': 0.6663,
        'cosine_overall': 0.7611,
        'acc': 46,
    }
}

# Academic color palette (colorblind-friendly)
COLORS = {
    'nude': '#E64B35',      # Red
    'srqat': '#4DBBD5',     # Blue
    'improvement': '#00A087', # Green
    'baseline': '#3C5488',   # Dark blue
}


def plot_overall_comparison(save_path='analysis/paper_figs'):
    """Plot overall L2 drift and cosine similarity comparison."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    models = ['Baseline\n(No SR-QAT)', 'SR-QAT\n(λ=1e⁻⁶)']
    x = np.arange(len(models))
    width = 0.5
    
    # L2 Drift (lower is better)
    ax1 = axes[0]
    l2_vals = [DATA['nude']['l2_overall'], DATA['srqat']['l2_overall']]
    bars1 = ax1.bar(x, l2_vals, width, color=[COLORS['nude'], COLORS['srqat']], 
                    edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Relative L2 Drift')
    ax1.set_title('(a) Activation Drift', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.2)
    
    # Add value labels
    for bar, val in zip(bars1, l2_vals):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement arrow
    ax1.annotate('', xy=(1, l2_vals[1]+0.05), xytext=(1, l2_vals[0]-0.05),
                arrowprops=dict(arrowstyle='->', color=COLORS['improvement'], lw=2))
    ax1.text(1.15, (l2_vals[0]+l2_vals[1])/2, '−31.1%', 
            color=COLORS['improvement'], fontsize=10, fontweight='bold', va='center')
    
    # Add "lower is better" indicator
    ax1.text(0.02, 0.98, '↓ lower is better', transform=ax1.transAxes,
            fontsize=8, color='gray', va='top', style='italic')
    
    # Cosine Similarity (higher is better)
    ax2 = axes[1]
    cos_vals = [DATA['nude']['cosine_overall'], DATA['srqat']['cosine_overall']]
    bars2 = ax2.bar(x, cos_vals, width, color=[COLORS['nude'], COLORS['srqat']],
                    edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('(b) Activation Similarity', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, val in zip(bars2, cos_vals):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement arrow
    ax2.annotate('', xy=(1, cos_vals[1]-0.03), xytext=(1, cos_vals[0]+0.03),
                arrowprops=dict(arrowstyle='->', color=COLORS['improvement'], lw=2))
    ax2.text(1.15, (cos_vals[0]+cos_vals[1])/2, '+56.0%',
            color=COLORS['improvement'], fontsize=10, fontweight='bold', va='center')
    
    # Add "higher is better" indicator
    ax2.text(0.02, 0.98, '↑ higher is better', transform=ax2.transAxes,
            fontsize=8, color='gray', va='top', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/activation_drift_overall.pdf', format='pdf')
    plt.savefig(f'{save_path}/activation_drift_overall.png', format='png')
    plt.close()
    print(f"Saved: {save_path}/activation_drift_overall.pdf/png")


def plot_layerwise_comparison(save_path='analysis/paper_figs'):
    """Plot layer-wise L2 drift and cosine similarity comparison."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    layers = DATA['layers']
    x = np.arange(len(layers))
    width = 0.35
    
    # L2 Drift by layer
    ax1 = axes[0]
    bars1_nude = ax1.bar(x - width/2, DATA['nude']['l2'], width, 
                         label='Baseline', color=COLORS['nude'],
                         edgecolor='black', linewidth=0.6)
    bars1_srqat = ax1.bar(x + width/2, DATA['srqat']['l2'], width,
                          label='SR-QAT', color=COLORS['srqat'],
                          edgecolor='black', linewidth=0.6)
    
    ax1.set_ylabel('Relative L2 Drift')
    ax1.set_title('(a) Layer-wise Activation Drift', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel('Layer')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(0, 1.4)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.text(len(layers)-0.5, 1.02, 'critical threshold', fontsize=8, color='gray', style='italic')
    
    # Cosine Similarity by layer
    ax2 = axes[1]
    bars2_nude = ax2.bar(x - width/2, DATA['nude']['cosine'], width,
                         label='Baseline', color=COLORS['nude'],
                         edgecolor='black', linewidth=0.6)
    bars2_srqat = ax2.bar(x + width/2, DATA['srqat']['cosine'], width,
                          label='SR-QAT', color=COLORS['srqat'],
                          edgecolor='black', linewidth=0.6)
    
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('(b) Layer-wise Activation Similarity', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel('Layer')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.text(0.1, 0.52, 'random threshold', fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/activation_drift_layerwise.pdf', format='pdf')
    plt.savefig(f'{save_path}/activation_drift_layerwise.png', format='png')
    plt.close()
    print(f"Saved: {save_path}/activation_drift_layerwise.pdf/png")


def plot_radar_chart(save_path='analysis/paper_figs'):
    """Plot radar chart for multi-dimensional comparison."""
    os.makedirs(save_path, exist_ok=True)
    
    # Metrics (normalized so that higher = better for all)
    categories = ['L2 Drift\n(inverted)', 'Cosine\nSimilarity', 'Conv2\nRobust', 
                  'Conv4\nRobust', 'FC2\nRobust']
    
    # Normalize: for L2, use 1 - value (so lower drift = higher score)
    nude_vals = [
        1 - DATA['nude']['l2_overall'],  # L2 inverted
        DATA['nude']['cosine_overall'],   # Cosine
        1 - DATA['nude']['l2'][0],        # Conv2 inverted
        1 - DATA['nude']['l2'][2],        # Conv4 inverted
        1 - DATA['nude']['l2'][4],        # FC2 inverted
    ]
    srqat_vals = [
        1 - DATA['srqat']['l2_overall'],
        DATA['srqat']['cosine_overall'],
        1 - DATA['srqat']['l2'][0],
        1 - DATA['srqat']['l2'][2],
        1 - DATA['srqat']['l2'][4],
    ]
    
    # Number of variables
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    nude_vals += nude_vals[:1]
    srqat_vals += srqat_vals[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Plot
    ax.plot(angles, nude_vals, 'o-', linewidth=2, label='Baseline', color=COLORS['nude'])
    ax.fill(angles, nude_vals, alpha=0.25, color=COLORS['nude'])
    ax.plot(angles, srqat_vals, 's-', linewidth=2, label='SR-QAT', color=COLORS['srqat'])
    ax.fill(angles, srqat_vals, alpha=0.25, color=COLORS['srqat'])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    
    plt.title('Fault Tolerance Profile\n(higher = better)', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}/activation_drift_radar.pdf', format='pdf')
    plt.savefig(f'{save_path}/activation_drift_radar.png', format='png')
    plt.close()
    print(f"Saved: {save_path}/activation_drift_radar.pdf/png")


def plot_combined_figure(save_path='analysis/paper_figs'):
    """Create a combined publication figure with all visualizations."""
    os.makedirs(save_path, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # ----- Row 1: Overall metrics -----
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    
    models = ['Baseline', 'SR-QAT']
    x = np.arange(len(models))
    width = 0.5
    
    # (a) L2 Drift
    l2_vals = [DATA['nude']['l2_overall'], DATA['srqat']['l2_overall']]
    bars1 = ax1.bar(x, l2_vals, width, color=[COLORS['nude'], COLORS['srqat']],
                    edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Relative L2 Drift')
    ax1.set_title('(a) Overall L2 Drift', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.2)
    for bar, val in zip(bars1, l2_vals):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    ax1.text(0.5, -0.15, '↓ lower is better', transform=ax1.transAxes,
            ha='center', fontsize=8, color='gray', style='italic')
    
    # (b) Cosine Similarity
    cos_vals = [DATA['nude']['cosine_overall'], DATA['srqat']['cosine_overall']]
    bars2 = ax2.bar(x, cos_vals, width, color=[COLORS['nude'], COLORS['srqat']],
                    edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('(b) Overall Cosine Similarity', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars2, cos_vals):
        ax2.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    ax2.text(0.5, -0.15, '↑ higher is better', transform=ax2.transAxes,
            ha='center', fontsize=8, color='gray', style='italic')
    
    # (c) Radar chart
    categories = ['L2↓', 'Cos↑', 'Conv2', 'Conv4', 'FC2']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    nude_radar = [1 - DATA['nude']['l2_overall'], DATA['nude']['cosine_overall'],
                  1 - DATA['nude']['l2'][0], 1 - DATA['nude']['l2'][2], 1 - DATA['nude']['l2'][4]]
    srqat_radar = [1 - DATA['srqat']['l2_overall'], DATA['srqat']['cosine_overall'],
                   1 - DATA['srqat']['l2'][0], 1 - DATA['srqat']['l2'][2], 1 - DATA['srqat']['l2'][4]]
    nude_radar += nude_radar[:1]
    srqat_radar += srqat_radar[:1]
    
    ax3.plot(angles, nude_radar, 'o-', linewidth=1.5, label='Baseline', color=COLORS['nude'], markersize=4)
    ax3.fill(angles, nude_radar, alpha=0.2, color=COLORS['nude'])
    ax3.plot(angles, srqat_radar, 's-', linewidth=1.5, label='SR-QAT', color=COLORS['srqat'], markersize=4)
    ax3.fill(angles, srqat_radar, alpha=0.2, color=COLORS['srqat'])
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=8)
    ax3.set_ylim(0, 1)
    ax3.set_title('(c) Robustness Profile', fontweight='bold', pad=15)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    
    # ----- Row 2: Layer-wise breakdown -----
    ax4 = fig.add_subplot(gs[1, :2])
    ax5 = fig.add_subplot(gs[1, 2])
    
    layers = DATA['layers']
    x_layer = np.arange(len(layers))
    width_layer = 0.35
    
    # (d) Layer-wise L2 and Cosine
    ax4_twin = ax4.twinx()
    
    # L2 as bars
    bars_nude = ax4.bar(x_layer - width_layer/2, DATA['nude']['l2'], width_layer,
                        label='Baseline L2', color=COLORS['nude'], alpha=0.7, edgecolor='black', linewidth=0.5)
    bars_srqat = ax4.bar(x_layer + width_layer/2, DATA['srqat']['l2'], width_layer,
                         label='SR-QAT L2', color=COLORS['srqat'], alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Cosine as lines
    line_nude, = ax4_twin.plot(x_layer, DATA['nude']['cosine'], 'o--', color=COLORS['nude'],
                               label='Baseline Cos', markersize=6, linewidth=1.5)
    line_srqat, = ax4_twin.plot(x_layer, DATA['srqat']['cosine'], 's-', color=COLORS['srqat'],
                                label='SR-QAT Cos', markersize=6, linewidth=1.5)
    
    ax4.set_ylabel('Relative L2 Drift (bars)', color='black')
    ax4_twin.set_ylabel('Cosine Similarity (lines)', color='black')
    ax4.set_xlabel('Layer')
    ax4.set_xticks(x_layer)
    ax4.set_xticklabels(layers)
    ax4.set_ylim(0, 1.4)
    ax4_twin.set_ylim(0, 1.0)
    ax4.set_title('(d) Layer-wise Analysis', fontweight='bold')
    
    # Combined legend
    handles1, labels1 = ax4.get_legend_handles_labels()
    handles2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=8, ncol=2)
    
    # (e) Improvement summary
    improvements = {
        'L2 Drift': -31.1,
        'Cosine Sim': 56.0,
        'Conv2 L2': -45.5,
        'Conv4 L2': -40.4,
        'FC2 L2': -26.5,
    }
    
    y_pos = np.arange(len(improvements))
    values = list(improvements.values())
    colors = [COLORS['improvement'] if v > 0 else COLORS['nude'] for v in values]
    
    bars = ax5.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(list(improvements.keys()))
    ax5.set_xlabel('Improvement (%)')
    ax5.set_title('(e) SR-QAT Improvement', fontweight='bold')
    ax5.axvline(x=0, color='black', linewidth=0.8)
    ax5.set_xlim(-60, 70)
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 2 if val > 0 else val - 2
        ha = 'left' if val > 0 else 'right'
        ax5.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
                va='center', ha=ha, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/activation_drift_combined.pdf', format='pdf')
    plt.savefig(f'{save_path}/activation_drift_combined.png', format='png')
    plt.close()
    print(f"Saved: {save_path}/activation_drift_combined.pdf/png")


def plot_heatmap_comparison(save_path='analysis/paper_figs'):
    """Create a heatmap showing layer-wise improvements."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    layers = DATA['layers']
    metrics = ['L2 Drift', 'Cosine Similarity']
    
    # Calculate improvements
    l2_improvement = [(n - s) / n * 100 for n, s in zip(DATA['nude']['l2'], DATA['srqat']['l2'])]
    cos_improvement = [(s - n) / n * 100 for n, s in zip(DATA['nude']['cosine'], DATA['srqat']['cosine'])]
    
    data = np.array([l2_improvement, cos_improvement])
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=70)
    
    # Labels
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(layers)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Layer')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(layers)):
            val = data[i, j]
            color = 'white' if abs(val) > 30 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', color=color, fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=15)
    
    plt.title('SR-QAT Improvement by Layer and Metric', fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(f'{save_path}/activation_drift_heatmap.pdf', format='pdf')
    plt.savefig(f'{save_path}/activation_drift_heatmap.png', format='png')
    plt.close()
    print(f"Saved: {save_path}/activation_drift_heatmap.pdf/png")


def main():
    save_path = 'analysis/paper_figs'
    
    print("Generating publication-quality figures...")
    print("=" * 50)
    
    plot_overall_comparison(save_path)
    plot_layerwise_comparison(save_path)
    plot_radar_chart(save_path)
    plot_combined_figure(save_path)
    plot_heatmap_comparison(save_path)
    
    print("=" * 50)
    print(f"All figures saved to: {save_path}/")
    print("\nGenerated files:")
    print("  - activation_drift_overall.pdf/png      (Figure for overall metrics)")
    print("  - activation_drift_layerwise.pdf/png    (Figure for layer-wise analysis)")
    print("  - activation_drift_radar.pdf/png        (Radar chart for robustness profile)")
    print("  - activation_drift_combined.pdf/png     (Combined multi-panel figure)")
    print("  - activation_drift_heatmap.pdf/png      (Heatmap of improvements)")


if __name__ == '__main__':
    main()

