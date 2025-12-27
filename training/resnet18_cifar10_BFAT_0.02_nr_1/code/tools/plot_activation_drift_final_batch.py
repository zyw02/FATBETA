import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib.gridspec import GridSpec

def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)
    l2_relative = df['l2_relative'].mean()
    cosine_similarity = df['cosine_similarity'].mean()
    
    # Parse layer-wise
    all_l2_by_layer = []
    all_cos_by_layer = []
    for _, row in df.iterrows():
        s_l2 = row['l2_by_layer'].replace('np.float64(', '').replace('np.float32(', '').replace(')', '')
        s_cos = row['cosine_by_layer'].replace('np.float64(', '').replace('np.float32(', '').replace(')', '')
        all_l2_by_layer.append(ast.literal_eval(s_l2))
        all_cos_by_layer.append(ast.literal_eval(s_cos))
        
    def aggregate(dict_list):
        if not dict_list: return {}
        agg = {}
        for k in dict_list[0].keys():
            agg[k] = np.mean([d[k] for d in dict_list if k in d])
        return agg

    return {
        'l2_overall': l2_relative,
        'cosine_overall': cosine_similarity,
        'l2_by_layer': aggregate(all_l2_by_layer),
        'cosine_by_layer': aggregate(all_cos_by_layer)
    }

def main():
    BER = 0.04
    output_dir = "analysis/final_batch_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {
        'Baseline': f'analysis/baseline/activation_drift_ber{BER}.csv',
        'BFAT_TRUNC': f'analysis/bfat_trunc/activation_drift_ber{BER}.csv',
        'BFAT_0.02': f'analysis/bfat_02/activation_drift_ber{BER}.csv',
        'BFAT_Dual': f'analysis/bfat_dual/activation_drift_ber{BER}.csv',
        'BFAT_Combined': f'analysis/bfat_combined/activation_drift_ber{BER}.csv'
    }
    
    data = {}
    for name, path in paths.items():
        if os.path.exists(path):
            data[name] = load_and_aggregate(path)
        else:
            print(f"Warning: {path} not found!")

    if not data:
        print("No data loaded. Exit.")
        return

    # 1. Prepare Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1, 1])
    
    colors = ['#E64B35', '#3C5488', '#4DBBD5', '#00A087', '#F39B7F']
    model_names = list(data.keys())
    
    # --- (A) Overall Bar Chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    l2_overalls = [data[m]['l2_overall'] for m in model_names]
    cos_overalls = [data[m]['cosine_overall'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width/2, l2_overalls, width, label='L2 Relative Drift', color=colors, alpha=0.8, edgecolor='black')
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + width/2, cos_overalls, width, label='Cosine Similarity', color=colors, alpha=0.4, hatch='//', edgecolor='black')
    
    ax1.set_ylabel('L2 Relative Drift (lower is better)')
    ax1_twin.set_ylabel('Cosine Similarity (higher is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.set_title('(A) Overall Activation Drift & Similarity (BER=0.04)', fontweight='bold')
    
    # --- (B) Radar Chart ---
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    # Pick 5 key layers or metrics
    # ResNet18 layers: layer1.0.conv1, layer2.0.conv1, layer3.0.conv1, layer4.0.conv1, fc
    # We'll just pick 5 representative indices
    layers_available = list(data['Baseline']['l2_by_layer'].keys())
    step = max(1, len(layers_available) // 5)
    radar_layers = layers_available[::step][:5]
    
    categories = ['L2_Total(Inv)', 'Cos_Total'] + [f"{l[:10]}.." for l in radar_layers]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for i, m in enumerate(model_names):
        # Normalize: L2 is 1/(1+val), Cos is val
        # For layer-wise L2, also 1/(1+val)
        values = [1.0 / (1.0 + data[m]['l2_overall']), data[m]['cosine_overall']]
        for l in radar_layers:
            values.append(1.0 / (1.0 + data[m]['l2_by_layer'].get(l, 0)))
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=m, color=colors[i])
        ax2.fill(angles, values, alpha=0.1, color=colors[i])
        
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('(B) Robustness Fingerprint (Inverted L2 & Cosine)', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # --- (C) Layer-wise L2 Drift (Line Plot) ---
    ax3 = fig.add_subplot(gs[1, 0])
    layers_sorted = sorted(layers_available, key=lambda x: layers_available.index(x))
    for i, m in enumerate(model_names):
        y = [data[m]['l2_by_layer'][l] for l in layers_sorted]
        ax3.plot(layers_sorted, y, 'o-', label=m, color=colors[i], markersize=4)
    
    ax3.set_ylabel('Relative L2 Drift')
    ax3.set_title('(C) Fault Propagation: Layer-wise L2 Drift', fontweight='bold')
    ax3.set_xticks(range(len(layers_sorted)))
    ax3.set_xticklabels([l.split('.')[-1] for l in layers_sorted], rotation=45, ha='right', size=8)
    ax3.legend()

    # --- (D) Layer-wise Cosine Similarity (Line Plot) ---
    ax4 = fig.add_subplot(gs[1, 1])
    for i, m in enumerate(model_names):
        y = [data[m]['cosine_by_layer'][l] for l in layers_sorted]
        ax4.plot(layers_sorted, y, 's-', label=m, color=colors[i], markersize=4)
        
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('(D) Feature Preservation: Layer-wise Cosine Similarity', fontweight='bold')
    ax4.set_xticks(range(len(layers_sorted)))
    ax4.set_xticklabels([l.split('.')[-1] for l in layers_sorted], rotation=45, ha='right', size=8)
    ax4.set_ylim(0, 1.1)
    ax4.legend()

    plt.tight_layout()
    final_path = os.path.join(output_dir, "resnet18_drift_analysis_combined.png")
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Final paper-style combined plot saved to: {final_path}")

if __name__ == "__main__":
    main()

