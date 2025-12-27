#!/usr/bin/env python3
"""
Compare activation drift between different models under SEU faults.
Generates comprehensive analysis for paper.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Compare activation drift between models')
    parser.add_argument('--nude_csv', type=str, required=True,
                       help='CSV file for nude model activation drift')
    parser.add_argument('--srqat_csv', type=str, required=True,
                       help='CSV file for SR-QAT model activation drift')
    parser.add_argument('--output_dir', type=str, default='analysis/comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--nude_label', type=str, default='Nude (Clean120)',
                       help='Label for nude model')
    parser.add_argument('--srqat_label', type=str, default='SR-QAT (λ=1e-6)',
                       help='Label for SR-QAT model')
    return parser.parse_args()

def load_and_process_data(csv_path):
    """Load CSV and extract metrics."""
    df = pd.read_csv(csv_path)

    # 聚合所有行的数据（均值）
    l2_relative = df['l2_relative'].mean()
    cosine_similarity = df['cosine_similarity'].mean()

    # 解析层级数据并取均值
    import ast
    all_l2_by_layer = []
    all_cosine_by_layer = []
    
    for _, row in df.iterrows():
        try:
            # 稳健性处理：替换掉可能导致 ast.literal_eval 报错的 numpy 字符串
            s_l2 = row['l2_by_layer'].replace('np.float64(', '').replace('np.float32(', '').replace(')', '')
            s_cos = row['cosine_by_layer'].replace('np.float64(', '').replace('np.float32(', '').replace(')', '')
            
            all_l2_by_layer.append(ast.literal_eval(s_l2))
            all_cosine_by_layer.append(ast.literal_eval(s_cos))
        except Exception as e:
            print(f"Warning: Failed to parse row: {e}")
            continue

    # 聚合字典：对每个层级的所有数值求平均
    def aggregate_dicts(dict_list):
        if not dict_list: return {}
        agg = {}
        keys = dict_list[0].keys()
        for k in keys:
            agg[k] = np.mean([d[k] for d in dict_list if k in d])
        return agg

    return {
        'l2_relative': l2_relative,
        'cosine_similarity': cosine_similarity,
        'l2_by_layer': aggregate_dicts(all_l2_by_layer),
        'cosine_by_layer': aggregate_dicts(all_cosine_by_layer)
    }

def create_comparison_plot(nude_data, srqat_data, nude_label, srqat_label, output_dir):
    """Create comprehensive comparison plot."""

    # Extract data
    models = [nude_label, srqat_label]
    l2_vals = [nude_data['l2_relative'], srqat_data['l2_relative']]
    cos_vals = [nude_data['cosine_similarity'], srqat_data['cosine_similarity']]

    # Get common layers
    layers = list(set(nude_data['l2_by_layer'].keys()) & set(srqat_data['l2_by_layer'].keys()))
    layers.sort()

    l2_nude = [nude_data['l2_by_layer'][l] for l in layers]
    l2_srqat = [srqat_data['l2_by_layer'][l] for l in layers]
    cos_nude = [nude_data['cosine_by_layer'][l] for l in layers]
    cos_srqat = [srqat_data['cosine_by_layer'][l] for l in layers]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Overall metrics comparison
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, l2_vals, width, label='L2 Relative Drift', alpha=0.8)
    ax1.set_ylabel('L2 Relative Drift')
    ax1.set_title('Overall Activation Drift Comparison (BER=0.1)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, alpha=0.3)

    # Add values on bars
    for i, v in enumerate(l2_vals):
        ax1.text(i - width/2, v + 0.01, '.3f', ha='center', va='bottom')

    ax2.bar(x - width/2, cos_vals, width, label='Cosine Similarity', alpha=0.8, color='orange')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Overall Activation Similarity Comparison (BER=0.1)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(True, alpha=0.3)

    # Add values on bars
    for i, v in enumerate(cos_vals):
        ax2.text(i - width/2, v + 0.01, '.3f', ha='center', va='bottom')

    # Layer-wise L2 drift
    x_layer = np.arange(len(layers))
    width_layer = 0.35

    ax3.bar(x_layer - width_layer/2, l2_nude, width_layer, label=nude_label, alpha=0.8)
    ax3.bar(x_layer + width_layer/2, l2_srqat, width_layer, label=srqat_label, alpha=0.8)
    ax3.set_ylabel('L2 Relative Drift')
    ax3.set_title('Layer-wise L2 Drift Comparison')
    ax3.set_xticks(x_layer)
    ax3.set_xticklabels(layers, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Layer-wise cosine similarity
    ax4.bar(x_layer - width_layer/2, cos_nude, width_layer, label=nude_label, alpha=0.8)
    ax4.bar(x_layer + width_layer/2, cos_srqat, width_layer, label=srqat_label, alpha=0.8)
    ax4.set_ylabel('Cosine Similarity')
    ax4.set_title('Layer-wise Cosine Similarity Comparison')
    ax4.set_xticks(x_layer)
    ax4.set_xticklabels(layers, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'activation_drift_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path

def create_summary_table(nude_data, srqat_data, nude_label, srqat_label, output_dir):
    """Create summary table."""

    # Calculate improvements
    l2_improvement = ((nude_data['l2_relative'] - srqat_data['l2_relative']) / nude_data['l2_relative']) * 100
    cos_improvement = ((srqat_data['cosine_similarity'] - nude_data['cosine_similarity']) / nude_data['cosine_similarity']) * 100

    summary_data = {
        'Model': [nude_label, srqat_label, 'Improvement'],
        'L2 Relative Drift': [
            '.4f',
            '.4f',
            '+.1f' if l2_improvement > 0 else '.1f'
        ],
        'Cosine Similarity': [
            '.4f',
            '.4f',
            '+.1f' if cos_improvement > 0 else '.1f'
        ]
    }

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'activation_drift_summary.csv')
    df_summary.to_csv(summary_path, index=False)

    return summary_path

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading nude model data...")
    nude_data = load_and_process_data(args.nude_csv)

    print("Loading SR-QAT model data...")
    srqat_data = load_and_process_data(args.srqat_csv)

    # Create comparison plot
    print("Creating comparison plot...")
    plot_path = create_comparison_plot(nude_data, srqat_data, args.nude_label, args.srqat_label, args.output_dir)

    # Create summary table
    print("Creating summary table...")
    summary_path = create_summary_table(nude_data, srqat_data, args.nude_label, args.srqat_label, args.output_dir)

    print("\nComparison complete!")
    print(f"- Comparison plot: {plot_path}")
    print(f"- Summary table: {summary_path}")

    # Print key findings
    l2_improvement = ((nude_data['l2_relative'] - srqat_data['l2_relative']) / nude_data['l2_relative']) * 100
    cos_improvement = ((srqat_data['cosine_similarity'] - nude_data['cosine_similarity']) / nude_data['cosine_similarity']) * 100

    print("\n" + "="*70)
    print("Key Findings (Layer-wise Activation Drift Analysis)")
    print("="*70)
    print(f"\n{args.nude_label}:")
    print(f"  - L2 Relative Drift: {nude_data['l2_relative']:.4f}")
    print(f"  - Cosine Similarity: {nude_data['cosine_similarity']:.4f}")
    print(f"\n{args.srqat_label}:")
    print(f"  - L2 Relative Drift: {srqat_data['l2_relative']:.4f}")
    print(f"  - Cosine Similarity: {srqat_data['cosine_similarity']:.4f}")
    print(f"\nImprovement with SR-QAT:")
    print(f"  - L2 Drift Reduction: {l2_improvement:.1f}% (lower is better)")
    print(f"  - Cosine Similarity Improvement: {cos_improvement:.1f}% (higher is better)")

    print("\n" + "-"*70)
    print("Interpretation:")
    print("-"*70)
    print("  L2 Relative Drift = ||act_faulted - act_clean|| / ||act_clean||")
    print("  - Lower L2 drift means the fault causes less deviation in activations")
    print("  Cosine Similarity = cos(act_faulted, act_clean)")
    print("  - Higher similarity means the activation direction is better preserved")
    print("\nConclusion:")
    if l2_improvement > 0 and cos_improvement > 0:
        print(f"  SR-QAT model exhibits {l2_improvement:.1f}% lower activation drift")
        print(f"  and {cos_improvement:.1f}% higher activation similarity under SEU faults.")
        print("  This indicates SR-QAT effectively LIMITS FAULT PROPAGATION through the network.")
    print("="*70)
if __name__ == '__main__':
    main()
