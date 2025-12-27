#!/usr/bin/env python3
"""
Analyze layer-wise activation drift under SEU faults.
Compares clean vs. faulted activations to quantify fault propagation.
"""

import os
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import random

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze layer-wise activation drift under SEU faults')
    parser.add_argument('--config', type=str, required=True,
                       help='Evaluation config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--bit_width_config', type=str, default=None,
                       help='Bit width config file (default: from config)')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Output directory for results')
    parser.add_argument('--ber', type=float, default=1e-1,
                       help='Bit Error Rate')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of batches to analyze')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                       help='Random seeds for fault injection')
    parser.add_argument('--arch', type=str, default=None,
                       help='Model architecture (auto-detected from checkpoint if not provided)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset (auto-detected from config)')
    return parser.parse_args()

def get_activation_hook(layer_name, activations_dict):
    """Create forward hook to capture activations."""
    def hook(module, input, output):
        activations_dict[layer_name] = output.detach().cpu().numpy()
    return hook

def get_quantized_layer_names(model):
    """Get names of quantized layers."""
    from quan import QuanConv2d, QuanLinear
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'bits') and module.bits is not None and len(module.bits) > 1:
                layer_names.append(name)

    return layer_names

def compute_activation_metrics(clean_acts, faulted_acts):
    """Compute activation drift metrics."""
    metrics = {}

    # Relative L2 distance
    l2_distances = []
    cosine_sims = []

    for layer_name in clean_acts:
        clean_act = clean_acts[layer_name].flatten()
        faulted_act = faulted_acts[layer_name].flatten()

        # L2 distance (normalized by clean activation norm)
        l2_dist = np.linalg.norm(clean_act - faulted_act) / (np.linalg.norm(clean_act) + 1e-8)
        l2_distances.append(l2_dist)

        # Cosine similarity
        cos_sim = np.dot(clean_act, faulted_act) / (np.linalg.norm(clean_act) * np.linalg.norm(faulted_act) + 1e-8)
        cosine_sims.append(cos_sim)

    metrics['l2_relative'] = float(np.mean(l2_distances))
    metrics['cosine_similarity'] = float(np.mean(cosine_sims))
    metrics['l2_by_layer'] = {k: float(v) for k, v in zip(clean_acts.keys(), l2_distances)}
    metrics['cosine_by_layer'] = {k: float(v) for k, v in zip(clean_acts.keys(), cosine_sims)}

    return metrics

def setup_hooks(model, target_layers):
    """Setup forward hooks on target layers."""
    hooks = []
    activations_dict = {}

    for layer_name in target_layers:
        # Find the module by name
        module = model
        for part in layer_name.split('.'):
            module = getattr(module, part)

        hook = get_activation_hook(layer_name, activations_dict)
        hooks.append(module.register_forward_hook(hook))

    return hooks, activations_dict

def analyze_activation_drift(model, dataloader, target_layers, ber, seed, num_batches, device, FaultInjector):
    """Analyze activation drift for one seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()
    fault_injector = FaultInjector(model, enable_statistics=True, mode='ber', ber=ber)

    all_metrics = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f'Seed {seed}')):
            if batch_idx >= num_batches:
                break

            inputs = inputs.to(device)

            # Clean forward pass
            hooks, clean_acts = setup_hooks(model, target_layers)
            _ = model(inputs)
            [h.remove() for h in hooks]

            # Faulted forward pass
            hooks, faulted_acts = setup_hooks(model, target_layers)
            fault_injector.enable()
            _ = model(inputs)
            fault_injector.disable()
            [h.remove() for h in hooks]

            # Compute metrics
            metrics = compute_activation_metrics(clean_acts, faulted_acts)
            metrics['batch_idx'] = batch_idx
            all_metrics.append(metrics)

    return all_metrics

def main():
    args = parse_args()

    # Import modules
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from util.fault_injector import FaultInjector
    from model import create_model
    from quan import find_modules_to_quantize, replace_module_by_names
    from util.checkpoint import load_checkpoint
    from util.data_loader import init_dataloader
    from util.config import get_config

    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv

    arch = args.arch or config.arch
    dataset_name = args.dataset or config.dataloader.dataset

    # Set determinism
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create model
    model = create_model(arch, dataset=dataset_name, pre_trained=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Apply quantization
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)

    # Load checkpoint
    load_checkpoint(model, args.checkpoint, model_device=device)

    # Setup dataset
    _, _, val_loader, _, _ = init_dataloader(config.dataloader, arch)

    # Do a dummy forward pass to initialize output_size
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(val_loader))
        inputs = inputs.to(device)
        _ = model(inputs)

    # Set bit width configuration (use max target_bits for dynamic layers)
    target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
    max_target_bit = max(target_bits) if target_bits else 6

    from quan.func import QuanConv2d, QuanLinear
    from util.mpq import switch_bit_width

    print(f"Setting bit width to {max_target_bit}-bit for all layers...")
    dynamic_layers_set = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                continue
            if hasattr(config.quan, 'excepts') and name in config.quan.excepts:
                continue
            module.bits = (max_target_bit, max_target_bit)
            dynamic_layers_set += 1
    print(f"Set bits for {dynamic_layers_set} dynamic layers")

    # Switch bit width and do another dummy forward pass
    switch_bit_width(model, quan_scheduler=config.quan, wbit=max_target_bit, abits=max_target_bit)
    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    # Get target layers (quantized layers for AlexNet)
    target_layers = get_quantized_layer_names(model)

    print(f"Target layers: {target_layers}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze for each seed
    all_results = []
    for seed in args.seeds:
        print(f"\nAnalyzing seed {seed}...")
        metrics = analyze_activation_drift(
            model, val_loader, target_layers, args.ber, seed,
            args.num_batches, device, FaultInjector
        )

        for m in metrics:
            m['seed'] = seed
            all_results.append(m)

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, f'activation_drift_ber{args.ber}.csv')
    df.to_csv(csv_path, index=False)

    # Compute summary statistics
    summary = df.groupby('seed').agg({
        'l2_relative': ['mean', 'std'],
        'cosine_similarity': ['mean', 'std']
    }).round(4)

    print("\nSummary statistics:")
    print(summary)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # L2 drift
    ax1.errorbar(summary.index, summary['l2_relative']['mean'],
                yerr=summary['l2_relative']['std'], marker='o', capsize=5)
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Relative L2 Drift')
    ax1.set_title(f'Activation Drift (BER={args.ber})')
    ax1.grid(True, alpha=0.3)

    # Cosine similarity
    ax2.errorbar(summary.index, summary['cosine_similarity']['mean'],
                yerr=summary['cosine_similarity']['std'], marker='s', capsize=5)
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Activation Similarity (BER={args.ber})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f'activation_drift_ber{args.ber}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Layer-wise analysis
    layer_l2 = defaultdict(list)
    layer_cos = defaultdict(list)

    for _, row in df.iterrows():
        for layer, l2_val in row['l2_by_layer'].items():
            layer_l2[layer].append(l2_val)
        for layer, cos_val in row['cosine_by_layer'].items():
            layer_cos[layer].append(cos_val)

    # Plot layer-wise
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    layers = list(layer_l2.keys())
    l2_means = [np.mean(layer_l2[l]) for l in layers]
    l2_stds = [np.std(layer_l2[l]) for l in layers]

    ax1.bar(range(len(layers)), l2_means, yerr=l2_stds, capsize=5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Relative L2 Drift')
    ax1.set_title(f'Layer-wise L2 Drift (BER={args.ber})')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    cos_means = [np.mean(layer_cos[l]) for l in layers]
    cos_stds = [np.std(layer_cos[l]) for l in layers]

    ax2.bar(range(len(layers)), cos_means, yerr=cos_stds, capsize=5)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title(f'Layer-wise Cosine Similarity (BER={args.ber})')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    layer_plot_path = os.path.join(args.output_dir, f'layerwise_activation_drift_ber{args.ber}.png')
    plt.savefig(layer_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {args.output_dir}")
    print(f"- Summary CSV: {csv_path}")
    print(f"- Summary plot: {plot_path}")
    print(f"- Layer-wise plot: {layer_plot_path}")

if __name__ == '__main__':
    main()
