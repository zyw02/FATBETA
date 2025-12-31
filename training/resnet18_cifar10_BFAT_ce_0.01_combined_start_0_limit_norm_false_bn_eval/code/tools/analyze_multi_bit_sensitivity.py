#!/usr/bin/env python
"""
Multi-Bit Sensitivity Analysis Tool

This script systematically analyzes layer sensitivity across different bit-width configurations.
It helps identify which layers are consistently sensitive and how their sensitivity scales with bit-width.

Usage:
    python tools/analyze_multi_bit_sensitivity.py \
        --config configs/eval/eval_alexnet_cifar10_sensitive_stage1.yaml \
        --resume_path training/alexnet_cifar10_sensitive_stage1/best.pth.tar \
        --ber 1e-1
"""

import argparse
import sys
import os
import json
from pathlib import Path
import torch
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from util.config import get_config
from model import create_model
from util.data_loader import init_dataloader
from util.checkpoint import load_checkpoint
from util.fault_injector import FaultInjector
from quan import find_modules_to_quantize, replace_module_by_names
from util.utils import preprocess_model
from util.mpq import switch_bit_width


def compute_accuracy(model: torch.nn.Module, data_loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    if total == 0:
        return 0.0
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Analyze Multi-Bit Layer Sensitivity')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit Error Rate for injection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for quick evaluation')
    parser.add_argument('--bit_configs', type=str, default=None,
                        help='Comma-separated list like "w2a2,w4a4". If not provided, '
                             'use all supported bits shared by weight/activation quantizers.')
    parser.add_argument('--summary_path', type=str, default=None,
                        help='Optional path to save sensitivity summary as JSON')
    
    args = parser.parse_args()
    
    # --- Setup ---
    # Load config (hacky argv swap to use get_config)
    original_argv = sys.argv.copy()
    sys.argv = ['analyze_multi_bit_sensitivity.py', args.config]
    try:
        template_path = Path(__file__).parent.parent / 'template.yaml'
        if template_path.exists():
            configs = get_config(str(template_path))
        else:
            import yaml
            import munch
            with open(args.config, 'r') as f:
                configs = munch.munchify(yaml.safe_load(f))
    finally:
        sys.argv = original_argv

    # Defaults required by downstream utilities (validate/update_meter expect these fields)
    default_attrs = {
        'local_rank': 0,
        'rank': 0,
        'world_size': 1,
        'enable_dynamic_bit_training': True,
        'split_aw_cands': False,
        'smoothing': 0.0,
    }
    for attr, value in default_attrs.items():
        if not hasattr(configs, attr):
            setattr(configs, attr, value)
    configs.post_training_batchnorm_calibration = False
    
    # Logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    
    # Device & Seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs.device = device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Create & Load Model
    model = create_model(configs.arch, dataset=getattr(configs.dataloader, 'dataset', 'cifar10'))
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    load_checkpoint(model, args.resume_path, device, strict=False)
    
    # Data Loader
    train_loader, val_loader, test_loader, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    if args.num_samples > 0:
        indices = torch.randperm(len(test_loader.dataset))[:args.num_samples]
        test_subset = torch.utils.data.Subset(test_loader.dataset, indices)
        eval_loader = torch.utils.data.DataLoader(test_subset, batch_size=configs.dataloader.batch_size, shuffle=False, num_workers=4)
    else:
        eval_loader = test_loader
    
    # --- Discover quantized layers and supported bits ---
    quantized_layers = []
    supported_weight_bits = set()
    supported_act_bits = set()
    dynamic_weight_bits = None
    dynamic_act_bits = None
    for name, m in model.named_modules():
        if hasattr(m, 'quan_w_fn') and m.quan_w_fn is not None:
            quantized_layers.append(name)
            bit_mapping = getattr(m.quan_w_fn, 'bit_mapping', None)
            if bit_mapping:
                supported_weight_bits.update(bit_mapping.keys())
                if getattr(m, 'fixed_bits', None) is None:
                    current_bits = set(bit_mapping.keys())
                    dynamic_weight_bits = current_bits if dynamic_weight_bits is None else dynamic_weight_bits.intersection(current_bits)
        if hasattr(m, 'quan_a_fn') and m.quan_a_fn is not None:
            bit_mapping = getattr(m.quan_a_fn, 'bit_mapping', None)
            if bit_mapping:
                supported_act_bits.update(bit_mapping.keys())
                if getattr(m, 'fixed_bits', None) is None:
                    current_bits = set(bit_mapping.keys())
                    dynamic_act_bits = current_bits if dynamic_act_bits is None else dynamic_act_bits.intersection(current_bits)
    
    if not quantized_layers:
        raise RuntimeError("No quantized layers found — cannot run sensitivity analysis.")

    if not supported_weight_bits:
        supported_weight_bits = {2, 3, 4, 5, 6}
    if not supported_act_bits:
        supported_act_bits = supported_weight_bits.copy()

    weight_candidates = dynamic_weight_bits if dynamic_weight_bits else supported_weight_bits
    act_candidates = dynamic_act_bits if dynamic_act_bits else supported_act_bits
    shared_bits = sorted(weight_candidates.intersection(act_candidates), reverse=True)
    
    # --- Analysis Configuration ---
    if args.bit_configs:
        bit_configs = []
        for token in args.bit_configs.split(','):
            token = token.strip()
            if not token:
                continue
            if not (token.startswith('w') and 'a' in token):
                raise ValueError(f"Invalid bit config format: {token}. Expected wXaY.")
            w_part, a_part = token[1:].split('a')
            try:
                w_bit = int(w_part)
                a_bit = int(a_part)
            except ValueError as exc:
                raise ValueError(f"Invalid bit numbers in {token}") from exc
            if w_bit not in supported_weight_bits:
                logging.warning("Skipping unsupported weight bit-width %s in %s", w_bit, token)
                continue
            if a_bit not in supported_act_bits:
                logging.warning("Skipping unsupported activation bit-width %s in %s", a_bit, token)
                continue
            bit_configs.append((w_bit, a_bit))
        if not bit_configs:
            raise ValueError("No valid bit configurations provided after filtering unsupported bits.")
    else:
        if shared_bits:
            bit_configs = [(b, b) for b in shared_bits]
        else:
            max_act = max(supported_act_bits)
            bit_configs = [(b, max_act) for b in sorted(supported_weight_bits, reverse=True)]
    results = {layer: {} for layer in quantized_layers}
    
    print(f"\n{'='*80}")
    print(f"Multi-Bit Sensitivity Analysis (BER={args.ber:.0e})")
    print(f"{'='*80}")
    
    for w_bit, a_bit in bit_configs:
        config_name = f"w{w_bit}a{a_bit}"
        print(f"\nAnalyzing Configuration: {config_name}")
        print("-" * 40)
        
        # 1. Set global bit-width
        # Note: We exclude first/last layers from this global switch usually, 
        # but for sensitivity analysis we might want to test them too if they are in `quantized_layers`.
        # `switch_bit_width` usually respects `excepts` in config.
        switch_bit_width(model, configs.quan, wbit=w_bit, abits=a_bit)
        
        # Initialize (Dummy forward)
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32).to(device))
            
        # 2. Baseline Accuracy for this config
        logger.info(f"Measuring baseline for {config_name}...")
        clean_acc = compute_accuracy(model, eval_loader, device)
        print(f"  Baseline Clean Acc: {clean_acc:.2f}%")
        
        # 3. Layer-wise injection
        for layer_name in quantized_layers:
            injector = FaultInjector(
                model, mode="ber", ber=args.ber, device=device,
                enable_in_inference=True, seed=args.seed,
                whitelist_layer=layer_name,
                skip_first_last=False # We want to test ALL layers, even first/last
            )
            
            injector.enable()
            acc = compute_accuracy(model, eval_loader, device)
            injector.disable()
            
            drop = clean_acc - acc
            results[layer_name][config_name] = drop
            print(f"  {layer_name:<20} Drop: {drop:.2f}% (Acc: {acc:.2f}%)")

    # --- Summary & Ranking ---
    print(f"\n{'='*80}")
    print("SENSITIVITY SUMMARY (Accuracy Drop %)")
    print(f"{'='*80}")
    
    df = pd.DataFrame.from_dict(results, orient='index')
    df['avg_drop'] = df.mean(axis=1)
    df = df.sort_values('avg_drop', ascending=False)
    
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df)
    
    print(f"\n{'='*80}")
    print("SENSITIVITY RANKING (Most Sensitive -> Least Sensitive)")
    print(f"{'='*80}")
    ranking = []
    for i, (layer, row) in enumerate(df.iterrows()):
        print(f"{i+1}. {layer} (Avg Drop: {row['avg_drop']:.2f}%)")
        ranking.append({'layer': layer, 'avg_drop': float(row['avg_drop'])})

    if args.summary_path:
        summary = {
            'ber': args.ber,
            'bit_configs': [{'weight_bits': w, 'act_bits': a} for w, a in bit_configs],
            'results': {
                layer: {cfg: float(val) for cfg, val in metrics.items()}
                for layer, metrics in results.items()
            },
            'avg_drop': {layer: float(value) for layer, value in df['avg_drop'].to_dict().items()},
            'ranking': ranking,
            'meta': {
                'config': args.config,
                'checkpoint': args.resume_path,
                'num_samples': args.num_samples,
            },
        }
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {summary_path}")

if __name__ == '__main__':
    main()

