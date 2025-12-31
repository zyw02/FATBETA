#!/usr/bin/env python
"""
Layer-wise Fault Sensitivity Analysis Tool

This script systematically injects faults into one layer at a time to identify
which layers are most sensitive to bit-flips. This helps in manual mixed-precision
tuning (e.g., keeping sensitive layers at higher bit-widths).

Usage:
    python tools/analyze_layer_sensitivity.py \
        --config configs/eval/eval_alexnet_cifar10_sensitive_stage1.yaml \
        --resume_path training/alexnet_cifar10_sensitive_stage1/best.pth.tar \
        --ber 1e-1 \
        --force_w2a2  # Force base configuration to w2a2
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from util.config import get_config, init_logger
from model import create_model
from util.data_loader import init_dataloader
from util.checkpoint import load_checkpoint
from util.dist import logger_info
from util.fault_injector import FaultInjector
from process import validate
from util.monitor import ProgressMonitor
from quan import find_modules_to_quantize, replace_module_by_names
from util.utils import preprocess_model
from util.mpq import switch_bit_width

def main():
    parser = argparse.ArgumentParser(description='Analyze Layer-wise Fault Sensitivity')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit Error Rate for injection')
    parser.add_argument('--force_w2a2', action='store_true', help='Force all layers (except first/last) to w2a2')
    parser.add_argument('--force_w2a6', action='store_true', help='Force all layers (except first/last) to w2a6')
    parser.add_argument('--force_w6a6', action='store_true', help='Force all layers (except first/last) to w6a6')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for quick evaluation (0 for full test set)')
    
    args = parser.parse_args()
    
    # Load config using get_config to properly handle __delete__ markers and template merging
    # We need to temporarily override sys.argv to avoid argparse conflicts
    import sys
    import yaml
    import munch
    
    # Save original argv
    original_argv = sys.argv.copy()
    # Temporarily replace argv with just the config file to let get_config parse it
    sys.argv = ['analyze_layer_sensitivity.py', args.config]
    
    try:
        # Use get_config to properly merge with template.yaml and handle __delete__
        from util.config import get_config
        template_path = Path(__file__).parent.parent / 'template.yaml'
        if template_path.exists():
            configs = get_config(str(template_path))
        else:
            # Fallback: load directly if template doesn't exist
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            configs = munch.munchify(cfg)
    finally:
        # Restore original argv
        sys.argv = original_argv

    # Set some defaults that might be missing
    if not hasattr(configs, 'local_rank'):
        configs.local_rank = 0
    if not hasattr(configs, 'enable_dynamic_bit_training'):
        configs.enable_dynamic_bit_training = True
    if not hasattr(configs, 'split_aw_cands'):
        configs.split_aw_cands = False
    if not hasattr(configs, 'smoothing'):
        configs.smoothing = 0.0
    if not hasattr(configs, 'world_size'):
        configs.world_size = 1
    if not hasattr(configs, 'rank'):
        configs.rank = 0

    # Setup logger
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs.device = device
    
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Create model
    model = create_model(configs.arch, dataset=getattr(configs.dataloader, 'dataset', 'cifar10'))
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(model, args.resume_path, device, strict=False)
    
    # Set Bit-width
    # Default behavior: use what's in checkpoint or config.
    # But for sensitivity analysis, we usually want to test a specific baseline (e.g., w2a2 or w2a6).
    if args.force_w2a2:
        logger.info("Forcing model to w2a2 (except first/last layers)...")
        switch_bit_width(model, configs.quan, wbit=2, abits=2)
    elif args.force_w2a6:
        logger.info("Forcing model to w2a6 (except first/last layers)...")
        switch_bit_width(model, configs.quan, wbit=2, abits=6)
    elif args.force_w6a6:
        logger.info("Forcing model to w6a6 (except first/last layers)...")
        switch_bit_width(model, configs.quan, wbit=6, abits=6)
    
    # Dummy forward to init
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32).to(device))
    
    # Data Loader
    # Use a subset for speed if requested
    train_loader, val_loader, test_loader, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    
    if args.num_samples > 0:
        indices = torch.randperm(len(test_loader.dataset))[:args.num_samples]
        test_subset = torch.utils.data.Subset(test_loader.dataset, indices)
        eval_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=configs.dataloader.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        logger.info(f"Using subset of {args.num_samples} samples for fast analysis")
    else:
        eval_loader = test_loader
        logger.info("Using full test set")

    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Baseline Acc
    logger.info("Evaluating Baseline Accuracy (Clean)...")
    baseline_acc = validate(eval_loader, model, criterion, -1, ProgressMonitor(logger), configs, eval_predefined_arch=[(32, None, None)])
    baseline_acc = baseline_acc[0] if isinstance(baseline_acc, list) else baseline_acc
    logger.info(f"Baseline Acc: {baseline_acc:.2f}%")
    
    # Identify quantized layers
    quantized_layers = []
    for name, m in model.named_modules():
        if hasattr(m, 'quan_w_fn') or hasattr(m, 'bits') or hasattr(m, 'fixed_bits'):
            # Filter valid quantized layers
            if hasattr(m, 'quan_w_fn') and m.quan_w_fn is not None:
                quantized_layers.append(name)
    
    logger.info(f"Found {len(quantized_layers)} quantized layers to test.")
    
    results = []
    
    print(f"{'Layer':<40} {'Clean':<8} {'Faulted':<8} {'Drop':<8}")
    print("-" * 70)
    
    for layer_name in quantized_layers:
        # Create injector for JUST this layer
        # Note: skip_first_last=False ensures we can test first/last layers if we want
        injector = FaultInjector(
            model, mode="ber", ber=args.ber, device=device,
            enable_in_inference=True, seed=args.seed,
            whitelist_layer=layer_name,  # KEY: Only inject this layer
            skip_first_last=False 
        )
        
        injector.enable()
        
        # Evaluate
        # We use eval_predefined_arch=[(32, None, None)] to prevent validate from changing bits
        acc = validate(eval_loader, model, criterion, -1, ProgressMonitor(logger), configs, eval_predefined_arch=[(32, None, None)])
        acc = acc[0] if isinstance(acc, list) else acc
        
        injector.disable()
        
        drop = baseline_acc - acc
        print(f"{layer_name:<40} {baseline_acc:<8.2f} {acc:<8.2f} {drop:<8.2f}")
        
        results.append({
            'layer': layer_name,
            'clean_acc': baseline_acc,
            'faulted_acc': acc,
            'drop': drop
        })
        
    # Sort by drop (descending)
    results.sort(key=lambda x: x['drop'], reverse=True)
    
    print("\n" + "="*70)
    print("SENSITIVITY RANKING (Most Sensitive First)")
    print("="*70)
    print(f"{'Rank':<5} {'Layer':<40} {'Drop':<10} {'Faulted Acc':<15}")
    print("-" * 70)
    
    for i, res in enumerate(results):
        print(f"{i+1:<5} {res['layer']:<40} {res['drop']:<10.2f} {res['faulted_acc']:<15.2f}")
        
    # Suggestion
    print("\nSuggested Actions:")
    print("Consider increasing bit-width for the top 3-5 layers above.")

if __name__ == '__main__':
    main()

