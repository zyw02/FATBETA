#!/usr/bin/env python3
"""
General tool for testing SEU fault tolerance.
Supports any model architecture registered in model.py.
Supports loading standard or EMA weights.
"""

import argparse
import sys
from pathlib import Path
import os

import torch
import random
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy on the full test/val set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='General SEU fault tolerance test')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA weights from checkpoint')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--ber', type=float, nargs='+', default=[1e-1], help='Bit error rate(s)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exclude_layers', type=str, nargs='+', default=None, help='Layers to exclude from fault injection')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip evaluation without faults')
    parser.add_argument('--skip_fault', action='store_true', help='Skip fault injection evaluation')
    parser.add_argument('--skip_msb', action='store_true', help='Skip MSB injection')
    parser.add_argument('--only_msb', action='store_true', help='Only inject faults on MSB')
    parser.add_argument('--dynamic_bits', type=int, default=None, help='Force bit-width for dynamic layers')
    
    args = parser.parse_args()
    
    # Load config
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Step 1: Create Model
    print(f"Creating model: {config.arch} for {config.dataloader.dataset}")
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # Step 2: Apply Quantization
    print("Applying quantization wrappers...")
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # Step 3: Load Checkpoint
    print(f"Loading checkpoint: {args.ckpt} (use_ema={args.use_ema})")
    load_checkpoint(model, args.ckpt, model_device=device, use_ema=args.use_ema)
    
    # Step 4: Data Loader
    print("Initializing dataloader...")
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # Step 5: Initialize Model (output_size etc.)
    print("Warm-up forward pass...")
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(test_loader))
        _ = model(inputs.to(device))
    
    # Step 6: Bit-width Configuration
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    else:
        # Auto-configure based on target_bits
        if args.dynamic_bits is not None:
            max_target_bit = args.dynamic_bits
        else:
            target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
            max_target_bit = max(target_bits) if target_bits else 6
        
        print(f"Setting bit-width to {max_target_bit} for all dynamic layers...")
        from quan.func import QuanConv2d, QuanLinear
        from util.qat import set_bit_width
        from util.mpq import switch_bit_width
        
        # Collect dynamic layers
        dynamic_layers_names = []
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    continue
                if hasattr(config.quan, 'excepts') and name in config.quan.excepts:
                    continue
                dynamic_layers_names.append(name)
        
        # Set bit-width
        w_bits_list = [max_target_bit] * len(dynamic_layers_names)
        a_bits_list = [max_target_bit] * len(dynamic_layers_names)
        set_bit_width(model, w_bits_list, a_bits_list)
        
        # Force bit-width in modules for FaultInjector
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    continue
                module.bits = (max_target_bit, max_target_bit)
                if hasattr(module, 'quan_w_fn') and module.quan_w_fn is not None:
                    if hasattr(module.quan_w_fn, 'bits'):
                        module.quan_w_fn.bits = max_target_bit
        
        switch_bit_width(model, quan_scheduler=config.quan, wbit=max_target_bit, abits=max_target_bit)
        
        # Final warm-up after bit-width switch
        with torch.no_grad():
            _ = model(inputs.to(device))
    
    # Step 7: Evaluation
    if not args.skip_baseline:
        print("\nTest 1: Baseline (No Fault Injection)")
        acc_base = evaluate_model(model, test_loader, device)
        print(f"故障注入前准确率: {acc_base:.2f}%")
    
    if not args.skip_fault:
        for ber in args.ber:
            print(f"\nTest 2: Fault Injection (BER={ber}, SkipMSB={args.skip_msb}, OnlyMSB={args.only_msb})")
            injector = FaultInjector(
                model=model,
                mode='ber',
                ber=ber,
                device=device,
                enable_in_inference=True,
                seed=args.seed,
                enable_statistics=True,
                exclude_layers=args.exclude_layers,
                skip_msb=args.skip_msb,
                only_msb=args.only_msb
            )
            injector.enable()
            acc_fault = evaluate_model(model, test_loader, device)
            injector.disable()
            print(f"故障注入后准确率: {acc_fault:.2f}%")
            
            if injector.enable_statistics:
                stats = injector.get_flip_statistics()
                if stats:
                    total_flipped = sum(s['flipped_bits'] for s in stats.values())
                    total_bits = sum(s['total_bits'] for s in stats.values())
                    print(f"Bit Flip Stats: {total_flipped}/{total_bits} ({100.*total_flipped/max(1, total_bits):.4f}%)")

if __name__ == '__main__':
    main()

