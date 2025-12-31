#!/usr/bin/env python
"""评估敏感通道修复器的SEU容错性"""

import argparse
import sys
import os
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from util.config import get_config
from model import create_model
from util.data_loader import init_dataloader
from util.checkpoint import load_checkpoint
from util.dist import logger_info
from util.fault_injector import FaultInjector
from util.sensitive_features import compute_activation_baseline
from util.sensitive_restorer import SensitiveActivationCollector, SensitiveChannelRestorer
from quan import find_modules_to_quantize, replace_module_by_names
from util.utils import preprocess_model
import logging

def parse_ber_list(ber_str):
    if not ber_str:
        return []
    return [float(x.strip()) for x in ber_str.split(',') if x.strip()]

def main():
    parser = argparse.ArgumentParser(description='Evaluate sensitive restorer with fault injection')
    parser.add_argument('--config', type=str, required=True, help='Path to eval config YAML')
    parser.add_argument('--bit_width_config', type=str, required=True, help='Path to bit-width config JSON')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint with sensitive_restorer')
    parser.add_argument('--sensitive_baseline', type=str, required=True, help='Path to sensitive_baseline.pth file')
    parser.add_argument('--ber_list', type=str, default="0.0,2e-2,3e-2,4e-2,5e-2,8e-2", help='BER values to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials per BER')
    parser.add_argument('--config_index', type=int, default=0, help='Bit-width config index')
    parser.add_argument('--use_default_bits', action='store_true', help='Use default 2-bit instead of loading bit-width config (simulate first evaluation)')
    
    args = parser.parse_args()
    
    # Load config
    import sys as sys_module
    original_argv = sys_module.argv.copy()
    sys_module.argv = ['eval_sensitive_restorer.py', args.config]
    try:
        configs = get_config(args.config)
    finally:
        sys_module.argv = original_argv
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger()
    
    # Create model
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=False)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Initialize output_size by doing a dummy forward pass
    model.eval()
    with torch.no_grad():
        input_size = 32 if configs.dataloader.dataset in ['cifar10', 'cifar100'] else 224
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        try:
            _ = model(dummy_input)
            logger_info(logger, "✓ Model output_size initialized")
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
    
    # Load and set bit-width configuration (CRITICAL for mixed-precision models)
    if args.use_default_bits:
        # Simulate first evaluation: use default 2-bit instead of loading bit-width config
        from util.mpq import switch_bit_width
        default_bit = configs.quan.weight.bit  # Usually 2
        logger_info(logger, f"Using default bit-width: {default_bit}-bit (simulating first evaluation)")
        switch_bit_width(model, quan_scheduler=configs.quan, wbit=default_bit, abits=default_bit)
        logger_info(logger, f"✓ Default bit-width applied: {default_bit}-bit for all dynamic layers")
    else:
        from util.fault_injector import setup_model_with_bit_width_config
        logger_info(logger, f"Loading bit-width configuration from: {args.bit_width_config}")
        try:
            weight_bits, act_bits = setup_model_with_bit_width_config(
                model,
                args.bit_width_config,
                config_index=args.config_index,
                verbose=True
            )
            logger_info(logger, f"✓ Bit-width configuration loaded: {len(weight_bits)} layers")
        except Exception as e:
            logger.error(f"Failed to load bit-width configuration: {e}")
            import traceback
            traceback.print_exc()
            return
    
    model.eval()
    logger_info(logger, "Model loaded and configured")
    
    # Load sensitive baseline stats
    baseline_stats = torch.load(args.sensitive_baseline, map_location=device)
    
    # Prepare stats for device
    prepared_stats = {}
    feature_dim = 0
    for name, entry in baseline_stats.items():
        indices = entry.get("indices", [])
        if not indices:
            continue
        prepared_stats[name] = {
            "indices": indices,
            "mean": entry["mean"].to(device),
            "std": entry["std"].to(device),
        }
        feature_dim += len(indices)
    
    # Load sensitive restorer
    if 'sensitive_restorer' not in checkpoint:
        raise ValueError("Checkpoint does not contain sensitive_restorer!")
    
    restorer = SensitiveChannelRestorer(
        feature_dim,
        configs.dataloader.num_classes,
        hidden_dim=128,
    ).to(device)
    restorer.load_state_dict(checkpoint['sensitive_restorer'])
    restorer.eval()
    logger_info(logger, f"✓ Loaded sensitive restorer ({sum(p.numel() for p in restorer.parameters())} parameters)")
    
    # Create data loaders
    _, _, test_loader, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    
    # Create collector
    collector = SensitiveActivationCollector(model, prepared_stats)
    
    # Parse BER list
    ber_list = parse_ber_list(args.ber_list)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print("\n" + "="*80)
    print("SEU Fault Tolerance Evaluation - Sensitive Channel Restorer")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"BER values: {ber_list}")
    print(f"Trials per BER: {args.num_trials}")
    print("="*80 + "\n")
    
    results = []
    
    for ber in ber_list:
        logger_info(logger, f"\nTesting BER = {ber:.2e}")
        
        trial_accs_model = []
        trial_accs_restored = []
        
        for trial in range(args.num_trials):
            fault_injector = FaultInjector(
                model=model,
                mode="ber",
                ber=ber,
                enable_in_training=False,
                enable_in_inference=True,
                seed=args.seed + trial,
                skip_first_last=False,
            )
            fault_injector.enable()
            
            correct_model = 0
            correct_restored = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Clean forward
                    collector.clear()
                    fault_injector.disable()
                    logits_clean = model(inputs)
                    
                    # Faulted forward
                    collector.clear()
                    fault_injector.enable()
                    logits_faulted = model(inputs)
                    pred_faulted = logits_faulted.argmax(dim=1)
                    correct_model += (pred_faulted == targets).sum().item()
                    
                    # Restored forward
                    features = collector.build_feature_vector(device)
                    if features is not None:
                        logits_restored, gate = restorer(logits_faulted, features)
                        pred_restored = logits_restored.argmax(dim=1)
                        correct_restored += (pred_restored == targets).sum().item()
                    else:
                        correct_restored += (pred_faulted == targets).sum().item()
                    
                    total += inputs.size(0)
            
            fault_injector.disable()
            
            acc_model = 100.0 * correct_model / total
            acc_restored = 100.0 * correct_restored / total
            trial_accs_model.append(acc_model)
            trial_accs_restored.append(acc_restored)
            
            logger_info(logger, f"  Trial {trial+1}: Model={acc_model:.2f}% | Restored={acc_restored:.2f}% | Gain={acc_restored-acc_model:+.2f}%")
        
        avg_model = sum(trial_accs_model) / len(trial_accs_model)
        avg_restored = sum(trial_accs_restored) / len(trial_accs_restored)
        std_model = (sum((x - avg_model)**2 for x in trial_accs_model) / len(trial_accs_model))**0.5 if len(trial_accs_model) > 1 else 0.0
        std_restored = (sum((x - avg_restored)**2 for x in trial_accs_restored) / len(trial_accs_restored))**0.5 if len(trial_accs_restored) > 1 else 0.0
        
        results.append((ber, avg_model, avg_restored, std_model, std_restored))
    
    collector.remove()
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"{'BER':<12} {'Model Acc':<15} {'Restored Acc':<15} {'Gain':<10} {'Std (Model)':<12} {'Std (Restored)':<12}")
    print("-"*80)
    
    baseline_model = results[0][1] if results else 0.0
    
    for ber, avg_model, avg_restored, std_model, std_restored in results:
        gain = avg_restored - avg_model
        print(f"{ber:<12.2e} {avg_model:<15.2f} {avg_restored:<15.2f} {gain:<+10.2f} {std_model:<12.2f} {std_restored:<12.2f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
