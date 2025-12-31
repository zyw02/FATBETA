"""
Evaluation script for Gradient Statistics-based Error Restorer

Tests the gradient statistics restorer on stage1 trained AlexNet with weight bit-flips.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from model.model import create_model
from model.alexnet_cifar import alexnet_cifar
from util.data_loader import init_dataloader
from util.config import get_config
from util.fault_injector import FaultInjector
from util.gradient_statistics_restorer import create_gradient_statistics_restorer
from util.sensitive_layer_restorer import create_sensitive_layer_restorer
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.utils import set_global_seed
from util.mpq import switch_bit_width, switch_bit_width_bn


def evaluate_with_restorer(
    model: nn.Module,
    test_loader: DataLoader,
    restorer,
    device: torch.device,
    ber_values: list,
    seed: int = 42,
    skip_first_last: bool = False,
) -> dict:
    """
    Evaluate model with gradient statistics restorer at different BER values
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        restorer: Gradient statistics restorer
        device: Device to use
        ber_values: List of BER values to test
        seed: Random seed for fault injection (default: 42)
    
    Returns:
        results: Dict with accuracy results for each BER
    """
    results = {}
    
    for ber in ber_values:
        print(f"\n{'='*60}")
        print(f"Testing BER = {ber:.2e}")
        print(f"{'='*60}")
        
        # Create a new FaultInjector for each BER test (same as eval_with_fault_injection.py)
        # This ensures consistent behavior: base_seed + hash(layer_name) for each layer
        fault_injector = FaultInjector(
            model=model,
            mode="ber",
            ber=ber,
            device=device,
            enable_in_training=False,
            enable_in_inference=True,
            use_random_flip_in_training=False,
            skip_first_last=skip_first_last,  # Control whether to skip first and last layers
            seed=seed,  # Use fixed seed for reproducibility (same as eval_with_fault_injection.py)
            seed_list=None,  # Don't use seed_list to ensure explicit seed is used
        )
        
        correct = 0
        total = 0
        
        model.eval()
        if hasattr(restorer, 'set_operating_point'):
            try:
                restorer.set_operating_point(ber)
            except Exception:
                pass
        elif hasattr(restorer, 'set_ber'):
            try:
                restorer.set_ber(ber)
            except Exception:
                pass
        fault_injector.enable()
        restorer.enable()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass with fault injection and restoration
                outputs = model(inputs)
                
                # Get predictions
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if (batch_idx + 1) % 50 == 0:
                    acc = 100. * correct / total
                    print(f"  Batch {batch_idx+1}/{len(test_loader)}, Acc: {acc:.2f}%")
        
        fault_injector.disable()
        restorer.disable()
        
        accuracy = 100. * correct / total
        results[ber] = accuracy
        print(f"\nBER {ber:.2e}: Accuracy = {accuracy:.2f}%")
        
        # Print flip statistics for this BER
        print(f"\n故障注入统计 (BER={ber:.2e}):")
        fault_injector.print_flip_statistics(verbose=True)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gradient Statistics Restorer')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                        help='Path to stage1 checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--k', type=float, default=4.0,
                        help='Chebyshev parameter k (default: 4.0)')
    parser.add_argument('--kernel_size', type=int, default=3, choices=[3, 9],
                        help='Kernel size for gradient refinement (3 or 9, default: 3)')
    parser.add_argument('--num_statistics_batches', type=int, default=50,
                        help='Number of batches for statistics collection (default: 50)')
    parser.add_argument('--ber_values', type=float, nargs='+',
                        default=[1e-5, 1e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
                        help='BER values to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--bit_width_config', type=str, default=None,
                        help='Path to bit-width configuration JSON file (optional)')
    parser.add_argument('--config_index', type=int, default=0,
                        help='Configuration index to use from JSON file (default: 0)')
    parser.add_argument('--force_w2a2', action='store_true', default=False,
                        help='Force all non-fixed_bits layers to use 2-bit weights and 2-bit activations')
    parser.add_argument('--force_w2a6', action='store_true', default=False,
                        help='Force all non-fixed_bits layers to use 2-bit weights and 6-bit activations')
    parser.add_argument('--force_w6a6', action='store_true', default=False,
                        help='Force all non-fixed_bits layers to use 6-bit weights and 6-bit activations')
    parser.add_argument('--skip_first_last', action='store_true', default=False,
                        help='Skip fault injection on first and last layers (default: False, inject in all layers)')
    parser.add_argument('--restorer_mode', type=str, default='sensitive', choices=['sensitive', 'gradient'],
                        help='Restorer implementation to use')
    parser.add_argument('--layer_profile', type=str, default=None,
                        help='Path to layer profile (required for sensitive mode)')
    parser.add_argument('--fault_layer_profile', type=str, default=None,
                        help='Optional fault profile for BER-aware blending')
    parser.add_argument('--fault_profile_ber', type=float, default=1e-1,
                        help='BER corresponding to the fault profile')
    parser.add_argument('--sensitive_layers', type=str, default=None,
                        help='Comma-separated sensitive layer names (optional)')
    parser.add_argument('--sensitive_z_thresh', type=float, default=3.0,
                        help='Z-score threshold for sensitive restorer')
    parser.add_argument('--sensitive_std_ratio_bounds', type=float, nargs=2, default=[0.5, 2.0],
                        help='Lower and upper bounds for std ratio (sensitive restorer)')
    parser.add_argument('--sensitive_clip_margin', type=float, default=1.25,
                        help='Clip margin for sensitive restorer')
    parser.add_argument('--ber_policy', type=str, default=None,
                        help='JSON file specifying BER-specific restorer policies')
    parser.add_argument('--repair_mode', type=str, default='rule',
                        choices=['rule', 'mlp', 'mlp_local', 'mlp_poly', 'ms_residual', 'lightweight_denoiser', 'denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor'],
                        help='Repair head implementation for sensitive restorer')
    parser.add_argument('--repair_head_ckpt', type=str, default=None,
                        help='Path to pretrained learning repair head checkpoint (MLP mode)')
    parser.add_argument('--mlp_hidden_dim', type=int, default=64,
                        help='Hidden size for MLP repair head')
    
    args = parser.parse_args()
    
    set_global_seed(seed=args.seed)
    device = torch.device(args.device)
    
    # Load config - get_config expects config file as positional arg
    import sys
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], args.config]
    try:
        configs = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    # Create model
    print("Creating model...")
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.stage1_ckpt}...")
    load_checkpoint(model, args.stage1_ckpt)
    print("Model loaded successfully.")
    
    # Load and set bit-width configuration
    # If force_w2a2 is True, set all non-fixed_bits layers to 2-bit
    # If force_w2a6 is True, set all non-fixed_bits layers to w2a6 (2-bit weights, 6-bit activations)
    # If JSON config is provided, use it; otherwise, use max target bit width
    if args.force_w2a2:
        print("Setting all non-fixed_bits layers to 2-bit weights and 2-bit activations (w2a2)...")
        # Set all non-fixed_bits layers to 2-bit
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if name not in getattr(configs.quan, 'excepts', {}):
                    if hasattr(module, 'bits'):
                        module.bits = (2, 2)
        # Also update BN layers
        switch_bit_width_bn(model, 2, 2)
        print("✓ All non-fixed_bits layers set to w2a2")
        print(f"Note: Layers in excepts (features.0, classifier.6) will remain 8-bit (fixed_bits)")
    elif args.force_w2a6:
        print("Setting all non-fixed_bits layers to 2-bit weights and 6-bit activations (w2a6)...")
        # Set all non-fixed_bits layers to w2a6
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if name not in getattr(configs.quan, 'excepts', {}):
                    if hasattr(module, 'bits'):
                        module.bits = (2, 6)
        # Also update BN layers (use 6-bit for activations)
        switch_bit_width_bn(model, 2, 6)
        print("✓ All non-fixed_bits layers set to w2a6")
        print(f"Note: Layers in excepts (features.0, classifier.6) will remain 8-bit (fixed_bits)")
    elif args.force_w6a6:
        print("Setting all non-fixed_bits layers to 6-bit weights and 6-bit activations (w6a6)...")
        # Set all non-fixed_bits layers to w6a6
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if name not in getattr(configs.quan, 'excepts', {}):
                    if hasattr(module, 'bits'):
                        module.bits = (6, 6)
        # Also update BN layers (use 6-bit for activations)
        switch_bit_width_bn(model, 6, 6)
        print("✓ All non-fixed_bits layers set to w6a6")
        print(f"Note: Layers in excepts (features.0, classifier.6) will remain 8-bit (fixed_bits)")
    elif args.bit_width_config and args.bit_width_config.strip():
        print(f"Loading bit-width configuration from: {args.bit_width_config}")
        try:
            from util.fault_injector import setup_model_with_bit_width_config
            weight_bits, act_bits = setup_model_with_bit_width_config(
                model,
                args.bit_width_config,
                config_index=args.config_index,
                verbose=True
            )
            print(f"✓ Bit-width configuration loaded: {len(weight_bits)} layers")
            print(f"Note: Layers in excepts (features.0, classifier.6) will remain 8-bit (fixed_bits)")
        except Exception as e:
            print(f"Failed to load bit-width configuration: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Switch to maximum target bit width (6-bit)
        # Note: first_layer (features.0) and last_layer (classifier.6) in excepts will remain 8-bit
        target_bits = configs.target_bits if hasattr(configs, 'target_bits') else [6, 5, 4, 3, 2]
        max_bit = max(target_bits)
        print(f"Switching model to {max_bit}-bit quantization (max target-bit)...")
        print(f"Note: Layers in excepts (features.0, classifier.6) will remain 8-bit")
        
        switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
        print(f"Model switched to {max_bit}-bit successfully.")
    
    # Verify excepts layers still have 8-bit (for both JSON config and max bit cases)
    excepts_layers = []
    if hasattr(configs.quan, 'excepts'):
        for name, module in model.named_modules():
            if name in configs.quan.excepts:
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    excepts_layers.append((name, module.fixed_bits))
                elif hasattr(module, 'bits') and module.bits is not None:
                    excepts_layers.append((name, module.bits))
    
    if excepts_layers:
        print("Verifying excepts layers (should remain 8-bit):")
        for name, expected_bits in excepts_layers:
            module = dict(model.named_modules())[name]
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                current_bits = module.fixed_bits
                print(f"  {name}: fixed_bits={current_bits} (should be 8-bit)")
            elif hasattr(module, 'bits') and module.bits is not None:
                current_bits = module.bits
                print(f"  {name}: bits={current_bits} (should be 8-bit)")
                if current_bits != expected_bits:
                    print(f"    WARNING: {name} bit width changed! This should not happen.")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, configs.arch)
    
    # Initialize model output_size with a dummy forward pass (CRITICAL for quantized models)
    # This must be done AFTER setting bit width, so quantization uses correct bit width
    print("Initializing model output_size with a dummy forward pass...")
    model.eval()
    with torch.no_grad():
        # Determine input size based on dataset
        dataset = configs.dataloader.dataset
        input_size = 32 if dataset in ['cifar10', 'cifar100'] else 224
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        try:
            _ = model(dummy_input)
            print("✓ Model output_size initialized")
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    def parse_layer_list(raw: Optional[str]) -> Optional[List[str]]:
        if not raw:
            return None
        items = [tok.strip() for tok in raw.split(',')]
        return [i for i in items if i]

    target_layers = parse_layer_list(args.sensitive_layers)
    ber_policy = None
    if args.ber_policy:
        with open(args.ber_policy, 'r') as f:
            ber_policy = json.load(f)

    if args.restorer_mode == 'gradient':
        print(f"Creating gradient statistics restorer (k={args.k}, kernel_size={args.kernel_size})...")
        print(f"Collecting statistics from {args.num_statistics_batches} batches...")
        restorer = create_gradient_statistics_restorer(
            model=model,
            data_loader=test_loader,
            k=args.k,
            kernel_size=args.kernel_size,
            num_statistics_batches=args.num_statistics_batches,
            layer_names=target_layers,
        )
        print("Restorer created successfully.")
    else:
        if not args.layer_profile:
            raise ValueError("Sensitive restorer requires --layer_profile")
        std_bounds = tuple(args.sensitive_std_ratio_bounds)
        print(f"Creating sensitive-layer restorer from {args.layer_profile}")
        restorer = create_sensitive_layer_restorer(
            model=model,
            profile_path=args.layer_profile,
            target_layers=target_layers,
            z_thresh=args.sensitive_z_thresh,
            std_ratio_bounds=std_bounds,
            clip_margin=args.sensitive_clip_margin,
            repair_mode=args.repair_mode,
            fault_profile_path=args.fault_layer_profile,
            fault_profile_ber=args.fault_profile_ber,
            ber_policy=ber_policy,
            repair_head_ckpt=args.repair_head_ckpt,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )
        print("Sensitive-layer restorer ready.")
    
    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    results = evaluate_with_restorer(
        model=model,
        test_loader=test_loader,
        restorer=restorer,
        device=device,
        ber_values=args.ber_values,
        seed=args.seed,
        skip_first_last=args.skip_first_last,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"{'BER':<15} {'Accuracy (%)':<15}")
    print("-" * 30)
    for ber in sorted(results.keys()):
        print(f"{ber:<15.2e} {results[ber]:<15.2f}")
    print("="*60)
    
    # Compare with baseline (no restorer)
    print("\nEvaluating baseline (no restorer)...")
    baseline_results = {}
    for ber in args.ber_values:
        # Create a new FaultInjector for each BER test (same as eval_with_fault_injection.py)
        fault_injector = FaultInjector(
            model=model,
            mode="ber",
            ber=ber,
            device=device,
            enable_in_training=False,
            enable_in_inference=True,
            use_random_flip_in_training=False,
            skip_first_last=args.skip_first_last,  # Control whether to skip first and last layers
            seed=args.seed,
            seed_list=None,
        )
        fault_injector.enable()
        restorer.disable()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        baseline_results[ber] = 100. * correct / total
        fault_injector.disable()
        
        # Print flip statistics for this BER
        print(f"\n故障注入统计 (BER={ber:.2e}):")
        fault_injector.print_flip_statistics(verbose=True)
    
    print("\n" + "="*60)
    print("Comparison: With Restorer vs Baseline")
    print("="*60)
    print(f"{'BER':<15} {'Baseline (%)':<15} {'Restorer (%)':<15} {'Improvement':<15}")
    print("-" * 60)
    for ber in sorted(results.keys()):
        baseline_acc = baseline_results[ber]
        restorer_acc = results[ber]
        improvement = restorer_acc - baseline_acc
        print(f"{ber:<15.2e} {baseline_acc:<15.2f} {restorer_acc:<15.2f} {improvement:+.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()

