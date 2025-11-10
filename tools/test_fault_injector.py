#!/usr/bin/env python
"""
Test script for Fault Injector tool

This script tests the fault injection functionality on a quantized model.
Usage:
    python tools/test_fault_injector.py --config configs/eval/eval_resnet18_cifar10_single_gpu.yaml --ber 1e-6
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from main import main as main_func
from util.fault_injector import FaultInjector
from util.dist import is_master
import logging

def test_fault_injector():
    """Test fault injector with a simple model forward pass."""
    parser = argparse.ArgumentParser(description='Test Fault Injector')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to evaluation config YAML file')
    parser.add_argument('--bit_width_config', type=str, default=None,
                        help='Path to bit-width configuration JSON file (for mixed-precision models)')
    parser.add_argument('--ber', type=float, default=1e-6,
                        help='Bit-error-rate for fault injection')
    parser.add_argument('--enable_in_training', action='store_true', default=False,
                        help='Enable fault injection during training mode')
    parser.add_argument('--enable_in_inference', action='store_true', default=True,
                        help='Enable fault injection during inference mode (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Import config loader
    from util.config import get_config
    from model import create_model
    from util.data_loader import init_dataloader
    from util.checkpoint import load_checkpoint
    from util.dist import init_logger
    
    # Load config
    configs = get_config(args.config)
    
    # Initialize logger
    script_dir = Path.cwd()
    output_dir = script_dir / configs.output_dir
    log_dir = init_logger(configs.name, output_dir)
    logger = logging.getLogger()
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs.device = device
    
    # Create model
    dataset = getattr(configs.dataloader, 'dataset', 'cifar10')
    num_classes = getattr(configs.dataloader, 'num_classes', 10)
    model = create_model(configs.arch, dataset=dataset, pre_trained=configs.pre_trained)
    model = model.to(device)
    
    # Load checkpoint if specified
    if hasattr(configs, 'resume') and hasattr(configs.resume, 'path') and configs.resume.path:
        checkpoint_path = configs.resume.path
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            model = load_checkpoint(model, checkpoint_path, device)
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    # Setup quantization (if needed)
    # This is typically done in main.py, but for testing we'll do it here
    # For a complete test, you might want to load a fully quantized and trained model
    
    # Load bit-width configuration if specified (for mixed-precision models)
    if args.bit_width_config:
        if not os.path.exists(args.bit_width_config):
            logger.error(f"Bit-width config file not found: {args.bit_width_config}")
            return
        
        logger.info(f"Loading bit-width configuration from: {args.bit_width_config}")
        from util.fault_injector import setup_model_with_bit_width_config
        try:
            weight_bits, act_bits = setup_model_with_bit_width_config(
                model,
                args.bit_width_config,
                config_index=0,
                verbose=True
            )
            logger.info(f"✓ Bit-width configuration loaded and set on model")
        except Exception as e:
            logger.error(f"Failed to load bit-width configuration: {e}")
            return
    else:
        logger.info("No bit-width config specified. Assuming model already has bit-widths configured.")
    
    # Create fault injector
    injector = FaultInjector(
        model=model,
        mode="ber",
        ber=args.ber,
        device=device,
        enable_in_training=args.enable_in_training,
        enable_in_inference=args.enable_in_inference,
        seed=args.seed,
    )
    
    logger.info(f"Fault Injector initialized:")
    logger.info(f"  Mode: {injector.mode}")
    logger.info(f"  BER: {injector.ber}")
    logger.info(f"  Enable in training: {injector.enable_in_training}")
    logger.info(f"  Enable in inference: {injector.enable_in_inference}")
    
    # Test: Create dummy input and run forward pass
    logger.info("\n=== Testing Fault Injector ===")
    
    # Create dummy input based on dataset
    if dataset in ['cifar10', 'cifar100']:
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
    else:  # ImageNet
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Test without fault injection
    model.eval()
    with torch.no_grad():
        output_no_fault = model(dummy_input)
        logger.info(f"Output shape (no fault): {output_no_fault.shape}")
        logger.info(f"Output sample (no fault): {output_no_fault[0, :5]}")
    
    # Test with fault injection
    injector.enable()
    logger.info("\nFault injection enabled")
    
    model.eval()
    with torch.no_grad():
        output_with_fault = model(dummy_input)
        logger.info(f"Output shape (with fault): {output_with_fault.shape}")
        logger.info(f"Output sample (with fault): {output_with_fault[0, :5]}")
    
    # Check if outputs differ
    diff = torch.abs(output_no_fault - output_with_fault)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    logger.info(f"\nOutput difference:")
    logger.info(f"  Max difference: {max_diff:.6f}")
    logger.info(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff > 0:
        logger.info("✓ Fault injection is working (outputs differ)")
    else:
        logger.warning("⚠ Fault injection may not be working (outputs are identical)")
        logger.warning("  This could mean:")
        logger.warning("    1. BER is too low to cause flips")
        logger.warning("    2. Model weights are not quantized")
        logger.warning("    3. Model is using FP32 weights")
    
    # Disable fault injection
    injector.disable()
    logger.info("\nFault injection disabled")
    
    # Verify restoration
    model.eval()
    with torch.no_grad():
        output_restored = model(dummy_input)
        diff_restored = torch.abs(output_no_fault - output_restored)
        max_diff_restored = diff_restored.max().item()
        
        if max_diff_restored < 1e-6:
            logger.info("✓ Model restored correctly (outputs match)")
        else:
            logger.error("✗ Model restoration failed (outputs differ)")
    
    logger.info("\n=== Test Complete ===")


if __name__ == "__main__":
    test_fault_injector()

