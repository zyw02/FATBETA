#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector
from util.qat import set_bit_width
from util.mpq import switch_bit_width
from quan.func import QuanConv2d, QuanLinear

def evaluate_model(model, dataloader, device, limit=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if limit and i >= limit:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='Fast MobileNetV2 BER Sweep (In-process)')
    parser.add_argument('--bits', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', type=str, default="/workspace/FATBETA/training/mobilenetv2_cifar_nude_bfat/mobilenetv2_cifar_nude_bfat_checkpoint.pth.tar")
    parser.add_argument('--config', type=str, default="configs/training/train_mobilenetv2_cifar_qat.yaml")
    parser.add_argument('--bers', type=str, default="1e-4,1e-3,2e-3,4e-3,8e-3,1e-2,1.2e-2,1.4e-2,1.8e-2,2e-2")
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of batches for evaluation')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    bers = [float(x) for x in args.bers.split(',')]
    
    # 1. Load Config & Model
    # We need to spoof sys.argv because get_config() internally parses it
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv

    model = create_model(config.arch, dataset=config.dataloader.dataset)
    model = model.to(device)
    
    # 2. Quantization Setup
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 3. Load Checkpoint
    load_checkpoint(model, args.ckpt, model_device=device, use_ema=not args.no_ema)
    
    # 4. Data Loader
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 5. Warm-up (Populate output_size etc.)
    print("Warm-up forward pass...")
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(test_loader))
        model(inputs.to(device))

    # 6. Set Bit-width
    bits = args.bits
    dynamic_layers = [n for n, m in model.named_modules() if isinstance(m, (QuanConv2d, QuanLinear)) and not (hasattr(m, 'fixed_bits') and m.fixed_bits)]
    set_bit_width(model, [bits]*len(dynamic_layers), [bits]*len(dynamic_layers))
    for m in model.modules():
        if isinstance(m, (QuanConv2d, QuanLinear)):
            m.bits = (bits, bits)
            if hasattr(m, 'quan_w_fn') and m.quan_w_fn: m.quan_w_fn.bits = bits
    switch_bit_width(model, quan_scheduler=config.quan, wbit=bits, abits=bits)

    # Final warm-up after bit-width switch
    with torch.no_grad():
        model(inputs.to(device))

    # Initialize Injector once (this will print "Wrapped X layers..." log)
    injector = FaultInjector(model, mode='ber', ber=bers[0], device=device, seed=args.seed)
    injector.enable()

    # Header (Print this AFTER initialization logs to keep table clean)
    C_RESET, C_BOLD, C_TITLE = "\033[0m", "\033[1m", "\033[36m"
    C_COL1, C_COL2, C_COL3 = "\033[32m", "\033[33m", "\033[35m"
    ema_status = "Standard" if args.no_ema else "EMA"
    print(f"\n{C_BOLD}{C_TITLE}Fast MobileNetV2 W{bits}A{bits} {ema_status} BER Sweep{C_RESET}")
    if args.limit: print(f"Using partial test set: first {args.limit} batches")
    print(f"{'BER':<10} | {C_COL1}{'All Bits':<12}{C_RESET} | {C_COL2}{'Skip MSB':<12}{C_RESET} | {C_COL3}{'Only MSB':<12}{C_RESET}")
    print("-" * 55)

    # Sweep Loop
    for ber in bers:
        results = []
        injector.ber = ber  # Update BER dynamically
        
        # Modes: (skip_msb, only_msb)
        for skip_msb, only_msb in [(False, False), (True, False), (False, True)]:
            injector.skip_msb = skip_msb
            injector.only_msb = only_msb
            acc = evaluate_model(model, test_loader, device, limit=args.limit)
            results.append(acc)
        
        print(f"{ber:<10} | {C_COL1}{results[0]:<12.2f}{C_RESET} | {C_COL2}{results[1]:<12.2f}{C_RESET} | {C_COL3}{results[2]:<12.2f}{C_RESET}")
    
    injector.disable()

if __name__ == '__main__':
    main()

