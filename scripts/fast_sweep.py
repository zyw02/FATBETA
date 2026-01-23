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
from util.qat import set_bit_width, get_quantized_layers
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
    parser = argparse.ArgumentParser(description='Fast R20 BER Sweep (In-process)')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--seeds', type=str, default="42,52,62,72,82", help='Comma-separated list of seeds (e.g., 62,63,64)')
    parser.add_argument('--ckpts', type=str, nargs='+', default=["/workspace/FATBETA/training/texp/r56_safety_zone_p004_bfat_c10/r56_safety_zone_p004_bfat_c10_checkpoint.pth.tar","/workspace/FATBETA/training/texp/r56_boundary_zone_p02_bfat_c10/r56_boundary_zone_p02_bfat_c10_checkpoint.pth.tar",
                                                                 "/workspace/FATBETA/training/texp/r56_collapse_zone_p025_bfat_c10/r56_collapse_zone_p025_bfat_c10_checkpoint.pth.tar"], help='List of checkpoint paths')
    parser.add_argument('--config', type=str, default="configs/training/r20.yaml")
    parser.add_argument('--bers', type=str, default="0.0,1e-6,1e-5,1e-4,1e-3,2e-3,4e-3,8e-3,1e-2,1.5e-2,2e-2,2.5e-2,3e-2,3.5e-2,4e-2,4.5e-2,5e-2,5.5e-2,6e-2,6.5e-2,7e-2,7.5e-2,8e-2,8.5e-2,9e-2,9.5e-2,1e-1,2e-1,4e-1,5e-1")
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of batches for evaluation')
    parser.add_argument('--log', type=str, default="results.log", help='Path to log file')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bers = [float(x) for x in args.bers.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    torch.manual_seed(seeds[0])
    
    # 1. Load Config & Model
    # We need to spoof sys.argv because get_config() internally parses it
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv

    # Force reduce workers to prevent hang
    config.dataloader.workers = 4
    print(f"Dataset: {config.dataloader.dataset}, Workers: {config.dataloader.workers}")
    model = create_model(config.arch, dataset=config.dataloader.dataset)
    model = model.to(device)
    
    # 2. Quantization Setup
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 4. Data Loader
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # Header logic
    C_RESET, C_BOLD, C_TITLE = "\033[0m", "\033[1m", "\033[36m"
    C_COL1, C_COL2, C_COL3 = "\033[32m", "\033[33m", "\033[35m"
    ema_status = "Standard" if args.no_ema else "EMA"
    seed_str = f"Seeds: {args.seeds}"
    bits = args.bits

    # Initialize Injector once
    injector = FaultInjector(model, mode='ber', ber=bers[0], device=device, seed=seeds[0])
    
    # Prepare Log File
    log_f = open(args.log, 'w') if args.log else None

    try:
        for ckpt_path in args.ckpts:
            print(f"\n{C_BOLD}Processing Checkpoint: {ckpt_path}{C_RESET}")
            if log_f:
                log_f.write(f"Checkpoint: {ckpt_path}\n")
                log_f.write(f"{'BER':<10} | {'All Bits':<15} | {'Skip MSB':<15} | {'Only MSB':<15}\n")
                log_f.write("-" * 65 + "\n")

            # 3. Load Checkpoint
            if not os.path.exists(ckpt_path):
                print(f"Checkout not found: {ckpt_path}, skipping...")
                continue
            load_checkpoint(model, ckpt_path, model_device=device, use_ema=not args.no_ema)
            
            # 5. Warm-up (Populate output_size etc.)
            print("Warm-up forward pass...")
            model.eval()
            with torch.no_grad():
                inputs, _ = next(iter(test_loader))
                model(inputs.to(device))

            # 6. Set Bit-width
            q_layers, _ = get_quantized_layers(model)
            set_bit_width(model, [bits]*len(q_layers), [bits]*len(q_layers))
            for m in model.modules():
                if isinstance(m, (QuanConv2d, QuanLinear)):
                    m.bits = (bits, bits)
                    if hasattr(m, 'quan_w_fn') and m.quan_w_fn: m.quan_w_fn.bits = bits
            switch_bit_width(model, quan_scheduler=config.quan, wbit=bits, abits=bits)

            # Final warm-up after bit-width switch
            with torch.no_grad():
                model(inputs.to(device))

            injector.enable()
            print(f"{C_BOLD}{C_TITLE}Fast {config.arch.upper()} W{bits}A{bits} {ema_status} BER Sweep ({seed_str}){C_RESET}")
            if args.limit: print(f"Using partial test set: first {args.limit} batches")
            print(f"{'BER':<10} | {C_COL1}{'All Bits':<15}{C_RESET} | {C_COL2}{'Skip MSB':<15}{C_RESET} | {C_COL3}{'Only MSB':<15}{C_RESET}")
            print("-" * 65)

            # Sweep Loop
            for ber in bers:
                mode_data = [[], [], []]
                injector.ber = ber  # Update BER dynamically
                
                for seed in seeds:
                    injector.seed = seed
                    # Modes: (skip_msb, only_msb)
                    for i, (skip_msb, only_msb) in enumerate([(False, False), (True, False), (False, True)]):
                        injector.skip_msb = skip_msb
                        injector.only_msb = only_msb
                        acc = evaluate_model(model, test_loader, device, limit=args.limit)
                        mode_data[i].append(acc)
                
                # Calculate mean and std for each mode
                row_results = []
                plain_results = []
                for data in mode_data:
                    mean = sum(data) / len(data)
                    std = 0.0
                    if len(data) > 1:
                        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
                        std = variance ** 0.5
                    row_results.append(f"{mean:6.2f}±{std:<5.2f}")
                    plain_results.append(f"{mean:6.2f}±{std:<5.2f}")
                
                print(f"{ber:<10} | {C_COL1}{row_results[0]:<15}{C_RESET} | {C_COL2}{row_results[1]:<15}{C_RESET} | {C_COL3}{row_results[2]:<15}{C_RESET}")
                if log_f:
                    log_f.write(f"{ber:<10} | {plain_results[0]:<15} | {plain_results[1]:<15} | {plain_results[2]:<15}\n")
                    log_f.flush()

            if log_f:
                log_f.write("\n")
            injector.disable()
            
    finally:
        if log_f:
            log_f.close()

if __name__ == '__main__':
    main()

