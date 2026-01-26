#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
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
from util.dist import init_dist_nccl_backend, is_master, setup_print

def evaluate_model(model, dataloader, device, limit=None):
    model.eval()
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if limit and i >= limit:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()
    
    # Aggregate results from all GPUs
    if dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        
    return 100. * correct.item() / total.item() if total.item() > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description='Fast R20 BER Sweep (DDP)')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--seeds', type=str, default="200,201,202,203,204", help='Comma-separated list of seeds')
    # parser.add_argument('--ckpts', type=str, nargs='+', default=["/workspace/FATBETA/training/texp/r56_w_res_ortho_bfat_c10/r56_w_res_ortho_bfat_c10_checkpoint.pth.tar",
    #                                                              "/workspace/FATBETA/training/texp/r56_baseline_bfat_c10/r56_baseline_bfat_c10_checkpoint.pth.tar",
    #                                                              "/workspace/FATBETA/training/texp/r56_w_res_direction_bfat_c10/r56_w_res_direction_bfat_c10_checkpoint.pth.tar"])
    # parser.add_argument('--ckpts', type=str, nargs='+', default=["/workspace/FATBETA/training/texp/deit_res_golden_ortho_checkpoint.pth.tar"])
    parser.add_argument('--ckpts', type=str, nargs='+', default=["/workspace/FATBETA/training/texp/r56_bucket_c10/r56_bucket_c10_checkpoint.pth.tar"])
    parser.add_argument('--config', type=str, default="configs/training/r20.yaml")
    parser.add_argument('--bers', type=str, default="0.0,1e-6,1e-5,1e-4,1e-3,2e-3,4e-3,8e-3,1e-2,1.5e-2,2e-2,2.5e-2,3e-2,3.5e-2,4e-2,4.5e-2,5e-2,5.5e-2,6e-2,6.5e-2,7e-2,7.5e-2,8e-2,8.5e-2,9e-2,9.5e-2,1e-1,2e-1,4e-1,5e-1")
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of batches per GPU')
    parser.add_argument('--log', type=str, default="results200_g.log", help='Path to log file')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # 0. Initialize DDP
    init_dist_nccl_backend(args)
    device = torch.device(f'cuda:{args.local_rank}')
    setup_print(is_master())

    bers = [float(x) for x in args.bers.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    torch.manual_seed(seeds[0])
    
    # 1. Load Config & Model
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv

    config.dataloader.workers = 4
    config.local_rank = args.local_rank
    
    print(f"Dataset: {config.dataloader.dataset}, Workers: {config.dataloader.workers}, Rank: {args.rank}")
    
    model = create_model(config.arch, dataset=config.dataloader.dataset)
    
    # 2. Quantization Setup
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    model = model.to(device)
    
    # 3. DDP Wrapping
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # 4. Data Loader
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # Header logic
    C_RESET, C_BOLD, C_TITLE = "\033[0m", "\033[1m", "\033[36m"
    C_COL1, C_COL2, C_COL3 = "\033[32m", "\033[33m", "\033[35m"
    ema_status = "Standard" if args.no_ema else "EMA"
    seed_str = f"Seeds: {args.seeds}"
    bits = args.bits

    # Initialize Injector once
    eval_model = model.module if args.distributed else model
    injector = FaultInjector(eval_model, mode='ber', ber=bers[0], device=device, seed=seeds[0])
    
    # Prepare Log File
    log_f = open(args.log, 'w') if args.log and is_master() else None

    try:
        for ckpt_path in args.ckpts:
            print(f"\n{C_BOLD}Processing Checkpoint: {ckpt_path}{C_RESET}")
            if log_f:
                log_f.write(f"Checkpoint: {ckpt_path}\n")
                log_f.write(f"{'BER':<10} | {'All Bits':<15} | {'Skip MSB':<15} | {'Only MSB':<15}\n")
                log_f.write("-" * 65 + "\n")

            # 5. Warm-up (Populate output_size etc.)
            print("Warm-up forward pass...")
            model.eval()
            with torch.no_grad():
                inputs, _ = next(iter(test_loader))
                model(inputs.to(device))

            # 6. Set Bit-width
            q_layers, _ = get_quantized_layers(eval_model)
            set_bit_width(eval_model, [bits]*len(q_layers), [bits]*len(q_layers))
            for m in eval_model.modules():
                if isinstance(m, (QuanConv2d, QuanLinear)):
                    m.bits = (bits, bits)
                    if hasattr(m, 'quan_w_fn') and m.quan_w_fn: m.quan_w_fn.bits = bits
            switch_bit_width(eval_model, quan_scheduler=config.quan, wbit=bits, abits=bits)

            # 7. Load Checkpoint
            print(f"Loading checkpoint: {ckpt_path}")
            load_checkpoint(eval_model, ckpt_path, model_device=device, use_ema=not args.no_ema)

            # 8. Lock Quantizers
            for m in eval_model.modules():
                if hasattr(m, 'quan_w_fn') and m.quan_w_fn and hasattr(m.quan_w_fn, 'init_state'):
                    m.quan_w_fn.init_state.fill_(1)
                if hasattr(m, 'quan_a_fn') and m.quan_a_fn and hasattr(m.quan_a_fn, 'init_state'):
                    m.quan_a_fn.init_state.fill_(1)

            # Final warm-up/calibration
            with torch.no_grad():
                model(inputs.to(device))

            print("Checkpoint loaded, calibrated, and quantizers locked.")

            injector.enable()
            print(f"{C_BOLD}{C_TITLE}Fast {config.arch.upper()} W{bits}A{bits} {ema_status} BER Sweep ({seed_str}){C_RESET}")
            if args.limit: print(f"Using partial test set: first {args.limit} batches per GPU")
            print(f"{'BER':<10} | {C_COL1}{'All Bits':<15}{C_RESET} | {C_COL2}{'Skip MSB':<15}{C_RESET} | {C_COL3}{'Only MSB':<15}{C_RESET}")
            print("-" * 65)

            # Sweep Loop
            for ber in bers:
                mode_data = [[], [], []]
                injector.ber = ber
                
                for seed in seeds:
                    injector.seed = seed
                    for i, (skip_msb, only_msb) in enumerate([(False, False), (True, False), (False, True)]):
                        injector.skip_msb = skip_msb
                        injector.only_msb = only_msb
                        acc = evaluate_model(model, test_loader, device, limit=args.limit)
                        mode_data[i].append(acc)
                
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
                if log_f and is_master():
                    log_f.write(f"{ber:<10} | {plain_results[0]:<15} | {plain_results[1]:<15} | {plain_results[2]:<15}\n")
                    log_f.flush()

            if log_f and is_master():
                log_f.write("\n")
            injector.disable()
            
    finally:
        if log_f:
            log_f.close()
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
