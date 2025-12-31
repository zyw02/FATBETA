#!/usr/bin/env python3
"""
测试格雷码编码对故障注入的影响

仅对 features.0 层使用格雷码编码，其他层使用标准二进制编码。
比较使用格雷码前后的模型准确率变化。
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.checkpoint import load_checkpoint
from model.model import create_model
from util.config import get_config
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.data_loader import init_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Test Gray Code fault injection on features.0')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config file')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                       help='Path to stage1 checkpoint')
    parser.add_argument('--bit_width_config', type=str, default=None,
                       help='Path to bit width config JSON (optional)')
    parser.add_argument('--ber', type=float, default=1e-2,
                       help='Bit error rate (default: 1e-2)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of test samples (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    return parser.parse_args()


def evaluate_model(model, test_loader, device, num_samples=None):
    """评估模型准确率（快速版本，只处理指定数量的样本）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if num_samples is not None and total >= num_samples:
                break
            
            # 如果当前批次会超过 num_samples，只取需要的部分
            batch_size = inputs.size(0)
            if num_samples is not None and total + batch_size > num_samples:
                needed = num_samples - total
                inputs = inputs[:needed]
                targets = targets[:needed]
                
            print(f"  Processing batch {batch_idx + 1}, moving to device...", flush=True)
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(f"  Running model forward pass...", flush=True)
            
            try:
                # 添加同步点，确保之前的操作完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"  CUDA synchronized, starting forward...", flush=True)
                
                outputs = model(inputs)
                
                # 添加同步点，确保前向传播完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"  Forward pass completed, computing predictions...", flush=True)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print(f"  Batch {batch_idx + 1} completed: {correct}/{total} correct", flush=True)
            except Exception as e:
                print(f"  ERROR during forward pass: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            if num_samples is not None and total >= num_samples:
                break
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置（需要临时修改 sys.argv，因为 get_config 从 sys.argv 读取）
    print(f"Loading config from {args.config}...")
    original_argv = sys.argv.copy()
    sys.argv = ['test_gray_code_fault_injection.py', args.config]
    try:
        configs = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    # 设置一些默认值（如果缺失）
    if not hasattr(configs, 'local_rank'):
        configs.local_rank = 0
    if not hasattr(configs, 'enable_dynamic_bit_training'):
        configs.enable_dynamic_bit_training = True
    if not hasattr(configs, 'split_aw_cands'):
        configs.split_aw_cands = False
    
    # 创建模型
    print("Creating model...")
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = model.to(device)
    
    # 预处理模型（量化等）
    print("Preprocessing model...")
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    
    # 加载 checkpoint（必须在设置 bit_width_config 之前，因为 checkpoint 可能覆盖 bits）
    print(f"Loading checkpoint from {args.stage1_ckpt}...")
    load_checkpoint(model, args.stage1_ckpt)
    print("Model loaded successfully.")
    
    # 如果提供了位宽配置，加载它（必须在 checkpoint 之后，确保 bits 被正确设置）
    if args.bit_width_config:
        print(f"Loading bit width config from {args.bit_width_config}...")
        setup_model_with_bit_width_config(
            model,
            json_path=args.bit_width_config,
            config_index=0,
            verbose=True
        )
    
    # 初始化数据加载器（使用测试集，不是训练集）
    print("Initializing data loader (using TEST set, not training set)...")
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)
    print(f"Test loader size: {len(test_loader)} batches")
    
    # 确保模型在评估模式
    model.eval()
    
    # 测试1：无故障注入（baseline）
    print("\n" + "="*80)
    print("Test 1: Baseline (no fault injection)")
    print("="*80)
    baseline_acc, baseline_correct, baseline_total = evaluate_model(
        model, test_loader, device, num_samples=args.num_samples
    )
    print(f"Baseline Accuracy: {baseline_acc:.2f}% ({baseline_correct}/{baseline_total})")
    
    # 测试2：标准二进制编码 + 故障注入
    print("\n" + "="*80)
    print(f"Test 2: Standard Binary Encoding + Fault Injection (BER={args.ber})")
    print("="*80)
    injector_binary = FaultInjector(
        model=model,
        mode="ber",
        ber=args.ber,
        device=device,
        enable_in_training=False,
        enable_in_inference=True,
        seed=args.seed,
        gray_code_layers=None,  # 不使用格雷码
        enable_statistics=True,  # 启用统计以验证故障注入
    )
    print("Enabling Binary FaultInjector...")
    injector_binary.enable()
    print("Binary FaultInjector enabled. Starting evaluation...")
    binary_acc, binary_correct, binary_total = evaluate_model(
        model, test_loader, device, num_samples=args.num_samples
    )
    print("Evaluation completed. Disabling Binary FaultInjector...")
    # 打印故障注入统计信息
    if hasattr(injector_binary, 'enable_statistics') and injector_binary.enable_statistics:
        injector_binary.print_flip_statistics()
    injector_binary.disable()
    print("Binary FaultInjector disabled.")
    print(f"Binary Encoding Accuracy: {binary_acc:.2f}% ({binary_correct}/{binary_total})")
    print(f"Accuracy Drop: {baseline_acc - binary_acc:.2f}%")
    
    # 确保模型状态清理干净
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared. Ready for Test 3.")
    
    # 测试3：格雷码编码（仅 features.0）+ 故障注入
    print("\n" + "="*80)
    print(f"Test 3: Gray Code Encoding (features.0 only) + Fault Injection (BER={args.ber})")
    print("="*80)
    print("Creating Gray Code FaultInjector...")
    injector_gray = FaultInjector(
        model=model,
        mode="ber",
        ber=args.ber,
        device=device,
        enable_in_training=False,
        enable_in_inference=True,
        seed=args.seed,  # 使用相同seed，但格雷码编码应该产生不同的结果
        gray_code_layers=['features.0'],  # 仅对 features.0 使用格雷码
        enable_statistics=True,  # 启用统计以验证故障注入
    )
    print(f"Gray code layers: {injector_gray.gray_code_layers}")
    print("Enabling Gray Code FaultInjector...")
    injector_gray.enable()
    print("Gray Code FaultInjector enabled. Starting evaluation...")
    gray_acc, gray_correct, gray_total = evaluate_model(
        model, test_loader, device, num_samples=args.num_samples
    )
    print("Evaluation completed. Disabling Gray Code FaultInjector...")
    injector_gray.disable()
    print("Gray Code FaultInjector disabled.")
    print(f"Gray Code Encoding Accuracy: {gray_acc:.2f}% ({gray_correct}/{gray_total})")
    print(f"Accuracy Drop: {baseline_acc - gray_acc:.2f}%")
    
    # 总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Baseline Accuracy:           {baseline_acc:.2f}%")
    print(f"Binary Encoding Accuracy:   {binary_acc:.2f}% (Drop: {baseline_acc - binary_acc:.2f}%)")
    print(f"Gray Code Encoding Accuracy: {gray_acc:.2f}% (Drop: {baseline_acc - gray_acc:.2f}%)")
    print(f"Improvement:                 {gray_acc - binary_acc:.2f}%")
    print("="*80)
    
    if gray_acc > binary_acc:
        print("✓ Gray Code encoding shows improvement!")
    elif gray_acc < binary_acc:
        print("✗ Gray Code encoding shows degradation.")
    else:
        print("= Gray Code encoding shows no difference.")


if __name__ == '__main__':
    main()

