#!/usr/bin/env python3
"""
测试原始模型在故障注入下的准确率（Baseline）

不涉及任何编码（OLM、格雷码等），仅测试标准二进制编码 + 故障注入的效果。

对比：
1. Baseline（无故障）
2. 标准二进制编码 + 故障注入（不同BER）

使用方法：
    python tools/test_fault_injection_baseline.py \
        --config configs/training/train_alexnet_cifar10_learnable_olm_fat.yaml \
        --ckpt training/alexnet_cifar10_learnable_olm_fat/alexnet_cifar10_learnable_olm_fat_checkpoint.pth.tar \
        --ber 1e-2
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config


def evaluate_model(model, dataloader, device):
    """评估模型准确率（在整个验证集上）"""
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
    parser = argparse.ArgumentParser(description='Test baseline fault injection (no encoding)')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--ber', type=float, default=1e-2, help='Bit error rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_ber_list', action='store_true', 
                       help='Test multiple BER values: [1e-3, 1e-2, 5e-2, 1e-1]')
    
    args = parser.parse_args()
    
    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    # 创建模型
    print("="*80)
    print("原始模型故障注入测试（Baseline，无编码）")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"设备: {device}")
    print()
    
    print("步骤1: 创建模型...")
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 应用量化
    print("步骤2: 应用量化...")
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载checkpoint
    print(f"步骤3: 加载checkpoint: {args.ckpt}")
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 加载bit-width配置（必须在checkpoint之后，因为需要设置dynamic bit-width）
    # 对于动态位宽训练的模型，checkpoint加载后需要重新设置bit-width
    print("步骤4: 设置bit-width配置...")
    if args.bit_width_config:
        print(f"  从文件加载bit-width配置: {args.bit_width_config}")
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    else:
        # 如果没有提供bit-width配置文件，从训练配置中读取target_bits并设置
        target_bits = getattr(config, 'target_bits', [8])
        enable_dynamic = getattr(config, 'enable_dynamic_bit_training', False)
        
        # switch_bit_width 需要单个值，不能是列表
        # 对于动态位宽模型，在推理时我们需要设置一个具体的bit-width
        # 使用 target_bits 的最大值作为默认值
        if isinstance(target_bits, list):
            if len(target_bits) > 0:
                # 对于动态位宽，使用最大值（这样可以覆盖所有可能的bit-width）
                bit_width_value = max(target_bits)
            else:
                bit_width_value = 8
        else:
            bit_width_value = int(target_bits)
        
        from util.mpq import switch_bit_width
        if enable_dynamic:
            print(f"  从配置读取: enable_dynamic_bit_training=True, target_bits={target_bits}")
            print(f"  设置所有dynamic layers为 {bit_width_value}-bit (使用target_bits的最大值)")
        else:
            print(f"  使用配置: enable_dynamic_bit_training={enable_dynamic}, target_bits={target_bits}")
            print(f"  设置所有层为 {bit_width_value}-bit")
        
        switch_bit_width(model, quan_scheduler=config.quan, wbit=bit_width_value, abits=bit_width_value)
    
    # 检查模型的bit-width设置
    print("步骤5: 检查模型的bit-width设置...")
    from util.qat import get_quantized_layers
    from quan.func import QuanConv2d, QuanLinear
    dynamic_count = 0
    fixed_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                fixed_count += 1
                wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
                if isinstance(wbits, torch.Tensor):
                    wbits = int(wbits.item())
                print(f"  {name}: fixed_bits={wbits}")
            elif hasattr(module, 'bits') and module.bits is not None:
                dynamic_count += 1
                wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
                if isinstance(wbits, torch.Tensor):
                    wbits = int(wbits.item()) if wbits.numel() == 1 else int(wbits[0].item())
                print(f"  {name}: bits={wbits}")
    print(f"  总计: {dynamic_count} 个dynamic layers, {fixed_count} 个fixed layers")
    print()
    
    model.eval()
    
    # 准备数据
    print("步骤6: 准备数据...")
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * test_loader.batch_size
    print(f"验证集大小: {total_samples} 样本")
    print()
    
    # Test 1: Baseline（无故障）
    print("="*80)
    print("Test 1: Baseline (无故障注入)")
    print("="*80)
    accuracy_baseline = evaluate_model(model, test_loader, device)
    print(f"准确率: {accuracy_baseline:.2f}%")
    print()
    
    # Test 2: 标准二进制编码 + 故障注入
    if args.test_ber_list:
        # 测试多个BER值
        ber_list = [1e-3, 1e-2, 5e-2, 1e-1]
        print("="*80)
        print("Test 2: 标准二进制编码 + 故障注入（多个BER值）")
        print("="*80)
        
        results = []
        for ber in ber_list:
            print(f"\nBER = {ber:.0e} ({ber})")
            print("-" * 80)
            
            injector = FaultInjector(
                model=model,
                mode='ber',
                ber=ber,
                device=device,
                enable_in_inference=True,
                seed=args.seed,
                enable_statistics=True
            )
            injector.enable()
            accuracy = evaluate_model(model, test_loader, device)
            injector.disable()
            
            drop = accuracy_baseline - accuracy
            drop_percent = (drop / accuracy_baseline * 100) if accuracy_baseline > 0 else 0
            
            print(f"准确率: {accuracy:.2f}%")
            print(f"相对Baseline下降: {drop:.2f}% ({drop_percent:.1f}%)")
            
            results.append({
                'ber': ber,
                'accuracy': accuracy,
                'drop': drop,
                'drop_percent': drop_percent
            })
        
        # 打印总结表格
        print()
        print("="*80)
        print("总结表格")
        print("="*80)
        print(f"{'BER':<15} {'准确率':<15} {'下降':<15} {'下降率':<15}")
        print("-" * 80)
        print(f"{'Baseline':<15} {accuracy_baseline:>13.2f}% {'-':<15} {'-':<15}")
        for r in results:
            print(f"{r['ber']:<15.0e} {r['accuracy']:>13.2f}% {r['drop']:>13.2f}% {r['drop_percent']:>13.1f}%")
        print("="*80)
    else:
        # 测试单个BER值
        print("="*80)
        print(f"Test 2: 标准二进制编码 + 故障注入 (BER={args.ber})")
        print("="*80)
        
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            enable_statistics=True
        )
        injector.enable()
        accuracy_faulted = evaluate_model(model, test_loader, device)
        injector.disable()
        
        drop = accuracy_baseline - accuracy_faulted
        drop_percent = (drop / accuracy_baseline * 100) if accuracy_baseline > 0 else 0
        
        print(f"准确率: {accuracy_faulted:.2f}%")
        print(f"相对Baseline下降: {drop:.2f}% ({drop_percent:.1f}%)")
        print()
        
        # 打印统计信息（如果启用）
        if hasattr(injector, 'print_statistics'):
            print("故障注入统计信息:")
            injector.print_statistics()
    
    print()
    print("="*80)
    print("测试完成")
    print("="*80)
    print(f"Baseline准确率: {accuracy_baseline:.2f}%")
    if args.test_ber_list:
        print("\n不同BER下的准确率:")
        for r in results:
            print(f"  BER={r['ber']:.0e}: {r['accuracy']:.2f}% (下降 {r['drop']:.2f}%)")
    else:
        print(f"故障注入后准确率 (BER={args.ber}): {accuracy_faulted:.2f}% (下降 {drop:.2f}%)")
    print("="*80)


if __name__ == '__main__':
    main()

