#!/usr/bin/env python3
"""
使用改进的OLM编码器训练（考虑梯度信息和多bit翻转）

改进点：
1. 基于梯度信息衡量权重的重要性
2. 考虑多bit翻转（高BER情况下）
3. 使用加权LRobust

使用方法：
    python tools/train_olm_encoder_improved.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --output olm_encoding_features_0_improved.json \
        --method simulated_annealing \
        --max_iterations 3000 \
        --ber 0.1 \
        --consider_multi_bit \
        --max_hamming_dist 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.olm_encoder import collect_quantized_value_distribution
from util.improved_olm_encoder import (
    compute_weight_importance_by_gradient,
    optimize_olm_mapping_improved
)


def main():
    parser = argparse.ArgumentParser(description='Train improved OLM encoder')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, required=True, help='Layer name(s), comma-separated')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--method', type=str, default='simulated_annealing', 
                       choices=['greedy', 'simulated_annealing'],
                       help='Optimization method')
    parser.add_argument('--max_iterations', type=int, default=3000,
                       help='Max iterations for simulated annealing')
    parser.add_argument('--ber', type=float, default=0.1,
                       help='Bit-error-rate (for multi-bit flip consideration)')
    parser.add_argument('--consider_multi_bit', action='store_true',
                       help='Consider multi-bit flips (for high BER)')
    parser.add_argument('--max_hamming_dist', type=int, default=3,
                       help='Maximum Hamming distance to consider (default: 3)')
    parser.add_argument('--use_gradient', action='store_true',
                       help='Use gradient information for weight importance')
    parser.add_argument('--gradient_samples', type=int, default=100,
                       help='Number of samples for gradient computation')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_fault_injection', action='store_true',
                       help='Test fault injection after training')
    parser.add_argument('--test_ber', type=float, default=0.1,
                       help='BER for fault injection test')
    
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
    
    # 解析层名称
    layer_names = [name.strip() for name in args.layer.split(',')]
    
    print("="*80)
    print("改进的OLM编码器训练")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"目标层: {', '.join(layer_names)}")
    print(f"优化方法: {args.method}")
    print(f"最大迭代次数: {args.max_iterations}")
    print(f"BER: {args.ber}")
    print(f"考虑多bit翻转: {args.consider_multi_bit}")
    if args.consider_multi_bit:
        print(f"最大Hamming距离: {args.max_hamming_dist}")
    print(f"使用梯度信息: {args.use_gradient}")
    print()
    
    # 创建模型
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    
    # 加载checkpoint
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 准备损失函数（用于梯度计算）
    criterion = nn.CrossEntropyLoss()
    
    # 存储所有层的映射
    all_olm_mappings = {}
    
    # 对每个层进行优化
    for layer_idx, layer_name in enumerate(layer_names):
        print(f"\n{'='*80}")
        print(f"处理层 {layer_idx+1}/{len(layer_names)}: {layer_name}")
        print(f"{'='*80}")
        
        # 获取位宽
        module = dict(model.named_modules())[layer_name]
        wbits = None
        if hasattr(module, 'bits') and module.bits is not None:
            wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
        elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
            wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
        
        if wbits is None:
            print(f"  ⚠️  跳过 {layer_name}：没有位宽配置")
            continue
        
        if isinstance(wbits, torch.Tensor):
            wbits = int(wbits.item())
        else:
            wbits = int(wbits)
        
        print(f"  位宽: {wbits}-bit")
        
        # 收集量化值分布
        print(f"  收集量化值分布...")
        distribution = collect_quantized_value_distribution(model, layer_name, num_samples=-1)
        print(f"  分布大小: {len(distribution)} 个唯一值")
        
        # 计算权重重要性（如果启用）
        weight_importance = None
        if args.use_gradient:
            print(f"  计算权重重要性（基于梯度信息）...")
            try:
                weight_importance = compute_weight_importance_by_gradient(
                    model, layer_name, test_loader, criterion, device,
                    num_samples=args.gradient_samples
                )
                print(f"  权重重要性: {len(weight_importance)} 个值")
            except Exception as e:
                print(f"  ⚠️  计算权重重要性失败: {e}")
                print(f"  继续使用均匀权重...")
        
        # 优化OLM映射
        print(f"  优化OLM映射（方法: {args.method}）...")
        import time
        start_time = time.time()
        
        value_to_code, code_to_value, lrobust = optimize_olm_mapping_improved(
            distribution=distribution,
            k=wbits,
            method=args.method,
            max_iterations=args.max_iterations,
            weight_importance=weight_importance,
            ber=args.ber,
            consider_multi_bit=args.consider_multi_bit,
            max_hamming_dist=args.max_hamming_dist
        )
        
        elapsed_time = time.time() - start_time
        print(f"  完成! LRobust: {lrobust:.4f}, 耗时: {elapsed_time:.2f}秒")
        
        # 保存映射
        all_olm_mappings[layer_name] = {
            'bit_width': wbits,
            'method': args.method,
            'lrobust': lrobust,
            'ber': args.ber,
            'consider_multi_bit': args.consider_multi_bit,
            'max_hamming_dist': args.max_hamming_dist if args.consider_multi_bit else 1,
            'use_gradient': args.use_gradient,
            'value_to_code': {str(k): v for k, v in value_to_code.items()},
            'code_to_value': {str(k): v for k, v in code_to_value.items()},
            'distribution': {str(k): v for k, v in distribution.items()}
        }
    
    # 保存结果
    output_data = {
        'layers': layer_names,
        'num_layers': len(layer_names),
        'layer_mappings': all_olm_mappings
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"结果已保存到: {args.output}")
    print(f"{'='*80}")
    
    # 测试故障注入（如果启用）
    if args.test_fault_injection:
        print(f"\n{'='*80}")
        print("测试故障注入")
        print(f"{'='*80}")
        
        # 构建OLM映射字典
        olm_layers_dict = {}
        for layer_name in layer_names:
            if layer_name in all_olm_mappings:
                value_to_code = {
                    int(k): int(v) 
                    for k, v in all_olm_mappings[layer_name]['value_to_code'].items()
                }
                olm_layers_dict[layer_name] = value_to_code
        
        # 初始化FaultInjector
        fault_injector = FaultInjector(
            model=model,
            mode='ber',
            ber=args.test_ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            olm_layers=olm_layers_dict,
            enable_statistics=True
        )
        
        # 设置code_to_value
        olm_code_to_value_dict = {}
        for layer_name in layer_names:
            if layer_name in all_olm_mappings:
                code_to_value = {
                    int(k): int(v)
                    for k, v in all_olm_mappings[layer_name]['code_to_value'].items()
                }
                olm_code_to_value_dict[layer_name] = code_to_value
        
        fault_injector.olm_code_to_value = olm_code_to_value_dict
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        
        fault_injector.enable()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        fault_injector.disable()
        
        accuracy = 100. * correct / total
        print(f"OLM编码 + 故障注入准确率: {accuracy:.2f}%")
        
        # 打印统计信息
        fault_injector._process_pending_statistics()
        fault_injector.print_statistics()


if __name__ == '__main__':
    main()

