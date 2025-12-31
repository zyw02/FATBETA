#!/usr/bin/env python3
"""
混合保护OLM编码器训练脚本

方案设计：
- bit[0,1]（最低2位）给bit7做冗余保护（3倍冗余中的2位）
- bit2-6（5位）做OLM编码
- 存储开销：8位 → 8位（无增加）

编码原理：
1. 提取bit7的值
2. 使用bit0-1存储bit7的2份冗余副本（共3份：bit7本身 + bit0 + bit1）
3. 对bit2-6进行OLM编码优化
4. 组合回完整编码

使用方法：
    python tools/train_olm_encoder_hybrid.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --output olm_encoding_hybrid.json \
        --method genetic \
        --use_sensitivity \
        --max_iterations 100000
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
from util.olm_encoder_enhanced import (
    collect_quantized_value_distribution_with_sensitivity,
    optimize_olm_mapping_enhanced
)


def extract_bit7(value: int, k: int) -> int:
    """提取bit7的值（0或1）"""
    thd_neg = -(1 << (k - 1))
    code_shifted = value - thd_neg
    bit7_idx = k - 1
    bit7 = (code_shifted >> bit7_idx) & 1
    return int(bit7)


def extract_bits_2_to_6(value: int, k: int) -> int:
    """提取bit2-6的值（5位，范围0-31）"""
    thd_neg = -(1 << (k - 1))
    code_shifted = value - thd_neg
    bits_2_to_6 = 0
    for i in range(2, 7):  # bit2到bit6
        bit_val = (code_shifted >> i) & 1
        bits_2_to_6 |= (bit_val << (i - 2))
    return bits_2_to_6


def create_hybrid_mapping(
    distribution: dict,
    sensitivity: dict,
    k: int,
    method: str = 'genetic',
    max_iterations: int = 100000,
    use_sensitivity: bool = True,
    population_size: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> tuple:
    """
    创建混合保护映射
    
    方案：
    - bit7: 使用bit0-1做冗余保护（3倍冗余：bit7本身 + bit0 + bit1）
    - bit2-6: OLM编码（5位，32个值）
    - bit0-1: 存储bit7的冗余副本
    
    Args:
        distribution: 完整量化值的分布
        sensitivity: 完整量化值的敏感度
        k: 总位宽（8）
        method: 优化方法
        max_iterations: 最大迭代次数
        use_sensitivity: 是否使用敏感度
        population_size: 种群大小
        crossover_rate: 交叉率
        mutation_rate: 变异率
        elite_size: 精英数量
    
    Returns:
        (value_to_code, code_to_value, bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust)
    """
    n_olm_bits = 5  # bit2-6
    n_olm_values = 1 << n_olm_bits  # 32
    
    # 收集bit2-6的值分布和敏感度
    olm_distribution = {}
    olm_sensitivity = {}
    
    for full_value, freq in distribution.items():
        bits_2_to_6_value = extract_bits_2_to_6(full_value, k)
        if bits_2_to_6_value not in olm_distribution:
            olm_distribution[bits_2_to_6_value] = 0
            olm_sensitivity[bits_2_to_6_value] = 0.0
        olm_distribution[bits_2_to_6_value] += freq
        if use_sensitivity and full_value in sensitivity:
            olm_sensitivity[bits_2_to_6_value] += sensitivity[full_value]
    
    print(f"  Bit2-6 OLM编码:")
    print(f"    唯一值数: {len(olm_distribution)}")
    print(f"    编码空间: {n_olm_values}")
    
    # 优化bit2-6的OLM映射
    bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust = optimize_olm_mapping_enhanced(
        distribution=olm_distribution,
        sensitivity=olm_sensitivity if use_sensitivity else {v: 1.0 for v in olm_distribution.keys()},
        k=n_olm_bits,
        method=method,
        max_iterations=max_iterations,
        use_surjective=False,
        top_k_values=10,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_size=elite_size
    )
    
    # 构建完整值的映射
    value_to_code = {}
    code_to_value = {}
    
    thd_neg = -(1 << (k - 1))
    
    for full_value, freq in distribution.items():
        # 提取bit7
        bit7 = extract_bit7(full_value, k)
        
        # 提取bit2-6的值
        bits_2_to_6_value = extract_bits_2_to_6(full_value, k)
        
        # 获取bit2-6的OLM编码
        if bits_2_to_6_value in bits_2_to_6_value_to_code:
            bits_2_to_6_code = bits_2_to_6_value_to_code[bits_2_to_6_value]
        else:
            bits_2_to_6_code = bits_2_to_6_value
        
        # 构建完整编码：
        # - bit7: 保持原值
        # - bit0-1: 存储bit7的冗余副本（用于错误纠正）
        # - bit2-6: 使用OLM编码
        full_code_shifted = full_value - thd_neg  # [0, 255]
        
        # 步骤1: 清除bit0-1和bit2-6的原始值
        full_code_shifted &= ~((1 << 0) | (1 << 1))  # 清除bit0-1
        for i in range(2, 7):
            full_code_shifted &= ~(1 << i)  # 清除bit2-6
        
        # 步骤2: 设置bit0-1为bit7的冗余副本
        if bit7:
            full_code_shifted |= (1 << 0)  # bit0 = bit7
            full_code_shifted |= (1 << 1)  # bit1 = bit7
        else:
            # bit0和bit1保持0（因为已经清除）
            pass
        
        # 步骤3: 设置bit2-6的OLM编码值
        for i in range(5):  # bit2-6共5位
            bit_val = (bits_2_to_6_code >> i) & 1
            bit_idx = i + 2  # bit2-6
            if bit_val:
                full_code_shifted |= (1 << bit_idx)
            else:
                full_code_shifted &= ~(1 << bit_idx)
        
        # 步骤4: bit7保持不变（已经是正确的）
        
        # 转换回原始范围
        full_code = full_code_shifted + thd_neg
        
        value_to_code[full_value] = full_code
        code_to_value[full_code] = full_value
    
    return value_to_code, code_to_value, bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust


def main():
    parser = argparse.ArgumentParser(description='Train hybrid OLM encoder (bit0-1 for bit7 redundancy, bit2-6 for OLM)')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, required=True, help='Layer name(s), comma-separated')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--method', type=str, default='genetic', 
                       choices=['greedy', 'simulated_annealing', 'genetic'],
                       help='Optimization method (default: genetic)')
    parser.add_argument('--max_iterations', type=int, default=100000,
                       help='Max iterations for optimization')
    parser.add_argument('--use_sensitivity', action='store_true',
                       help='Use sensitivity-weighted LRobust')
    parser.add_argument('--gradient_samples', type=int, default=-1,
                       help='Number of samples for gradient computation (-1 means use entire training set)')
    parser.add_argument('--population_size', type=int, default=200,
                       help='Population size for genetic algorithm')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                       help='Crossover rate for genetic algorithm')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                       help='Mutation rate for genetic algorithm')
    parser.add_argument('--elite_size', type=int, default=5,
                       help='Elite size for genetic algorithm')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    print("混合保护OLM编码器训练")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"目标层: {', '.join(layer_names)}")
    print(f"保护方案:")
    print(f"  - bit7: 使用bit0-1做冗余保护（3倍冗余：bit7 + bit0 + bit1）")
    print(f"  - bit2-6: OLM编码（5位，32个值）")
    print(f"  - bit0-1: 存储bit7的冗余副本")
    print(f"优化方法: {args.method}")
    print(f"使用敏感度权重: {args.use_sensitivity}")
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
    
    # 准备损失函数
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
        
        if wbits != 8:
            print(f"  ⚠️  警告：当前层位宽为{wbits}位，混合保护方案设计为8位")
            print(f"     将跳过此层或使用适配方案")
            continue
        
        print(f"  位宽: {wbits}-bit")
        
        # 收集量化值分布和敏感度
        if args.use_sensitivity:
            print(f"  收集量化值分布和敏感度...")
            distribution, sensitivity = collect_quantized_value_distribution_with_sensitivity(
                model, layer_name, test_loader, criterion, device,
                num_samples=args.gradient_samples
            )
        else:
            from util.olm_encoder_enhanced import collect_quantized_value_distribution_with_sensitivity
            distribution, _ = collect_quantized_value_distribution_with_sensitivity(
                model, layer_name, test_loader, criterion, device,
                num_samples=-1, use_sensitivity=False
            )
            sensitivity = {v: 1.0 for v in distribution.keys()}
        
        print(f"  分布大小: {len(distribution)} 个唯一值")
        
        # 创建混合保护映射
        print(f"  优化混合保护OLM映射（方法: {args.method}）...")
        import time
        start_time = time.time()
        
        value_to_code, code_to_value, bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust = create_hybrid_mapping(
            distribution=distribution,
            sensitivity=sensitivity,
            k=wbits,
            method=args.method,
            max_iterations=args.max_iterations,
            use_sensitivity=args.use_sensitivity,
            population_size=args.population_size,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            elite_size=args.elite_size
        )
        
        elapsed_time = time.time() - start_time
        print(f"  完成! LRobust: {lrobust:.4f}, 耗时: {elapsed_time:.2f}秒")
        
        # 保存映射
        all_olm_mappings[layer_name] = {
            'bit_width': wbits,
            'method': args.method,
            'lrobust': lrobust,
            'use_sensitivity': args.use_sensitivity,
            'protection_scheme': 'hybrid',
            'bit7_redundancy': 'bit0-1存储bit7的冗余副本（3倍冗余）',
            'bit2_6_olm': 'bit2-6使用OLM编码（5位，32个值）',
            'value_to_code': {str(k): v for k, v in value_to_code.items()},
            'code_to_value': {str(k): v for k, v in code_to_value.items()},
            'bits_2_to_6_value_to_code': {str(k): v for k, v in bits_2_to_6_value_to_code.items()},
            'bits_2_to_6_code_to_value': {str(k): v for k, v in bits_2_to_6_code_to_value.items()},
            'distribution': {str(k): v for k, v in distribution.items()},
            'sensitivity': {str(k): v for k, v in sensitivity.items()} if args.use_sensitivity else None
        }
    
    # 保存结果
    output_data = {
        'layers': layer_names,
        'num_layers': len(layer_names),
        'protection_scheme': 'hybrid',
        'description': 'bit0-1给bit7做冗余保护，bit2-6做OLM编码',
        'layer_mappings': all_olm_mappings
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"结果已保存到: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()



