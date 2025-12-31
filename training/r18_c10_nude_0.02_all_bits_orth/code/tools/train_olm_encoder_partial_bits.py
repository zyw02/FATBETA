#!/usr/bin/env python3
"""
部分位保护OLM编码器训练脚本

只对指定位范围（如bit5-7）进行OLM编码保护，其他位使用标准二进制编码
这样可以：
1. 减少编码映射的复杂度（只需要映射部分位的组合）
2. 降低存储开销
3. 提高编码/解码速度
4. 专注于保护最重要的位

使用方法：
    python tools/train_olm_encoder_partial_bits.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --output olm_encoding_partial_bits.json \
        --protected_bits "5,6,7" \
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


def extract_protected_bits_value(value: int, protected_bits: list, k: int) -> int:
    """
    从完整的量化值中提取被保护位的值
    
    Args:
        value: 完整的量化值（例如 -128 到 127）
        protected_bits: 被保护的位索引列表（例如 [5, 6, 7]）
        k: 总位宽（例如 8）
    
    Returns:
        被保护位的值（例如 0 到 7，如果保护bit5-7）
    """
    # 转换为非负范围 [0, 2^k-1]
    thd_neg = -(1 << (k - 1))
    code_shifted = value - thd_neg  # 现在在 [0, 2^k-1]
    
    # 提取被保护位的值
    protected_value = 0
    for bit_idx in protected_bits:
        if bit_idx < k:
            bit_val = (code_shifted >> bit_idx) & 1
            protected_value |= (bit_val << (bit_idx - min(protected_bits)))
    
    return protected_value


def create_partial_bits_mapping(
    distribution: dict,
    sensitivity: dict,
    k: int,
    protected_bits: list,
    method: str = 'genetic',
    max_iterations: int = 100000,
    use_sensitivity: bool = True,
    population_size: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> tuple:
    """
    创建部分位保护的OLM映射
    
    只对被保护的位进行编码映射，其他位保持不变
    
    Args:
        distribution: 完整量化值的分布
        sensitivity: 完整量化值的敏感度
        k: 总位宽
        protected_bits: 被保护的位索引列表（例如 [5, 6, 7]）
        method: 优化方法
        max_iterations: 最大迭代次数
        use_sensitivity: 是否使用敏感度
        population_size: 种群大小
        crossover_rate: 交叉率
        mutation_rate: 变异率
        elite_size: 精英数量
    
    Returns:
        (value_to_code, code_to_value, protected_value_to_code, protected_code_to_value, lrobust)
        - value_to_code: 完整值到完整编码的映射（用于兼容现有接口）
        - code_to_value: 完整编码到完整值的映射（用于兼容现有接口）
        - protected_value_to_code: 被保护位的值到编码的映射
        - protected_code_to_value: 被保护位的编码到值的映射
        - lrobust: LRobust值
    """
    n_protected_bits = len(protected_bits)
    n_protected_codes = 1 << n_protected_bits
    
    # 收集被保护位的值分布和敏感度
    protected_distribution = {}
    protected_sensitivity = {}
    
    for full_value, freq in distribution.items():
        protected_value = extract_protected_bits_value(full_value, protected_bits, k)
        if protected_value not in protected_distribution:
            protected_distribution[protected_value] = 0
            protected_sensitivity[protected_value] = 0.0
        protected_distribution[protected_value] += freq
        if use_sensitivity and full_value in sensitivity:
            protected_sensitivity[protected_value] += sensitivity[full_value]
    
    print(f"  被保护位: bit{protected_bits[0]}-bit{protected_bits[-1]} ({n_protected_bits}位)")
    print(f"  被保护位的唯一值数: {len(protected_distribution)}")
    print(f"  被保护位的编码空间: {n_protected_codes}")
    
    # 优化被保护位的映射
    protected_value_to_code, protected_code_to_value, lrobust = optimize_olm_mapping_enhanced(
        distribution=protected_distribution,
        sensitivity=protected_sensitivity if use_sensitivity else {v: 1.0 for v in protected_distribution.keys()},
        k=n_protected_bits,
        method=method,
        max_iterations=max_iterations,
        use_surjective=False,
        top_k_values=10,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_size=elite_size
    )
    
    # 构建完整值的映射（用于兼容现有接口）
    # 对于每个完整值，提取被保护位的值，查找对应的编码，然后组合回完整编码
    value_to_code = {}
    code_to_value = {}
    
    thd_neg = -(1 << (k - 1))
    
    for full_value, freq in distribution.items():
        # 提取被保护位的值
        protected_value = extract_protected_bits_value(full_value, protected_bits, k)
        
        # 获取被保护位的编码
        if protected_value in protected_value_to_code:
            protected_code = protected_value_to_code[protected_value]
        else:
            # 如果被保护位的值没有映射，使用原值
            protected_code = protected_value
        
        # 构建完整编码：被保护位使用OLM编码，其他位保持不变
        full_code_shifted = full_value - thd_neg  # [0, 2^k-1]
        
        # 清除被保护位的值
        for bit_idx in protected_bits:
            full_code_shifted &= ~(1 << bit_idx)
        
        # 设置被保护位的编码值
        for i, bit_idx in enumerate(protected_bits):
            bit_val = (protected_code >> i) & 1
            if bit_val:
                full_code_shifted |= (1 << bit_idx)
            else:
                full_code_shifted &= ~(1 << bit_idx)
        
        # 转换回原始范围
        full_code = full_code_shifted + thd_neg
        
        value_to_code[full_value] = full_code
        code_to_value[full_code] = full_value
    
    return value_to_code, code_to_value, protected_value_to_code, protected_code_to_value, lrobust


def main():
    parser = argparse.ArgumentParser(description='Train partial bits OLM encoder (only protect specified bits)')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, required=True, help='Layer name(s), comma-separated')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--protected_bits', type=str, required=True, 
                       help='Protected bit indices, comma-separated (e.g., "5,6,7" for bit5-bit7)')
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
    parser.add_argument('--test_fault_injection', action='store_true',
                       help='Test fault injection after training')
    parser.add_argument('--test_ber', type=float, default=0.1,
                       help='BER for fault injection test')
    
    args = parser.parse_args()
    
    # 解析被保护的位
    protected_bits = [int(b.strip()) for b in args.protected_bits.split(',')]
    protected_bits = sorted(protected_bits)
    
    if not protected_bits or any(b < 0 for b in protected_bits):
        raise ValueError(f"Invalid protected_bits: {args.protected_bits}")
    
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
    print("部分位保护OLM编码器训练")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"目标层: {', '.join(layer_names)}")
    print(f"被保护位: bit{protected_bits[0]}-bit{protected_bits[-1]} ({protected_bits})")
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
        
        print(f"  位宽: {wbits}-bit")
        
        # 检查被保护的位是否有效
        if any(b >= wbits for b in protected_bits):
            print(f"  ⚠️  警告：被保护位 {protected_bits} 超出位宽范围 [0, {wbits-1}]")
            protected_bits_valid = [b for b in protected_bits if b < wbits]
            if not protected_bits_valid:
                print(f"  ❌ 错误：没有有效的被保护位")
                continue
            protected_bits = protected_bits_valid
            print(f"  → 使用有效的被保护位: {protected_bits}")
        
        # 收集量化值分布和敏感度
        if args.use_sensitivity:
            print(f"  收集量化值分布和敏感度...")
            distribution, sensitivity = collect_quantized_value_distribution_with_sensitivity(
                model, layer_name, test_loader, criterion, device,
                num_samples=args.gradient_samples
            )
        else:
            # 不使用敏感度时，直接收集量化值分布
            from util.olm_encoder_enhanced import collect_quantized_value_distribution_with_sensitivity
            distribution, _ = collect_quantized_value_distribution_with_sensitivity(
                model, layer_name, test_loader, criterion, device,
                num_samples=-1, use_sensitivity=False
            )
            sensitivity = {v: 1.0 for v in distribution.keys()}
        
        print(f"  分布大小: {len(distribution)} 个唯一值")
        
        # 创建部分位保护映射
        print(f"  优化部分位保护OLM映射（方法: {args.method}）...")
        import time
        start_time = time.time()
        
        value_to_code, code_to_value, protected_value_to_code, protected_code_to_value, lrobust = create_partial_bits_mapping(
            distribution=distribution,
            sensitivity=sensitivity,
            k=wbits,
            protected_bits=protected_bits,
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
            'protected_bits': protected_bits,
            'protected_bits_count': len(protected_bits),
            'value_to_code': {str(k): v for k, v in value_to_code.items()},
            'code_to_value': {str(k): v for k, v in code_to_value.items()},
            'protected_value_to_code': {str(k): v for k, v in protected_value_to_code.items()},
            'protected_code_to_value': {str(k): v for k, v in protected_code_to_value.items()},
            'distribution': {str(k): v for k, v in distribution.items()},
            'sensitivity': {str(k): v for k, v in sensitivity.items()} if args.use_sensitivity else None
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
        print(f"部分位保护OLM编码 + 故障注入准确率: {accuracy:.2f}%")
        
        # 打印统计信息
        fault_injector._process_pending_statistics()
        fault_injector.print_statistics()


if __name__ == '__main__':
    main()

