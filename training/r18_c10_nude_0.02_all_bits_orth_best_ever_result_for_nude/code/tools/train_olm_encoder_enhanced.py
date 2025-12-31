#!/usr/bin/env python3
"""
增强的OLM编码器训练脚本（基于Gemini建议）

改进点：
1. Hessian感知的加权目标函数（使用梯度平方和作为敏感度权重）
2. 多对一映射（利用空闲编码空间）
3. 支持遗传算法搜索（可选）

使用方法：
    python tools/train_olm_encoder_enhanced.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --output olm_encoding_enhanced.json \
        --method greedy \
        --use_sensitivity \
        --use_surjective \
        --top_k_values 10 \
        --gradient_samples 100
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


def main():
    parser = argparse.ArgumentParser(description='Train enhanced OLM encoder (Gemini suggestions)')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, required=True, help='Layer name(s), comma-separated')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--method', type=str, default='genetic', 
                       choices=['greedy', 'simulated_annealing', 'genetic'],
                       help='Optimization method (default: genetic, recommended)')
    parser.add_argument('--max_iterations', type=int, default=100000,
                       help='Max iterations for simulated annealing or max generations for genetic algorithm (default: 100000, recommended: 100000-1000000 for large search spaces)')
    parser.add_argument('--use_sensitivity', action='store_true',
                       help='Use sensitivity-weighted LRobust (Hessian-aware)')
    parser.add_argument('--use_surjective', action='store_true',
                       help='Use surjective mapping (multi-to-one encoding)')
    parser.add_argument('--top_k_values', type=int, default=10,
                       help='Number of top values to assign multiple codes (default: 10)')
    parser.add_argument('--gradient_samples', type=int, default=-1,
                       help='Number of samples for gradient computation (-1 means use entire training set, default: -1)')
    parser.add_argument('--population_size', type=int, default=200,
                       help='Population size for genetic algorithm (default: 50)')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                       help='Crossover rate for genetic algorithm (default: 0.8)')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                       help='Mutation rate for genetic algorithm (default: 0.1)')
    parser.add_argument('--elite_size', type=int, default=5,
                       help='Elite size for genetic algorithm (default: 5)')
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
    print("增强的OLM编码器训练（基于Gemini建议）")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"目标层: {', '.join(layer_names)}")
    print(f"优化方法: {args.method}")
    print(f"使用敏感度权重: {args.use_sensitivity}")
    print(f"使用多对一映射: {args.use_surjective}")
    if args.use_surjective:
        print(f"前K个值分配多个编码: {args.top_k_values}")
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
        
        # 收集量化值分布和敏感度
        if args.use_sensitivity:
            print(f"  收集量化值分布和敏感度（梯度平方和）...")
            distribution, sensitivity = collect_quantized_value_distribution_with_sensitivity(
                model, layer_name, test_loader, criterion, device,
                num_samples=args.gradient_samples
            )
            print(f"  分布大小: {len(distribution)} 个唯一值")
            if sensitivity and len(sensitivity) > 0:
                sens_values = list(sensitivity.values())
                print(f"  敏感度范围: [{min(sens_values):.4f}, {max(sens_values):.4f}]")
        else:
            print(f"  收集量化值分布...")
            from util.olm_encoder import collect_quantized_value_distribution
            distribution = collect_quantized_value_distribution(model, layer_name, num_samples=-1)
            sensitivity = {v: 1.0 for v in distribution.keys()}  # 均匀敏感度
            print(f"  分布大小: {len(distribution)} 个唯一值")
        
        # 检查分布大小
        if len(distribution) <= 1:
            print(f"  ⚠️  警告：只有 {len(distribution)} 个唯一量化值，OLM优化意义不大")
            print(f"  → 使用简单映射（所有编码映射到同一个值）")
            # 创建简单的映射
            n_codes = 1 << wbits
            if len(distribution) == 1:
                value = list(distribution.keys())[0]
                value_to_code = {value: 0}
                code_to_value = {0: value}
                # 填充剩余编码
                for code in range(1, n_codes):
                    code_to_value[code] = value
                lrobust = 0.0  # 只有一个值，LRobust为0（没有邻居误差）
            else:
                # 没有值，创建默认映射
                value_to_code = {}
                code_to_value = {}
                default_value = 0
                for code in range(n_codes):
                    code_to_value[code] = default_value
                lrobust = float('inf')
            
            # 保存简单映射
            all_olm_mappings[layer_name] = {
                'bit_width': wbits,
                'method': 'simple',
                'lrobust': lrobust,
                'use_sensitivity': args.use_sensitivity,
                'use_surjective': args.use_surjective,
                'top_k_values': args.top_k_values if args.use_surjective else None,
                'value_to_code': {str(k): v for k, v in value_to_code.items()},
                'code_to_value': {str(k): v for k, v in code_to_value.items()},
                'distribution': {str(k): v for k, v in distribution.items()},
                'sensitivity': {str(k): v for k, v in sensitivity.items()} if args.use_sensitivity else None,
                'note': 'Simple mapping due to insufficient unique values'
            }
            continue
        
        # 优化OLM映射
        print(f"  优化OLM映射（方法: {args.method}）...")
        import time
        start_time = time.time()
        
        # 准备遗传算法参数（如果使用）
        genetic_kwargs = {}
        if args.method == 'genetic':
            genetic_kwargs = {
                'population_size': args.population_size,
                'crossover_rate': args.crossover_rate,
                'mutation_rate': args.mutation_rate,
                'elite_size': args.elite_size
            }
        
        value_to_code, code_to_value, lrobust = optimize_olm_mapping_enhanced(
            distribution=distribution,
            sensitivity=sensitivity,
            k=wbits,
            method=args.method,
            max_iterations=args.max_iterations,
            use_surjective=args.use_surjective,
            top_k_values=args.top_k_values,
            **genetic_kwargs
        )
        
        elapsed_time = time.time() - start_time
        print(f"  完成! LRobust: {lrobust:.4f}, 耗时: {elapsed_time:.2f}秒")
        
        # 验证映射的双射性
        def verify_bijective(vtc, ctv):
            """验证映射是否为双射"""
            # 检查value_to_code：每个值应该对应唯一编码
            code_to_values = {}
            for value, code in vtc.items():
                if code not in code_to_values:
                    code_to_values[code] = []
                code_to_values[code].append(value)
            code_conflicts = {code: values for code, values in code_to_values.items() if len(values) > 1}
            
            # 检查code_to_value：每个编码应该对应唯一值
            value_to_codes = {}
            for code, value in ctv.items():
                if value not in value_to_codes:
                    value_to_codes[value] = []
                value_to_codes[value].append(code)
            value_conflicts = {value: codes for value, codes in value_to_codes.items() if len(codes) > 1}
            
            # 检查双向一致性
            inconsistencies = []
            for value, code in vtc.items():
                if code in ctv:
                    if ctv[code] != value:
                        inconsistencies.append((value, code, ctv[code]))
            
            return code_conflicts, value_conflicts, inconsistencies
        
        code_conflicts, value_conflicts, inconsistencies = verify_bijective(value_to_code, code_to_value)
        
        if code_conflicts:
            print(f"    ⚠️  警告：发现 {len(code_conflicts)} 个编码冲突（多个值映射到同一个编码）")
            for code, values in list(code_conflicts.items())[:5]:
                print(f"      编码 {code} 被 {len(values)} 个值映射: {values[:3]}...")
        if value_conflicts:
            print(f"    ⚠️  警告：发现 {len(value_conflicts)} 个值冲突（多个编码映射到同一个值）")
            for value, codes in list(value_conflicts.items())[:5]:
                print(f"      值 {value} 被 {len(codes)} 个编码映射: {codes[:3]}...")
        if inconsistencies:
            print(f"    ⚠️  警告：发现 {len(inconsistencies)} 个不一致映射")
            for value, code, expected in inconsistencies[:5]:
                print(f"      value_to_code[{value}] = {code}, 但 code_to_value[{code}] = {expected}")
        
        if not code_conflicts and not value_conflicts and not inconsistencies:
            print(f"    ✅ 映射是双射的（每个值对应唯一编码，每个编码对应唯一值）")
        
        # 保存映射
        all_olm_mappings[layer_name] = {
            'bit_width': wbits,
            'method': args.method,
            'lrobust': lrobust,
            'use_sensitivity': args.use_sensitivity,
            'use_surjective': args.use_surjective,
            'top_k_values': args.top_k_values if args.use_surjective else None,
            'value_to_code': {str(k): v for k, v in value_to_code.items()},
            'code_to_value': {str(k): v for k, v in code_to_value.items()},
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
        print(f"增强OLM编码 + 故障注入准确率: {accuracy:.2f}%")
        
        # 打印统计信息
        fault_injector._process_pending_statistics()
        fault_injector.print_statistics()


if __name__ == '__main__':
    main()

