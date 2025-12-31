#!/usr/bin/env python3
"""
分析Learnable OLM和传统OLM的编码差异，以及故障注入后的权重变化

使用方法:
    python tools/analyze_olm_differences.py \
        --config configs/training/train_alexnet_cifar10_learnable_olm_fat.yaml \
        --ckpt training/alexnet_cifar10_learnable_olm_fat_v2/alexnet_cifar10_learnable_olm_fat_v2_checkpoint.pth.tar \
        --layer features.0 \
        --ber 1e-2 \
        --num_samples 100
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from quan.func import QuanConv2d, QuanLinear
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.learnable_olm import LearnableOLMManager
from util.olm_encoder import create_olm_encoder, collect_quantized_value_distribution, optimize_olm_mapping
from util.qat import get_quantized_layers

# 导入加载函数
import importlib.util
spec = importlib.util.spec_from_file_location("test_script", "tools/test_learnable_olm_vs_traditional_olm.py")
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
load_learnable_olm_from_checkpoint = test_module.load_learnable_olm_from_checkpoint


def collect_weight_statistics(model, layer_name: str, num_samples: int = 100):
    """收集权重的量化值分布"""
    module = dict(model.named_modules())[layer_name]
    if not isinstance(module, (QuanConv2d, QuanLinear)):
        raise ValueError(f"Layer {layer_name} is not a quantized layer")
    
    # 获取权重
    weight = module.weight.data
    scale = module.scale.data if hasattr(module, 'scale') else torch.tensor(1.0)
    bits = module.bits if hasattr(module, 'bits') else 8
    
    # 量化
    thd_neg = -(1 << (bits - 1))
    thd_pos = (1 << (bits - 1)) - 1
    
    # 采样
    if weight.numel() > num_samples:
        indices = torch.randperm(weight.numel())[:num_samples]
        sampled_weights = weight.view(-1)[indices]
    else:
        sampled_weights = weight.view(-1)
    
    # 量化值
    quantized = torch.round(sampled_weights / scale).clamp(thd_neg, thd_pos)
    
    return {
        'quantized_values': quantized.cpu().numpy(),
        'original_weights': sampled_weights.cpu().numpy(),
        'scale': scale.item() if isinstance(scale, torch.Tensor) else scale,
        'bits': bits,
        'thd_neg': thd_neg,
        'thd_pos': thd_pos,
    }


def compare_encodings(
    traditional_mapping: Dict[int, int],
    learnable_encoder,
    quantized_values: np.ndarray,
    thd_neg: int,
    thd_pos: int
) -> Dict:
    """比较传统OLM和Learnable OLM的编码差异"""
    results = {
        'traditional_mapping': traditional_mapping,
        'learnable_mapping': {},
        'encoding_differences': [],
        'conflicts': [],
    }
    
    # 获取Learnable OLM的映射
    value_to_code_learnable, code_to_value_learnable = learnable_encoder.get_hard_mapping()
    results['learnable_mapping'] = value_to_code_learnable
    
    # 比较每个量化值的编码
    unique_values = np.unique(quantized_values)
    for value in unique_values:
        value_int = int(value)
        if value_int in traditional_mapping:
            traditional_code = traditional_mapping[value_int]
        else:
            traditional_code = None
        
        if value_int in value_to_code_learnable:
            learnable_code = value_to_code_learnable[value_int]
        else:
            learnable_code = None
        
        if traditional_code != learnable_code:
            results['encoding_differences'].append({
                'value': value_int,
                'traditional_code': traditional_code,
                'learnable_code': learnable_code,
            })
    
    # 检查冲突（多个量化值映射到同一个编码）
    # 传统OLM
    traditional_code_to_values = defaultdict(list)
    for value, code in traditional_mapping.items():
        traditional_code_to_values[code].append(value)
    traditional_conflicts = {code: values for code, values in traditional_code_to_values.items() if len(values) > 1}
    
    # Learnable OLM
    learnable_code_to_values = defaultdict(list)
    for value, code in value_to_code_learnable.items():
        learnable_code_to_values[code].append(value)
    learnable_conflicts = {code: values for code, values in learnable_code_to_values.items() if len(values) > 1}
    
    results['traditional_conflicts'] = dict(traditional_conflicts)
    results['learnable_conflicts'] = dict(learnable_conflicts)
    
    return results


def analyze_fault_injection_impact(
    model,
    layer_name: str,
    traditional_mapping: Dict[int, int],
    learnable_encoder,
    ber: float,
    num_samples: int = 1000
) -> Dict:
    """分析故障注入对权重的影响"""
    module = dict(model.named_modules())[layer_name]
    if not isinstance(module, (QuanConv2d, QuanLinear)):
        raise ValueError(f"Layer {layer_name} is not a quantized layer")
    
    weight = module.weight.data.clone()
    scale = module.scale.data if hasattr(module, 'scale') else torch.tensor(1.0)
    bits = module.bits if hasattr(module, 'bits') else 8
    
    thd_neg = -(1 << (bits - 1))
    thd_pos = (1 << (bits - 1)) - 1
    n_levels = (1 << bits) - 1
    
    # 量化原始权重
    quantized_original = torch.round(weight / scale).clamp(thd_neg, thd_pos)
    
    # 采样分析
    if weight.numel() > num_samples:
        indices = torch.randperm(weight.numel())[:num_samples]
        sampled_indices = indices
    else:
        sampled_indices = torch.arange(weight.numel())
    
    sampled_quantized = quantized_original.view(-1)[sampled_indices]
    
    results = {
        'traditional_olm': {
            'original_values': sampled_quantized.cpu().numpy().tolist(),
            'encoded_values': [],
            'faulted_encoded_values': [],
            'decoded_values': [],
            'weight_differences': [],
            'mse': 0.0,
            'max_diff': 0.0,
        },
        'learnable_olm': {
            'original_values': sampled_quantized.cpu().numpy().tolist(),
            'encoded_values': [],
            'faulted_encoded_values': [],
            'decoded_values': [],
            'weight_differences': [],
            'mse': 0.0,
            'max_diff': 0.0,
        },
    }
    
    # 传统OLM分析
    print(f"  分析传统OLM故障注入影响...")
    traditional_encoded = []
    traditional_decoded = []
    
    for value_int in sampled_quantized.cpu().numpy():
        value_int = int(value_int)
        # 编码
        if value_int in traditional_mapping:
            code = traditional_mapping[value_int]
        else:
            code = value_int - thd_neg  # identity mapping
        
        traditional_encoded.append(code)
        
        # 模拟位翻转（简化：随机翻转一个bit）
        if torch.rand(1).item() < ber:
            bit_to_flip = torch.randint(0, bits, (1,)).item()
            code_faulted = code ^ (1 << bit_to_flip)
            code_faulted = max(0, min(n_levels, code_faulted))
        else:
            code_faulted = code
        
        # 解码（需要反向映射）
        # 这里简化处理，假设有反向映射
        code_to_value_traditional = {v: k for k, v in traditional_mapping.items()}
        if code_faulted in code_to_value_traditional:
            decoded_value = code_to_value_traditional[code_faulted]
        else:
            decoded_value = code_faulted + thd_neg  # identity mapping
        
        traditional_decoded.append(decoded_value)
    
    results['traditional_olm']['encoded_values'] = traditional_encoded
    results['traditional_olm']['decoded_values'] = traditional_decoded
    results['traditional_olm']['weight_differences'] = [
        abs(orig - decoded) for orig, decoded in zip(sampled_quantized.cpu().numpy(), traditional_decoded)
    ]
    results['traditional_olm']['mse'] = np.mean([d**2 for d in results['traditional_olm']['weight_differences']])
    results['traditional_olm']['max_diff'] = max(results['traditional_olm']['weight_differences']) if results['traditional_olm']['weight_differences'] else 0
    
    # Learnable OLM分析
    print(f"  分析Learnable OLM故障注入影响...")
    learnable_encoded = []
    learnable_decoded = []
    
    quantized_tensor = sampled_quantized.to(weight.device).int()
    # 编码
    encoded_tensor = learnable_encoder.encode(quantized_tensor, training=False)
    learnable_encoded = encoded_tensor.cpu().numpy().tolist()
    
    # 模拟位翻转
    encoded_faulted = encoded_tensor.clone()
    for i in range(len(encoded_faulted)):
        if torch.rand(1).item() < ber:
            bit_to_flip = torch.randint(0, bits, (1,)).item()
            encoded_faulted[i] = encoded_faulted[i] ^ (1 << bit_to_flip)
            encoded_faulted[i] = max(0, min(n_levels, int(encoded_faulted[i])))
    
    # 解码
    decoded_tensor = learnable_encoder.decode(encoded_faulted.int(), training=False)
    learnable_decoded = decoded_tensor.cpu().numpy().tolist()
    
    results['learnable_olm']['encoded_values'] = learnable_encoded
    results['learnable_olm']['decoded_values'] = learnable_decoded
    results['learnable_olm']['weight_differences'] = [
        abs(orig - decoded) for orig, decoded in zip(sampled_quantized.cpu().numpy(), learnable_decoded)
    ]
    results['learnable_olm']['mse'] = np.mean([d**2 for d in results['learnable_olm']['weight_differences']])
    results['learnable_olm']['max_diff'] = max(results['learnable_olm']['weight_differences']) if results['learnable_olm']['weight_differences'] else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description='分析Learnable OLM和传统OLM的差异')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--layer', type=str, required=True, help='Layer name to analyze')
    parser.add_argument('--ber', type=float, default=1e-2, help='Bit error rate')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for analysis')
    parser.add_argument('--output', type=str, default='olm_analysis.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载配置和模型
    print("="*80)
    print("Learnable OLM vs Traditional OLM 差异分析")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"分析层: {args.layer}")
    print(f"BER: {args.ber}")
    print()
    
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    # 创建模型
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
    
    # 设置bit-width
    setup_model_with_bit_width_config(model, config)
    
    # 收集权重统计
    print(f"步骤4: 收集权重统计...")
    weight_stats = collect_weight_statistics(model, args.layer, num_samples=args.num_samples)
    
    # 生成传统OLM映射
    print(f"步骤5: 生成传统OLM映射...")
    distribution = collect_quantized_value_distribution(model, args.layer, num_samples=1000)
    module = dict(model.named_modules())[args.layer]
    bits = module.bits if hasattr(module, 'bits') else 8
    traditional_mapping, _, _ = optimize_olm_mapping(distribution, bits, method='greedy')
    print(f"  传统OLM映射大小: {len(traditional_mapping)}")
    
    # 加载Learnable OLM
    print(f"步骤6: 加载Learnable OLM...")
    learnable_olm_manager = load_learnable_olm_from_checkpoint(
        args.ckpt, model, [args.layer], device, config
    )
    
    if learnable_olm_manager is None or args.layer not in learnable_olm_manager.encoders:
        print(f"  ❌ 无法加载Learnable OLM编码器")
        return
    
    learnable_encoder = learnable_olm_manager.encoders[args.layer]
    print(f"  Learnable OLM已加载")
    
    # 比较编码
    print(f"步骤7: 比较编码差异...")
    encoding_comparison = compare_encodings(
        traditional_mapping,
        learnable_encoder,
        weight_stats['quantized_values'],
        weight_stats['thd_neg'],
        weight_stats['thd_pos']
    )
    
    print(f"  编码差异数量: {len(encoding_comparison['encoding_differences'])}")
    print(f"  传统OLM冲突数量: {len(encoding_comparison['traditional_conflicts'])}")
    print(f"  Learnable OLM冲突数量: {len(encoding_comparison['learnable_conflicts'])}")
    
    # 分析故障注入影响
    print(f"步骤8: 分析故障注入影响...")
    fault_analysis = analyze_fault_injection_impact(
        model,
        args.layer,
        traditional_mapping,
        learnable_encoder,
        args.ber,
        num_samples=args.num_samples
    )
    
    print(f"  传统OLM MSE: {fault_analysis['traditional_olm']['mse']:.6f}")
    print(f"  传统OLM最大差异: {fault_analysis['traditional_olm']['max_diff']}")
    print(f"  Learnable OLM MSE: {fault_analysis['learnable_olm']['mse']:.6f}")
    print(f"  Learnable OLM最大差异: {fault_analysis['learnable_olm']['max_diff']}")
    
    # 保存结果
    print(f"步骤9: 保存分析结果...")
    results = {
        'layer_name': args.layer,
        'ber': args.ber,
        'weight_statistics': {
            'bits': weight_stats['bits'],
            'thd_neg': weight_stats['thd_neg'],
            'thd_pos': weight_stats['thd_pos'],
            'unique_values_count': len(np.unique(weight_stats['quantized_values'])),
        },
        'encoding_comparison': {
            'encoding_differences_count': len(encoding_comparison['encoding_differences']),
            'traditional_conflicts_count': len(encoding_comparison['traditional_conflicts']),
            'learnable_conflicts_count': len(encoding_comparison['learnable_conflicts']),
            'encoding_differences': encoding_comparison['encoding_differences'][:20],  # 只保存前20个
            'traditional_conflicts': {str(k): v for k, v in list(encoding_comparison['traditional_conflicts'].items())[:10]},
            'learnable_conflicts': {str(k): v for k, v in list(encoding_comparison['learnable_conflicts'].items())[:10]},
        },
        'fault_injection_analysis': {
            'traditional_olm': {
                'mse': fault_analysis['traditional_olm']['mse'],
                'max_diff': fault_analysis['traditional_olm']['max_diff'],
                'mean_diff': np.mean(fault_analysis['traditional_olm']['weight_differences']),
                'std_diff': np.std(fault_analysis['traditional_olm']['weight_differences']),
            },
            'learnable_olm': {
                'mse': fault_analysis['learnable_olm']['mse'],
                'max_diff': fault_analysis['learnable_olm']['max_diff'],
                'mean_diff': np.mean(fault_analysis['learnable_olm']['weight_differences']),
                'std_diff': np.std(fault_analysis['learnable_olm']['weight_differences']),
            },
        },
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  结果已保存到: {args.output}")
    print()
    print("="*80)
    print("分析总结")
    print("="*80)
    print(f"编码差异: {len(encoding_comparison['encoding_differences'])} 个量化值使用了不同的编码")
    print(f"传统OLM冲突: {len(encoding_comparison['traditional_conflicts'])} 个编码被多个量化值使用")
    print(f"Learnable OLM冲突: {len(encoding_comparison['learnable_conflicts'])} 个编码被多个量化值使用")
    print()
    print("故障注入影响:")
    print(f"  传统OLM - MSE: {fault_analysis['traditional_olm']['mse']:.6f}, 最大差异: {fault_analysis['traditional_olm']['max_diff']}")
    print(f"  Learnable OLM - MSE: {fault_analysis['learnable_olm']['mse']:.6f}, 最大差异: {fault_analysis['learnable_olm']['max_diff']}")
    print()
    print(f"详细结果已保存到: {args.output}")


if __name__ == '__main__':
    main()

