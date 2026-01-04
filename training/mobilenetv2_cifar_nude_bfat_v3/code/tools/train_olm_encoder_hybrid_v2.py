#!/usr/bin/env python3
"""
混合保护OLM编码器训练脚本 V2

方案设计：
- bit7, bit0, bit1: 做冗余（3倍冗余保护bit7）
- bit6-2: 提取出来当做无符号数，做OLM训练（5位，32个值）

映射关系：
- 编码：多对一（多个原始二进制码可以对应同一个OLM码值）
- 解码：一对一（每个OLM编码能且仅能对应唯一的一个二进制数）

使用方法：
    python tools/train_olm_encoder_hybrid_v2.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --output olm_encoding_hybrid_v2.json \
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
from collections import defaultdict

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
from util.olm_encoder import collect_quantized_value_distribution


def extract_bit7(value: int, k: int) -> int:
    """提取bit7的值（0或1）"""
    thd_neg = -(1 << (k - 1))
    code_shifted = value - thd_neg
    bit7_idx = k - 1
    bit7 = (code_shifted >> bit7_idx) & 1
    return int(bit7)


def extract_bits_2_to_6(value: int, k: int) -> int:
    """提取bit2-6的值（5位，范围0-31），当做无符号数"""
    thd_neg = -(1 << (k - 1))
    code_shifted = value - thd_neg
    bits_2_to_6 = 0
    for i in range(2, 7):  # bit2到bit6
        bit_val = (code_shifted >> i) & 1
        bits_2_to_6 |= (bit_val << (i - 2))
    return bits_2_to_6


def create_hybrid_mapping_v2(
    distribution: dict,
    sensitivity: dict,
    k: int,
    method: str = 'genetic',
    max_iterations: int = 100000,
    use_sensitivity: bool = True,
    ber: float = 0.1,
    population_size: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> tuple:
    """
    创建混合保护映射 V2
    
    方案：
    - bit7, bit0, bit1: 做冗余（3倍冗余保护bit7）
    - bit6-2: 提取出来当做无符号数，做OLM训练（5位，32个值）
    
    映射关系：
    - 编码：一对多（同一个原始二进制数可以对应多个OLM码值）
    - 解码：一对一（每个OLM编码能且仅能对应唯一的一个二进制数）
    
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
    # 注意：多个原始值可能对应同一个bit2-6值
    olm_distribution = defaultdict(int)
    olm_sensitivity = defaultdict(float)
    olm_value_to_original_values = defaultdict(list)  # 记录每个bit2-6值对应的原始值列表
    
    for full_value, freq in distribution.items():
        bits_2_to_6_value = extract_bits_2_to_6(full_value, k)
        olm_distribution[bits_2_to_6_value] += freq
        olm_value_to_original_values[bits_2_to_6_value].append(full_value)
        if use_sensitivity and full_value in sensitivity:
            olm_sensitivity[bits_2_to_6_value] += sensitivity[full_value]
    
    print(f"  Bit2-6 OLM编码:")
    print(f"    实际唯一值数: {len(olm_distribution)}")
    print(f"    编码空间: {n_olm_values} (必须全部使用)")
    
    # 关键：32个OLM编码必须全部使用
    # 如果实际值少于32个，需要确保所有32个编码都被使用
    
    # 优化bit2-6的OLM映射
    olm_distribution_dict = dict(olm_distribution)
    # 纯方案B：不使用sensitivity，全部设为1.0（最终加权由weight_mode控制）
    olm_sensitivity_dict = {v: 1.0 for v in olm_distribution_dict.keys()}
    
    # 方案B要求：在 5-bit 域上做 32↔32 的映射（置换/双射），避免“多对一解码”带来的信息损失。
    # 若某些 bit2-6 值在当前层的分布中未出现，则补齐 freq=0、sens=0，使优化器仍能输出完整双射映射。
    for v in range(n_olm_values):
        olm_distribution_dict.setdefault(v, 0)
        olm_sensitivity_dict.setdefault(v, 1.0)
    
    # 调用优化函数，但需要确保所有32个编码都被使用
    bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust = optimize_olm_mapping_enhanced(
        distribution=olm_distribution_dict,
        sensitivity=olm_sensitivity_dict,
        k=n_olm_bits,
        method=method,
        max_iterations=max_iterations,
        use_surjective=False,
        top_k_values=10,
        objective="bsc_mse",
        ber=float(ber),
        weight_mode="freq",
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_size=elite_size
    )
    
    # 验证：确保所有32个编码都被使用
    used_codes = set(bits_2_to_6_code_to_value.keys())
    all_codes = set(range(n_olm_values))
    unused_codes = all_codes - used_codes
    
    if unused_codes:
        print(f"    警告: 有 {len(unused_codes)} 个编码未使用: {sorted(unused_codes)}")
        print(f"    将未使用的编码映射到高频/高敏感度的值")
        
        # 为未使用的编码分配映射
        # 策略：将未使用的编码映射到频率最高或敏感度最高的bit2-6值
        sorted_values_by_importance = sorted(
            olm_distribution_dict.keys(),
            key=lambda v: olm_distribution_dict.get(v, 0) * olm_sensitivity_dict.get(v, 1.0),
            reverse=True
        )
        
        # 为每个未使用的编码分配一个映射
        for unused_code in sorted(unused_codes):
            # 选择最重要的值（如果还有值可用）
            if sorted_values_by_importance:
                # 选择最重要的值，但可以重复使用（多对一）
                target_value = sorted_values_by_importance[0]
                bits_2_to_6_code_to_value[unused_code] = target_value
                print(f"      编码 {unused_code} → bit2-6值 {target_value} (与编码 {bits_2_to_6_value_to_code.get(target_value)} 共享)")
            else:
                # 如果没有值可用，使用默认值0
                bits_2_to_6_code_to_value[unused_code] = 0
                print(f"      编码 {unused_code} → bit2-6值 0 (默认)")
    
    # 验证：确保所有32个编码都被使用
    used_codes_after = set(bits_2_to_6_code_to_value.keys())
    if used_codes_after == all_codes:
        print(f"    ✅ 所有32个编码都已使用")
    else:
        print(f"    ⚠️  仍有 {len(all_codes - used_codes_after)} 个编码未使用")
    
    # 构建完整值的映射
    # 编码：一对多（同一个原始值可以对应多个编码值）
    # 解码：一对一（每个编码只能对应一个原始值）
    value_to_codes = defaultdict(list)  # 一个值对应多个编码
    code_to_value = {}  # 一个编码对应一个值
    
    thd_neg = -(1 << (k - 1))
    
    # 第一步：为每个bit2-6的OLM编码值分配一个唯一的原始值（用于解码）
    # 选择频率最高或敏感度最高的原始值作为代表
    bits_2_to_6_code_to_representative_value = {}
    for bits_2_to_6_code, bits_2_to_6_value in bits_2_to_6_code_to_value.items():
        # 找到所有对应这个bit2-6值的原始值
        original_values = olm_value_to_original_values[bits_2_to_6_value]
        
        # 选择频率最高或敏感度最高的作为代表值
        if use_sensitivity:
            # 使用频率×敏感度的加权和
            best_value = max(original_values, 
                           key=lambda v: distribution.get(v, 0) * sensitivity.get(v, 0.0))
        else:
            # 只使用频率
            best_value = max(original_values, 
                           key=lambda v: distribution.get(v, 0))
        
        bits_2_to_6_code_to_representative_value[bits_2_to_6_code] = best_value
    
    # 第二步：构建编码映射（一对多）
    # 对于每个原始值，生成多个编码值
    # 策略：
    # 1. 如果bit2-6值映射到多个OLM编码（因为32个编码必须全部使用），使用这些编码
    # 2. 通过改变bit0-1的冗余副本生成多个编码
    
    # 首先，找出每个bit2-6值对应的所有OLM编码（可能多个）
    bits_2_to_6_value_to_codes = defaultdict(list)  # 一个bit2-6值可能对应多个OLM编码
    for code, value in bits_2_to_6_code_to_value.items():
        bits_2_to_6_value_to_codes[value].append(code)
    
    print(f"  Bit2-6值到OLM编码的映射:")
    for value, codes in sorted(bits_2_to_6_value_to_codes.items()):
        if len(codes) > 1:
            print(f"    bit2-6值 {value} → OLM编码 {codes} (多对一)")
        else:
            print(f"    bit2-6值 {value} → OLM编码 {codes[0]} (一对一)")
    
    # 为每个原始值生成多个编码值
    value_to_code_primary = {}
    
    for full_value, freq in distribution.items():
        # 提取bit7
        bit7 = extract_bit7(full_value, k)
        
        # 提取bit2-6的值
        bits_2_to_6_value = extract_bits_2_to_6(full_value, k)
        
        # 获取这个bit2-6值对应的所有OLM编码（可能多个）
        if bits_2_to_6_value in bits_2_to_6_value_to_codes:
            available_olm_codes = bits_2_to_6_value_to_codes[bits_2_to_6_value]
        else:
            # 如果bit2-6值没有映射，使用原值作为编码
            available_olm_codes = [bits_2_to_6_value]
        
        # 为这个原始值生成多个编码值
        # 策略：只通过bit2-6的不同OLM编码生成多个编码值
        # bit0-1保持标准冗余（bit0=bit7, bit1=bit7），不改变
        # 这样既保证了冗余保护，又实现了一对多编码
        
        codes_for_value = []
        
        # 为每个可用的OLM编码生成一个编码值
        for bits_2_to_6_code in available_olm_codes:
            # 构建完整编码
            full_code_shifted = full_value - thd_neg  # [0, 255]
            
            # 清除bit0-1和bit2-6的原始值
            full_code_shifted &= ~((1 << 0) | (1 << 1))  # 清除bit0-1
            for i in range(2, 7):
                full_code_shifted &= ~(1 << i)  # 清除bit2-6
            
            # 设置bit0-1为bit7的标准冗余副本（保持不变，始终等于bit7）
            if bit7:
                full_code_shifted |= (1 << 0)  # bit0 = bit7
                full_code_shifted |= (1 << 1)  # bit1 = bit7
            # else: bit0和bit1保持0（因为已经清除，且bit7=0）
            
            # 设置bit2-6的OLM编码值（这是生成多个编码的关键）
            for i in range(5):  # bit2-6共5位
                bit_val = (bits_2_to_6_code >> i) & 1
                bit_idx = i + 2  # bit2-6
                if bit_val:
                    full_code_shifted |= (1 << bit_idx)
                else:
                    full_code_shifted &= ~(1 << bit_idx)
            
            # 转换回原始范围
            full_code = full_code_shifted + thd_neg
            codes_for_value.append(full_code)
        
        # 保存主要编码（第一个）
        value_to_code_primary[full_value] = codes_for_value[0]
        # 保存所有编码（一对多）
        value_to_codes[full_value] = codes_for_value
    
    # 第三步：构建解码映射（多对一）
    # 对于每个原始值生成的所有编码值，都映射回同一个原始值
    # 例如：原始值12的编码值230, 231, 232, 233都解码回12
    
    # 首先，为所有生成的编码值建立解码映射
    for full_value, codes_list in value_to_codes.items():
        for full_code in codes_list:
            # 所有编码值都解码回同一个原始值
            code_to_value[full_code] = full_value
    
    # 然后，为所有可能的编码值（包括未映射的）建立解码映射
    for code_shifted in range(256):  # 所有可能的编码值
        full_code = code_shifted - 128
        
        # 如果这个编码值已经在value_to_codes中，跳过（已经映射）
        if full_code in code_to_value:
            continue
        
        # 对于未映射的编码值，使用标准解码流程
        # 提取bit7（可能已故障）
        bit7_original = (code_shifted >> 7) & 1
        bit7_redundant_0 = (code_shifted >> 0) & 1
        bit7_redundant_1 = (code_shifted >> 1) & 1
        
        # 多数投票纠正bit7
        bit7_corrected = 1 if (bit7_original + bit7_redundant_0 + bit7_redundant_1) >= 2 else 0
        
        # 提取bit2-6的OLM编码
        bits_2_to_6_code = 0
        for i in range(5):
            bit_val = (code_shifted >> (i + 2)) & 1
            bits_2_to_6_code |= (bit_val << i)
        
        # 使用OLM反向映射找到bit2-6的值
        if bits_2_to_6_code in bits_2_to_6_code_to_value:
            bits_2_to_6_value = bits_2_to_6_code_to_value[bits_2_to_6_code]
        else:
            # 如果编码没有映射，使用编码本身
            bits_2_to_6_value = bits_2_to_6_code
        
        # 获取代表原始值
        if bits_2_to_6_code in bits_2_to_6_code_to_representative_value:
            representative_value = bits_2_to_6_code_to_representative_value[bits_2_to_6_code]
        else:
            # 如果没有代表值，从bit2-6值重建
            reconstructed_shifted = 0
            reconstructed_shifted |= (bit7_corrected << 7)  # bit7
            for i in range(5):
                bit_val = (bits_2_to_6_value >> i) & 1
                reconstructed_shifted |= (bit_val << (i + 2))  # bit2-6
            representative_value = reconstructed_shifted + thd_neg
        
        # 重新构建解码值（使用纠正后的bit7）
        decoded_shifted = 0
        decoded_shifted |= (bit7_corrected << 7)  # 使用纠正后的bit7
        for i in range(5):
            bit_val = (bits_2_to_6_value >> i) & 1
            decoded_shifted |= (bit_val << (i + 2))  # bit2-6
        decoded_value = decoded_shifted + thd_neg
        
        code_to_value[full_code] = decoded_value
    
    # 为了支持真正的"一对多"编码，我们需要修改value_to_code
    # 让一个原始值可以对应多个编码值
    # 这里我们使用value_to_code_primary作为主要映射（用于兼容现有接口）
    # 但保留value_to_codes用于一对多编码
    value_to_code = value_to_code_primary  # 主要编码（用于兼容）
    
    # 注意：value_to_codes包含一对多映射，可以在后续扩展中使用
    # 当前实现中，每个原始值至少有一个编码（主要编码）
    # 如果需要真正的"一对多"，可以为同一个原始值生成多个编码（例如，通过不同的bit2-6 OLM编码选择）
    
    return value_to_code, code_to_value, bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust, value_to_codes


def main():
    parser = argparse.ArgumentParser(description='Train hybrid OLM encoder V2 (bit7/bit0/bit1 redundancy, bit2-6 OLM with many-to-one encoding)')
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
    parser.add_argument('--ber', type=float, default=0.1,
                       help='BER for BSC objective when optimizing OLM mapping (default: 0.1)')
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
    print("混合保护OLM编码器训练 V2")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"目标层: {', '.join(layer_names)}")
    print(f"保护方案:")
    print(f"  - bit7, bit0, bit1: 做冗余（3倍冗余保护bit7）")
    print(f"  - bit6-2: 提取出来当做无符号数，做OLM训练（5位，32个值）")
    print(f"映射关系:")
    print(f"  - 编码：一对多（同一个原始二进制数可以对应多个OLM码值）")
    print(f"  - 解码：一对一（每个OLM编码能且仅能对应唯一的一个二进制数）")
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
        
        # 收集量化值分布（不使用sensitivity/梯度信息：纯方案B）
        print(f"  收集量化值分布（不收集梯度sensitivity）...")
        distribution = collect_quantized_value_distribution(model, layer_name, num_samples=args.gradient_samples)
        sensitivity = {v: 1.0 for v in distribution.keys()}  # 占位，不参与加权
        
        print(f"  分布大小: {len(distribution)} 个唯一值")
        
        # 创建混合保护映射
        print(f"  优化混合保护OLM映射（方法: {args.method}）...")
        import time
        start_time = time.time()
        
        value_to_code, code_to_value, bits_2_to_6_value_to_code, bits_2_to_6_code_to_value, lrobust, value_to_codes = create_hybrid_mapping_v2(
            distribution=distribution,
            sensitivity=sensitivity,
            k=wbits,
            method=args.method,
            max_iterations=args.max_iterations,
            use_sensitivity=False,
            ber=args.ber,
            population_size=args.population_size,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            elite_size=args.elite_size
        )
        
        elapsed_time = time.time() - start_time
        print(f"  完成! LRobust: {lrobust:.4f}, 耗时: {elapsed_time:.2f}秒")
        
        # 验证映射关系
        print(f"  验证映射关系...")
        # 检查编码：一对多
        one_to_many_count = sum(1 for value, codes in value_to_codes.items() if len(codes) > 1)
        print(f"    编码（一对多）: {one_to_many_count} 个原始值对应多个编码值")
        
        # 检查解码：一对一
        one_to_one_count = len(code_to_value)
        print(f"    解码（一对一）: {one_to_one_count} 个编码对应唯一解码值")
        
        # 保存映射
        all_olm_mappings[layer_name] = {
            'bit_width': wbits,
            'method': args.method,
            'lrobust': lrobust,
            'use_sensitivity': args.use_sensitivity,
            'protection_scheme': 'hybrid_v2',
            'bit7_redundancy': 'bit7, bit0, bit1做冗余（3倍冗余）',
            'bit2_6_olm': 'bit6-2提取出来当做无符号数，做OLM训练（5位，32个值）',
            'encoding_relation': 'one-to-many (同一个原始值可以映射到多个编码)',
            'decoding_relation': 'one-to-one (每个编码只能对应唯一的一个解码值)',
            'value_to_code': {str(k): v for k, v in value_to_code.items()},  # 主要编码（一对一，用于兼容）
            'value_to_codes': {str(k): v for k, v in value_to_codes.items()},  # 一对多编码（一个值对应多个编码）
            'code_to_value': {str(k): v for k, v in code_to_value.items()},
            'bits_2_to_6_value_to_code': {str(k): v for k, v in bits_2_to_6_value_to_code.items()},
            'bits_2_to_6_code_to_value': {str(k): v for k, v in bits_2_to_6_code_to_value.items()},
            'distribution': {str(k): v for k, v in distribution.items()},
            'sensitivity': {str(k): v for k, v in sensitivity.items()} if args.use_sensitivity else None,
            'one_to_many_count': one_to_many_count,
            'one_to_one_count': one_to_one_count
        }
    
    # 保存结果
    output_data = {
        'layers': layer_names,
        'num_layers': len(layer_names),
        'protection_scheme': 'hybrid_v2',
        'description': 'bit7/bit0/bit1做冗余，bit6-2提取出来做OLM（一对多编码，一对一解码）',
        'layer_mappings': all_olm_mappings
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"结果已保存到: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

