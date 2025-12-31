#!/usr/bin/env python3
"""
测试Learnable OLM的编解码一致性

检查在没有故障注入的情况下，编码后再解码是否能恢复原始权重值
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from quan.func import QuanConv2d, QuanLinear
from util.checkpoint import load_checkpoint
from util.config import get_config
# 不需要导入setup_model_with_bit_width_config，因为bit-width已经在checkpoint中
from util.learnable_olm import LearnableOLMManager
from util.qat import get_quantized_layers

# 导入加载函数
import importlib.util
spec = importlib.util.spec_from_file_location("test_script", "tools/test_learnable_olm_vs_traditional_olm.py")
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
load_learnable_olm_from_checkpoint = test_module.load_learnable_olm_from_checkpoint


def test_encode_decode_consistency(model, layer_name: str, learnable_encoder, device):
    """测试编码解码一致性"""
    module = dict(model.named_modules())[layer_name]
    if not isinstance(module, (QuanConv2d, QuanLinear)):
        raise ValueError(f"Layer {layer_name} is not a quantized layer")
    
    weight = module.weight.data.clone()
    scale = module.scale.data if hasattr(module, 'scale') else torch.tensor(1.0)
    
    # 获取权重位宽（bits可能是tuple (weight_bits, act_bits)）
    bits_raw = module.bits if hasattr(module, 'bits') else 8
    if isinstance(bits_raw, (list, tuple)):
        bits = bits_raw[0]  # 权重位宽
    elif isinstance(bits_raw, torch.Tensor):
        bits = bits_raw.item() if bits_raw.numel() == 1 else int(bits_raw[0].item())
    else:
        bits = int(bits_raw)
    
    thd_neg = -(1 << (bits - 1))
    thd_pos = (1 << (bits - 1)) - 1
    
    # 量化原始权重
    quantized_original = torch.round(weight / scale).clamp(thd_neg, thd_pos)
    
    # 编码
    quantized_int = quantized_original.to(device).int()
    encoded = learnable_encoder.encode(quantized_int, training=False)
    
    # 解码
    decoded = learnable_encoder.decode(encoded.int(), training=False)
    
    # 转换回量化值范围（如果需要）
    decoded_quantized = decoded.clamp(thd_neg, thd_pos)
    
    # 计算差异
    diff = (quantized_original.to(device) - decoded_quantized).abs()
    
    # 统计
    total_elements = quantized_original.numel()
    zero_diff = (diff == 0).sum().item()
    non_zero_diff = total_elements - zero_diff
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    mse = (diff ** 2).mean().item()
    
    # 检查是否有系统性偏差
    mean_original = quantized_original.float().mean().item()
    mean_decoded = decoded_quantized.float().mean().item()
    mean_bias = mean_original - mean_decoded
    
    return {
        'layer_name': layer_name,
        'total_elements': total_elements,
        'zero_diff_count': zero_diff,
        'zero_diff_ratio': zero_diff / total_elements if total_elements > 0 else 0,
        'non_zero_diff_count': non_zero_diff,
        'non_zero_diff_ratio': non_zero_diff / total_elements if total_elements > 0 else 0,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'mse': mse,
        'mean_original': mean_original,
        'mean_decoded': mean_decoded,
        'mean_bias': mean_bias,
        'original_range': [quantized_original.min().item(), quantized_original.max().item()],
        'decoded_range': [decoded_quantized.min().item(), decoded_quantized.max().item()],
    }


def main():
    parser = argparse.ArgumentParser(description='测试Learnable OLM编解码一致性')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("Learnable OLM 编解码一致性测试")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print()
    
    # 加载配置和模型
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
    
    # 步骤3.5: 设置位宽
    print("步骤3.5: 设置位宽...")
    # 获取target_bits的最大值
    target_bits = getattr(config, 'target_bits', [8])
    max_target_bit = max(target_bits) if target_bits else 8
    print(f"  target_bits: {target_bits}, 使用最大值: {max_target_bit}")
    
    # 获取fixed_bits配置（从quan.excepts中读取）
    excepts_bits_width = 8  # 默认值
    if hasattr(config, 'quan') and hasattr(config.quan, 'excepts'):
        excepts_bits_width = getattr(config.quan.excepts, 'excepts_bits_width', 8)
    print(f"  fixed_bits层使用: {excepts_bits_width}位")
    
    # 设置位宽
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                # fixed_bits层：使用yaml配置的值
                if isinstance(module.fixed_bits, (list, tuple)):
                    module.bits = (excepts_bits_width, excepts_bits_width)
                else:
                    module.bits = (excepts_bits_width, excepts_bits_width)
                print(f"  {name}: fixed_bits -> {excepts_bits_width}位")
            elif hasattr(module, 'bits') and module.bits is not None:
                # 动态位宽层：使用target_bits的最大值
                module.bits = (max_target_bit, max_target_bit)
                print(f"  {name}: 动态位宽 -> {max_target_bit}位")
            else:
                # 默认使用target_bits的最大值
                module.bits = (max_target_bit, max_target_bit)
                print(f"  {name}: 默认 -> {max_target_bit}位")
    
    # 获取所有量化层（直接遍历，不依赖get_quantized_layers，避免output_size问题）
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            layer_names.append(name)
    
    print(f"找到 {len(layer_names)} 个量化层: {layer_names}")
    print()
    
    # 加载Learnable OLM
    print("步骤4: 加载Learnable OLM...")
    try:
        # 手动加载Learnable OLM，避免get_quantized_layers的问题
        checkpoint = torch.load(args.ckpt, map_location=device)
        
        # 获取bit-width配置
        bit_widths = {}
        for layer_name in layer_names:
            module = dict(model.named_modules())[layer_name]
            if isinstance(module, (QuanConv2d, QuanLinear)):
                wbits = None
                if hasattr(module, 'bits') and module.bits is not None:
                    wbits = module.bits
                    if isinstance(wbits, (list, tuple)):
                        wbits = wbits[0] if len(wbits) > 0 else 8
                    if isinstance(wbits, torch.Tensor):
                        wbits = wbits.item() if wbits.numel() == 1 else int(wbits[0].item())
                    bit_widths[layer_name] = int(wbits)
                elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    wbits = module.fixed_bits
                    if isinstance(wbits, (list, tuple)):
                        wbits = wbits[0] if len(wbits) > 0 else 8
                    if isinstance(wbits, torch.Tensor):
                        wbits = wbits.item() if wbits.numel() == 1 else int(wbits[0].item())
                    bit_widths[layer_name] = int(wbits)
                else:
                    bit_widths[layer_name] = 8  # 默认8bit
        
        # 创建LearnableOLMManager
        learnable_olm_config = getattr(config, 'learnable_olm', None)
        if learnable_olm_config is None:
            init_method = 'identity'
            temperature = 1.0
            use_straight_through = True
        else:
            init_method = getattr(learnable_olm_config, 'init_method', 'identity')
            temperature = getattr(learnable_olm_config, 'temperature', 1.0)
            use_straight_through = getattr(learnable_olm_config, 'use_straight_through', True)
        
        learnable_olm_manager = LearnableOLMManager(
            model=model,
            layer_names=layer_names,
            bit_widths=bit_widths,
            device=device,
            init_method=init_method,
            temperature=temperature,
            use_straight_through=use_straight_through,
        )
        
        # 加载编码器参数（LearnableOLMManager没有load_state_dict方法，需要逐层加载）
        checkpoint_olm_state = checkpoint.get('learnable_olm_state', None)
        extras = checkpoint.get('extras', {})
        learnable_olm_state = extras.get('learnable_olm_state', None)
        
        if checkpoint_olm_state:
            print("  📦 从checkpoint['learnable_olm_state']加载...")
            for layer_name, layer_state in checkpoint_olm_state.items():
                if layer_name in learnable_olm_manager.encoders:
                    encoder = learnable_olm_manager.encoders[layer_name]
                    try:
                        encoder.load_state_dict(layer_state, strict=False)
                        print(f"  ✅ 已加载 {layer_name} 的编码器参数")
                    except Exception as e:
                        print(f"  ⚠️  加载 {layer_name} 的编码器参数失败: {e}")
        elif learnable_olm_state:
            print("  📦 从extras.learnable_olm_state加载...")
            for layer_name, layer_state in learnable_olm_state.items():
                if layer_name in learnable_olm_manager.encoders:
                    encoder = learnable_olm_manager.encoders[layer_name]
                    try:
                        encoder.load_state_dict(layer_state, strict=False)
                        print(f"  ✅ 已加载 {layer_name} 的编码器参数")
                    except Exception as e:
                        print(f"  ⚠️  加载 {layer_name} 的编码器参数失败: {e}")
        else:
            print(f"  ⚠️  Checkpoint中没有Learnable OLM状态，使用默认初始化")
        
        # 设置编码器为eval模式
        for encoder in learnable_olm_manager.encoders.values():
            encoder.eval()
        
        print(f"  ✅ Learnable OLM已加载，包含层: {list(learnable_olm_manager.encoders.keys())}")
        print()
    except Exception as e:
        print(f"  ❌ 加载Learnable OLM失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试每一层的编解码一致性
    print("步骤5: 测试编解码一致性...")
    print("="*80)
    
    all_results = []
    for layer_name in layer_names:
        if layer_name not in learnable_olm_manager.encoders:
            print(f"  ⚠️  {layer_name}: 没有Learnable OLM编码器，跳过")
            continue
        
        print(f"  测试层: {layer_name}")
        try:
            result = test_encode_decode_consistency(
                model, layer_name, learnable_olm_manager.encoders[layer_name], device
            )
            all_results.append(result)
            
            print(f"    总元素数: {result['total_elements']:,}")
            print(f"    完全一致: {result['zero_diff_count']:,} ({result['zero_diff_ratio']*100:.2f}%)")
            print(f"    有差异: {result['non_zero_diff_count']:,} ({result['non_zero_diff_ratio']*100:.2f}%)")
            print(f"    最大差异: {result['max_diff']:.2f}")
            print(f"    平均差异: {result['mean_diff']:.6f}")
            print(f"    MSE: {result['mse']:.6f}")
            print(f"    原始均值: {result['mean_original']:.2f}, 解码均值: {result['mean_decoded']:.2f}, 偏差: {result['mean_bias']:.2f}")
            print(f"    原始范围: [{result['original_range'][0]}, {result['original_range'][1]}]")
            print(f"    解码范围: [{result['decoded_range'][0]}, {result['decoded_range'][1]}]")
            
            if result['non_zero_diff_ratio'] > 0.01:  # 如果超过1%的元素有差异
                print(f"    ⚠️  警告: 超过1%的元素在编解码后发生变化！")
            if abs(result['mean_bias']) > 0.1:  # 如果有明显的系统性偏差
                print(f"    ⚠️  警告: 存在明显的系统性偏差！")
            if result['max_diff'] > 1:  # 如果最大差异超过1
                print(f"    ⚠️  警告: 最大差异超过1，可能存在严重问题！")
            
            print()
        except Exception as e:
            print(f"    ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # 总结
    print("="*80)
    print("测试总结")
    print("="*80)
    
    if all_results:
        total_elements = sum(r['total_elements'] for r in all_results)
        total_zero_diff = sum(r['zero_diff_count'] for r in all_results)
        total_non_zero_diff = sum(r['non_zero_diff_count'] for r in all_results)
        max_max_diff = max(r['max_diff'] for r in all_results)
        avg_mean_diff = np.mean([r['mean_diff'] for r in all_results])
        avg_mse = np.mean([r['mse'] for r in all_results])
        
        print(f"总元素数: {total_elements:,}")
        print(f"完全一致: {total_zero_diff:,} ({total_zero_diff/total_elements*100:.2f}%)")
        print(f"有差异: {total_non_zero_diff:,} ({total_non_zero_diff/total_elements*100:.2f}%)")
        print(f"最大差异: {max_max_diff:.2f}")
        print(f"平均差异: {avg_mean_diff:.6f}")
        print(f"平均MSE: {avg_mse:.6f}")
        print()
        
        if total_non_zero_diff / total_elements > 0.01:
            print("⚠️  严重警告: 超过1%的元素在编解码后发生变化！")
            print("   这说明Learnable OLM的编解码流程存在问题！")
        if max_max_diff > 1:
            print("⚠️  严重警告: 最大差异超过1，说明存在严重的编解码错误！")
        
        # 找出问题最严重的层
        worst_layers = sorted(all_results, key=lambda x: x['non_zero_diff_ratio'], reverse=True)[:3]
        print()
        print("问题最严重的层（按差异比例排序）:")
        for i, r in enumerate(worst_layers, 1):
            print(f"  {i}. {r['layer_name']}: {r['non_zero_diff_ratio']*100:.2f}% 有差异, 最大差异: {r['max_diff']:.2f}")


if __name__ == '__main__':
    main()

