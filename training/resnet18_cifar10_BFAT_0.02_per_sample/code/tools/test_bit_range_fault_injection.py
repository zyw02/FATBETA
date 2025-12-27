#!/usr/bin/env python3
"""
逐位故障注入测试：从最低位开始，逐步增加故障注入的位范围

测试方案：
- 第1轮：仅对bit0做故障注入（BER=1e-1）
- 第2轮：对bit0-bit1做故障注入（BER=1e-1）
- 第3轮：对bit0-bit2做故障注入（BER=1e-1）
- ...
- 第8轮：对bit0-bit7做故障注入（BER=1e-1）

统计每次迭代后的模型准确率
"""

import argparse
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


class BitRangeFaultInjector(FaultInjector):
    """
    扩展FaultInjector，支持按位范围进行故障注入
    """
    def __init__(self, *args, bit_range=None, **kwargs):
        """
        Args:
            bit_range: (start_bit, end_bit) 元组，指定要注入故障的位范围（包含end_bit）
                      例如 (0, 2) 表示对bit0、bit1、bit2进行故障注入
                      如果为None，则对所有位进行故障注入（默认行为）
        """
        super().__init__(*args, **kwargs)
        self.bit_range = bit_range  # (start_bit, end_bit)
    
    def _generate_flip_mask(self, N: int, k: int, device: torch.device, 
                           layer_name=None, mask_seed=None) -> torch.Tensor:
        """
        生成翻转掩码，但只对指定位范围进行故障注入
        
        Args:
            N: 权重数量
            k: 位宽
            bit_range: (start_bit, end_bit) 元组，指定要注入故障的位范围
        
        Returns:
            flip_mask: shape [N, k]，True表示该位需要翻转
        """
        if self.bit_range is None:
            # 默认行为：对所有位进行故障注入
            return super()._generate_flip_mask(N, k, device, layer_name, mask_seed)
        
        start_bit, end_bit = self.bit_range
        if start_bit < 0 or end_bit >= k or start_bit > end_bit:
            raise ValueError(f"Invalid bit_range: {start_bit}-{end_bit} for {k}-bit quantization")
        
        # 生成完整的翻转掩码
        full_mask = super()._generate_flip_mask(N, k, device, layer_name, mask_seed)
        
        # 只保留指定位范围的翻转
        mask = torch.zeros_like(full_mask)
        mask[:, start_bit:end_bit+1] = full_mask[:, start_bit:end_bit+1]
        
        return mask
    
    def _inject_on_quantized_tensor(self, x_q: torch.Tensor, k: int, scale: torch.Tensor, 
                                    layer_name=None, forward_seed=None, layer_name_for_stats=None) -> torch.Tensor:
        """
        重写故障注入方法，使用自定义的位范围翻转掩码
        """
        if self.bit_range is None:
            # 默认行为
            return super()._inject_on_quantized_tensor(x_q, k, scale, layer_name, forward_seed, layer_name_for_stats)
        
        # 使用自定义的位范围故障注入
        # 直接调用父类方法，但先修改_generate_flip_mask的行为
        # 由于_generate_flip_mask已经被重写，所以直接调用父类方法即可
        return super()._inject_on_quantized_tensor(x_q, k, scale, layer_name, forward_seed, layer_name_for_stats)


def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='逐位故障注入测试')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, default='features.0', help='Layer name to test')
    parser.add_argument('--ber', type=float, default=0.1, help='Bit-error-rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
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
    print("创建模型...")
    dataset = config.dataloader.dataset if hasattr(config.dataloader, 'dataset') else 'cifar10'
    model = create_model(config.arch, dataset=dataset, pre_trained=getattr(config, 'pre_trained', False))
    model = model.to(device)
    
    # 应用量化
    print("应用量化...")
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        print("加载bit-width配置...")
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    
    # 加载checkpoint
    print("加载checkpoint...")
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 获取层位宽
    module = dict(model.named_modules())[args.layer]
    wbits = None
    if hasattr(module, 'bits') and module.bits is not None:
        wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
    
    if wbits is None:
        print(f"❌ 错误：层 {args.layer} 没有位宽配置")
        return
    
    if isinstance(wbits, torch.Tensor):
        wbits = int(wbits.item())
    else:
        wbits = int(wbits)
    
    print("="*80)
    print("逐位故障注入测试")
    print("="*80)
    print(f"测试层: {args.layer}")
    print(f"位宽: {wbits}-bit")
    print(f"BER: {args.ber}")
    print(f"测试范围: bit0 到 bit{wbits-1}")
    print()
    
    # Test 0: Baseline（无故障）
    print("Test 0: Baseline (无故障注入)")
    accuracy_baseline = evaluate_model(model, test_loader, device)
    print(f"  准确率: {accuracy_baseline:.2f}%")
    print()
    
    # 逐位范围测试
    results = []
    results.append({
        'bit_range': 'None',
        'bits': 'Baseline',
        'accuracy': accuracy_baseline,
        'drop': 0.0
    })
    
    # 第一轮：累积测试（bit0, bit0-bit1, ..., bit0-bit7）
    print("="*80)
    print("第一轮：累积测试（从bit0开始逐步增加）")
    print("="*80)
    
    for end_bit in range(wbits):
        bit_range = (0, end_bit)
        bits_str = f"bit0-bit{end_bit}" if end_bit > 0 else "bit0"
        
        print(f"\nTest {end_bit+1}: 故障注入范围 = {bits_str}")
        print("-" * 80)
        
        # 创建BitRangeFaultInjector
        injector = BitRangeFaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            bit_range=bit_range,
            enable_statistics=False
        )
        
        # 只对指定层启用故障注入
        # 使用whitelist_layer来限制只对features.0层进行故障注入
        injector.whitelist_layer = args.layer
        injector.enable()
        
        # 评估
        accuracy = evaluate_model(model, test_loader, device)
        drop = accuracy_baseline - accuracy
        
        injector.disable()
        injector.whitelist_layer = None  # 重置
        
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  相对Baseline下降: {drop:.2f}%")
        
        results.append({
            'bit_range': bit_range,
            'bits': bits_str,
            'accuracy': accuracy,
            'drop': drop
        })
    
    # 打印总结
    print("\n" + "="*80)
    print("测试总结 - 累积测试")
    print("="*80)
    print(f"{'测试':<15} {'位范围':<15} {'准确率':<12} {'相对Baseline下降':<20}")
    print("-" * 80)
    for i, result in enumerate(results):
        if i == 0:
            print(f"Test {i:<13} {result['bits']:<15} {result['accuracy']:>10.2f}% {result['drop']:>18.2f}%")
        else:
            print(f"Test {i:<13} {result['bits']:<15} {result['accuracy']:>10.2f}% {result['drop']:>18.2f}%")
    
    # 第二轮：单独测试每个位（只对单个位注入故障）
    print("\n" + "="*80)
    print("第二轮：单独测试每个位（只对单个位注入故障）")
    print("="*80)
    
    individual_results = []
    for bit_idx in range(wbits):
        bit_range = (bit_idx, bit_idx)
        bits_str = f"bit{bit_idx} only"
        
        print(f"\nTest Individual {bit_idx+1}: 故障注入范围 = {bits_str}")
        print("-" * 80)
        
        # 创建BitRangeFaultInjector
        injector = BitRangeFaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            bit_range=bit_range,
            enable_statistics=False,
            whitelist_layer=args.layer
        )
        
        # 只对指定层启用故障注入
        injector.enable()
        
        # 评估
        accuracy = evaluate_model(model, test_loader, device)
        drop = accuracy_baseline - accuracy
        
        injector.disable()
        
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  相对Baseline下降: {drop:.2f}%")
        
        individual_results.append({
            'bit_idx': bit_idx,
            'bits': bits_str,
            'accuracy': accuracy,
            'drop': drop
        })
    
    print("\n" + "="*80)
    print("测试总结 - 单独测试每个位")
    print("="*80)
    print(f"{'位索引':<10} {'位范围':<20} {'准确率':<12} {'相对Baseline下降':<20}")
    print("-" * 80)
    for result in individual_results:
        print(f"bit{result['bit_idx']:<8} {result['bits']:<20} {result['accuracy']:>10.2f}% {result['drop']:>18.2f}%")
    
    print("="*80)
    print("\n分析：")
    print(f"  - Baseline准确率: {accuracy_baseline:.2f}%")
    
    # 累积测试分析
    if len(results) > 1:
        print(f"\n累积测试：")
        print(f"  - 最低准确率: {min(r['accuracy'] for r in results[1:]):.2f}% (位范围: {results[min(range(1, len(results)), key=lambda i: results[i]['accuracy'])]['bits']})")
        print(f"  - 最高准确率: {max(r['accuracy'] for r in results[1:]):.2f}% (位范围: {results[max(range(1, len(results)), key=lambda i: results[i]['accuracy'])]['bits']})")
        print(f"  - 最大下降: {max(r['drop'] for r in results[1:]):.2f}%")
        
        # 计算每增加一位的影响
        print(f"\n  每增加一位的边际影响（累积测试）：")
        for i in range(1, len(results)):
            prev_acc = results[i-1]['accuracy']
            curr_acc = results[i]['accuracy']
            impact = prev_acc - curr_acc
            print(f"    {results[i-1]['bits']:<15} → {results[i]['bits']:<15}: {impact:>6.2f}% 下降 (累计: {results[i]['drop']:.2f}%)")
    
    # 单独测试分析
    if individual_results:
        print(f"\n单独测试（每个位的独立影响）：")
        sorted_individual = sorted(individual_results, key=lambda x: x['drop'], reverse=True)
        print(f"  - 影响最大的位: bit{sorted_individual[0]['bit_idx']} (下降 {sorted_individual[0]['drop']:.2f}%)")
        print(f"  - 影响最小的位: bit{sorted_individual[-1]['bit_idx']} (下降 {sorted_individual[-1]['drop']:.2f}%)")
        
        print(f"\n  按影响大小排序：")
        for i, result in enumerate(sorted_individual):
            print(f"    {i+1}. bit{result['bit_idx']}: {result['drop']:>6.2f}% 下降 (准确率: {result['accuracy']:.2f}%)")
        
        # 分析bit7的重要性
        bit7_result = next((r for r in individual_results if r['bit_idx'] == wbits-1), None)
        if bit7_result:
            print(f"\n  bit7（最高位/符号位）单独测试：")
            print(f"    - 准确率: {bit7_result['accuracy']:.2f}%")
            print(f"    - 相对Baseline下降: {bit7_result['drop']:.2f}%")
            print(f"    - 结论: bit7的独立影响是 {bit7_result['drop']:.2f}%")
        
        # 对比累积测试中bit0-bit7的下降和单独测试bit7的下降
        cumulative_bit7_drop = results[-1]['drop'] if results else 0
        if bit7_result:
            print(f"\n  对比分析：")
            print(f"    - 累积测试（bit0-bit7）下降: {cumulative_bit7_drop:.2f}%")
            print(f"    - 单独测试（bit7 only）下降: {bit7_result['drop']:.2f}%")
            print(f"    - 差异: {cumulative_bit7_drop - bit7_result['drop']:.2f}%")
            print(f"    - 说明: 如果差异很大，说明bit0-bit6与bit7有交互影响")


if __name__ == '__main__':
    main()

