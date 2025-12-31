#!/usr/bin/env python3
"""
测试bit7翻转方向的影响

分别测试：
1. bit7从0翻转到1（负数变正数）
2. bit7从1翻转到0（正数变负数）

使用方法：
    python tools/test_bit7_flip_direction.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --ber 0.1 \
        --seed 42
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


class DirectionalBit7FaultInjector(FaultInjector):
    """
    定向bit7故障注入器
    
    只对bit7进行故障注入，并且可以指定翻转方向：
    - flip_0_to_1: 只对bit7=0的权重进行0→1翻转
    - flip_1_to_0: 只对bit7=1的权重进行1→0翻转
    """
    
    def __init__(self, *args, flip_0_to_1=True, flip_1_to_0=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip_0_to_1 = flip_0_to_1
        self.flip_1_to_0 = flip_1_to_0
    
    def _generate_directional_flip_mask(self, flat_code: torch.Tensor, k: int, device: torch.device, 
                                          layer_name=None, mask_seed=None) -> torch.Tensor:
        """
        生成定向翻转掩码
        
        只对bit7进行翻转，并且根据原始bit7的值决定是否翻转：
        - 如果原始bit7=0且flip_0_to_1=True，则翻转
        - 如果原始bit7=1且flip_1_to_0=True，则翻转
        """
        N = flat_code.numel()
        bit7_idx = k - 1  # bit7是最高位
        
        # 提取bit7的值
        flat_int64 = flat_code.to(torch.int64)
        bit7_values = ((flat_int64 >> bit7_idx) & 1).to(torch.bool)
        
        # 生成完整的翻转掩码（使用父类方法）
        full_mask = super()._generate_flip_mask(N, k, device, layer_name, mask_seed)
        
        # 创建定向掩码：只对bit7进行翻转，并且根据方向过滤
        directional_mask = torch.zeros_like(full_mask)
        
        if self.flip_0_to_1:
            # 只对bit7=0的权重进行0→1翻转
            mask_0_to_1 = full_mask[:, bit7_idx] & (~bit7_values)
            directional_mask[:, bit7_idx] = mask_0_to_1
        
        if self.flip_1_to_0:
            # 只对bit7=1的权重进行1→0翻转
            mask_1_to_0 = full_mask[:, bit7_idx] & bit7_values
            directional_mask[:, bit7_idx] = mask_1_to_0
        
        return directional_mask
    
    def _inject_on_quantized_tensor(
        self, x_q: torch.Tensor, k: int, scale: torch.Tensor, 
        layer_name=None, forward_seed=None, layer_name_for_stats=None
    ) -> torch.Tensor:
        """
        重写故障注入方法，使用定向翻转掩码
        """
        device = x_q.device if self.device is None else self.device
        
        # 判断是否使用格雷码或OLM编码
        use_gray_code = (len(self.gray_code_layers) > 0 and 
                        layer_name is not None and 
                        layer_name in self.gray_code_layers)
        use_olm = (len(self.olm_layers) > 0 and 
                  layer_name is not None and 
                  layer_name in self.olm_layers)
        
        if use_gray_code and use_olm:
            raise ValueError(f"Layer {layer_name} cannot use both Gray Code and OLM encoding")
        
        # Handle scale
        if isinstance(scale, torch.Tensor):
            s = scale.to(device)
            if s.dim() > 0 and s.numel() > 1:
                while s.dim() < x_q.dim():
                    s = s.unsqueeze(-1)
        else:
            s = torch.tensor(float(scale), device=device, dtype=x_q.dtype)
        
        # Compute quantization thresholds
        thd_neg = -(1 << (k - 1))
        thd_pos = (1 << (k - 1)) - 1
        n_levels = (1 << k) - 1
        
        # Step 1: Convert quantized value to integer code
        code_f = torch.round(x_q / s)
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        code_shifted = code_f - thd_neg  # [0, n_levels]
        
        # Determine code dtype
        code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
        code = code_shifted.to(code_dtype).clamp(0, n_levels)
        
        # Step 2: Apply Gray Code or OLM encoding if needed
        if use_gray_code:
            code = code ^ (code >> 1)
        elif use_olm:
            code_original = code_shifted + thd_neg
            value_to_code = self.olm_layers[layer_name]
            lookup_table = torch.arange(n_levels + 1, dtype=code_dtype, device=device)
            for val, enc in value_to_code.items():
                idx = val - thd_neg
                if 0 <= idx <= n_levels:
                    lookup_table[idx] = enc
            code = lookup_table[code_shifted.clamp(0, n_levels).long()]
        
        # Flatten for bit operations
        flat = code.view(-1)
        N = flat.numel()
        
        # Step 3: Generate directional flip mask (only for bit7)
        mask_seed = forward_seed if forward_seed is not None else self.seed
        flip_mask = self._generate_directional_flip_mask(flat, k, device, layer_name, mask_seed)
        
        # Statistics (if enabled)
        if self.enable_statistics:
            total_bits = N * k
            total_params = N
            stats_key = layer_name_for_stats if layer_name_for_stats is not None else (layer_name if layer_name is not None else "unknown")
            flip_mask_sum = flip_mask.sum()
            affected_params_sum = (flip_mask.sum(dim=1) > 0).sum()
            self._pending_stats.append((flip_mask_sum, total_bits, total_params, stats_key, affected_params_sum))
        
        # Step 4: Apply bit flips
        bit_positions = torch.arange(k, device=device, dtype=torch.int64)
        bit_weights = (1 << bit_positions).to(torch.int64)
        flat_int64 = flat.to(torch.int64)
        if flat_int64.device != device:
            flat_int64 = flat_int64.to(device)
        bits = ((flat_int64.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)
        flipped_bits = bits ^ flip_mask
        flat_faulted = (flipped_bits.to(torch.int64) * bit_weights).sum(-1)
        flat_faulted = flat_faulted.clamp(0, n_levels).to(code_dtype)
        
        # Step 5: Apply Gray Code or OLM decoding if needed
        if use_gray_code:
            flat_faulted = self._gray_to_binary(flat_faulted, k)
        elif use_olm:
            code_to_value = self.olm_code_to_value[layer_name]
            max_code = max(code_to_value.keys()) if code_to_value else n_levels
            reverse_lookup = torch.arange(max_code + 1, dtype=code_dtype, device=device)
            for enc, val in code_to_value.items():
                if 0 <= enc <= max_code:
                    reverse_lookup[enc] = val - thd_neg
            flat_faulted = reverse_lookup[flat_faulted.clamp(0, max_code).long()].clamp(0, n_levels)
        
        # Reshape back
        code_faulted = flat_faulted.view_as(code)
        code_f = code_faulted.to(x_q.dtype) + thd_neg
        x_faulted = code_f * s
        
        return x_faulted


def evaluate_model(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def count_bit7_distribution(model, layer_name, device):
    """统计bit7的分布（0和1的数量）"""
    module = dict(model.named_modules())[layer_name]
    if not hasattr(module, 'quan_w_fn') or module.quan_w_fn is None:
        return None, None
    
    quantizer = module.quan_w_fn
    wbits = None
    if hasattr(module, 'bits') and module.bits is not None:
        wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
    
    if wbits is None:
        return None, None
    
    if isinstance(wbits, torch.Tensor):
        wbits = int(wbits.item())
    else:
        wbits = int(wbits)
    
    bit7_idx = wbits - 1
    
    with torch.no_grad():
        scale = quantizer.get_scale(wbits, detach=True)
        weight_q = quantizer(module.weight, wbits, is_activation=False)
        
        if isinstance(scale, torch.Tensor):
            if scale.dim() > 0 and scale.numel() > 1:
                while scale.dim() < weight_q.dim():
                    scale = scale.unsqueeze(-1)
                code_f = torch.round(weight_q / scale)
            else:
                code_f = torch.round(weight_q / scale.item())
        else:
            code_f = torch.round(weight_q / scale)
        
        thd_neg = -(1 << (wbits - 1))
        thd_pos = (1 << (wbits - 1)) - 1
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        code_shifted = code_f - thd_neg
        
        code = code_shifted.int().cpu()
        flat_code = code.view(-1)
        
        # 提取bit7
        bit7_values = ((flat_code.to(torch.int64) >> bit7_idx) & 1)
        count_0 = (bit7_values == 0).sum().item()
        count_1 = (bit7_values == 1).sum().item()
        
        return count_0, count_1


def main():
    parser = argparse.ArgumentParser(description='Test bit7 flip direction impact')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, required=True, help='Layer name')
    parser.add_argument('--ber', type=float, default=0.1, help='Bit error rate')
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
    
    print("="*80)
    print("Bit7翻转方向影响测试")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"测试层: {args.layer}")
    print(f"BER: {args.ber}")
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
    
    # 统计bit7分布
    count_0, count_1 = count_bit7_distribution(model, args.layer, device)
    if count_0 is not None and count_1 is not None:
        total = count_0 + count_1
        print(f"Bit7分布统计（层 {args.layer}）:")
        print(f"  bit7=0 (负数)的数量: {count_0} ({100.*count_0/total:.2f}%)")
        print(f"  bit7=1 (正数)的数量: {count_1} ({100.*count_1/total:.2f}%)")
        print()
    
    # 评估Baseline
    print("Test 0: Baseline (无故障注入)")
    print("-" * 80)
    accuracy_baseline = evaluate_model(model, test_loader, device)
    print(f"  准确率: {accuracy_baseline:.2f}%")
    print()
    
    results = []
    
    # Test 1: 只测试bit7从0翻转到1（负数变正数）
    print("="*80)
    print("Test 1: Bit7 0→1 翻转（负数变正数）")
    print("="*80)
    injector_0_to_1 = DirectionalBit7FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        flip_0_to_1=True,
        flip_1_to_0=False,
        enable_statistics=False,
        whitelist_layer=args.layer
    )
    
    injector_0_to_1.enable()
    accuracy_0_to_1 = evaluate_model(model, test_loader, device)
    injector_0_to_1.disable()
    
    drop_0_to_1 = accuracy_baseline - accuracy_0_to_1
    print(f"  准确率: {accuracy_0_to_1:.2f}%")
    print(f"  相对Baseline下降: {drop_0_to_1:.2f}%")
    print()
    
    results.append({
        'test': 'Bit7 0→1',
        'accuracy': accuracy_0_to_1,
        'drop': drop_0_to_1,
        'description': '负数变正数'
    })
    
    # Test 2: 只测试bit7从1翻转到0（正数变负数）
    print("="*80)
    print("Test 2: Bit7 1→0 翻转（正数变负数）")
    print("="*80)
    injector_1_to_0 = DirectionalBit7FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        flip_0_to_1=False,
        flip_1_to_0=True,
        enable_statistics=False,
        whitelist_layer=args.layer
    )
    
    injector_1_to_0.enable()
    accuracy_1_to_0 = evaluate_model(model, test_loader, device)
    injector_1_to_0.disable()
    
    drop_1_to_0 = accuracy_baseline - accuracy_1_to_0
    print(f"  准确率: {accuracy_1_to_0:.2f}%")
    print(f"  相对Baseline下降: {drop_1_to_0:.2f}%")
    print()
    
    results.append({
        'test': 'Bit7 1→0',
        'accuracy': accuracy_1_to_0,
        'drop': drop_1_to_0,
        'description': '正数变负数'
    })
    
    # Test 3: 双向翻转（作为对比）
    print("="*80)
    print("Test 3: Bit7 双向翻转（0→1 和 1→0）")
    print("="*80)
    injector_both = DirectionalBit7FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        flip_0_to_1=True,
        flip_1_to_0=True,
        enable_statistics=False,
        whitelist_layer=args.layer
    )
    
    injector_both.enable()
    accuracy_both = evaluate_model(model, test_loader, device)
    injector_both.disable()
    
    drop_both = accuracy_baseline - accuracy_both
    print(f"  准确率: {accuracy_both:.2f}%")
    print(f"  相对Baseline下降: {drop_both:.2f}%")
    print()
    
    results.append({
        'test': 'Bit7 双向',
        'accuracy': accuracy_both,
        'drop': drop_both,
        'description': '0→1 和 1→0'
    })
    
    # 打印总结
    print("="*80)
    print("测试总结")
    print("="*80)
    print(f"{'测试':<20} {'描述':<20} {'准确率':<12} {'相对Baseline下降':<20}")
    print("-" * 80)
    print(f"{'Baseline':<20} {'无故障注入':<20} {accuracy_baseline:>10.2f}% {'0.00%':>18}")
    for result in results:
        print(f"{result['test']:<20} {result['description']:<20} {result['accuracy']:>10.2f}% {result['drop']:>18.2f}%")
    
    print()
    print("="*80)
    print("分析")
    print("="*80)
    
    if count_0 is not None and count_1 is not None:
        print(f"Bit7分布:")
        print(f"  bit7=0 (负数): {count_0} ({100.*count_0/total:.2f}%)")
        print(f"  bit7=1 (正数): {count_1} ({100.*count_1/total:.2f}%)")
        print()
    
    print(f"翻转方向影响:")
    print(f"  Bit7 0→1 (负数变正数): {drop_0_to_1:.2f}% 下降")
    print(f"  Bit7 1→0 (正数变负数): {drop_1_to_0:.2f}% 下降")
    print(f"  双向翻转: {drop_both:.2f}% 下降")
    print()
    
    if abs(drop_0_to_1 - drop_1_to_0) > 1.0:
        if drop_0_to_1 > drop_1_to_0:
            print(f"  ✅ 结论: Bit7 0→1 翻转影响更大（{drop_0_to_1 - drop_1_to_0:.2f}%差异）")
            print(f"     说明: 负数变正数对模型性能影响更大")
        else:
            print(f"  ✅ 结论: Bit7 1→0 翻转影响更大（{drop_1_to_0 - drop_0_to_1:.2f}%差异）")
            print(f"     说明: 正数变负数对模型性能影响更大")
    else:
        print(f"  ✅ 结论: Bit7两种翻转方向的影响相近（差异<1%）")
        print(f"     说明: 符号位翻转的影响与方向无关")


if __name__ == '__main__':
    main()


