#!/usr/bin/env python3
"""
对比新旧fault_injector版本对features.0层的处理差异
确保使用相同的故障注入bit索引，对比原始值、编码值、解码值等
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config


def get_quantized_tensor(model, layer_name, device):
    """获取指定层的量化tensor（用于故障注入）"""
    module = dict(model.named_modules())[layer_name]
    if not hasattr(module, 'quan_w_fn') or module.quan_w_fn is None:
        raise ValueError(f"Layer {layer_name} has no quantization function")
    
    quantizer = module.quan_w_fn
    
    # 检查位宽配置（支持bits和fixed_bits）
    wbits = None
    if hasattr(module, 'bits') and module.bits is not None:
        wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
    
    if wbits is None:
        raise ValueError(f"Layer {layer_name} has no bit-width configuration (neither bits nor fixed_bits)")
    
    # 递归处理嵌套的列表/元组
    while isinstance(wbits, (list, tuple)) and len(wbits) > 0:
        wbits = wbits[0]
    
    if isinstance(wbits, torch.Tensor):
        wbits = int(wbits.item())
    else:
        wbits = int(wbits)
    
    with torch.no_grad():
        weight_q = quantizer(module.weight, wbits, is_activation=False)
    
    return weight_q, wbits


def simulate_fault_injection_old(weight_q, wbits, scale, value_to_code, code_to_value, device, seed, layer_name):
    """模拟旧版本的故障注入过程"""
    thd_neg = -(1 << (wbits - 1))
    thd_pos = (1 << (wbits - 1)) - 1
    n_levels = (1 << wbits) - 1
    
    # Step 1: 计算量化值
    if isinstance(scale, torch.Tensor):
        if scale.dim() > 0 and scale.numel() > 1:
            while scale.dim() < weight_q.dim():
                scale = scale.unsqueeze(-1)
            code_f = torch.round(weight_q.to(device) / scale)
        else:
            code_f = torch.round(weight_q.to(device) / scale.item())
    else:
        code_f = torch.round(weight_q.to(device) / scale)
    
    code_f = torch.clamp(code_f, thd_neg, thd_pos)
    
    # Shift to non-negative range [0, 2^k-1] for bit operations
    code_shifted = code_f - thd_neg  # Now in [0, 2^k-1]
    
    # Use compact integer dtype for efficiency (与fault_injector保持一致)
    code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
    code = code_shifted.to(code_dtype).clamp(0, n_levels)
    
    # Step 2: OLM编码
    lookup_table = torch.arange(n_levels + 1, dtype=code_dtype, device=device)
    for val, enc in value_to_code.items():
        idx = val - thd_neg
        if 0 <= idx <= n_levels:
            lookup_table[idx] = enc
    encoded = lookup_table[code]  # code已经是code_dtype，可以直接索引
    
    # Step 3: 故障注入（使用固定seed生成flip_mask）
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    flat = encoded.view(-1)
    N = flat.numel()
    k = wbits
    
    # 生成flip_mask
    p = 0.1
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    flip_mask = torch.rand((N, k), generator=generator, device=device) < p
    
    # 位翻转
    bit_positions = torch.arange(k, device=device, dtype=torch.int64)
    bit_weights = (1 << bit_positions).to(torch.int64)
    flat_int64 = flat.to(torch.int64)
    bits = ((flat_int64.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)
    flipped_bits = bits ^ flip_mask
    flat_faulted = (flipped_bits.to(torch.int64) * bit_weights).sum(-1)
    flat_faulted = flat_faulted.clamp(0, n_levels).to(code_dtype)
    encoded_faulted = flat_faulted.view_as(encoded)
    
    # Step 4: OLM解码（旧版本）
    max_code = max(code_to_value.keys()) if code_to_value else n_levels
    reverse_lookup = torch.arange(max_code + 1, dtype=code_dtype, device=device)
    for enc, val in code_to_value.items():
        if 0 <= enc <= max_code:
            reverse_lookup[enc] = val - thd_neg
    # encoded_faulted已经是code_dtype，可以直接索引（与fault_injector保持一致）
    decoded_faulted = reverse_lookup[encoded_faulted.clamp(0, max_code)].clamp(0, n_levels)
    
    # 转换回量化值
    code_faulted_shifted = decoded_faulted.to(weight_q.dtype) + thd_neg
    weight_q_faulted = code_faulted_shifted * scale
    
    return {
        'original_quantized': code_int.cpu(),
        'encoded': encoded.cpu(),
        'decoded': reverse_lookup[encoded.clamp(0, max_code).long()].clamp(0, n_levels).cpu(),
        'encoded_faulted': encoded_faulted.cpu(),
        'decoded_faulted': decoded_faulted.cpu(),
        'faulted_quantized': code_faulted_shifted.cpu().int(),
        'flip_mask': flip_mask.cpu()
    }


def simulate_fault_injection_new(weight_q, wbits, scale, value_to_code, code_to_value, device, seed, layer_name):
    """模拟新版本的故障注入过程"""
    thd_neg = -(1 << (wbits - 1))
    thd_pos = (1 << (wbits - 1)) - 1
    n_levels = (1 << wbits) - 1
    
    # Step 1: 计算量化值
    if isinstance(scale, torch.Tensor):
        if scale.dim() > 0 and scale.numel() > 1:
            while scale.dim() < weight_q.dim():
                scale = scale.unsqueeze(-1)
            code_f = torch.round(weight_q.to(device) / scale)
        else:
            code_f = torch.round(weight_q.to(device) / scale.item())
    else:
        code_f = torch.round(weight_q.to(device) / scale)
    
    code_f = torch.clamp(code_f, thd_neg, thd_pos)
    code_int = code_f.int()
    code_shifted = code_int - thd_neg
    
    # Step 2: OLM编码（新版本）
    code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
    lookup_table = torch.arange(n_levels + 1, dtype=code_dtype, device=device)
    for val, enc in value_to_code.items():
        idx = val - thd_neg
        if 0 <= idx <= n_levels:
            lookup_table[idx] = enc
    code_shifted_clamped = code_shifted.clamp(0, n_levels).to(torch.long)
    encoded = lookup_table[code_shifted_clamped].to(code_dtype)
    
    # Step 3: 故障注入（使用相同的seed和flip_mask）
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    flat = encoded.view(-1)
    N = flat.numel()
    k = wbits
    
    # 生成flip_mask（与旧版本相同）
    p = 0.1
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    flip_mask = torch.rand((N, k), generator=generator, device=device) < p
    
    # 位翻转
    bit_positions = torch.arange(k, device=device, dtype=torch.int64)
    bit_weights = (1 << bit_positions).to(torch.int64)
    flat_int64 = flat.to(torch.int64)
    bits = ((flat_int64.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)
    flipped_bits = bits ^ flip_mask
    flat_faulted = (flipped_bits.to(torch.int64) * bit_weights).sum(-1)
    flat_faulted = flat_faulted.clamp(0, n_levels).to(code_dtype)
    encoded_faulted = flat_faulted.view_as(encoded)
    
    # Step 4: OLM解码（新版本）
    # 推断位宽
    if code_to_value:
        max_code_in_mapping = max(int(enc) for enc in code_to_value.keys())
        inferred_k = 0
        inferred_n_levels = 1
        while inferred_n_levels <= max_code_in_mapping:
            inferred_k += 1
            inferred_n_levels = 1 << inferred_k
        olm_n_levels = inferred_n_levels
    else:
        olm_n_levels = n_levels
    
    # 创建reverse_lookup（新版本的填充逻辑）
    reverse_lookup = torch.arange(olm_n_levels, dtype=code_dtype, device=device)
    for enc, val in code_to_value.items():
        enc_int = int(enc)
        if 0 <= enc_int < olm_n_levels:
            val_int = int(val)
            val_shifted = val_int - thd_neg
            val_shifted_clamped = max(0, min(n_levels - 1, val_shifted))  # 新版本的clamp
            reverse_lookup[enc_int] = val_shifted_clamped
    
    # 解码
    flat_faulted_clamped = encoded_faulted.view(-1).clamp(0, olm_n_levels - 1).long()
    decoded_faulted = reverse_lookup[flat_faulted_clamped].clamp(0, n_levels - 1)  # 新版本的clamp
    decoded_faulted = decoded_faulted.view_as(encoded_faulted)
    
    # 转换回量化值
    code_faulted_shifted = decoded_faulted.to(weight_q.dtype) + thd_neg
    weight_q_faulted = code_faulted_shifted * scale
    
    return {
        'original_quantized': code_int.cpu(),
        'encoded': encoded.cpu(),
        'decoded': reverse_lookup[encoded.view(-1).clamp(0, olm_n_levels - 1).long()].clamp(0, n_levels - 1).view_as(encoded).cpu(),
        'encoded_faulted': encoded_faulted.cpu(),
        'decoded_faulted': decoded_faulted.cpu(),
        'faulted_quantized': code_faulted_shifted.cpu().int(),
        'flip_mask': flip_mask.cpu()
    }


def main():
    parser = argparse.ArgumentParser(description='Compare old and new fault_injector')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--olm_json', type=str, required=True, help='OLM encoding JSON file')
    parser.add_argument('--layer', type=str, default='features.0', help='Layer to compare')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for fault injection')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of weight values to compare')
    
    args = parser.parse_args()
    
    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载OLM映射
    with open(args.olm_json, 'r') as f:
        olm_data = json.load(f)
    
    if 'layer_mappings' in olm_data:
        olm_layers_dict = {}
        for layer_name, layer_data in olm_data['layer_mappings'].items():
            olm_layers_dict[layer_name] = layer_data['value_to_code']
    else:
        olm_layers_dict = {args.layer: olm_data['value_to_code']}
    
    if args.layer not in olm_layers_dict:
        raise ValueError(f"Layer {args.layer} not found in OLM mappings")
    
    # 创建模型
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    else:
        # 验证时：fixed层使用8-bit，dynamic层使用6-bit
        from util.mpq import switch_bit_width
        target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
        max_bit = max(target_bits) if target_bits else 6
        print(f"未提供bit_width_config，使用target_bits的最大值: {max_bit}-bit (dynamic层)")
        print(f"注意: fixed_bits层（features.0, classifier.6）将保持8-bit")
        switch_bit_width(model, quan_scheduler=config.quan, wbit=max_bit, abits=max_bit)
        
        # 确保fixed层使用8-bit
        for name, module in model.named_modules():
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                # fixed层已经配置为8-bit，不需要修改
                pass
    
    # 加载checkpoint
    load_checkpoint(model, args.ckpt, model_device=device, lean=True)
    model.eval()
    
    # 获取量化tensor和OLM映射
    weight_q, wbits = get_quantized_tensor(model, args.layer, device)
    module = dict(model.named_modules())[args.layer]
    quantizer = module.quan_w_fn
    scale = quantizer.get_scale(wbits, detach=True)
    
    value_to_code_raw = olm_layers_dict[args.layer]
    # 确保键和值都是整数类型
    value_to_code = {int(k): int(v) for k, v in value_to_code_raw.items()}
    code_to_value = {int(code): int(value) for value, code in value_to_code.items()}
    
    # 使用旧版本处理
    print("="*80)
    print("使用旧版本fault_injector处理...")
    print("="*80)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    result_old = simulate_fault_injection_old(weight_q, wbits, scale, value_to_code, code_to_value, device, args.seed, args.layer)
    
    # 使用新版本处理
    print("="*80)
    print("使用新版本fault_injector处理...")
    print("="*80)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    result_new = simulate_fault_injection_new(weight_q, wbits, scale, value_to_code, code_to_value, device, args.seed, args.layer)
    
    # 验证flip_mask是否相同
    flip_mask_old = result_old['flip_mask']
    flip_mask_new = result_new['flip_mask']
    if not torch.equal(flip_mask_old, flip_mask_new):
        print("⚠️  警告：新旧版本的flip_mask不一致！")
        print(f"  差异数量: {(flip_mask_old != flip_mask_new).sum().item()}")
    else:
        print("✓ flip_mask一致，故障注入bit索引相同")
    
    # 对比结果
    print("="*80)
    print(f"对比结果 - 层: {args.layer}")
    print("="*80)
    
    # 展平所有tensor以便对比
    orig_old = result_old['original_quantized'].view(-1)
    orig_new = result_new['original_quantized'].view(-1)
    encoded_old = result_old['encoded'].view(-1)
    encoded_new = result_new['encoded'].view(-1)
    decoded_old = result_old['decoded'].view(-1)
    decoded_new = result_new['decoded'].view(-1)
    faulted_old = result_old['faulted_quantized'].view(-1)
    faulted_new = result_new['faulted_quantized'].view(-1)
    encoded_faulted_old = result_old['encoded_faulted'].view(-1)
    encoded_faulted_new = result_new['encoded_faulted'].view(-1)
    decoded_faulted_old = result_old['decoded_faulted'].view(-1)
    decoded_faulted_new = result_new['decoded_faulted'].view(-1)
    
    # 选择前num_samples个值进行对比
    num_samples = min(args.num_samples, orig_old.numel())
    indices = torch.randperm(orig_old.numel())[:num_samples]
    
    print(f"\n对比前 {num_samples} 个权重值（随机采样）:")
    print("-" * 200)
    header = f"{'索引':<8} {'原始量化值':<14} {'编码值(旧)':<14} {'编码值(新)':<14} {'解码值(旧)':<14} {'解码值(新)':<14} "
    header += f"{'故障后量化值(旧)':<20} {'故障后量化值(新)':<20} {'故障后编码值(旧)':<20} {'故障后编码值(新)':<20} "
    header += f"{'故障后解码值(旧)':<20} {'故障后解码值(新)':<20}"
    print(header)
    print("-" * 200)
    
    differences = []
    for idx in indices:
        i = idx.item()
        orig_val = orig_old[i].item()
        enc_old = encoded_old[i].item()
        enc_new = encoded_new[i].item()
        dec_old = decoded_old[i].item()
        dec_new = decoded_new[i].item()
        fault_old = faulted_old[i].item()
        fault_new = faulted_new[i].item()
        enc_fault_old = encoded_faulted_old[i].item()
        enc_fault_new = encoded_faulted_new[i].item()
        dec_fault_old = decoded_faulted_old[i].item()
        dec_fault_new = decoded_faulted_new[i].item()
        
        # 检查是否有差异
        if enc_old != enc_new or dec_old != dec_new or fault_old != fault_new or \
           enc_fault_old != enc_fault_new or dec_fault_old != dec_fault_new:
            differences.append(i)
            marker = " ***"
        else:
            marker = ""
        
        print(f"{i:<8} {orig_val:<14.0f} {enc_old:<14.0f} {enc_new:<14.0f} {dec_old:<14.0f} {dec_new:<14.0f} "
              f"{fault_old:<20.0f} {fault_new:<20.0f} {enc_fault_old:<20.0f} {enc_fault_new:<20.0f} "
              f"{dec_fault_old:<20.0f} {dec_fault_new:<20.0f}{marker}")
    
    print("-" * 200)
    print(f"\n发现 {len(differences)} 个值存在差异（标记为 ***）")
    
    # 统计差异
    enc_diff = (encoded_old != encoded_new).sum().item()
    dec_diff = (decoded_old != decoded_new).sum().item()
    fault_diff = (faulted_old != faulted_new).sum().item()
    enc_fault_diff = (encoded_faulted_old != encoded_faulted_new).sum().item()
    dec_fault_diff = (decoded_faulted_old != decoded_faulted_new).sum().item()
    
    print(f"\n总体差异统计:")
    print(f"  编码值差异: {enc_diff} / {orig_old.numel()} ({100*enc_diff/orig_old.numel():.2f}%)")
    print(f"  解码值差异: {dec_diff} / {orig_old.numel()} ({100*dec_diff/orig_old.numel():.2f}%)")
    print(f"  故障后量化值差异: {fault_diff} / {orig_old.numel()} ({100*fault_diff/orig_old.numel():.2f}%)")
    print(f"  故障后编码值差异: {enc_fault_diff} / {orig_old.numel()} ({100*enc_fault_diff/orig_old.numel():.2f}%)")
    print(f"  故障后解码值差异: {dec_fault_diff} / {orig_old.numel()} ({100*dec_fault_diff/orig_old.numel():.2f}%)")
    
    # 如果有差异，显示一些具体的差异示例
    if differences:
        print(f"\n差异示例（前10个）:")
        print("-" * 200)
        print(header)
        print("-" * 200)
        for i in differences[:10]:
            orig_val = orig_old[i].item()
            enc_old = encoded_old[i].item()
            enc_new = encoded_new[i].item()
            dec_old = decoded_old[i].item()
            dec_new = decoded_new[i].item()
            fault_old = faulted_old[i].item()
            fault_new = faulted_new[i].item()
            enc_fault_old = encoded_faulted_old[i].item()
            enc_fault_new = encoded_faulted_new[i].item()
            dec_fault_old = decoded_faulted_old[i].item()
            dec_fault_new = decoded_faulted_new[i].item()
            
            print(f"{i:<8} {orig_val:<14.0f} {enc_old:<14.0f} {enc_new:<14.0f} {dec_old:<14.0f} {dec_new:<14.0f} "
                  f"{fault_old:<20.0f} {fault_new:<20.0f} {enc_fault_old:<20.0f} {enc_fault_new:<20.0f} "
                  f"{dec_fault_old:<20.0f} {dec_fault_new:<20.0f} ***")


if __name__ == '__main__':
    main()
