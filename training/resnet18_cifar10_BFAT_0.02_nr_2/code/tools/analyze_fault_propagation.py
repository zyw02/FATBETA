"""
分析 SEU 故障从权重到激活值的传播机制
解释为什么会出现稀疏错误而不是整通道错误
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from util.checkpoint import load_checkpoint
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.config import get_config
from util.data_loader import init_dataloader
from util.utils import preprocess_model
from quan import find_modules_to_quantize, replace_module_by_names


def analyze_weight_fault_impact(model, layer_name, data, ber, device):
    """分析权重故障对输出的影响"""
    model.eval()
    
    # 获取目标层
    modules = dict(model.named_modules())
    target_module = modules.get(layer_name)
    if target_module is None:
        print(f"Layer {layer_name} not found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing {layer_name}")
    print(f"{'='*60}")
    
    # 1. 获取 clean 权重
    with torch.no_grad():
        # 获取量化后的权重
        if hasattr(target_module, 'quan_w_fn') and target_module.quan_w_fn is not None:
            wbits = target_module.bits[0] if target_module.bits else 8
            weight_clean = target_module.quan_w_fn(target_module.weight, wbits, is_activation=False)
        else:
            weight_clean = target_module.weight
        
        print(f"\n1. Weight Statistics:")
        print(f"   Shape: {weight_clean.shape}")
        print(f"   Mean: {weight_clean.mean().item():.6f}")
        print(f"   Std: {weight_clean.std().item():.6f}")
        print(f"   Min: {weight_clean.min().item():.6f}")
        print(f"   Max: {weight_clean.max().item():.6f}")
        
        # 2. 计算故障注入后的权重
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=ber,
            device=device,
            enable_in_inference=True,
            seed=42,
            skip_first_last=False,
        )
        injector.enable()
        
        # 重新获取权重（这次会经过故障注入）
        if hasattr(target_module, 'quan_w_fn') and target_module.quan_w_fn is not None:
            wbits = target_module.bits[0] if target_module.bits else 8
            weight_fault = target_module.quan_w_fn(target_module.weight, wbits, is_activation=False)
        else:
            weight_fault = target_module.weight
        
        injector.disable()
        
        # 3. 计算权重差异
        weight_diff = (weight_fault - weight_clean).abs()
        num_flipped_weights = (weight_diff > 1e-6).sum().item()
        total_weights = weight_clean.numel()
        flip_ratio = num_flipped_weights / total_weights
        
        print(f"\n2. Weight Fault Statistics (BER={ber}):")
        print(f"   Total weights: {total_weights}")
        print(f"   Flipped weights: {num_flipped_weights} ({100*flip_ratio:.2f}%)")
        print(f"   Max weight change: {weight_diff.max().item():.6f}")
        print(f"   Mean weight change: {weight_diff.mean().item():.6f}")
        print(f"   Mean change (only flipped): {weight_diff[weight_diff > 1e-6].mean().item():.6f}")
        
        # 4. 分析每个输出通道受影响的权重数量
        if len(weight_clean.shape) == 4:  # Conv2d: [out_channels, in_channels, H, W]
            out_channels, in_channels, kh, kw = weight_clean.shape
            print(f"\n3. Per-Output-Channel Analysis:")
            print(f"   Weight shape: [out_channels={out_channels}, in_channels={in_channels}, kernel={kh}x{kw}]")
            
            # 每个输出通道的权重数量
            weights_per_channel = in_channels * kh * kw
            
            # 每个输出通道有多少权重被翻转
            weight_diff_reshaped = weight_diff.view(out_channels, -1)  # [out_channels, in_channels*kh*kw]
            flipped_per_channel = (weight_diff_reshaped > 1e-6).sum(dim=1)  # [out_channels]
            flip_ratio_per_channel = flipped_per_channel.float() / weights_per_channel
            
            print(f"   Weights per output channel: {weights_per_channel}")
            print(f"   Flipped weights per channel:")
            print(f"     Min: {flipped_per_channel.min().item()}, Max: {flipped_per_channel.max().item()}")
            print(f"     Mean: {flipped_per_channel.float().mean().item():.2f}")
            print(f"     Channels with >0 flips: {(flipped_per_channel > 0).sum().item()} / {out_channels}")
            print(f"     Channels with >10% flips: {(flip_ratio_per_channel > 0.1).sum().item()} / {out_channels}")
            print(f"     Channels with >50% flips: {(flip_ratio_per_channel > 0.5).sum().item()} / {out_channels}")
        
        # 5. 分析故障对输出的影响
        print(f"\n4. Output Impact Analysis:")
        
        # 获取输入激活值
        input_acts = {}
        def input_hook(name):
            def hook(module, input, output):
                input_acts[name] = input[0].detach().clone()
            return hook
        
        input_hook_handle = target_module.register_forward_hook(input_hook(layer_name))
        
        # Clean forward
        with torch.no_grad():
            _ = model(data)
        input_act = input_acts[layer_name]
        
        print(f"   Input activation shape: {input_act.shape}")
        print(f"   Input activation stats: mean={input_act.mean().item():.6f}, std={input_act.std().item():.6f}")
        
        # 计算 clean 输出
        if isinstance(target_module, nn.Conv2d) or hasattr(target_module, '__class__') and 'Conv' in target_module.__class__.__name__:
            output_clean = F.conv2d(input_act, weight_clean, 
                                  stride=target_module.stride, 
                                  padding=target_module.padding,
                                  groups=target_module.groups if hasattr(target_module, 'groups') else 1)
        else:
            output_clean = F.linear(input_act, weight_clean)
        
        # 计算 fault 输出
        output_fault = F.conv2d(input_act, weight_fault,
                               stride=target_module.stride,
                               padding=target_module.padding,
                               groups=target_module.groups if hasattr(target_module, 'groups') else 1) if len(weight_clean.shape) == 4 else F.linear(input_act, weight_fault)
        
        output_diff = (output_fault - output_clean).abs()
        
        print(f"   Output shape: {output_clean.shape}")
        print(f"   Output diff stats:")
        print(f"     Max: {output_diff.max().item():.6f}")
        print(f"     Mean: {output_diff.mean().item():.6f}")
        print(f"     Std: {output_diff.std().item():.6f}")
        
        # 6. 分析每个输出通道的错误
        if len(output_clean.shape) == 4:  # [B, C, H, W]
            B, C, H, W = output_clean.shape
            output_diff_per_channel = output_diff.mean(dim=(0, 2, 3))  # [C] - 每个通道的平均错误
            
            print(f"\n5. Per-Output-Channel Error:")
            print(f"   Output channels: {C}")
            print(f"   Mean error per channel:")
            print(f"     Min: {output_diff_per_channel.min().item():.6f}")
            print(f"     Max: {output_diff_per_channel.max().item():.6f}")
            print(f"     Mean: {output_diff_per_channel.mean().item():.6f}")
            print(f"     Channels with error > threshold: {(output_diff_per_channel > output_diff_per_channel.mean()).sum().item()} / {C}")
            
            # 7. 分析空间错误分布
            output_diff_spatial = output_diff.mean(dim=1)  # [B, H, W] - 跨通道平均
            spatial_error_mean = output_diff_spatial.mean().item()
            spatial_error_std = output_diff_spatial.std().item()
            
            # 计算空间错误的一致性（如果整通道错误，空间错误应该比较均匀）
            spatial_error_flat = output_diff_spatial[0].cpu().numpy().flatten()
            spatial_error_cv = spatial_error_std / (spatial_error_mean + 1e-8)  # 变异系数
            
            print(f"\n6. Spatial Error Distribution:")
            print(f"   Mean spatial error: {spatial_error_mean:.6f}")
            print(f"   Std spatial error: {spatial_error_std:.6f}")
            print(f"   Coefficient of Variation (CV): {spatial_error_cv:.4f}")
            print(f"     (CV < 0.5: uniform/channel-wide, CV > 1.0: sparse/point-wise)")
            
            # 8. 关键发现
            print(f"\n7. Key Findings:")
            if flip_ratio < 0.05:
                print(f"   ✓ Low weight flip ratio ({100*flip_ratio:.2f}%) → Only few weights affected")
            else:
                print(f"   ⚠ High weight flip ratio ({100*flip_ratio:.2f}%) → Many weights affected")
            
            if spatial_error_cv > 1.0:
                print(f"   ✓ High spatial CV ({spatial_error_cv:.2f}) → SPARSE point-wise errors")
                print(f"     Reason: Errors concentrated in specific spatial locations")
            elif spatial_error_cv < 0.5:
                print(f"   ⚠ Low spatial CV ({spatial_error_cv:.2f}) → UNIFORM channel-wide errors")
                print(f"     Reason: Errors distributed evenly across spatial locations")
            else:
                print(f"   → Medium spatial CV ({spatial_error_cv:.2f}) → MIXED pattern")
            
            # 分析为什么会出现稀疏错误
            print(f"\n8. Why Sparse Errors?")
            print(f"   a) Low BER ({ber}) → Only {100*flip_ratio:.2f}% of weights flipped")
            print(f"   b) Each flipped weight affects output through convolution")
            print(f"   c) But input activation distribution may be sparse/uneven")
            print(f"   d) Quantization may mask some errors")
            print(f"   e) Spatial error CV = {spatial_error_cv:.2f} suggests {'sparse' if spatial_error_cv > 1.0 else 'uniform'} pattern")
        
        input_hook_handle.remove()


def main():
    parser = argparse.ArgumentParser(description='Analyze fault propagation from weights to activations')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--layer', type=str, default='features.0')
    parser.add_argument('--ber', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--bit_width_config', type=str, default=None)
    parser.add_argument('--config_index', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    import sys as sys_module
    original_argv = sys_module.argv.copy()
    sys_module.argv = ['analyze_fault_propagation.py', args.config]
    try:
        configs = get_config(args.config)
    finally:
        sys_module.argv = original_argv
    
    if not hasattr(configs, 'local_rank'):
        configs.local_rank = 0
    if not hasattr(configs, 'world_size'):
        configs.world_size = 1
    if not hasattr(configs, 'rank'):
        configs.rank = 0
    
    model = create_model(configs.arch, dataset=configs.dataloader.dataset)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    model.eval()
    
    load_checkpoint(model, args.ckpt, model_device=str(device), strict=False)
    
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, args.config_index)
    
    # 获取一批数据
    _, _, test_loader, _, _ = init_dataloader(configs.dataloader, configs.arch)
    data, _ = next(iter(test_loader))
    data = data.to(device)
    
    # 分析
    analyze_weight_fault_impact(model, args.layer, data, args.ber, device)


if __name__ == '__main__':
    main()


