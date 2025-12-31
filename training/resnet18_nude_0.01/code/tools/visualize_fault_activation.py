"""
可视化故障注入对激活值的影响
对比 clean 和 fault 模式下 features.0 的激活值
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from util.checkpoint import load_checkpoint
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.config import get_config
from util.data_loader import init_dataloader
from util.utils import preprocess_model
from quan import find_modules_to_quantize, replace_module_by_names
from quan.func import QuanConv2d, QuanLinear


def register_activation_hook(model, layer_name, activations_dict):
    """注册 hook 提取指定层的激活值"""
    def make_hook(name):
        def hook(module, input, output):
            activations_dict[name] = output.detach().clone()
        return hook
    
    modules = dict(model.named_modules())
    if layer_name in modules:
        hook = modules[layer_name].register_forward_hook(make_hook(layer_name))
        return hook
    return None


def visualize_activation_comparison(
    clean_acts: torch.Tensor,
    fault_acts: torch.Tensor,
    output_dir: Path,
    layer_name: str,
    sample_idx: int = 0,
):
    """
    可视化 clean 和 fault 激活值的对比
    
    Args:
        clean_acts: Clean 激活值 [B, C, H, W]
        fault_acts: Fault 激活值 [B, C, H, W]
        output_dir: 输出目录
        layer_name: 层名称
        sample_idx: 样本索引
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择第一个样本
    clean = clean_acts[sample_idx].cpu().numpy()  # [C, H, W]
    fault = fault_acts[sample_idx].cpu().numpy()   # [C, H, W]
    diff = fault - clean  # [C, H, W]
    
    C, H, W = clean.shape
    
    # 1. 统计信息对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{layer_name} Activation Statistics Comparison (Sample {sample_idx})', fontsize=14)
    
    # 1.1 通道均值对比
    clean_mean = clean.mean(axis=(1, 2))  # [C]
    fault_mean = fault.mean(axis=(1, 2))  # [C]
    axes[0, 0].plot(clean_mean, label='Clean', alpha=0.7)
    axes[0, 0].plot(fault_mean, label='Fault', alpha=0.7)
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Mean Activation')
    axes[0, 0].set_title('Channel Mean Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 1.2 通道标准差对比
    clean_std = clean.std(axis=(1, 2))  # [C]
    fault_std = fault.std(axis=(1, 2))  # [C]
    axes[0, 1].plot(clean_std, label='Clean', alpha=0.7)
    axes[0, 1].plot(fault_std, label='Fault', alpha=0.7)
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Std Activation')
    axes[0, 1].set_title('Channel Std Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 1.3 激活值分布直方图（所有通道）
    clean_flat = clean.flatten()
    fault_flat = fault.flatten()
    axes[1, 0].hist(clean_flat, bins=50, alpha=0.5, label='Clean', density=True)
    axes[1, 0].hist(fault_flat, bins=50, alpha=0.5, label='Fault', density=True)
    axes[1, 0].set_xlabel('Activation Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Activation Value Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 1.4 差值统计
    diff_flat = diff.flatten()
    axes[1, 1].hist(diff_flat, bins=50, alpha=0.7, color='red')
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Difference (Fault - Clean)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Difference Distribution\n(Mean={diff_flat.mean():.4f}, Std={diff_flat.std():.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = output_dir / f'{layer_name}_stats_comparison.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved statistics comparison to {stats_path}")
    
    # 2. 特征图热力图对比（选择前 16 个通道）
    num_channels_to_show = min(16, C)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'{layer_name} Feature Map Comparison (First {num_channels_to_show} Channels, Sample {sample_idx})', fontsize=14)
    
    for i in range(num_channels_to_show):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 显示 clean 和 fault 的差值
        diff_channel = diff[i]
        im = ax.imshow(diff_channel, cmap='RdBu_r', vmin=-np.abs(diff_channel).max(), vmax=np.abs(diff_channel).max())
        ax.set_title(f'Channel {i}\nDiff Range: [{diff_channel.min():.3f}, {diff_channel.max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    heatmap_path = output_dir / f'{layer_name}_heatmap_comparison.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap comparison to {heatmap_path}")
    
    # 3. 单个通道详细对比（选择差异最大的通道）
    diff_magnitude = np.abs(diff).mean(axis=(1, 2))  # [C]
    max_diff_channel = int(np.argmax(diff_magnitude))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'{layer_name} Channel {max_diff_channel} Detailed Comparison (Max Diff Channel)', fontsize=14)
    
    # 3.1 Clean 特征图
    im1 = axes[0, 0].imshow(clean[max_diff_channel], cmap='viridis')
    axes[0, 0].set_title(f'Clean (Mean={clean[max_diff_channel].mean():.3f}, Std={clean[max_diff_channel].std():.3f})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 3.2 Fault 特征图
    im2 = axes[0, 1].imshow(fault[max_diff_channel], cmap='viridis')
    axes[0, 1].set_title(f'Fault (Mean={fault[max_diff_channel].mean():.3f}, Std={fault[max_diff_channel].std():.3f})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3.3 差值图
    diff_channel = diff[max_diff_channel]
    im3 = axes[1, 0].imshow(diff_channel, cmap='RdBu_r', vmin=-np.abs(diff_channel).max(), vmax=np.abs(diff_channel).max())
    axes[1, 0].set_title(f'Difference (Fault - Clean)\nMax Abs Diff: {np.abs(diff_channel).max():.3f}')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 3.4 差值直方图
    axes[1, 1].hist(diff_channel.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Difference Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Difference Distribution\n(Mean={diff_channel.mean():.4f}, Std={diff_channel.std():.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    detail_path = output_dir / f'{layer_name}_channel_{max_diff_channel}_detailed.png'
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved detailed comparison to {detail_path}")
    
    # 4. SEU 故障模式分析：稀疏点 vs 整通道
    print(f"\nAnalyzing SEU fault patterns...")
    
    # 4.1 计算每个空间位置的错误幅度
    abs_diff = np.abs(diff)  # [C, H, W]
    spatial_error = abs_diff.mean(axis=0)  # [H, W] - 跨通道平均
    channel_error = abs_diff.mean(axis=(1, 2))  # [C] - 跨空间平均
    
    # 4.2 识别异常点（超过阈值的空间位置）
    error_threshold = np.percentile(abs_diff.flatten(), 95)  # 95分位数作为阈值
    anomaly_mask = abs_diff > error_threshold  # [C, H, W]
    anomaly_spatial = anomaly_mask.any(axis=0)  # [H, W] - 至少一个通道异常的位置
    anomaly_channels = anomaly_mask.any(axis=(1, 2))  # [C] - 至少一个位置异常的通道
    
    # 4.2.1 分析每个通道的错误分布模式
    channel_sparsity = []  # 每个通道的稀疏度
    for ch in range(C):
        ch_errors = abs_diff[ch]
        ch_threshold = np.percentile(ch_errors.flatten(), 95)
        ch_anomaly_pixels = (ch_errors > ch_threshold).sum()
        ch_sparsity = 1 - (ch_anomaly_pixels / ch_errors.size)
        channel_sparsity.append(ch_sparsity)
    channel_sparsity = np.array(channel_sparsity)
    
    # 4.3 可视化故障模式
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{layer_name} SEU Fault Pattern Analysis (Sample {sample_idx})', fontsize=14)
    
    # 4.3.1 空间错误分布（跨通道平均）
    im1 = axes[0, 0].imshow(spatial_error, cmap='hot')
    axes[0, 0].set_title(f'Spatial Error Distribution (Mean across channels)\nMax: {spatial_error.max():.4f}, Mean: {spatial_error.mean():.4f}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 4.3.2 异常点空间分布
    im2 = axes[0, 1].imshow(anomaly_spatial.astype(float), cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Anomaly Spatial Mask (Threshold={error_threshold:.4f})\n{anomaly_spatial.sum()} / {anomaly_spatial.size} pixels ({100*anomaly_spatial.sum()/anomaly_spatial.size:.2f}%)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 4.3.3 通道错误分布
    axes[0, 2].bar(range(len(channel_error)), channel_error)
    axes[0, 2].axhline(error_threshold, color='red', linestyle='--', label=f'Threshold ({error_threshold:.4f})')
    axes[0, 2].set_xlabel('Channel')
    axes[0, 2].set_ylabel('Mean Absolute Error')
    axes[0, 2].set_title(f'Channel-wise Error Distribution\n{anomaly_channels.sum()} / {len(anomaly_channels)} channels affected')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4.3.4 单个通道的详细错误分布（选择错误最大的通道）
    max_error_channel = int(np.argmax(channel_error))
    im4 = axes[1, 0].imshow(abs_diff[max_error_channel], cmap='hot')
    axes[1, 0].set_title(f'Channel {max_error_channel} Error Map (Max Error Channel)\nMax: {abs_diff[max_error_channel].max():.4f}, Mean: {abs_diff[max_error_channel].mean():.4f}')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 4.3.5 错误的空间分布直方图
    axes[1, 1].hist(spatial_error.flatten(), bins=50, alpha=0.7, color='orange')
    axes[1, 1].axvline(error_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({error_threshold:.4f})')
    axes[1, 1].set_xlabel('Spatial Error Magnitude')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Spatial Error Distribution\n(Mean={spatial_error.mean():.4f}, Std={spatial_error.std():.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 4.3.6 故障模式总结
    num_anomaly_pixels = anomaly_spatial.sum()
    num_anomaly_channels = anomaly_channels.sum()
    total_pixels = anomaly_spatial.size
    total_channels = len(anomaly_channels)
    
    # 计算稀疏度：异常像素占比
    sparsity = 1 - (num_anomaly_pixels / total_pixels)
    channel_affected_ratio = num_anomaly_channels / total_channels
    
    axes[1, 2].axis('off')
    summary_text = f"""
SEU Fault Pattern Summary
==========================

Spatial Pattern:
  - Total pixels: {total_pixels}
  - Anomaly pixels: {num_anomaly_pixels} ({100*num_anomaly_pixels/total_pixels:.2f}%)
  - Sparsity: {sparsity:.4f} (1.0 = fully sparse, 0.0 = fully dense)
  
Channel Pattern:
  - Total channels: {total_channels}
  - Affected channels: {num_anomaly_channels} ({100*channel_affected_ratio:.2f}%)
  
Fault Type Analysis:
"""
    if sparsity > 0.9:
        fault_type = "SPARSE (Point-wise errors)"
    elif sparsity > 0.5:
        fault_type = "MIXED (Partial spatial errors)"
    else:
        fault_type = "DENSE (Widespread errors)"
    
    if channel_affected_ratio < 0.1:
        channel_type = "FEW channels affected"
    elif channel_affected_ratio < 0.5:
        channel_type = "SOME channels affected"
    else:
        channel_type = "MOST channels affected"
    
    summary_text += f"  - Spatial: {fault_type}\n"
    summary_text += f"  - Channel: {channel_type}\n"
    summary_text += f"\nConclusion:\n"
    
    if sparsity > 0.9 and channel_affected_ratio < 0.1:
        summary_text += "  → SPARSE POINT-WISE ERRORS\n"
        summary_text += "    故障表现为稀疏的点状错误，\n"
        summary_text += "    只有少数像素和通道受影响。"
    elif sparsity > 0.5 and channel_affected_ratio > 0.5:
        summary_text += "  → CHANNEL-WIDE ERRORS\n"
        summary_text += "    故障表现为整通道变异，\n"
        summary_text += "    多个通道整体受影响。"
    else:
        summary_text += "  → MIXED PATTERN\n"
        summary_text += "    故障表现为混合模式，\n"
        summary_text += "    既有局部点错误，也有通道级影响。"
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    pattern_path = output_dir / f'{layer_name}_seu_pattern_analysis.png'
    plt.savefig(pattern_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved SEU pattern analysis to {pattern_path}")
    
    # 4.4 通道级详细分析：哪些通道受影响最严重，以及每个通道的错误模式
    top_affected_channels = np.argsort(channel_error)[-8:][::-1]  # Top 8
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    fig.suptitle(f'{layer_name} Channel-wise SEU Impact: Sparse Points vs Channel-wide Errors', fontsize=14)
    
    # 4.4.1 受影响最严重的通道（Top 8）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(range(len(top_affected_channels)), channel_error[top_affected_channels])
    ax1.set_yticks(range(len(top_affected_channels)))
    ax1.set_yticklabels([f'Ch {i}' for i in top_affected_channels])
    ax1.set_xlabel('Mean Absolute Error')
    ax1.set_title('Top 8 Most Affected Channels')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 4.4.2 通道错误 vs 通道索引（散点图）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(range(len(channel_error)), channel_error, alpha=0.6, s=20)
    ax2.axhline(error_threshold, color='red', linestyle='--', label=f'Threshold')
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Channel Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 4.4.3 每个通道的稀疏度分析
    channel_sparsity = []
    for ch in range(C):
        ch_errors = abs_diff[ch]
        ch_threshold = np.percentile(ch_errors.flatten(), 95)
        ch_anomaly_pixels = (ch_errors > ch_threshold).sum()
        ch_sparsity = 1 - (ch_anomaly_pixels / ch_errors.size)
        channel_sparsity.append(ch_sparsity)
    channel_sparsity = np.array(channel_sparsity)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(range(len(channel_sparsity)), channel_sparsity, alpha=0.6, s=20, c=channel_error, cmap='hot')
    ax3.set_xlabel('Channel Index')
    ax3.set_ylabel('Sparsity (1.0=sparse, 0.0=dense)')
    ax3.set_title('Channel Sparsity (Color=Error Magnitude)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(ax3.collections[0], ax=ax3, label='Error')
    
    # 4.4.4 错误模式总结（稀疏 vs 密集）
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    sparse_channels = (channel_sparsity > 0.9).sum()
    dense_channels = (channel_sparsity < 0.5).sum()
    mixed_channels = C - sparse_channels - dense_channels
    
    pattern_text = f"""
Channel Error Pattern Summary
=============================
Total Channels: {C}

Sparse (Point-wise):
  Channels: {sparse_channels} ({100*sparse_channels/C:.1f}%)
  → 错误集中在少数像素点

Dense (Channel-wide):
  Channels: {dense_channels} ({100*dense_channels/C:.1f}%)
  → 错误分布在整个通道

Mixed:
  Channels: {mixed_channels} ({100*mixed_channels/C:.1f}%)
  → 混合模式
"""
    ax4.text(0.1, 0.5, pattern_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    # 4.4.5-8 显示前4个受影响通道的错误空间分布
    for i, ch_idx in enumerate(top_affected_channels[:4]):
        row = 1
        col = i
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(abs_diff[ch_idx], cmap='hot')
        ch_sparsity = channel_sparsity[ch_idx]
        pattern_type = "SPARSE" if ch_sparsity > 0.9 else ("DENSE" if ch_sparsity < 0.5 else "MIXED")
        ax.set_title(f'Ch {ch_idx} ({pattern_type})\nSparsity={ch_sparsity:.2f}, Error={channel_error[ch_idx]:.4f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    channel_path = output_dir / f'{layer_name}_channel_pattern_analysis.png'
    plt.savefig(channel_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved channel pattern analysis to {channel_path}")
    
    # 4. 保存数值统计到文本文件
    stats_text = f"""
{layer_name} Activation Comparison Statistics
=============================================
Sample Index: {sample_idx}
Shape: {clean.shape}

Overall Statistics:
------------------
Clean:
  Mean: {clean_flat.mean():.6f}
  Std:  {clean_flat.std():.6f}
  Min:  {clean_flat.min():.6f}
  Max:  {clean_flat.max():.6f}
  
Fault:
  Mean: {fault_flat.mean():.6f}
  Std:  {fault_flat.std():.6f}
  Min:  {fault_flat.min():.6f}
  Max:  {fault_flat.max():.6f}

Difference (Fault - Clean):
  Mean: {diff_flat.mean():.6f}
  Std:  {diff_flat.std():.6f}
  Min:  {diff_flat.min():.6f}
  Max:  {diff_flat.max():.6f}
  Max Abs: {np.abs(diff_flat).max():.6f}

Channel-wise Statistics (Top 10 channels with largest difference):
-------------------------------------------------------------------
"""
    diff_per_channel = np.abs(diff).mean(axis=(1, 2))
    top_channels = np.argsort(diff_per_channel)[-10:][::-1]
    for ch_idx in top_channels:
        stats_text += f"Channel {ch_idx}:\n"
        stats_text += f"  Clean: mean={clean_mean[ch_idx]:.4f}, std={clean_std[ch_idx]:.4f}\n"
        stats_text += f"  Fault: mean={fault_mean[ch_idx]:.4f}, std={fault_std[ch_idx]:.4f}\n"
        stats_text += f"  Diff:  mean={diff[ch_idx].mean():.4f}, std={diff[ch_idx].std():.4f}, max_abs={np.abs(diff[ch_idx]).max():.4f}\n"
        stats_text += "\n"
    
    stats_file = output_dir / f'{layer_name}_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write(stats_text)
    print(f"  Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize fault injection impact on activations')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--layer', type=str, default='features.0', help='Layer name to visualize')
    parser.add_argument('--ber', type=float, default=1e-2, help='Bit error rate')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--output_dir', type=str, default='visualizations/fault_activation', help='Output directory')
    parser.add_argument('--bit_width_config', type=str, default=None, help='Bit width config JSON')
    parser.add_argument('--config_index', type=int, default=0, help='Config index')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置和模型
    print("Loading model...")
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['visualize_fault_activation.py', args.config]
    try:
        configs = get_config(args.config)
    finally:
        sys.argv = original_argv
    
    # 设置默认值
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
    
    # 加载 checkpoint
    load_checkpoint(model, args.ckpt, model_device=str(device), strict=False)
    
    # 设置位宽配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, args.config_index)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(configs.dataloader, configs.arch)
    
    # 获取一批数据
    data, _ = next(iter(test_loader))
    data = data.to(device)
    
    print(f"Processing {args.num_samples} samples with BER={args.ber}...")
    
    # 1. 提取 clean 激活值
    clean_activations = {}
    hook_clean = register_activation_hook(model, args.layer, clean_activations)
    
    with torch.no_grad():
        _ = model(data)
    
    clean_act = clean_activations[args.layer]  # [B, C, H, W]
    hook_clean.remove()
    
    print(f"  Clean activation shape: {clean_act.shape}")
    
    # 2. 提取 fault 激活值
    fault_activations = {}
    hook_fault = register_activation_hook(model, args.layer, fault_activations)
    
    injector = FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        skip_first_last=False,
    )
    injector.enable()
    
    with torch.no_grad():
        _ = model(data)
    
    fault_act = fault_activations[args.layer]  # [B, C, H, W]
    hook_fault.remove()
    injector.disable()
    
    print(f"  Fault activation shape: {fault_act.shape}")
    
    # 3. 可视化对比
    print(f"\nGenerating visualizations...")
    visualize_activation_comparison(
        clean_act,
        fault_act,
        output_dir,
        args.layer,
        sample_idx=0,
    )
    
    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

