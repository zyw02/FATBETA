#!/usr/bin/env python3
"""
Train learning-based repair heads for sensitive layers using dumped activations.

Example:
python tools/train_sensitive_repair_head.py \
  --clean_dir activation_dumps/clean \
  --fault_dir activation_dumps/fault \
  --layers features.0,classifier.1 \
  --output checkpoints/mlp_heads.pt \
  --epochs 5 --batch_size 8 --lr 1e-3
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from util.sensitive_layer_restorer import (
    ChannelStatsComputer,
    LearningChannelRepair,
    PolynomialChannelRepair,
    QuantAwareResidualRepair,
    LightweightDenoiser,
    DenoiseRestorer,
    StackedDenoiseRestorer,
    RestorerV4,
    ImprovedRestorer,
    ActivationReconstructor,
    ChannelDetector,
)


def find_pairs(clean_dir: Path, fault_dir: Path, target_layers: List[str]) -> Dict[str, List[Tuple[Path, Path]]]:
    clean_map: Dict[str, Dict[int, Path]] = {}
    fault_map: Dict[str, Dict[int, Path]] = {}

    def register(directory: Path, storage: Dict[str, Dict[int, Path]]):
        for path in directory.glob('*.pt'):
            meta = torch.load(path, map_location='cpu')
            layer = meta['layer']
            batch = meta.get('batch', 0)
            if target_layers and layer not in target_layers:
                continue
            storage.setdefault(layer, {})[batch] = path

    register(clean_dir, clean_map)
    register(fault_dir, fault_map)

    pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    for layer, batches in fault_map.items():
        if layer not in clean_map:
            continue
        shared_batches = set(batches.keys()) & set(clean_map[layer].keys())
        if not shared_batches:
            continue
        pairs[layer] = [(fault_map[layer][b], clean_map[layer][b]) for b in sorted(shared_batches)]
    return pairs


class ActivationPairDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, Path]], max_samples: Optional[int] = None):
        if max_samples is not None:
            samples = samples[:max_samples]
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fault_path, clean_path = self.samples[idx]
        fault = torch.load(fault_path, map_location='cpu')['tensor'].float()
        clean = torch.load(clean_path, map_location='cpu')['tensor'].float()
        return fault, clean


def collate_activation_batches(batch):
    faults, cleans = zip(*batch)
    faults = torch.cat(faults, dim=0)
    cleans = torch.cat(cleans, dim=0)
    return faults, cleans


def build_repair_module(
    layer_stats: Dict,
    repair_mode: str,
    hidden_dim: int,
    clip_margin: float,
    act_bit: int,
    layer_name: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    num_stages: int = 1,
):
    if layer_stats is None:
        return None
    # 检查必要的键是否存在
    required_keys = ['channel_mean', 'channel_std']
    missing_keys = [key for key in required_keys if key not in layer_stats]
    if missing_keys:
        print(f"Warning: Missing required keys in layer_stats: {missing_keys}")
        return None
    num_channels = layer_stats['channel_mean'].numel()
    if repair_mode in ('mlp', 'mlp_local'):
        module = LearningChannelRepair(num_channels=num_channels, hidden_size=hidden_dim, clip_margin=clip_margin)
    elif repair_mode == 'mlp_poly':
        module = PolynomialChannelRepair(num_channels=num_channels, hidden_size=max(hidden_dim, 64), clip_margin=clip_margin)
    elif repair_mode == 'ms_residual':
        module = QuantAwareResidualRepair(
            num_channels=num_channels,
            hidden_size=max(hidden_dim, 64),
            clip_margin=clip_margin,
            act_bit=act_bit,
        )
    elif repair_mode == 'lightweight_denoiser':
        module = LightweightDenoiser(num_channels=num_channels, clip_margin=clip_margin)
    elif repair_mode == 'denoise_restorer':
        # 从下一层获取量化边界（如果提供了模型）
        quant_min = -5.0  # 默认值
        quant_max = 5.0   # 默认值
        
        if model is not None and layer_name is not None:
            from quan.func import QuanConv2d, QuanLinear
            from quan.quantizer.lsq import LsqQuan
            modules = dict(model.named_modules())
            layer_names = list(modules.keys())
            try:
                current_idx = layer_names.index(layer_name)
                # 从当前层之后查找第一个 QuanConv2d 或 QuanLinear
                for i in range(current_idx + 1, len(layer_names)):
                    next_layer = modules[layer_names[i]]
                    if isinstance(next_layer, (QuanConv2d, QuanLinear)):
                        print(f"[{layer_name}] Found next layer: {layer_names[i]} ({type(next_layer).__name__})")
                        
                        # 获取下一层的激活量化器
                        if hasattr(next_layer, 'quan_a_fn') and next_layer.quan_a_fn is not None:
                            quan_a_fn = next_layer.quan_a_fn
                            
                            # 获取激活位宽
                            if hasattr(next_layer, 'fixed_bits') and next_layer.fixed_bits is not None:
                                abits = next_layer.fixed_bits[1]
                            elif hasattr(next_layer, 'bits') and next_layer.bits is not None:
                                abits = next_layer.bits[1]
                            else:
                                break
                            
                            if abits is None or abits >= 32:
                                break
                            
                            # 获取 scale 并计算量化边界
                            if isinstance(quan_a_fn, LsqQuan):
                                scale = quan_a_fn.get_scale(abits, detach=True)
                                if scale is not None:
                                    # 计算量化阈值
                                    all_positive = quan_a_fn.all_positive
                                    symmetric = quan_a_fn.symmetric
                                    
                                    if all_positive:
                                        thd_neg = 0
                                        thd_pos = 2 ** abits - 1
                                    else:
                                        if symmetric:
                                            thd_neg = - 2 ** (abits - 1) + 1
                                            thd_pos = 2 ** (abits - 1) - 1
                                        else:
                                            thd_neg = - 2 ** (abits - 1)
                                            thd_pos = 2 ** (abits - 1) - 1
                                    
                                    # 计算量化边界
                                    quant_min = (scale * thd_neg).item()
                                    quant_max = (scale * thd_pos).item()
                                    print(f"[{layer_name}] Quant bounds from {layer_names[i]}: [{quant_min:.4f}, {quant_max:.4f}] (abits={abits}, scale={scale.item():.4f})")
                                    break
            except (ValueError, Exception) as e:
                print(f"[{layer_name}] Warning: Failed to get quant bounds from next layer: {e}")
        
        # 如果无法从下一层获取，使用统计信息估算（作为 fallback）
        if quant_min == -5.0 and quant_max == 5.0:
            max_abs = layer_stats.get('channel_max_abs', layer_stats['channel_mean'].abs())
            quant_max = max_abs.max().item() * clip_margin
            quant_min = -quant_max
            print(f"[{layer_name}] Using estimated quant bounds: [{quant_min:.4f}, {quant_max:.4f}]")
        
        # 创建 DenoiseRestorer 或 StackedDenoiseRestorer
        if num_stages > 1:
            # 堆叠多个 DenoiseRestorer
            module = StackedDenoiseRestorer(
                channels=num_channels, 
                quant_min=quant_min, 
                quant_max=quant_max, 
                next_layer_module=None,  # 训练时不需要，边界已设置
                num_stages=num_stages
            )
            print(f"[{layer_name}] Using StackedDenoiseRestorer with {num_stages} stages")
        else:
            # 单个 DenoiseRestorer
            module = DenoiseRestorer(
                channels=num_channels, 
                quant_min=quant_min, 
                quant_max=quant_max, 
                next_layer_module=None  # 训练时不需要，边界已设置
            )
    elif repair_mode == 'restorer_v4':
        # Restorer V4: 高容量结构
        quant_min = -5.0  # 默认值
        quant_max = 5.0   # 默认值
        
        if model is not None and layer_name is not None:
            from quan.func import QuanConv2d, QuanLinear
            from quan.quantizer.lsq import LsqQuan
            modules = dict(model.named_modules())
            layer_names = list(modules.keys())
            try:
                current_idx = layer_names.index(layer_name)
                for i in range(current_idx + 1, len(layer_names)):
                    next_layer = modules[layer_names[i]]
                    if isinstance(next_layer, (QuanConv2d, QuanLinear)):
                        print(f"[{layer_name}] Found next layer: {layer_names[i]} ({type(next_layer).__name__})")
                        
                        if hasattr(next_layer, 'quan_a_fn') and next_layer.quan_a_fn is not None:
                            quan_a_fn = next_layer.quan_a_fn
                            
                            if hasattr(next_layer, 'fixed_bits') and next_layer.fixed_bits is not None:
                                abits = next_layer.fixed_bits[1]
                            elif hasattr(next_layer, 'bits') and next_layer.bits is not None:
                                abits = next_layer.bits[1]
                            else:
                                break
                            
                            if abits is None or abits >= 32:
                                break
                            
                            if isinstance(quan_a_fn, LsqQuan):
                                scale = quan_a_fn.get_scale(abits, detach=True)
                                if scale is not None:
                                    all_positive = quan_a_fn.all_positive
                                    symmetric = quan_a_fn.symmetric
                                    
                                    if all_positive:
                                        thd_neg = 0
                                        thd_pos = 2 ** abits - 1
                                    else:
                                        if symmetric:
                                            thd_neg = - 2 ** (abits - 1) + 1
                                            thd_pos = 2 ** (abits - 1) - 1
                                        else:
                                            thd_neg = - 2 ** (abits - 1)
                                            thd_pos = 2 ** (abits - 1) - 1
                                    
                                    quant_min = (scale * thd_neg).item()
                                    quant_max = (scale * thd_pos).item()
                                    print(f"[{layer_name}] Quant bounds from {layer_names[i]}: [{quant_min:.4f}, {quant_max:.4f}] (abits={abits}, scale={scale.item():.4f})")
                                    break
            except (ValueError, Exception) as e:
                print(f"[{layer_name}] Warning: Failed to get quant bounds from next layer: {e}")
        
        if quant_min == -5.0 and quant_max == 5.0:
            max_abs = layer_stats.get('channel_max_abs', layer_stats['channel_mean'].abs())
            quant_max = max_abs.max().item() * clip_margin
            quant_min = -quant_max
            print(f"[{layer_name}] Using estimated quant bounds: [{quant_min:.4f}, {quant_max:.4f}]")
        
        # 使用 num_stages 作为 num_blocks（Restorer V4 的堆叠块数）
        # 增加块数以提升容量（默认 5 个块，如果 num_stages > 1 则使用 num_stages）
        num_blocks = max(num_stages, 5) if num_stages > 1 else 5
        if repair_mode == 'restorer_v4':
            module = RestorerV4(
                channels=num_channels,
                quant_min=quant_min,
                quant_max=quant_max,
                next_layer_module=None,  # 训练时不需要，边界已设置
                num_blocks=num_blocks,
                use_bn=False,  # 暂时禁用 BatchNorm，因为可能导致训练不稳定
                residual_scale=1.0,  # 可以尝试 0.1 来稳定训练
                expand_channels=0,  # 不扩展通道（可以尝试 2 来增加容量）
            )
            print(f"[{layer_name}] Using RestorerV4 (Enhanced) with {num_blocks} correction blocks, BatchNorm=False")
        elif repair_mode == 'improved_restorer':
            # 使用改进的 Restorer（更简单、更有效）
            module = ImprovedRestorer(
                channels=num_channels,
                quant_min=quant_min,
                quant_max=quant_max,
                next_layer_module=None,  # 训练时不需要，边界已设置
                hidden_ratio=2.0,  # 隐藏层通道数是输入的 2 倍
            )
            print(f"[{layer_name}] Using ImprovedRestorer with hidden_ratio=2.0")
        elif repair_mode == 'activation_reconstructor':
            # 使用激活值重建器（专门设计用于激活值重建）
            module = ActivationReconstructor(
                channels=num_channels,
                quant_min=quant_min,
                quant_max=quant_max,
                next_layer_module=None,  # 训练时不需要，边界已设置
                num_stages=2,  # 渐进式重建的阶段数
            )
            print(f"[{layer_name}] Using ActivationReconstructor with {2} stages")
    else:
        raise ValueError(f'Unsupported repair_mode for training: {repair_mode}')
    
    # 对于其他 repair_mode，update_reference 是必需的（用于设置参考统计）
    # 对于 denoise_restorer、stacked_denoise_restorer 和 restorer_v4，这是可选的（边界已设置）
    if repair_mode not in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4'):
        module.update_reference(
            layer_stats['channel_mean'].float(),
            layer_stats['channel_std'].float(),
            layer_stats.get('channel_max_abs', layer_stats['channel_mean'].abs()).float(),
        )
    return module


def train_layer_model(
    layer: str,
    dataset: ActivationPairDataset,
    layer_stats: Dict,
    repair_mode: str,
    hidden_dim: int,
    clip_margin: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    act_bit: int,
    lr_min: float,
    main_model: Optional[nn.Module] = None,
    num_stages: int = 1,
):
    module = build_repair_module(layer_stats, repair_mode, hidden_dim, clip_margin, act_bit, 
                                layer_name=layer, model=main_model, num_stages=num_stages)
    if module is None:
        raise ValueError(f"Layer {layer} missing statistics in profile; cannot train repair head.")
    model = module.to(device)
    
    # 优化器选择：denoise_restorer、stacked_denoise_restorer、restorer_v4、improved_restorer 和 activation_reconstructor 使用 Adam，ms_residual 使用 AdamW，其他使用 SGD
    use_adam = repair_mode in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor', 'lightweight_denoiser')
    use_adamw = repair_mode == 'ms_residual'
    
    if use_adam:
        # Restorer V4 和 ImprovedRestorer 使用更高的学习率和更小的 weight decay
        if repair_mode == 'restorer_v4':
            # 使用更高的初始学习率（5e-3）和更小的 weight decay
            effective_lr = max(lr, 5e-3)
            optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=1e-6, betas=(0.9, 0.999))
            print(f"[{layer}] Using enhanced Adam optimizer for RestorerV4: lr={effective_lr:.2e}, weight_decay=1e-6")
        elif repair_mode == 'improved_restorer':
            # ImprovedRestorer 使用适中的学习率
            effective_lr = max(lr, 2e-3)
            optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=1e-5, betas=(0.9, 0.999))
            print(f"[{layer}] Using Adam optimizer for ImprovedRestorer: lr={effective_lr:.2e}, weight_decay=1e-5")
        elif repair_mode == 'activation_reconstructor':
            # ActivationReconstructor 使用适中的学习率（因为使用了 BatchNorm，可以稍微高一点）
            effective_lr = max(lr, 3e-3)
            optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=1e-5, betas=(0.9, 0.999))
            print(f"[{layer}] Using Adam optimizer for ActivationReconstructor: lr={effective_lr:.2e}, weight_decay=1e-5")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=max(lr, 1e-4), weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_activation_batches)
    total_steps = max(1, epochs * len(loader))
    
    # 学习率调度器：denoise_restorer、stacked_denoise_restorer 和 restorer_v4 使用 step decay，其他使用 cosine annealing
    if repair_mode in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4'):
        if repair_mode == 'restorer_v4':
            # Restorer V4 使用更温和的衰减策略：每 40% epoch 衰减，衰减率 0.5
            step_size = max(1, int(epochs * 0.4))
            gamma = 0.5
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            print(f"[{layer}] Using StepLR for RestorerV4: step_size={step_size}, gamma={gamma}")
        else:
            # Step decay: 每 30% 的 epoch 衰减一次，衰减率为 0.1
            step_size = max(1, int(epochs * 0.3))
            gamma = 0.1
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr_min)

    quant_lambda = 0.5 if repair_mode == 'ms_residual' else 0.0
    if repair_mode == 'ms_residual' and layer.startswith('features.0'):
        quant_lambda = 0.0
    detector = ChannelDetector()
    detector.update_reference(layer_stats['channel_mean'].float(), layer_stats['channel_std'].float())
    if layer.startswith('features.0'):
        detector.set_params(z_thresh=2.0, std_ratio_bounds=(0.3, 3.0))
    for epoch in range(epochs):
        total_loss = 0.0
        for faults, cleans in loader:
            faults = faults.to(device)
            cleans = cleans.to(device)
            snapshot = ChannelStatsComputer.compute(faults)
            mask = detector.detect(snapshot).to(device)
            if not mask.any():
                mask = torch.ones_like(snapshot.mean, dtype=torch.bool, device=device)
            reg = 0.0
            quant_penalty = 0.0
            if repair_mode == 'mlp_poly':
                repaired, params = model.repair(faults, snapshot, mask, return_params=True)
                if params is not None:
                    scale, bias, quad = params
                    reg = (
                        0.1 * torch.mean((scale - 1.0) ** 2)
                        + 0.01 * torch.mean(bias ** 2)
                        + 0.1 * torch.mean(quad ** 2)
                    )
            elif repair_mode == 'ms_residual':
                repaired, params = model.repair(faults, snapshot, mask, return_params=True)
                residual = params.get('residual') if params else None
                if residual is not None:
                    reg = 0.001 * torch.mean(residual ** 2)
                if quant_lambda > 0.0:
                    quant_penalty = F.mse_loss(model.quantize(repaired), model.quantize(cleans))
            elif repair_mode == 'lightweight_denoiser':
                # Always-on 修复，mask 被忽略（为接口兼容性保留）
                repaired, params = model.repair(faults, snapshot, mask, return_params=True)
                residual = params.get('residual') if params else None
                if residual is not None:
                    # 轻量级 L2 正则化，防止残差过大
                    reg = 0.001 * torch.mean(residual ** 2)
            elif repair_mode in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor'):
                # Always-on 修复，mask 被忽略
                repaired, params = model.repair(faults, snapshot, mask, return_params=True)
                residual = params.get('residual') if params else None
                if residual is not None:
                    # Restorer V4、ImprovedRestorer 和 ActivationReconstructor 使用更小的正则化（因为容量更大）
                    if repair_mode in ('restorer_v4', 'improved_restorer', 'activation_reconstructor'):
                        reg = 0.0001 * torch.mean(residual ** 2)  # 更小的正则化
                    else:
                        reg = 0.001 * torch.mean(residual ** 2)  # 轻量级 L2 正则化
            else:
                repaired = model.repair(faults, snapshot, mask)
            
            # Restorer V4、ImprovedRestorer 和 ActivationReconstructor 可以使用 Smooth L1 Loss（对异常值更鲁棒）
            if repair_mode in ('restorer_v4', 'improved_restorer', 'activation_reconstructor'):
                # 尝试使用 Smooth L1 Loss（对异常值更鲁棒，可能有助于收敛）
                loss = F.smooth_l1_loss(repaired, cleans, beta=1.0) + reg + quant_lambda * quant_penalty
            else:
                loss = F.mse_loss(repaired, cleans) + reg + quant_lambda * quant_penalty

            optimizer.zero_grad()
            loss.backward()
            # Restorer V4、ImprovedRestorer 和 ActivationReconstructor 使用更大的梯度裁剪（因为网络更深）
            max_grad_norm = 5.0 if repair_mode in ('restorer_v4', 'improved_restorer', 'activation_reconstructor') else 2.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            # 对于非 step decay 的调度器，每个 step 更新一次
            if repair_mode not in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor'):
                scheduler.step()

            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        # 对于 StepLR，在每个 epoch 结束时更新
        if repair_mode in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor'):
            scheduler.step()
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{layer}] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.6f}, LR={current_lr:.2e}")

    return model.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Train learning repair heads for sensitive layers")
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--fault_dir', type=str, required=True)
    parser.add_argument('--layers', type=str, required=True, help='Comma-separated layer names')
    parser.add_argument('--output', type=str, required=True, help='Path to save state dict')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--clip_margin', type=float, default=1.25)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--repair_mode', type=str, default='ms_residual',
                        choices=['mlp', 'mlp_local', 'mlp_poly', 'ms_residual', 'lightweight_denoiser', 'denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor'])
    parser.add_argument('--num_stages', type=int, default=1, help='Number of stacked stages for denoise_restorer (default: 1, use stacked_denoise_restorer mode for >1)')
    parser.add_argument('--act_bit', type=int, default=6,
                        help='Activation bit width for quant-aware repair heads')
    parser.add_argument('--layer_profile', type=str, required=True, help='Path to clean layer statistics')
    parser.add_argument('--max_samples', type=int, default=None, help='Optional cap on activation pairs per layer')
    parser.add_argument('--model_config', type=str, default=None, help='Model config file (for getting next layer module)')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Model checkpoint (for getting next layer module)')
    return parser.parse_args()


def main():
    args = parse_args()
    target_layers = [layer.strip() for layer in args.layers.split(',') if layer.strip()]
    clean_dir = Path(args.clean_dir)
    fault_dir = Path(args.fault_dir)
    output_path = Path(args.output)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    profile = torch.load(args.layer_profile, map_location='cpu')
    layer_stats_map = profile.get('layers', {})
    
    # 如果提供了模型配置，加载模型以获取下一层模块
    main_model = None
    if args.model_config and args.model_ckpt:
        print(f"[Info] Loading model to get next layer modules...")
        from model import create_model
        from util.checkpoint import load_checkpoint
        from util.config import get_config
        from util.utils import preprocess_model
        from quan import find_modules_to_quantize, replace_module_by_names
        from util.mpq import switch_bit_width, switch_bit_width_bn
        
        import sys as sys_module
        original_argv = sys_module.argv.copy()
        sys_module.argv = ['train_sensitive_repair_head.py', args.model_config]
        try:
            configs = get_config(args.model_config)
        finally:
            sys_module.argv = original_argv
        
        if not hasattr(configs, 'local_rank'):
            configs.local_rank = 0
        if not hasattr(configs, 'world_size'):
            configs.world_size = 1
        if not hasattr(configs, 'rank'):
            configs.rank = 0
        
        main_model = create_model(configs.arch, dataset=configs.dataloader.dataset)
        main_model = preprocess_model(main_model, configs)
        main_model = replace_module_by_names(main_model, find_modules_to_quantize(main_model, configs))
        main_model = main_model.to(device)
        main_model.eval()
        load_checkpoint(main_model, args.model_ckpt, model_device=str(device), strict=False)
        
        # 设置为 w6a6（与数据收集时一致）
        switch_bit_width(main_model, configs.quan, wbit=6, abits=6)
        switch_bit_width_bn(main_model, 6, 6)
        print(f"[Info] Model loaded and set to w6a6")
    
    # 诊断信息：显示 profile 中包含的层
    print(f"[Diagnostic] Profile contains {len(layer_stats_map)} layers:")
    for layer_name in sorted(layer_stats_map.keys()):
        stats = layer_stats_map[layer_name]
        if stats is None:
            print(f"  - {layer_name}: None")
        elif isinstance(stats, dict):
            has_mean = 'channel_mean' in stats
            has_std = 'channel_std' in stats
            print(f"  - {layer_name}: dict (has_mean={has_mean}, has_std={has_std})")
        else:
            print(f"  - {layer_name}: {type(stats)}")
    print(f"[Diagnostic] Target layers: {target_layers}")

    pairs = find_pairs(clean_dir, fault_dir, target_layers)
    if not pairs:
        print("No matching clean/fault activation pairs found.")
        print(f"  Clean dir: {clean_dir}")
        print(f"  Fault dir: {fault_dir}")
        print(f"  Target layers: {target_layers}")
        # 调试信息：列出实际找到的文件
        if clean_dir.exists():
            clean_files = list(clean_dir.glob('*.pt'))
            print(f"  Found {len(clean_files)} files in clean dir")
            if clean_files:
                sample = torch.load(clean_files[0], map_location='cpu')
                print(f"    Sample: {clean_files[0].name}, layer={sample.get('layer')}, batch={sample.get('batch')}")
        else:
            print(f"  Clean dir does not exist!")
        if fault_dir.exists():
            fault_files = list(fault_dir.glob('*.pt'))
            print(f"  Found {len(fault_files)} files in fault dir")
            if fault_files:
                sample = torch.load(fault_files[0], map_location='cpu')
                print(f"    Sample: {fault_files[0].name}, layer={sample.get('layer')}, batch={sample.get('batch')}")
        else:
            print(f"  Fault dir does not exist!")
        return

    state_dict = {}
    for layer in target_layers:
        if layer not in pairs:
            print(f"[{layer}] No data available, skipping.")
            continue
        if layer not in layer_stats_map:
            print(f"[{layer}] Missing statistics in profile, skipping.")
            continue
        layer_stats = layer_stats_map[layer]
        # 检查 layer_stats 是否有效
        if layer_stats is None:
            print(f"[{layer}] Statistics entry is None, skipping.")
            continue
        if not isinstance(layer_stats, dict):
            print(f"[{layer}] Statistics entry is not a dict (type={type(layer_stats)}), skipping.")
            continue
        required_keys = ['channel_mean', 'channel_std']
        missing_keys = [key for key in required_keys if key not in layer_stats]
        if missing_keys:
            print(f"[{layer}] Missing required keys in statistics: {missing_keys}, skipping.")
            continue
        dataset = ActivationPairDataset(pairs[layer], max_samples=args.max_samples)
        if len(dataset) == 0:
            print(f"[{layer}] No valid samples, skipping.")
            continue
        repair_mode = args.repair_mode

        repair_head = train_layer_model(
            layer=layer,
            dataset=dataset,
            layer_stats=layer_stats,
            repair_mode=repair_mode,
            hidden_dim=args.hidden_dim,
            clip_margin=args.clip_margin,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            act_bit=args.act_bit,
            lr_min=args.lr_min,
            main_model=main_model,
            num_stages=args.num_stages,
        )
        state_dict[layer] = repair_head.state_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"Saved repair head state dict to {output_path}")


if __name__ == '__main__':
    main()

