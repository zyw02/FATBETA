"""
Sensitive Layer Restorer with BER-aware profiles and learning repair head support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from quan.func import QuanConv2d, QuanLinear


def _load_profile(path: str | Path) -> Dict:
    data = torch.load(path, map_location='cpu')
    if 'layers' not in data:
        raise ValueError(f'Layer profile at {path} missing "layers" field.')
    return data


def _resolve_layers(model: nn.Module, requested: Optional[Iterable[str]], profile_layers: Iterable[str]) -> List[str]:
    modules = dict(model.named_modules())
    valid_layers = []
    candidate = requested if requested else profile_layers
    for name in candidate:
        if name in modules and name in profile_layers:
            valid_layers.append(name)
    return valid_layers


def _channel_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor.detach().permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)
    if tensor.dim() == 2:
        return tensor.detach().transpose(0, 1).contiguous()
    raise ValueError(f'Unsupported activation dim={tensor.dim()}')


@dataclass
class ChannelSnapshot:
    mean: torch.Tensor
    std: torch.Tensor
    max_abs: torch.Tensor


class ChannelStatsComputer:
    @staticmethod
    def compute(activation: torch.Tensor) -> ChannelSnapshot:
        view = _channel_view(activation)
        mean = view.mean(dim=1)
        std = view.std(dim=1, unbiased=False)
        max_abs = view.abs().max(dim=1).values
        return ChannelSnapshot(mean=mean, std=std, max_abs=max_abs)


class ChannelDetector:
    def __init__(self, z_thresh: float = 3.0, std_ratio_bounds: Tuple[float, float] = (0.5, 2.0)):
        self.z_thresh = z_thresh
        self.std_low, self.std_high = std_ratio_bounds
        self.ref_mean = None
        self.ref_std = None

    def update_reference(self, mean: torch.Tensor, std: torch.Tensor):
        eps = 1e-6
        self.ref_mean = mean.clone()
        self.ref_std = torch.clamp(std.clone(), min=eps)

    def set_params(self, z_thresh: float, std_ratio_bounds: Tuple[float, float]):
        self.z_thresh = z_thresh
        self.std_low, self.std_high = std_ratio_bounds

    def detect(self, snapshot: ChannelSnapshot) -> torch.Tensor:
        if self.ref_mean is None:
            return torch.zeros_like(snapshot.mean, dtype=torch.bool)
        ref_mean = self.ref_mean.to(snapshot.mean.device)
        ref_std = self.ref_std.to(snapshot.std.device)
        mean_delta = torch.abs(snapshot.mean - ref_mean)
        z = mean_delta / ref_std
        std_ratio = torch.zeros_like(snapshot.std)
        std_ratio[:] = torch.where(
            snapshot.std > 0,
            snapshot.std / ref_std,
            torch.zeros_like(snapshot.std),
        )
        mask = (z > self.z_thresh) | (std_ratio < self.std_low) | (std_ratio > self.std_high)
        return mask


class ChannelRepair:
    def __init__(self, clip_margin: float = 1.25, scale_bounds: Tuple[float, float] = (0.25, 4.0)):
        self.clip_margin = clip_margin
        self.scale_bounds = scale_bounds
        self.ref_mean = None
        self.ref_std = None
        self.ref_max_abs = None

    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        eps = 1e-6
        self.ref_mean = mean.clone()
        self.ref_std = torch.clamp(std.clone(), min=eps)
        self.ref_max_abs = torch.clamp(max_abs.clone(), min=eps)

    def set_clip_margin(self, clip_margin: float):
        self.clip_margin = clip_margin

    def repair(self, activation: torch.Tensor, snapshot: ChannelSnapshot, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any() or self.ref_mean is None:
            return activation

        ref_mean = self.ref_mean.to(snapshot.mean.device)
        ref_std = self.ref_std.to(snapshot.std.device)
        ref_max = self.ref_max_abs.to(snapshot.max_abs.device)

        scale = torch.ones_like(snapshot.std)
        bias = torch.zeros_like(snapshot.mean)

        scale[mask] = torch.clamp(
            ref_std[mask] / torch.clamp(snapshot.std[mask], min=1e-5),
            min=self.scale_bounds[0],
            max=self.scale_bounds[1],
        )
        bias[mask] = ref_mean[mask] - scale[mask] * snapshot.mean[mask]

        reshape_dims = [1, -1] + ([1, 1] if activation.dim() == 4 else [])
        activation = activation * scale.view(*reshape_dims) + bias.view(*reshape_dims)

        if self.clip_margin > 0:
            limit = (ref_max * self.clip_margin).view(*reshape_dims)
            activation = torch.clamp(activation, min=-limit, max=limit)
        return activation


class LearningChannelRepair(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int = 32, clip_margin: float = 1.25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2),
        )
        self.clip_margin = clip_margin
        self.scale_bounds = (0.1, 5.0)
        self.ref_mean = torch.zeros(num_channels)
        self.ref_std = torch.ones(num_channels)
        self.ref_max_abs = torch.ones(num_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_clip_margin(self, clip_margin: float):
        self.clip_margin = clip_margin

    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        eps = 1e-6
        self.ref_mean = mean.clone()
        self.ref_std = torch.clamp(std.clone(), min=eps)
        self.ref_max_abs = torch.clamp(max_abs.clone(), min=eps)

    def forward_params(self, snapshot: ChannelSnapshot) -> Tuple[torch.Tensor, torch.Tensor]:
        device = snapshot.mean.device
        ref_mean = self.ref_mean.to(device)
        ref_std = torch.clamp(self.ref_std, min=1e-6).to(device)
        ref_max = self.ref_max_abs.to(device)
        feats = torch.stack([
            snapshot.mean.to(device) - ref_mean,
            snapshot.std.to(device) / ref_std - 1.0,
            snapshot.max_abs.to(device) / ref_max - 1.0,
        ], dim=1)
        mlp_device = next(self.mlp.parameters()).device
        if mlp_device != device:
            self.mlp = self.mlp.to(device)
        delta = self.mlp(feats)
        scale = torch.clamp(delta[:, 0] + 1.0, min=self.scale_bounds[0], max=self.scale_bounds[1])
        bias = delta[:, 1]
        return scale, bias

    def repair(self, activation: torch.Tensor, snapshot: ChannelSnapshot, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return activation
        scale, bias = self.forward_params(snapshot)
        reshape_dims = [1, -1] + ([1, 1] if activation.dim() == 4 else [])
        activation = activation * scale.view(*reshape_dims) + bias.view(*reshape_dims)
        if self.clip_margin > 0:
            limit = (self.ref_max_abs.to(scale.device) * self.clip_margin).view(*reshape_dims)
            activation = torch.clamp(activation, min=-limit, max=limit)
        return activation


class PolynomialChannelRepair(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int = 64, clip_margin: float = 1.25):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = 6  # mean_delta, std_ratio, max_ratio, global_mean, global_std, energy_ratio
        self.clip_margin = clip_margin
        self.scale_bounds = (0.05, 6.0)
        self.quad_bound = 2.0
        self.backbone = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
        self.ref_mean = torch.zeros(num_channels)
        self.ref_std = torch.ones(num_channels)
        self.ref_max_abs = torch.ones(num_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_clip_margin(self, clip_margin: float):
        self.clip_margin = clip_margin

    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        eps = 1e-6
        self.ref_mean = mean.clone()
        self.ref_std = torch.clamp(std.clone(), min=eps)
        self.ref_max_abs = torch.clamp(max_abs.clone(), min=eps)

    def _build_features(self, snapshot: ChannelSnapshot) -> torch.Tensor:
        device = snapshot.mean.device
        ref_mean = self.ref_mean.to(device)
        ref_std = torch.clamp(self.ref_std, min=1e-6).to(device)
        ref_max = torch.clamp(self.ref_max_abs, min=1e-6).to(device)

        mean_delta = snapshot.mean.to(device) - ref_mean
        std_ratio = snapshot.std.to(device) / ref_std - 1.0
        max_ratio = snapshot.max_abs.to(device) / ref_max - 1.0

        global_mean = torch.mean(mean_delta)
        global_std = torch.mean(std_ratio)
        energy_ratio = torch.mean(
            (snapshot.mean.to(device) ** 2 + snapshot.std.to(device) ** 2)
            / (ref_mean ** 2 + ref_std ** 2 + 1e-6) - 1.0
        )

        global_feat = torch.stack([
            torch.full_like(mean_delta, global_mean),
            torch.full_like(mean_delta, global_std),
        ], dim=1)
        energy_feat = torch.full_like(mean_delta, energy_ratio).unsqueeze(1)

        feats = torch.cat([
            mean_delta.unsqueeze(1),
            std_ratio.unsqueeze(1),
            max_ratio.unsqueeze(1),
            global_feat,
            energy_feat,
        ], dim=1)
        return feats

    def forward_params(self, snapshot: ChannelSnapshot) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = snapshot.mean.device
        feats = self._build_features(snapshot).to(device)
        backbone_device = next(self.backbone.parameters()).device
        if backbone_device != device:
            self.backbone = self.backbone.to(device)
        coeffs = self.backbone(feats)
        scale = torch.clamp(coeffs[:, 0] + 1.0, min=self.scale_bounds[0], max=self.scale_bounds[1])
        bias = coeffs[:, 1]
        quad = torch.tanh(coeffs[:, 2]) * self.quad_bound
        return scale, bias, quad

    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        if not mask.any():
            return (activation, None) if return_params else activation
        scale, bias, quad = self.forward_params(snapshot)
        mask = mask.to(activation.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scale = torch.where(mask, scale, torch.ones_like(scale))
        bias = torch.where(mask, bias, torch.zeros_like(bias))
        quad = torch.where(mask, quad, torch.zeros_like(quad))
        reshape_dims = [1, -1] + ([1, 1] if activation.dim() == 4 else [])
        centered = activation - snapshot.mean.to(activation.device).view(*reshape_dims)
        activation = (
            activation * scale.view(*reshape_dims)
            + bias.view(*reshape_dims)
            + quad.view(*reshape_dims) * centered * centered
        )
        if self.clip_margin > 0:
            limit = (self.ref_max_abs.to(scale.device) * self.clip_margin).view(*reshape_dims)
            activation = torch.clamp(activation, min=-limit, max=limit)
        if return_params:
            return activation, (scale, bias, quad)
        return activation


class QuantAwareResidualRepair(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int = 64, clip_margin: float = 1.25, act_bit: int = 2):
        super().__init__()
        self.num_channels = num_channels
        self.clip_margin = clip_margin
        self.act_bit = act_bit
        self.depthwise = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels)
        self.pointwise = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.scale_head = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )
        self.ref_mean = torch.zeros(num_channels)
        self.ref_std = torch.ones(num_channels)
        self.ref_max_abs = torch.ones(num_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        nn.init.zeros_(self.depthwise.bias)
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')
        nn.init.zeros_(self.pointwise.bias)
        for module in self.scale_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def set_clip_margin(self, clip_margin: float):
        self.clip_margin = clip_margin

    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        eps = 1e-6
        self.ref_mean = mean.clone()
        self.ref_std = torch.clamp(std.clone(), min=eps)
        self.ref_max_abs = torch.clamp(max_abs.clone(), min=eps)

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        max_abs = self.ref_max_abs.to(device)
        levels = 2 ** (self.act_bit - 1) - 1
        delta = torch.clamp(max_abs / levels, min=1e-6)
        view = [1, -1] + ([1, 1] if tensor.dim() == 4 else [])
        limit = max_abs.view(*view)
        delta_view = delta.view(*view)
        tensor = torch.clamp(tensor, -limit, limit)
        return torch.round(tensor / delta_view) * delta_view

    def _multi_scale(self, activation: torch.Tensor) -> torch.Tensor:
        pooled_2 = F.interpolate(F.adaptive_avg_pool2d(activation, 2), size=activation.shape[-2:])
        pooled_4 = F.interpolate(F.adaptive_avg_pool2d(activation, 4), size=activation.shape[-2:])
        return pooled_2 + pooled_4

    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        if not mask.any():
            return (activation, None) if return_params else activation
        device = activation.device
        original_dim = activation.dim()
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        mask = mask.to(device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        activation_device = activation.device
        self.depthwise = self.depthwise.to(activation_device)
        self.pointwise = self.pointwise.to(activation_device)
        self.scale_head = self.scale_head.to(activation_device)

        residual = self.depthwise(activation)
        residual = residual + self._multi_scale(activation)
        residual = self.pointwise(residual)
        # channel gating based on statistics
        stats = torch.stack([
            (snapshot.mean.to(device) - self.ref_mean.to(device)),
            snapshot.std.to(device) / torch.clamp(self.ref_std.to(device), min=1e-6) - 1.0,
            snapshot.max_abs.to(device) / torch.clamp(self.ref_max_abs.to(device), min=1e-6) - 1.0,
            torch.full_like(snapshot.mean.to(device), float(self.act_bit)),
            torch.full_like(snapshot.mean.to(device), mask.float().mean().item()),
            torch.zeros_like(snapshot.mean.to(device)),
        ], dim=1)
        coeffs = self.scale_head(stats)
        gain = torch.sigmoid(coeffs[:, 0])
        bias = coeffs[:, 1]
        quad = torch.tanh(coeffs[:, 2])
        view = [1, -1] + ([1, 1] if activation.dim() == 4 else [])
        mask_view = mask.view(*view)
        residual = residual * mask_view
        residual = gain.view(*view) * residual + bias.view(*view)
        if quad.abs().max() > 0:
            residual = residual + quad.view(*view) * (activation - snapshot.mean.to(device).view(*view)) ** 2
        activation = activation + residual
        if self.clip_margin > 0:
            limit = (self.ref_max_abs.to(device) * self.clip_margin).view(*view)
            activation = torch.clamp(activation, min=-limit, max=limit)
        params = {'residual': residual}
        if original_dim == 2:
            activation = activation.squeeze(-1).squeeze(-1)
            params = {'residual': residual.squeeze(-1).squeeze(-1)}
        if return_params:
            return activation, params
        return activation


class LightweightDenoiser(nn.Module):
    """
    轻量级去噪自编码器（方案A）。
    
    特点：
    - Always-on，无门控机制，完全可导
    - 利用空间邻域信息（3×3 depthwise conv）修复孤立噪点
    - 适合 SEU 这种脉冲噪声
    - 简单高效，loss 容易下降
    """
    def __init__(self, num_channels: int, clip_margin: float = 1.25):
        super().__init__()
        self.num_channels = num_channels
        self.clip_margin = clip_margin
        
        # 核心修复块：利用空间邻域信息修复孤立噪点
        self.depthwise = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=3, padding=1,
            groups=num_channels,  # 深度可分离，轻量
            bias=False
        )
        self.pointwise = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        
        # 参考统计（用于动态限幅）
        self.ref_max_abs = torch.ones(num_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')
    
    def set_clip_margin(self, clip_margin: float):
        self.clip_margin = clip_margin
    
    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        """更新参考统计（仅需要 max_abs 用于限幅）"""
        eps = 1e-6
        self.ref_max_abs = torch.clamp(max_abs.clone(), min=eps)
    
    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        """
        修复激活值（always-on，忽略 mask）。
        
        Args:
            activation: 输入激活值 [B, C, H, W] 或 [B, C]
            snapshot: 通道统计快照（未使用，为接口兼容性保留）
            mask: 通道掩码（未使用，为接口兼容性保留）
            return_params: 是否返回参数（用于训练时的正则化）
        
        Returns:
            修复后的激活值
        """
        device = activation.device
        original_dim = activation.dim()
        
        # 处理 2D 输入（linear 层）
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        
        # 确保模块在正确的设备上
        activation_device = activation.device
        self.depthwise = self.depthwise.to(activation_device)
        self.pointwise = self.pointwise.to(activation_device)
        
        # Always-on 修复：无门控，完全可导
        residual = self.depthwise(activation)
        residual = F.relu(residual)  # 非线性激活
        residual = self.pointwise(residual)
        
        # 残差连接
        activation = activation + residual
        
        # 动态限幅（关键！防止数值溢出）
        if self.clip_margin > 0:
            view = [1, -1] + ([1, 1] if activation.dim() == 4 else [])
            limit = (self.ref_max_abs.to(device) * self.clip_margin).view(*view)
            activation = torch.clamp(activation, min=-limit, max=limit)
        
        # 恢复原始维度
        if original_dim == 2:
            activation = activation.squeeze(-1).squeeze(-1)
            residual = residual.squeeze(-1).squeeze(-1)
        
        if return_params:
            return activation, {'residual': residual}
        return activation


class DenoiseRestorer(nn.Module):
    """
    轻量级残差去噪修复模块 (Residual Denoising Restorer)
    
    适用于 Conv 层后、ReLU 激活前的激活值修复。
    使用下一层激活量化器的边界进行最终限幅。
    """
    def __init__(self, channels: int = 64, quant_min: float = -5.0, quant_max: float = 5.0, 
                 next_layer_module: Optional[nn.Module] = None):
        """
        Args:
            channels: 当前层激活值的通道数 (AlexNet features.0 为 64)
            quant_min: 初始量化边界下限（备用，如果无法从下一层获取）
            quant_max: 初始量化边界上限（备用，如果无法从下一层获取）
            next_layer_module: 下一层模块（QuanConv2d 或 QuanLinear），用于获取激活量化边界
        """
        super().__init__()
        
        # 1. 存储下一层模块引用（用于动态获取激活量化边界）
        self.next_layer_module = next_layer_module
        
        # 2. 注册备用量化边界（如果无法从下一层获取时使用）
        self.register_buffer('q_min_fallback', torch.tensor(float(quant_min)))
        self.register_buffer('q_max_fallback', torch.tensor(float(quant_max)))
        
        # 2. 核心结构：残差预测分支 (Correction_Branch)
        self.residual_estimator = nn.Sequential(
            # --- 3x3 深度卷积 (Spatial Awareness) ---
            # dilation=2: 扩大感受野，从 3x3 扩展到 5x5 有效感受野
            # padding=2: 保持输出尺寸不变 (padding = dilation * (kernel_size - 1) / 2)
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, 
                      dilation=3, groups=channels, bias=True),
            
            # ReLU：非线性判断异常
            nn.ReLU(inplace=True),
            
            # --- 1x1 点卷积 (Channel Fusion / Output Correction Value) ---
            # 注意：最后不加 ReLU，以允许输出负修正量
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )
        
        # 3. 初始化：确保 Restorer 初始为 Identity Mapping
        # 这是残差网络设计的最佳实践，确保训练开始时 Loss 很低
        try:
            # 将最后的 1x1 卷积权重和偏置初始化为 0
            nn.init.constant_(self.residual_estimator[-1].weight, 0)
            nn.init.constant_(self.residual_estimator[-1].bias, 0)
        except:
            print("Warning: Failed to initialize last layer to zero.")
    
    def set_quant_bounds(self, quant_min: float, quant_max: float):
        """更新量化边界"""
        self.q_min_fallback.fill_(float(quant_min))
        self.q_max_fallback.fill_(float(quant_max))
    
    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        """更新参考统计（用于估算量化边界）"""
        # 从统计信息估算量化边界（如果未显式设置）
        # quant_max ≈ max_abs * 1.25 (clip_margin)
        # quant_min = -quant_max (假设对称量化)
        if self.q_max_fallback.item() == 5.0 and self.q_min_fallback.item() == -5.0:  # 默认值
            estimated_max = max_abs.max().item() * 1.25
            self.q_max_fallback.fill_(estimated_max)
            self.q_min_fallback.fill_(-estimated_max)
    
    def _get_quant_bounds_from_next_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从下一层的激活量化器获取量化边界。
        
        Returns:
            (q_min, q_max): 量化边界，如果无法获取则返回备用量化边界
        """
        if self.next_layer_module is None:
            return self.q_min_fallback, self.q_max_fallback
        
        try:
            # 获取下一层的激活量化器
            if not hasattr(self.next_layer_module, 'quan_a_fn'):
                return self.q_min_fallback, self.q_max_fallback
            
            quan_a_fn = self.next_layer_module.quan_a_fn
            if quan_a_fn is None:
                return self.q_min_fallback, self.q_max_fallback
            
            # 获取下一层的激活位宽
            if hasattr(self.next_layer_module, 'fixed_bits') and self.next_layer_module.fixed_bits is not None:
                abits = self.next_layer_module.fixed_bits[1]  # (wbits, abits)
            elif hasattr(self.next_layer_module, 'bits') and self.next_layer_module.bits is not None:
                abits = self.next_layer_module.bits[1]  # (wbits, abits)
            else:
                return self.q_min_fallback, self.q_max_fallback
            
            # 如果位宽 >= 32，表示不使用量化
            if abits is None or abits >= 32:
                return self.q_min_fallback, self.q_max_fallback
            
            # 获取 scale
            from quan.quantizer.lsq import LsqQuan
            if not isinstance(quan_a_fn, LsqQuan):
                return self.q_min_fallback, self.q_max_fallback
            
            scale = quan_a_fn.get_scale(abits, detach=True)
            if scale is None:
                return self.q_min_fallback, self.q_max_fallback
            
            # 计算量化阈值 (thd_neg, thd_pos)
            # 使用与 LsqQuan 相同的逻辑
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
            
            # 计算量化边界: scale * thd_neg, scale * thd_pos
            q_min = scale * thd_neg
            q_max = scale * thd_pos
            
            return q_min, q_max
            
        except Exception as e:
            # 如果出错，返回备用量化边界
            print(f"Warning: Failed to get quant bounds from next layer: {e}")
            return self.q_min_fallback, self.q_max_fallback
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 包含 SEU 错误的浮点激活值 [N, C, H, W]
        Returns:
            修复后的激活值 [N, C, H, W]
        """
        # 1. 计算预测的修正量 Delta
        noise_pred = self.residual_estimator(x)
        
        # 2. 残差连接 (Denoise)
        out = x + noise_pred
        
        # 3. 边界限幅 (Clipping)
        # 从下一层的激活量化器动态获取量化边界
        q_min, q_max = self._get_quant_bounds_from_next_layer()
        out = torch.clamp(out, min=q_min, max=q_max)
        
        return out
    
    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        """
        修复激活值（接口兼容性方法）。
        
        Args:
            activation: 输入激活值 [B, C, H, W] 或 [B, C]
            snapshot: 通道统计快照（未使用，为接口兼容性保留）
            mask: 通道掩码（未使用，为接口兼容性保留）
            return_params: 是否返回参数（用于训练时的正则化）
        
        Returns:
            修复后的激活值
        """
        device = activation.device
        original_dim = activation.dim()
        
        # 处理 2D 输入（linear 层）
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        
        # 确保模块在正确的设备上
        activation_device = activation.device
        self.residual_estimator = self.residual_estimator.to(activation_device)
        # q_min_fallback 和 q_max_fallback 是 buffer，会自动移动到正确的设备
        
        # 使用 forward 方法修复
        repaired = self.forward(activation)
        
        # 恢复原始维度
        if original_dim == 2:
            repaired = repaired.squeeze(-1).squeeze(-1)
            activation = activation.squeeze(-1).squeeze(-1)
        
        if return_params:
            residual = repaired - activation
            return repaired, {'residual': residual}
        return repaired


class StandardCorrectionUnit(nn.Module):
    """
    Standard Dilated Correction Unit (标准空洞修正单元) - 增强版
    
    Restorer V4 的核心单元，采用 ResNet-in-ResNet 风格，带有内部残差连接。
    增强版添加了 BatchNorm 来稳定训练，并支持残差缩放。
    结构：X_{i+1} = X_i + α * ReLU(BN(Standard Conv_{3×3, d=2}(X_i)))
    """
    def __init__(self, channels: int, use_bn: bool = True, residual_scale: float = 1.0):
        """
        Args:
            channels: 通道数
            use_bn: 是否使用 BatchNorm（默认 True，有助于稳定训练）
            residual_scale: 残差缩放因子（默认 1.0，可以尝试 0.1 来稳定训练）
        """
        super().__init__()
        # Standard Conv 3×3 with dilation=2
        # padding = dilation * (kernel_size - 1) / 2 = 2 * (3 - 1) / 2 = 2
        self.conv = nn.Conv2d(
            channels, channels, 
            kernel_size=3, 
            padding=2, 
            dilation=2, 
            bias=not use_bn  # 如果使用 BN，则不需要 bias
        )
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_scale = residual_scale
        
        # 初始化：使用零初始化（确保初始为 Identity Mapping）
        # 这对于残差网络很重要，确保训练开始时 loss 很低
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入激活值 [N, C, H, W]
        Returns:
            输出激活值 [N, C, H, W]
        """
        # X_{i+1} = X_i + α * ReLU(BN(Standard Conv_{3×3, d=2}(X_i)))
        residual = self.conv(x)
        if self.use_bn:
            residual = self.bn(residual)
        residual = self.relu(residual)
        return x + self.residual_scale * residual


class ActivationReconstructor(nn.Module):
    """
    激活值重建器：专门设计用于从损坏的激活值重建正确的激活值
    
    核心思想：
    1. **残差学习**：学习 clean - fault 的残差，而不是直接学习 clean
       - 残差通常比原始激活值更稀疏、更容易学习
       - 初始化为 Identity Mapping，确保训练开始时 loss 很低
    2. **多尺度特征提取**：结合局部（3x3）和全局（1x1）信息
       - 局部：捕获空间局部错误模式
       - 全局：捕获通道间关系和全局统计特征
    3. **注意力机制**：学习哪些区域需要修复
       - 空间注意力：关注错误发生的位置
       - 通道注意力：关注哪些通道受影响最大
    4. **渐进式重建**：通过多个阶段逐步修复
       - 每个阶段专注于不同尺度的错误
    5. **统计先验**：利用 clean 激活值的统计特征作为先验
    """
    def __init__(self, channels: int = 64, quant_min: float = -5.0, quant_max: float = 5.0, 
                 next_layer_module: Optional[nn.Module] = None, num_stages: int = 2):
        """
        Args:
            channels: 当前层激活值的通道数
            quant_min: 量化边界下限（备用）
            quant_max: 量化边界上限（备用）
            next_layer_module: 下一层模块（用于获取激活量化边界）
            num_stages: 渐进式重建的阶段数（默认 2）
        """
        super().__init__()
        
        self.next_layer_module = next_layer_module
        self.register_buffer('q_min_fallback', torch.tensor(float(quant_min)))
        self.register_buffer('q_max_fallback', torch.tensor(float(quant_max)))
        self.num_stages = num_stages
        
        # 每个阶段的结构：多尺度特征提取 + 注意力 + 残差生成
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = self._build_stage(channels, stage_idx=i)
            self.stages.append(stage)
    
    def _build_stage(self, channels: int, stage_idx: int) -> nn.Module:
        """构建单个重建阶段"""
        hidden_channels = channels * 2  # 隐藏层通道数是输入的 2 倍
        
        # 1. 多尺度特征提取
        local_conv = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        global_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        
        # 2. 特征融合
        fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # 3. 空间注意力：学习哪些空间位置需要修复
        spatial_attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),  # 输出 0-1 之间的注意力权重
        )
        
        # 4. 通道注意力：学习哪些通道受影响最大
        channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(hidden_channels, hidden_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 4, hidden_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),  # 输出 0-1 之间的注意力权重
        )
        
        # 5. 残差生成：生成修复残差
        residual_gen = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),  # 生成残差
        )
        
        # 初始化：确保初始为 Identity Mapping（残差为 0）
        nn.init.zeros_(local_conv.weight)
        nn.init.zeros_(global_conv.weight)
        nn.init.zeros_(fusion[0].weight)
        nn.init.zeros_(residual_gen[-1].weight)
        nn.init.zeros_(residual_gen[-1].bias)  # 残差初始为 0
        
        # 注意力机制初始化为均匀（不偏向任何位置/通道）
        nn.init.normal_(spatial_attention[0].weight, mean=0.0, std=0.01)
        nn.init.normal_(spatial_attention[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(spatial_attention[2].bias, 0.0)  # sigmoid 输出 ≈ 0.5
        
        nn.init.normal_(channel_attention[1].weight, mean=0.0, std=0.01)
        nn.init.normal_(channel_attention[3].weight, mean=0.0, std=0.01)
        nn.init.constant_(channel_attention[3].bias, 0.0)  # sigmoid 输出 ≈ 0.5
        
        return nn.ModuleDict({
            'local_conv': local_conv,
            'global_conv': global_conv,
            'fusion': fusion,
            'spatial_attention': spatial_attention,
            'channel_attention': channel_attention,
            'residual_gen': residual_gen,
        })
    
    def set_quant_bounds(self, quant_min: float, quant_max: float):
        """更新量化边界"""
        self.q_min_fallback.fill_(float(quant_min))
        self.q_max_fallback.fill_(float(quant_max))
    
    def set_clip_margin(self, clip_margin: float):
        """接口兼容性方法"""
        pass
    
    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        """更新参考统计"""
        if self.q_max_fallback.item() == 5.0 and self.q_min_fallback.item() == -5.0:
            estimated_max = max_abs.max().item() * 1.25
            self.q_max_fallback.fill_(estimated_max)
            self.q_min_fallback.fill_(-estimated_max)
    
    def _get_quant_bounds_from_next_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从下一层的激活量化器获取量化边界"""
        if self.next_layer_module is None:
            return self.q_min_fallback, self.q_max_fallback
        
        try:
            if not hasattr(self.next_layer_module, 'quan_a_fn'):
                return self.q_min_fallback, self.q_max_fallback
            
            quan_a_fn = self.next_layer_module.quan_a_fn
            if quan_a_fn is None:
                return self.q_min_fallback, self.q_max_fallback
            
            if hasattr(self.next_layer_module, 'fixed_bits') and self.next_layer_module.fixed_bits is not None:
                abits = self.next_layer_module.fixed_bits[1]
            elif hasattr(self.next_layer_module, 'bits') and self.next_layer_module.bits is not None:
                abits = self.next_layer_module.bits[1]
            else:
                return self.q_min_fallback, self.q_max_fallback
            
            if abits is None or abits >= 32:
                return self.q_min_fallback, self.q_max_fallback
            
            from quan.quantizer.lsq import LsqQuan
            if not isinstance(quan_a_fn, LsqQuan):
                return self.q_min_fallback, self.q_max_fallback
            
            scale = quan_a_fn.get_scale(abits, detach=True)
            if scale is None:
                return self.q_min_fallback, self.q_max_fallback
            
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
            
            q_min = scale * thd_neg
            q_max = scale * thd_pos
            
            return q_min, q_max
            
        except Exception as e:
            return self.q_min_fallback, self.q_max_fallback
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        渐进式重建激活值
        
        Args:
            x: 包含 SEU 错误的浮点激活值 [N, C, H, W]
        Returns:
            重建后的激活值 [N, C, H, W]
        """
        current = x
        
        # 通过多个阶段逐步修复
        for stage in self.stages:
            # 1. 多尺度特征提取
            local_feat = stage['local_conv'](current)
            global_feat = stage['global_conv'](current)
            
            # 2. 特征融合
            combined = torch.cat([local_feat, global_feat], dim=1)
            features = stage['fusion'](combined)
            
            # 3. 应用注意力机制
            spatial_att = stage['spatial_attention'](features)  # [N, 1, H, W]
            channel_att = stage['channel_attention'](features)  # [N, C, 1, 1]
            
            # 4. 生成残差
            residual = stage['residual_gen'](features)
            
            # 5. 应用注意力加权的残差
            attended_residual = residual * spatial_att * channel_att
            
            # 6. 残差连接：current = current + attended_residual
            current = current + attended_residual
        
        # 7. 边界限幅
        q_min, q_max = self._get_quant_bounds_from_next_layer()
        if isinstance(q_min, torch.Tensor):
            q_min = q_min.item() if q_min.numel() == 1 else q_min
        if isinstance(q_max, torch.Tensor):
            q_max = q_max.item() if q_max.numel() == 1 else q_max
        current = torch.clamp(current, min=q_min, max=q_max)
        
        return current
    
    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        """修复激活值（接口兼容性方法）"""
        device = activation.device
        original_dim = activation.dim()
        
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        
        # 确保模块在正确的设备上
        self.to(device)
        
        # 使用 forward 方法重建
        reconstructed = self.forward(activation)
        
        if original_dim == 2:
            reconstructed = reconstructed.squeeze(-1).squeeze(-1)
            activation = activation.squeeze(-1).squeeze(-1)
        
        if return_params:
            residual = reconstructed - activation
            return reconstructed, {'residual': residual}
        return reconstructed


class ImprovedRestorer(nn.Module):
    """
    改进的 Restorer：更简单、更有效的架构
    
    设计原则：
    1. 浅而宽的网络（2-3 层，但每层更宽）
    2. 门控机制：只在检测到错误时应用修复
    3. 多尺度特征提取：结合局部和全局信息
    4. 更好的初始化：确保初始为 Identity Mapping
    5. 自适应残差缩放：根据错误严重程度调整修复强度
    """
    def __init__(self, channels: int = 64, quant_min: float = -5.0, quant_max: float = 5.0, 
                 next_layer_module: Optional[nn.Module] = None, hidden_ratio: float = 2.0):
        """
        Args:
            channels: 当前层激活值的通道数
            quant_min: 量化边界下限（备用）
            quant_max: 量化边界上限（备用）
            next_layer_module: 下一层模块（用于获取激活量化边界）
            hidden_ratio: 隐藏层通道数相对于输入通道数的倍数（默认 2.0，增加容量）
        """
        super().__init__()
        
        self.next_layer_module = next_layer_module
        self.register_buffer('q_min_fallback', torch.tensor(float(quant_min)))
        self.register_buffer('q_max_fallback', torch.tensor(float(quant_max)))
        
        hidden_channels = int(channels * hidden_ratio)
        
        # 多尺度特征提取：结合局部（3x3）和全局（1x1）信息
        # 1. 局部特征提取（3x3 conv，捕获空间局部错误）
        self.local_conv = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        
        # 2. 全局特征提取（1x1 conv，捕获通道间关系）
        self.global_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        
        # 3. 特征融合和压缩
        self.fusion = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        
        # 4. 输出层：生成修正量 Δ
        self.output_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        
        # 5. 门控机制：学习何时应用修复（可选，用于减少对 clean 数据的影响）
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  # 输出 0-1 之间的门控值
        )
        
        # 初始化：确保初始为 Identity Mapping
        # 所有卷积层初始化为零，这样初始时 Δ ≈ 0
        nn.init.zeros_(self.local_conv.weight)
        nn.init.zeros_(self.local_conv.bias)
        nn.init.zeros_(self.global_conv.weight)
        nn.init.zeros_(self.global_conv.bias)
        nn.init.zeros_(self.fusion[1].weight)
        nn.init.zeros_(self.fusion[1].bias)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        
        # 门控层初始化为偏向不应用修复（初始门控值较小）
        nn.init.normal_(self.gate[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.normal_(self.gate[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate[2].bias, -2.0)  # 初始 sigmoid 输出 ≈ 0.12
    
    def set_quant_bounds(self, quant_min: float, quant_max: float):
        """更新量化边界"""
        self.q_min_fallback.fill_(float(quant_min))
        self.q_max_fallback.fill_(float(quant_max))
    
    def set_clip_margin(self, clip_margin: float):
        """接口兼容性方法"""
        pass
    
    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        """更新参考统计"""
        if self.q_max_fallback.item() == 5.0 and self.q_min_fallback.item() == -5.0:
            estimated_max = max_abs.max().item() * 1.25
            self.q_max_fallback.fill_(estimated_max)
            self.q_min_fallback.fill_(-estimated_max)
    
    def _get_quant_bounds_from_next_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从下一层的激活量化器获取量化边界"""
        if self.next_layer_module is None:
            return self.q_min_fallback, self.q_max_fallback
        
        try:
            if not hasattr(self.next_layer_module, 'quan_a_fn'):
                return self.q_min_fallback, self.q_max_fallback
            
            quan_a_fn = self.next_layer_module.quan_a_fn
            if quan_a_fn is None:
                return self.q_min_fallback, self.q_max_fallback
            
            if hasattr(self.next_layer_module, 'fixed_bits') and self.next_layer_module.fixed_bits is not None:
                abits = self.next_layer_module.fixed_bits[1]
            elif hasattr(self.next_layer_module, 'bits') and self.next_layer_module.bits is not None:
                abits = self.next_layer_module.bits[1]
            else:
                return self.q_min_fallback, self.q_max_fallback
            
            if abits is None or abits >= 32:
                return self.q_min_fallback, self.q_max_fallback
            
            from quan.quantizer.lsq import LsqQuan
            if not isinstance(quan_a_fn, LsqQuan):
                return self.q_min_fallback, self.q_max_fallback
            
            scale = quan_a_fn.get_scale(abits, detach=True)
            if scale is None:
                return self.q_min_fallback, self.q_max_fallback
            
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
            
            q_min = scale * thd_neg
            q_max = scale * thd_pos
            
            return q_min, q_max
            
        except Exception as e:
            return self.q_min_fallback, self.q_max_fallback
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 包含 SEU 错误的浮点激活值 [N, C, H, W]
        Returns:
            修复后的激活值 [N, C, H, W]
        """
        # 1. 多尺度特征提取
        local_feat = self.local_conv(x)  # 局部特征
        global_feat = self.global_conv(x)  # 全局特征
        
        # 2. 特征融合
        combined = torch.cat([local_feat, global_feat], dim=1)
        features = self.fusion(combined)
        
        # 3. 生成修正量 Δ
        delta = self.output_conv(features)
        
        # 4. 门控机制：学习何时应用修复
        gate_value = self.gate(x)  # [N, 1, H, W]
        
        # 5. 应用门控的修正量
        gated_delta = delta * gate_value
        
        # 6. 外部残差连接
        out = x + gated_delta
        
        # 7. 边界限幅
        q_min, q_max = self._get_quant_bounds_from_next_layer()
        if isinstance(q_min, torch.Tensor):
            q_min = q_min.item() if q_min.numel() == 1 else q_min
        if isinstance(q_max, torch.Tensor):
            q_max = q_max.item() if q_max.numel() == 1 else q_max
        out = torch.clamp(out, min=q_min, max=q_max)
        
        return out
    
    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        """修复激活值（接口兼容性方法）"""
        device = activation.device
        original_dim = activation.dim()
        
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        
        # 确保模块在正确的设备上
        self.to(device)
        
        # 使用 forward 方法修复
        repaired = self.forward(activation)
        
        if original_dim == 2:
            repaired = repaired.squeeze(-1).squeeze(-1)
            activation = activation.squeeze(-1).squeeze(-1)
        
        if return_params:
            residual = repaired - activation
            return repaired, {'residual': residual, 'gate': None}  # gate 信息可以后续添加
        return repaired


class RestorerV4(nn.Module):
    """
    Restorer V4: 高容量结构，解决 BER=10^-2 下的严重欠拟合 - 增强版
    
    结构：
    1. 外部残差连接：Y_repaired = Clamp(X_corrupted + Δ)
    2. 核心修正分支：由 N 个堆叠的 Standard Dilated Correction Unit 组成（带 BatchNorm）
    3. 最终映射：通过 1×1 卷积将特征映射到 Δ
    4. 增强：添加 BatchNorm、残差缩放、更深的网络
    """
    def __init__(self, channels: int = 64, quant_min: float = -5.0, quant_max: float = 5.0, 
                 next_layer_module: Optional[nn.Module] = None, num_blocks: int = 5, 
                 use_bn: bool = True, residual_scale: float = 1.0, expand_channels: int = 0):
        """
        Args:
            channels: 当前层激活值的通道数
            quant_min: 量化边界下限（备用）
            quant_max: 量化边界上限（备用）
            next_layer_module: 下一层模块（用于获取激活量化边界）
            num_blocks: 堆叠的 Standard Correction Unit 数量（默认 5，增加容量）
            use_bn: 是否在 Correction Unit 中使用 BatchNorm（默认 True）
            residual_scale: 残差缩放因子（默认 1.0）
            expand_channels: 是否扩展通道数（0 表示不扩展，>0 表示扩展的倍数）
        """
        super().__init__()
        
        # 存储下一层模块引用
        self.next_layer_module = next_layer_module
        
        # 注册备用量化边界
        self.register_buffer('q_min_fallback', torch.tensor(float(quant_min)))
        self.register_buffer('q_max_fallback', torch.tensor(float(quant_max)))
        
        # 可选的通道扩展（增加模型容量）
        if expand_channels > 0:
            self.expand_conv = nn.Conv2d(channels, channels * expand_channels, kernel_size=1, bias=False)
            self.expand_bn = nn.BatchNorm2d(channels * expand_channels) if use_bn else None
            self.project_conv = nn.Conv2d(channels * expand_channels, channels, kernel_size=1, bias=False)
            self.project_bn = nn.BatchNorm2d(channels) if use_bn else None
            inner_channels = channels * expand_channels
        else:
            self.expand_conv = None
            self.expand_bn = None
            self.project_conv = None
            self.project_bn = None
            inner_channels = channels
        
        # 核心修正分支：N 个堆叠的 Standard Correction Unit
        self.correction_blocks = nn.ModuleList([
            StandardCorrectionUnit(inner_channels, use_bn=use_bn, residual_scale=residual_scale) 
            for _ in range(num_blocks)
        ])
        
        # 最终 1×1 卷积：将特征映射到修正量 Δ
        self.final_conv = nn.Conv2d(inner_channels, channels, kernel_size=1, bias=True)
        
        # 初始化：使用零初始化（确保初始为 Identity Mapping）
        # 这对于残差网络很重要，确保训练开始时 loss 很低
        if self.expand_conv is not None:
            nn.init.zeros_(self.expand_conv.weight)
            nn.init.zeros_(self.project_conv.weight)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
    
    def set_quant_bounds(self, quant_min: float, quant_max: float):
        """更新量化边界"""
        self.q_min_fallback.fill_(float(quant_min))
        self.q_max_fallback.fill_(float(quant_max))
    
    def set_clip_margin(self, clip_margin: float):
        """
        设置 clip_margin（接口兼容性方法）
        
        RestorerV4 不使用 clip_margin，而是使用量化边界。
        此方法仅为接口兼容性保留。
        """
        # RestorerV4 不使用 clip_margin，但保留此方法以保持接口兼容性
        pass
    
    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        """更新参考统计（用于估算量化边界）"""
        if self.q_max_fallback.item() == 5.0 and self.q_min_fallback.item() == -5.0:
            estimated_max = max_abs.max().item() * 1.25
            self.q_max_fallback.fill_(estimated_max)
            self.q_min_fallback.fill_(-estimated_max)
    
    def _get_quant_bounds_from_next_layer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从下一层的激活量化器获取量化边界"""
        if self.next_layer_module is None:
            return self.q_min_fallback, self.q_max_fallback
        
        try:
            if not hasattr(self.next_layer_module, 'quan_a_fn'):
                return self.q_min_fallback, self.q_max_fallback
            
            quan_a_fn = self.next_layer_module.quan_a_fn
            if quan_a_fn is None:
                return self.q_min_fallback, self.q_max_fallback
            
            if hasattr(self.next_layer_module, 'fixed_bits') and self.next_layer_module.fixed_bits is not None:
                abits = self.next_layer_module.fixed_bits[1]
            elif hasattr(self.next_layer_module, 'bits') and self.next_layer_module.bits is not None:
                abits = self.next_layer_module.bits[1]
            else:
                return self.q_min_fallback, self.q_max_fallback
            
            if abits is None or abits >= 32:
                return self.q_min_fallback, self.q_max_fallback
            
            from quan.quantizer.lsq import LsqQuan
            if not isinstance(quan_a_fn, LsqQuan):
                return self.q_min_fallback, self.q_max_fallback
            
            scale = quan_a_fn.get_scale(abits, detach=True)
            if scale is None:
                return self.q_min_fallback, self.q_max_fallback
            
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
            
            q_min = scale * thd_neg
            q_max = scale * thd_pos
            
            return q_min, q_max
            
        except Exception as e:
            return self.q_min_fallback, self.q_max_fallback
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 包含 SEU 错误的浮点激活值 [N, C, H, W]
        Returns:
            修复后的激活值 [N, C, H, W]
        """
        # 1. 可选的通道扩展（增加模型容量）
        if self.expand_conv is not None:
            features = self.expand_conv(x)
            if self.expand_bn is not None:
                features = self.expand_bn(features)
            features = F.relu(features, inplace=True)
        else:
            features = x
        
        # 2. 通过堆叠的 Standard Correction Units
        for block in self.correction_blocks:
            features = block(features)
        
        # 3. 可选的通道投影（如果使用了扩展）
        if self.project_conv is not None:
            features = self.project_conv(features)
            if self.project_bn is not None:
                features = self.project_bn(features)
        
        # 4. 通过最终 1×1 卷积映射到修正量 Δ
        delta = self.final_conv(features)
        
        # 5. 外部残差连接：Y_repaired = X_corrupted + Δ
        out = x + delta
        
        # 6. 边界限幅 (Clipping)
        q_min, q_max = self._get_quant_bounds_from_next_layer()
        # 确保 q_min 和 q_max 是标量或与 out 兼容的形状
        if isinstance(q_min, torch.Tensor):
            q_min = q_min.item() if q_min.numel() == 1 else q_min
        if isinstance(q_max, torch.Tensor):
            q_max = q_max.item() if q_max.numel() == 1 else q_max
        out = torch.clamp(out, min=q_min, max=q_max)
        
        return out
    
    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        """
        修复激活值（接口兼容性方法）
        
        Args:
            activation: 输入激活值 [B, C, H, W] 或 [B, C]
            snapshot: 通道统计快照（未使用）
            mask: 通道掩码（未使用）
            return_params: 是否返回参数
        """
        device = activation.device
        original_dim = activation.dim()
        
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        
        # 确保模块在正确的设备上
        for block in self.correction_blocks:
            block.conv = block.conv.to(device)
            if hasattr(block, 'bn') and block.bn is not None:
                block.bn = block.bn.to(device)
        if self.expand_conv is not None:
            self.expand_conv = self.expand_conv.to(device)
            if self.expand_bn is not None:
                self.expand_bn = self.expand_bn.to(device)
        if self.project_conv is not None:
            self.project_conv = self.project_conv.to(device)
            if self.project_bn is not None:
                self.project_bn = self.project_bn.to(device)
        self.final_conv = self.final_conv.to(device)
        
        # 使用 forward 方法修复
        repaired = self.forward(activation)
        
        if original_dim == 2:
            repaired = repaired.squeeze(-1).squeeze(-1)
            activation = activation.squeeze(-1).squeeze(-1)
        
        if return_params:
            residual = repaired - activation
            return repaired, {'residual': residual}
        return repaired


class StackedDenoiseRestorer(nn.Module):
    """
    堆叠 DenoiseRestorer：通过多层逐步修复错误
    
    使用多个 DenoiseRestorer 模块堆叠，每个模块逐步修复错误，类似于残差网络。
    这可以帮助处理更严重的错误，并可能提升修复效果。
    """
    def __init__(self, channels: int = 64, quant_min: float = -5.0, quant_max: float = 5.0, 
                 next_layer_module: Optional[nn.Module] = None, num_stages: int = 2):
        """
        Args:
            channels: 当前层激活值的通道数
            quant_min: 量化边界下限
            quant_max: 量化边界上限
            next_layer_module: 下一层模块（用于获取量化边界）
            num_stages: 堆叠的 DenoiseRestorer 数量（默认 2）
        """
        super().__init__()
        self.num_stages = num_stages
        self.next_layer_module = next_layer_module
        
        # 创建多个 DenoiseRestorer 模块
        self.stages = nn.ModuleList([
            DenoiseRestorer(
                channels=channels,
                quant_min=quant_min,
                quant_max=quant_max,
                next_layer_module=None,  # 只有最后一个阶段需要 next_layer_module
            ) for _ in range(num_stages - 1)
        ])
        
        # 最后一个阶段使用 next_layer_module（用于获取量化边界）
        self.stages.append(
            DenoiseRestorer(
                channels=channels,
                quant_min=quant_min,
                quant_max=quant_max,
                next_layer_module=next_layer_module,
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过多个阶段逐步修复激活值
        
        Args:
            x: 包含 SEU 错误的浮点激活值 [N, C, H, W]
        Returns:
            修复后的激活值 [N, C, H, W]
        """
        out = x
        # 依次通过每个阶段
        for stage in self.stages:
            out = stage(out)
        return out
    
    def repair(
        self,
        activation: torch.Tensor,
        snapshot: ChannelSnapshot,
        mask: torch.Tensor,
        return_params: bool = False,
    ):
        """
        修复激活值（接口兼容性方法）
        
        Args:
            activation: 输入激活值 [B, C, H, W] 或 [B, C]
            snapshot: 通道统计快照（未使用）
            mask: 通道掩码（未使用）
            return_params: 是否返回参数
        """
        device = activation.device
        original_dim = activation.dim()
        
        # 处理 2D 输入
        if original_dim == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)
        
        # 确保所有阶段在正确的设备上
        for stage in self.stages:
            stage.residual_estimator = stage.residual_estimator.to(device)
        
        # 使用 forward 方法修复
        repaired = self.forward(activation)
        
        # 恢复原始维度
        if original_dim == 2:
            repaired = repaired.squeeze(-1).squeeze(-1)
            activation = activation.squeeze(-1).squeeze(-1)
        
        if return_params:
            residual = repaired - activation
            return repaired, {'residual': residual}
        return repaired
    
    def set_quant_bounds(self, quant_min: float, quant_max: float):
        """更新所有阶段的量化边界"""
        for stage in self.stages:
            stage.set_quant_bounds(quant_min, quant_max)
    
    def set_clip_margin(self, clip_margin: float):
        """
        设置 clip_margin（接口兼容性方法）
        
        StackedDenoiseRestorer 不使用 clip_margin，而是使用量化边界。
        此方法仅为接口兼容性保留。
        """
        # StackedDenoiseRestorer 不使用 clip_margin，但保留此方法以保持接口兼容性
        pass
    
    def update_reference(self, mean: torch.Tensor, std: torch.Tensor, max_abs: torch.Tensor):
        """更新所有阶段的参考统计"""
        for stage in self.stages:
            stage.update_reference(mean, std, max_abs)


class SensitiveLayerRestorer:
    def __init__(
        self,
        model: nn.Module,
        profile_path: str | Path,
        target_layers: Optional[Iterable[str]] = None,
        default_z_thresh: float = 3.0,
        default_std_ratio_bounds: Tuple[float, float] = (0.5, 2.0),
        default_clip_margin: float = 1.25,
        repair_mode: str = 'rule',
        fault_profile_path: Optional[str] = None,
        fault_profile_ber: float = 1e-1,
        ber_policy: Optional[Dict[str, Any]] = None,
        repair_head_state: Optional[Dict[str, Any]] = None,
        mlp_hidden_dim: int = 32,
    ):
        self.model = model
        self.clean_profile = _load_profile(profile_path)
        self.fault_profile = _load_profile(fault_profile_path) if fault_profile_path else None
        self.fault_profile_ber = fault_profile_ber
        profile_layers = self.clean_profile['layers']
        requested = list(target_layers) if target_layers else None

        self.layer_names = _resolve_layers(model, requested, profile_layers.keys())
        if not self.layer_names:
            raise ValueError('No valid target layers found for SensitiveLayerRestorer.')

        self.default_policy = {
            'mode': 'repair',
            'z_thresh': default_z_thresh,
            'std_ratio_bounds': default_std_ratio_bounds,
            'clip_margin': default_clip_margin,
        }
        self.policy_map = ber_policy or {}
        valid_modes = {'rule', 'mlp', 'mlp_local', 'mlp_poly', 'ms_residual', 'lightweight_denoiser', 'denoise_restorer', 'restorer_v4'}
        if repair_mode not in valid_modes:
            raise ValueError(f'Unsupported repair_mode {repair_mode}')
        if repair_mode == 'mlp':
            repair_mode = 'mlp_local'
        self.repair_mode = repair_mode
        self.learning_modes = {'mlp_local', 'mlp_poly', 'ms_residual', 'lightweight_denoiser', 'denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor'}
        self.hooks = []
        self.enabled = False
        self.detectors: Dict[str, ChannelDetector] = {}
        self.repairs: Dict[str, ChannelRepair | LearningChannelRepair] = {}
        self.active_policy = self.default_policy.copy()

        self.clean_stats_cache = {name: self.clean_profile['layers'][name] for name in self.layer_names}
        self.fault_stats_cache = {name: self.fault_profile['layers'][name] for name in self.layer_names} if self.fault_profile else None

        for name in self.layer_names:
            clean_stats = self.clean_stats_cache[name]
            detector = ChannelDetector(z_thresh=default_z_thresh, std_ratio_bounds=default_std_ratio_bounds)
            detector.update_reference(clean_stats['channel_mean'].float(), clean_stats['channel_std'].float())
            self.detectors[name] = detector

            if self.repair_mode == 'mlp_local':
                num_channels = clean_stats['channel_mean'].numel()
                repair = LearningChannelRepair(
                    num_channels=num_channels,
                    hidden_size=mlp_hidden_dim,
                    clip_margin=default_clip_margin,
                )
            elif self.repair_mode == 'mlp_poly':
                num_channels = clean_stats['channel_mean'].numel()
                repair = PolynomialChannelRepair(
                    num_channels=num_channels,
                    hidden_size=max(mlp_hidden_dim, 64),
                    clip_margin=default_clip_margin,
                )
            elif self.repair_mode == 'ms_residual':
                num_channels = clean_stats['channel_mean'].numel()
                repair = QuantAwareResidualRepair(
                    num_channels=num_channels,
                    hidden_size=max(mlp_hidden_dim, 64),
                    clip_margin=default_clip_margin,
                    act_bit=int(clean_stats.get('act_bit', 2)),
                )
            elif self.repair_mode == 'lightweight_denoiser':
                num_channels = clean_stats['channel_mean'].numel()
                repair = LightweightDenoiser(
                    num_channels=num_channels,
                    clip_margin=default_clip_margin,
                )
            elif self.repair_mode == 'denoise_restorer':
                num_channels = clean_stats['channel_mean'].numel()
                # 从统计信息估算量化边界（备用）
                max_abs = clean_stats.get('channel_max_abs', clean_stats['channel_mean'].abs())
                quant_max = max_abs.max().item() * default_clip_margin
                quant_min = -quant_max  # 假设对称量化
                
                # 查找下一层模块（用于动态获取激活量化边界）
                next_layer_module = self._find_next_quantized_layer(name)
                
                # 检查 checkpoint 中是否是堆叠的 DenoiseRestorer 或 RestorerV4
                num_stages = 1
                is_restorer_v4 = False
                num_blocks = 3
                
                if repair_head_state and name in repair_head_state:
                    state = repair_head_state[name]
                    # 检查是否是 RestorerV4（有 correction_blocks）
                    if any(k.startswith('correction_blocks.') for k in state.keys()):
                        is_restorer_v4 = True
                        block_keys = [k for k in state.keys() if k.startswith('correction_blocks.')]
                        block_indices = set()
                        for k in block_keys:
                            parts = k.split('.')
                            if len(parts) >= 2 and parts[0] == 'correction_blocks':
                                try:
                                    block_idx = int(parts[1])
                                    block_indices.add(block_idx)
                                except ValueError:
                                    pass
                        if block_indices:
                            num_blocks = max(block_indices) + 1
                            print(f"[{name}] Detected RestorerV4 with {num_blocks} correction blocks from checkpoint")
                    # 检查是否有 stages.0, stages.1 等键（StackedDenoiseRestorer）
                    elif any(k.startswith('stages.') for k in state.keys()):
                        stage_keys = [k for k in state.keys() if k.startswith('stages.')]
                        stage_indices = set()
                        for k in stage_keys:
                            parts = k.split('.')
                            if len(parts) >= 2 and parts[0] == 'stages':
                                try:
                                    stage_idx = int(parts[1])
                                    stage_indices.add(stage_idx)
                                except ValueError:
                                    pass
                        if stage_indices:
                            num_stages = max(stage_indices) + 1
                            print(f"[{name}] Detected StackedDenoiseRestorer with {num_stages} stages from checkpoint")
                
                if is_restorer_v4:
                    repair = RestorerV4(
                        channels=num_channels,
                        quant_min=quant_min,
                        quant_max=quant_max,
                        next_layer_module=next_layer_module,
                        num_blocks=num_blocks,
                    )
                elif num_stages > 1:
                    repair = StackedDenoiseRestorer(
                        channels=num_channels,
                        quant_min=quant_min,
                        quant_max=quant_max,
                        next_layer_module=next_layer_module,
                        num_stages=num_stages,
                    )
                else:
                    repair = DenoiseRestorer(
                        channels=num_channels,
                        quant_min=quant_min,
                        quant_max=quant_max,
                        next_layer_module=next_layer_module,
                    )
            elif self.repair_mode == 'restorer_v4':
                num_channels = clean_stats['channel_mean'].numel()
                max_abs = clean_stats.get('channel_max_abs', clean_stats['channel_mean'].abs())
                quant_max = max_abs.max().item() * default_clip_margin
                quant_min = -quant_max
                
                next_layer_module = self._find_next_quantized_layer(name)
                
                # 检查 checkpoint 中的块数量
                num_blocks = 3
                if repair_head_state and name in repair_head_state:
                    state = repair_head_state[name]
                    block_keys = [k for k in state.keys() if k.startswith('correction_blocks.')]
                    if block_keys:
                        block_indices = set()
                        for k in block_keys:
                            parts = k.split('.')
                            if len(parts) >= 2 and parts[0] == 'correction_blocks':
                                try:
                                    block_idx = int(parts[1])
                                    block_indices.add(block_idx)
                                except ValueError:
                                    pass
                        if block_indices:
                            num_blocks = max(block_indices) + 1
                            print(f"[{name}] Detected RestorerV4 with {num_blocks} correction blocks from checkpoint")
                
                repair = RestorerV4(
                    channels=num_channels,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    next_layer_module=next_layer_module,
                    num_blocks=num_blocks,
                )
            elif self.repair_mode == 'improved_restorer':
                num_channels = clean_stats['channel_mean'].numel()
                max_abs = clean_stats.get('channel_max_abs', clean_stats['channel_mean'].abs())
                quant_max = max_abs.max().item() * default_clip_margin
                quant_min = -quant_max
                
                next_layer_module = self._find_next_quantized_layer(name)
                
                repair = ImprovedRestorer(
                    channels=num_channels,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    next_layer_module=next_layer_module,
                    hidden_ratio=2.0,
                )
            else:
                repair = ChannelRepair(clip_margin=default_clip_margin)

            repair.update_reference(
                clean_stats['channel_mean'].float(),
                clean_stats['channel_std'].float(),
                clean_stats.get('channel_max_abs', clean_stats['channel_mean'].abs()).float(),
            )
            self.repairs[name] = repair

        if self.repair_mode in self.learning_modes and repair_head_state:
            for name, state in repair_head_state.items():
                if name in self.repairs and isinstance(self.repairs[name], nn.Module):
                    # 过滤掉 next_layer_module 相关的键（这是外部引用，不应该在 state_dict 中）
                    filtered_state = {k: v for k, v in state.items() if not k.startswith('next_layer_module.')}
                    if filtered_state:
                        # 使用 strict=False 来忽略不匹配的键（如 next_layer_module 相关的键）
                        self.repairs[name].load_state_dict(filtered_state, strict=False)

        self.set_operating_point(None)
    
    def _find_next_quantized_layer(self, current_layer_name: str) -> Optional[nn.Module]:
        """
        查找当前层之后的第一个量化层（QuanConv2d 或 QuanLinear）。
        
        Args:
            current_layer_name: 当前层的名称（如 'features.0'）
        
        Returns:
            下一层模块，如果找不到则返回 None
        """
        modules = dict(self.model.named_modules())
        layer_names = list(modules.keys())
        
        # 找到当前层在列表中的位置
        try:
            current_idx = layer_names.index(current_layer_name)
        except ValueError:
            return None
        
        # 从当前层之后查找第一个 QuanConv2d 或 QuanLinear
        for i in range(current_idx + 1, len(layer_names)):
            module = modules[layer_names[i]]
            if isinstance(module, (QuanConv2d, QuanLinear)):
                return module
        
        return None

    def learning_parameters(self):
        if self.repair_mode not in self.learning_modes:
            return []
        params = []
        for repair in self.repairs.values():
            if isinstance(repair, nn.Module):
                params += list(repair.parameters())
        return params

    def save_learning_state(self, path: str):
        if self.repair_mode not in self.learning_modes:
            raise RuntimeError('Learning state available only in learning modes.')
        state = {name: repair.state_dict() for name, repair in self.repairs.items()}
        torch.save(state, path)

    def load_learning_state(self, path: str):
        state = torch.load(path, map_location='cpu')
        for name, repair in self.repairs.items():
            if isinstance(repair, nn.Module) and name in state:
                repair.load_state_dict(state[name])

    def _blend_references(self, alpha: float):
        if not self.fault_stats_cache:
            return
        alpha = float(alpha)
        for name in self.layer_names:
            clean = self.clean_stats_cache[name]
            fault = self.fault_stats_cache.get(name, clean)
            mean = (1 - alpha) * clean['channel_mean'].float() + alpha * fault['channel_mean'].float()
            std = (1 - alpha) * clean['channel_std'].float() + alpha * fault['channel_std'].float()
            max_abs = (1 - alpha) * clean.get('channel_max_abs', clean['channel_mean'].abs()).float() \
                      + alpha * fault.get('channel_max_abs', fault['channel_mean'].abs()).float()
            self.detectors[name].update_reference(mean, std)
            self.repairs[name].update_reference(mean, std, max_abs)

    def set_operating_point(self, ber: Optional[float]):
        policy = self._resolve_policy(ber)
        self.active_policy = policy
        z_thresh = policy.get('z_thresh', self.default_policy['z_thresh'])
        std_bounds = tuple(policy.get('std_ratio_bounds', self.default_policy['std_ratio_bounds']))
        clip_margin = policy.get('clip_margin', self.default_policy['clip_margin'])

        for detector in self.detectors.values():
            detector.set_params(z_thresh, std_bounds)
        for repair in self.repairs.values():
            repair.set_clip_margin(clip_margin)

        alpha = policy.get('fault_alpha')
        if alpha is None and ber is not None and self.fault_stats_cache:
            alpha = min(max(ber / max(self.fault_profile_ber, 1e-6), 0.0), 1.0)
        if alpha is not None:
            self._blend_references(alpha)
        else:
            self._blend_references(0.0)

    def _resolve_policy(self, ber: Optional[float]) -> Dict[str, Any]:
        if not self.policy_map:
            return self.default_policy.copy()
        key = None
        if ber is not None:
            key = f"{float(ber):.0e}"
        policy = None
        if key and key in self.policy_map:
            policy = self.policy_map[key]
        elif 'default' in self.policy_map:
            policy = self.policy_map['default']
        else:
            policy = self.default_policy
        merged = self.default_policy.copy()
        merged.update(policy)
        return merged

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        self._register_hooks()

    def disable(self):
        if not self.enabled:
            return
        self.enabled = False
        self._remove_hooks()

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            module = modules.get(name)
            if module is None or not isinstance(module, (nn.Conv2d, nn.Linear, QuanConv2d, QuanLinear)):
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _make_hook(self, layer_name: str):
        detector = self.detectors[layer_name]
        repair = self.repairs[layer_name]
        # 判断是否是 always-on 模式（不依赖 mask）
        is_always_on = self.repair_mode in ('denoise_restorer', 'stacked_denoise_restorer', 'restorer_v4', 'improved_restorer', 'activation_reconstructor', 'lightweight_denoiser')

        def hook(_module, _inputs, output):
            if not self.enabled or output.dim() not in (2, 4):
                return output
            snapshot = ChannelStatsComputer.compute(output)
            policy = self.active_policy
            mode = policy.get('mode', 'repair')
            if mode == 'off':
                return output
            
            # 对于 always-on 模式，总是应用修复（不检查 mask）
            if is_always_on and mode != 'monitor':
                mask = detector.detect(snapshot)  # 仍然计算 mask（用于接口兼容性）
                output = repair.repair(output, snapshot, mask.to(output.device))
            else:
                # 对于其他模式，只有当检测到异常时才修复
                mask = detector.detect(snapshot)
                if mask.any() and mode != 'monitor':
                    output = repair.repair(output, snapshot, mask.to(output.device))
            return output

        return hook

    def repair_manual(self, layer_name: str, activation: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        snapshot = ChannelStatsComputer.compute(activation)
        detector = self.detectors[layer_name]
        repair = self.repairs[layer_name]
        mask = mask if mask is not None else detector.detect(snapshot)
        if mask.any():
            return repair.repair(activation, snapshot, mask.to(activation.device))
        return activation


def create_sensitive_layer_restorer(
    model: nn.Module,
    profile_path: str,
    target_layers: Optional[List[str]] = None,
    z_thresh: float = 3.0,
    std_ratio_bounds: Tuple[float, float] = (0.5, 2.0),
    clip_margin: float = 1.25,
    repair_mode: str = 'rule',
    fault_profile_path: Optional[str] = None,
    fault_profile_ber: float = 1e-1,
    ber_policy: Optional[Dict[str, Any]] = None,
    repair_head_ckpt: Optional[str] = None,
    mlp_hidden_dim: int = 32,
) -> SensitiveLayerRestorer:
    repair_state = None
    if repair_head_ckpt:
        repair_state = torch.load(repair_head_ckpt, map_location='cpu')
    return SensitiveLayerRestorer(
        model=model,
        profile_path=profile_path,
        target_layers=target_layers,
        default_z_thresh=z_thresh,
        default_std_ratio_bounds=std_ratio_bounds,
        default_clip_margin=clip_margin,
        repair_mode=repair_mode,
        fault_profile_path=fault_profile_path,
        fault_profile_ber=fault_profile_ber,
        ber_policy=ber_policy,
        repair_head_state=repair_state,
        mlp_hidden_dim=mlp_hidden_dim,
    )

