"""
Enhanced Sensitive Channel Restorer V2
改进的架构设计，提升容错性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from quan.func import QuanConv2d, QuanLinear


class SensitiveActivationCollectorV2:
    """增强的特征收集器：提取更丰富的特征"""
    def __init__(self, model, sensitive_info: Dict[str, Dict[str, List[int]]]):
        self.model = model.module if hasattr(model, "module") else model
        self.sensitive_info = sensitive_info
        self.handles = []
        self.buffers = {}
        self._register_hooks()

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        for name in self.sensitive_info.keys():
            if name not in modules:
                continue

            def make_hook(key):
                def hook(module, input, output):
                    self.buffers[key] = output.detach()
                return hook

            handle = modules[name].register_forward_hook(make_hook(name))
            self.handles.append(handle)

    def clear(self):
        self.buffers.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def build_feature_vector(self, device):
        """提取更丰富的特征：能量、均值、方差、最大值"""
        feats = []
        for name, info in self.sensitive_info.items():
            if name not in self.buffers:
                continue
            activations = self.buffers[name]
            idx = info["indices"]
            if len(idx) == 0:
                continue
            act_sel = activations[:, idx]
            
            if act_sel.dim() == 4:  # Conv layer
                # 空间维度上的统计
                energy = act_sel.pow(2).mean(dim=[2, 3])  # [B, C]
                mean = act_sel.mean(dim=[2, 3])  # [B, C]
                std = act_sel.std(dim=[2, 3])  # [B, C]
                max_val = act_sel.max(dim=3)[0].max(dim=2)[0]  # [B, C]
            else:  # Linear layer
                energy = act_sel.pow(2)  # [B, C]
                mean = act_sel  # [B, C]
                std = torch.zeros_like(mean)  # No spatial dim
                max_val = act_sel  # [B, C]
            
            # Normalize with baseline
            baseline_mean = info["mean"].to(device).unsqueeze(0)
            baseline_std = info["std"].to(device).unsqueeze(0)
            
            # 归一化特征
            energy_norm = (energy - baseline_mean) / (baseline_std + 1e-6)
            mean_norm = (mean - baseline_mean) / (baseline_std + 1e-6)
            std_norm = std / (baseline_std + 1e-6)
            max_norm = (max_val - baseline_mean) / (baseline_std + 1e-6)
            
            # 拼接多种特征
            layer_feat = torch.cat([energy_norm, mean_norm, std_norm, max_norm], dim=1)
            feats.append(layer_feat)
        
        if not feats:
            return None
        feature_vec = torch.cat(feats, dim=1)
        return feature_vec


class LayerAttention(nn.Module):
    """层间注意力机制：学习不同层的重要性"""
    def __init__(self, num_layers: int, embed_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_embedding = nn.Embedding(num_layers, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, layer_features: List[torch.Tensor]):
        """
        Args:
            layer_features: List of [B, C_i] tensors, one per layer
        Returns:
            attended_features: [B, sum(C_i)]
        """
        # Stack features: [B, num_layers, max_channels]
        max_channels = max(f.shape[1] for f in layer_features)
        stacked = []
        for f in layer_features:
            if f.shape[1] < max_channels:
                padding = torch.zeros(f.shape[0], max_channels - f.shape[1], 
                                    device=f.device, dtype=f.dtype)
                f = torch.cat([f, padding], dim=1)
            stacked.append(f)
        stacked = torch.stack(stacked, dim=1)  # [B, num_layers, max_channels]
        
        # Layer embeddings
        layer_ids = torch.arange(self.num_layers, device=stacked.device)
        layer_emb = self.layer_embedding(layer_ids).unsqueeze(0)  # [1, num_layers, embed_dim]
        
        # Add layer embeddings to features
        if stacked.shape[2] != layer_emb.shape[2]:
            # Project features to embed_dim
            proj = nn.Linear(stacked.shape[2], layer_emb.shape[2]).to(stacked.device)
            stacked = proj(stacked)
        
        attended, _ = self.attention(stacked + layer_emb, stacked + layer_emb, stacked + layer_emb)
        attended = self.norm(attended + stacked)
        
        # Flatten back
        B = attended.shape[0]
        attended_flat = attended.view(B, -1)
        return attended_flat


class ChannelAttention(nn.Module):
    """通道注意力机制：学习通道间的关系"""
    def __init__(self, feature_dim: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, C]
        B, C = x.shape
        x_avg = self.avg_pool(x.unsqueeze(1)).squeeze(1)  # [B, C]
        x_max = self.max_pool(x.unsqueeze(1)).squeeze(1)  # [B, C]
        attention = self.fc(x_avg + x_max)  # [B, C]
        return x * attention


class EnhancedSensitiveChannelRestorer(nn.Module):
    """
    增强的敏感通道修复器 V2
    
    改进点：
    1. 更丰富的特征提取（能量、均值、方差、最大值）
    2. 通道注意力机制
    3. 残差连接
    4. 细粒度gate（每个类别一个gate）
    5. 分层处理
    """
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 120, 
                 num_layers: int = 8, use_attention: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Feature projection with channel attention
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        if use_attention:
            self.channel_attention = ChannelAttention(hidden_dim)
        
        self.feature_proj2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Enhanced detector with residual
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.detector_gate = nn.Linear(hidden_dim // 4, num_classes)  # Per-class gate
        self.detector_residual = nn.Linear(hidden_dim, hidden_dim // 4)
        
        # Enhanced restorer with residual
        self.restorer_proj = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.restorer_mid = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.restorer_out = nn.Linear(hidden_dim, num_classes)
        self.restorer_residual = nn.Linear(hidden_dim + num_classes, num_classes)

    def forward(self, logits, features):
        # Feature projection
        embed = self.feature_proj(features)
        if self.use_attention:
            embed = self.channel_attention(embed)
        embed = self.feature_proj2(embed)
        
        # Detector with residual
        gate_feat = self.detector(embed)
        gate_residual = self.detector_residual(embed)
        gate_feat = gate_feat + gate_residual
        gate = torch.sigmoid(self.detector_gate(gate_feat))  # [B, num_classes]
        
        # Restorer with residual
        augmented = torch.cat([embed, logits], dim=1)
        delta = self.restorer_proj(augmented)
        delta = self.restorer_mid(delta)
        delta = self.restorer_out(delta)
        delta_residual = self.restorer_residual(augmented)
        delta = delta + delta_residual
        
        # Per-class gating
        restored = logits + gate * delta
        return restored, gate.mean(dim=1, keepdim=True)  # Return average gate for logging

