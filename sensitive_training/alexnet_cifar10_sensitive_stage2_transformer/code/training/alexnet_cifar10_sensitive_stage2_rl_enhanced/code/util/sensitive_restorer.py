import torch
import torch.nn as nn
from typing import Dict, List, Optional
from quan.func import QuanConv2d, QuanLinear


class SensitiveActivationCollector:
    def __init__(self, model, sensitive_info: Dict[str, Dict[str, List[int]]], baseline_stats=None):
        self.model = model.module if hasattr(model, "module") else model
        self.sensitive_info = sensitive_info
        self.baseline = baseline_stats
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

    def build_layer_features(self, device) -> Optional[List[torch.Tensor]]:
        """
        Builds a list of feature tensors, one for each sensitive layer.
        This preserves the layer-wise structure of the features.
        """
        if not self.buffers:
            return None

        all_layer_features = []
        for name, info in self.sensitive_info.items():
            if name not in self.buffers:
                continue
            activations = self.buffers[name]
            idx = info["indices"]
            if len(idx) == 0:
                continue
            act_sel = activations[:, idx]
            
            # Extract multiple statistics
            if act_sel.dim() == 4:  # Conv layer
                energy = act_sel.pow(2).mean(dim=[2, 3])
                mean_val = act_sel.mean(dim=[2, 3])
                std_val = act_sel.std(dim=[2, 3])
                max_val = act_sel.max(dim=3)[0].max(dim=2)[0]
            else:  # Linear layer
                energy = act_sel.pow(2)
                mean_val = act_sel
                std_val = torch.zeros_like(mean_val) # No spatial dim for std
                max_val = act_sel

            # Normalize statistics if baseline is available
            if self.baseline is not None and name in self.baseline:
                baseline = self.baseline[name]
                # Handle different baseline formats
                if "energy_mean" in baseline:
                    # New format with multiple statistics
                    norm_energy = (energy - baseline["energy_mean"].to(device)) / (baseline["energy_std"].to(device) + 1e-6)
                    norm_mean = (mean_val - baseline["mean_mean"].to(device)) / (baseline["mean_std"].to(device) + 1e-6)
                    norm_std = (std_val - baseline["std_mean"].to(device)) / (baseline["std_std"].to(device) + 1e-6)
                    norm_max = (max_val - baseline["max_mean"].to(device)) / (baseline["max_std"].to(device) + 1e-6)
                elif "mean" in baseline and isinstance(baseline["mean"], torch.Tensor):
                    # Baseline format: baseline["mean"] shape is [num_features, num_channels]
                    # For Conv layers: num_features = 16 (2x2 grid × 4 stats)
                    # For Linear layers: num_features = 4 (4 stats)
                    # We extract global stats (first 4 features for conv, all 4 for linear)
                    baseline_mean = baseline["mean"].to(device)  # [num_features, num_channels]
                    baseline_std = baseline["std"].to(device)    # [num_features, num_channels]
                    
                    # For conv layers, baseline has 16 features (grid-based), we need to average or use first 4
                    # For simplicity, we'll use the first 4 features (which correspond to the first grid cell)
                    # Or we can average across all grid cells for each stat type
                    if act_sel.dim() == 4 and baseline_mean.shape[0] == 16:
                        # Conv layer with grid-based baseline: average across grid cells
                        # Reshape: [16, C] -> [4, 4, C] -> [4, C] (average over grid cells)
                        baseline_mean_reshaped = baseline_mean.view(4, 4, -1).mean(dim=1)  # [4, C]
                        baseline_std_reshaped = baseline_std.view(4, 4, -1).mean(dim=1)    # [4, C]
                    else:
                        # Linear layer or already in correct format
                        baseline_mean_reshaped = baseline_mean[:4]  # [4, C] or [num_features, C]
                        baseline_std_reshaped = baseline_std[:4]
                    
                    # Now baseline_mean_reshaped is [4, C], we need to transpose to [C, 4] and index
                    # energy -> index 0, mean -> index 1, std -> index 2, max -> index 3
                    baseline_mean_T = baseline_mean_reshaped.T  # [C, 4]
                    baseline_std_T = baseline_std_reshaped.T    # [C, 4]
                    
                    # energy, mean_val, std_val, max_val are [B, C]
                    # baseline_mean_T[:, 0] is [C], we need to broadcast to [B, C]
                    norm_energy = (energy - baseline_mean_T[:, 0].unsqueeze(0)) / (baseline_std_T[:, 0].unsqueeze(0) + 1e-6)
                    norm_mean = (mean_val - baseline_mean_T[:, 1].unsqueeze(0)) / (baseline_std_T[:, 1].unsqueeze(0) + 1e-6)
                    norm_std = (std_val - baseline_mean_T[:, 2].unsqueeze(0)) / (baseline_std_T[:, 2].unsqueeze(0) + 1e-6)
                    norm_max = (max_val - baseline_mean_T[:, 3].unsqueeze(0)) / (baseline_std_T[:, 3].unsqueeze(0) + 1e-6)
                else:
                    # No normalization
                    norm_energy = energy
                    norm_mean = mean_val
                    norm_std = std_val
                    norm_max = max_val
            else:
                # No baseline available, use raw statistics
                norm_energy = energy
                norm_mean = mean_val
                norm_std = std_val
                norm_max = max_val
            
            # Concatenate features for the current layer
            layer_feature_tensor = torch.cat([norm_energy, norm_mean, norm_std, norm_max], dim=1)
            # Fill any NaNs or Infs
            layer_feature_tensor = torch.nan_to_num(layer_feature_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            all_layer_features.append(layer_feature_tensor)
        
        if not all_layer_features:
            return None
            
        return all_layer_features

    def build_feature_vector(self, device):
        """提取丰富的特征：能量、均值、方差、最大值"""
        feats = []
        for name, info in self.sensitive_info.items():
            if name not in self.buffers:
                continue
            activations = self.buffers[name]
            idx = info["indices"]
            if len(idx) == 0:
                continue
            act_sel = activations[:, idx]
            
            # Extract multiple statistics
            if act_sel.dim() == 4:  # Conv layer
                energy = act_sel.pow(2).mean(dim=[2, 3])  # [B, C]
                mean_val = act_sel.mean(dim=[2, 3])  # [B, C]
                std_val = act_sel.std(dim=[2, 3])  # [B, C]
                max_val = act_sel.max(dim=3)[0].max(dim=2)[0]  # [B, C]
            else:  # Linear layer
                energy = act_sel.pow(2)  # [B, C]
                mean_val = act_sel  # [B, C]
                std_val = torch.zeros_like(mean_val)  # No spatial dim
                max_val = act_sel  # [B, C]
            
            # Normalize with baseline statistics if available
            # Handle backward compatibility: old baseline only has "mean" and "std"
            if "energy_mean" in info:
                # New format with multiple statistics
                energy_mean = info["energy_mean"].to(device).unsqueeze(0)
                energy_std = info["energy_std"].to(device).unsqueeze(0)
                mean_mean = info["mean_mean"].to(device).unsqueeze(0)
                mean_std = info["mean_std"].to(device).unsqueeze(0)
                std_mean = info["std_mean"].to(device).unsqueeze(0)
                std_std = info["std_std"].to(device).unsqueeze(0)
                max_mean = info["max_mean"].to(device).unsqueeze(0)
                max_std = info["max_std"].to(device).unsqueeze(0)
                
                energy_norm = (energy - energy_mean) / (energy_std + 1e-6)
                mean_norm = (mean_val - mean_mean) / (mean_std + 1e-6)
                std_norm = (std_val - std_mean) / (std_std + 1e-6)
                max_norm = (max_val - max_mean) / (max_std + 1e-6)
                
                # Concatenate all features
                layer_feat = torch.cat([energy_norm, mean_norm, std_norm, max_norm], dim=1)
            elif "mean" in info and "std" in info:
                # Old format: only energy (backward compatibility)
                # Check if mean and std are tensors before using them
                try:
                    mean = info["mean"]
                    std = info["std"]
                    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
                        mean = mean.to(device).unsqueeze(0)
                        std = std.to(device).unsqueeze(0)
                        layer_feat = (energy - mean) / (std + 1e-6)
                    else:
                        # mean/std are not tensors, use raw features
                        layer_feat = torch.cat([energy, mean_val, std_val, max_val], dim=1)
                except (KeyError, AttributeError, TypeError):
                    # Fallback to raw features if anything goes wrong
                    layer_feat = torch.cat([energy, mean_val, std_val, max_val], dim=1)
            else:
                # No baseline statistics available, use raw features without normalization
                # Concatenate all features: energy, mean, std, max
                layer_feat = torch.cat([energy, mean_val, std_val, max_val], dim=1)
            
            feats.append(layer_feat)
        if not feats:
            return None
        feature_vec = torch.cat(feats, dim=1)
        return feature_vec



class SensitiveChannelRestorer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        # Enhanced feature projection with deeper architecture
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Enhanced detector
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        # Enhanced restorer with deeper architecture
        self.restorer = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits, features):
        embed = self.feature_proj(features)
        gate = self.detector(embed)
        augmented = torch.cat([embed, logits], dim=1)
        delta = self.restorer(augmented)
        restored = logits + gate * delta
        return restored, gate

