import torch
import torch.nn as nn
from typing import Dict, List, Optional
from quan.func import QuanConv2d, QuanLinear


class SensitiveActivationCollector:
    def __init__(self, model, sensitive_info: Dict[str, Dict[str, List[int]]], baseline_stats=None):
        self.model = model.module if hasattr(model, "module") else model
        self.sensitive_info = sensitive_info
        self.baseline = baseline_stats
        self.layer_features = {}
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

    def build_layer_features(self, activations):
        """Builds a feature vector for each layer from the collected activations."""
        layer_feats = []
        layer_names = []
        
        device = next(self.model.parameters()).device
        
        for name, info in self.sensitive_info.items():
            if name not in activations:
                continue

            idx = info['indices'] # Extract the list of indices from the dict
            if not idx:
                continue
                
            act = activations[name][:, idx, ...].clone()
            
            # For convolutional layers, use 2x2 grid-based stats to capture spatial information
            if act.dim() == 4:
                B, C, H, W = act.shape
                grid_size = 2
                
                # If the feature map is too small, fall back to global stats
                if H < grid_size or W < grid_size:
                    h_step, w_step = H, W
                else:
                    h_step, w_step = H // grid_size, W // grid_size

                all_grid_features = []
                # The baseline stats for a layer are shaped [num_features, num_channels]
                # We need to iterate over grid cells to correctly map features to stats
                for r in range(grid_size):
                    for c in range(grid_size):
                        grid_idx_base = (r * grid_size + c) * 4
                        if H < grid_size or W < grid_size:
                            grid = act
                        else:
                            grid = act[:, :, r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]

                        energy = grid.pow(2).mean(dim=[2, 3])
                        mean_val = grid.mean(dim=[2, 3])
                        std_val = grid.std(dim=[2, 3])
                        max_val = grid.flatten(2).max(dim=2)[0]
                        
                        # Normalize each stat with its baseline mean and std, transposing the stats for broadcasting
                        norm_energy = (energy - self.baseline[name]['mean'][grid_idx_base + 0, :]) / self.baseline[name]['std'][grid_idx_base + 0, :]
                        norm_mean = (mean_val - self.baseline[name]['mean'][grid_idx_base + 1, :]) / self.baseline[name]['std'][grid_idx_base + 1, :]
                        norm_std = (std_val - self.baseline[name]['mean'][grid_idx_base + 2, :]) / self.baseline[name]['std'][grid_idx_base + 2, :]
                        norm_max = (max_val - self.baseline[name]['mean'][grid_idx_base + 3, :]) / self.baseline[name]['std'][grid_idx_base + 3, :]
                        
                        all_grid_features.extend([norm_energy, norm_mean, norm_std, norm_max])
                
                # Concatenate all normalized features
                features = torch.cat(all_grid_features, dim=1)

            else: # For linear layers, use global stats
                energy = act.pow(2)
                mean_val = act
                std_val = torch.zeros_like(mean_val) # No spatial dim
                max_val = act
                
                norm_energy = (energy - self.baseline[name]['mean'][0, :]) / self.baseline[name]['std'][0, :]
                norm_mean = (mean_val - self.baseline[name]['mean'][1, :]) / self.baseline[name]['std'][1, :]
                # Std of std is likely 0, so we just use a zero vector for it.
                norm_std = torch.zeros_like(std_val)
                norm_max = (max_val - self.baseline[name]['mean'][3, :]) / self.baseline[name]['std'][3, :]

                features = torch.cat([norm_energy, norm_mean, norm_std, norm_max], dim=1)
            
            # Fill any NaNs or Infs that might result from division by zero std
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            self.layer_features[name] = features
            layer_feats.append(features)
            layer_names.append(name)
        
        return layer_feats, layer_names

    def build_feature_vector(self, device):
        """提取丰富的特征：能量、均值、方差、最大值"""
        feats = []
        for name, info in self.sensitive_info.items():
            if name not in self.buffers:
                continue
            activations = self.buffers[name]
            idx = info["indices"]
            if len(idx) == 0:
                act = activations
                if act.dim() == 4:
                    energy = act.pow(2).mean(dim=[2, 3]).mean(dim=0)
                else:
                    energy = act.pow(2).mean(dim=0)
                k = 1
                idx = torch.topk(energy, k, largest=True).indices.tolist()
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
            
            # Normalize with baseline statistics
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
            else:
                # Old format: only energy (backward compatibility)
                mean = info["mean"].to(device).unsqueeze(0)
                std = info["std"].to(device).unsqueeze(0)
                layer_feat = (energy - mean) / (std + 1e-6)
            
            feats.append(layer_feat)
        if not feats:
            return None
        feature_vec = torch.cat(feats, dim=1)
        return feature_vec

    def build_layer_features(self, device) -> Optional[List[torch.Tensor]]:
        """
        Builds a list of feature tensors, one for each sensitive layer.
        This preserves the layer-wise structure of the features.
        """
        if not self.buffers: # Changed from self.collected_activations to self.buffers
            return None

        all_layer_features = []
        feature_dims = []
        for name, info in self.sensitive_info.items(): # Iterate in a fixed order
            if name not in self.buffers: # Changed from self.stats to self.sensitive_info
                continue
            
            # Check if baseline exists for this layer
            has_baseline = (
                self.baseline is not None 
                and name in self.baseline 
                and isinstance(self.baseline[name], dict)
                and 'mean' in self.baseline[name] 
                and 'std' in self.baseline[name]
            )
            baseline_for_layer = self.baseline[name] if has_baseline else None

            activations = self.buffers[name]
            idx = info["indices"]
            if len(idx) == 0:
                act = activations
                if act.dim() == 4:
                    energy = act.pow(2).mean(dim=[2, 3]).mean(dim=0)
                else:
                    energy = act.pow(2).mean(dim=0)
                k = 1
                idx = torch.topk(energy, k, largest=True).indices.tolist()
            act_sel = activations[:, idx]
            
            # Extract multiple statistics from the activation tensor
            if act_sel.dim() == 4:  # Conv layer
                B, C, H, W = act_sel.shape
                grid_size = 2
                
                if H < grid_size or W < grid_size:
                    h_step, w_step = H, W
                else:
                    h_step, w_step = H // grid_size, W // grid_size

                grid_features = []
                for r in range(grid_size):
                    for c in range(grid_size):
                        grid_idx_base = (r * grid_size + c) * 4
                        if H < grid_size or W < grid_size:
                            grid = act_sel
                        else:
                            grid = act_sel[:, :, r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]

                        energy = grid.pow(2).mean(dim=[2, 3])
                        mean_val = grid.mean(dim=[2, 3])
                        std_val = grid.std(dim=[2, 3])
                        max_val = grid.flatten(2).max(dim=2)[0]
                        
                        # Normalize each stat with its baseline mean and std if available
                        if has_baseline:
                            norm_energy = (energy - baseline_for_layer['mean'][grid_idx_base + 0, :].to(device)) / (baseline_for_layer['std'][grid_idx_base + 0, :].to(device) + 1e-6)
                            norm_mean = (mean_val - baseline_for_layer['mean'][grid_idx_base + 1, :].to(device)) / (baseline_for_layer['std'][grid_idx_base + 1, :].to(device) + 1e-6)
                            norm_std = (std_val - baseline_for_layer['mean'][grid_idx_base + 2, :].to(device)) / (baseline_for_layer['std'][grid_idx_base + 2, :].to(device) + 1e-6)
                            norm_max = (max_val - baseline_for_layer['mean'][grid_idx_base + 3, :].to(device)) / (baseline_for_layer['std'][grid_idx_base + 3, :].to(device) + 1e-6)
                        else:
                            # Use raw features if no baseline
                            norm_energy = energy
                            norm_mean = mean_val
                            norm_std = std_val
                            norm_max = max_val
                        
                        grid_features.extend([norm_energy, norm_mean, norm_std, norm_max])
                
                layer_feature_tensor = torch.cat(grid_features, dim=1)

            else:  # Linear layer
                energy = act_sel.pow(2)
                mean_val = act_sel
                std_val = torch.zeros_like(mean_val) # No spatial dim for std
                max_val = act_sel

                # Normalize statistics if baseline is available
                if has_baseline:
                    norm_energy = (energy - baseline_for_layer['mean'][0, :].to(device)) / (baseline_for_layer['std'][0, :].to(device) + 1e-6)
                    norm_mean = (mean_val - baseline_for_layer['mean'][1, :].to(device)) / (baseline_for_layer['std'][1, :].to(device) + 1e-6)
                    norm_std = (std_val - baseline_for_layer['mean'][2, :].to(device)) / (baseline_for_layer['std'][2, :].to(device) + 1e-6) # Std of std is likely 0
                    norm_max = (max_val - baseline_for_layer['mean'][3, :].to(device)) / (baseline_for_layer['std'][3, :].to(device) + 1e-6)
                else:
                    # Use raw features if no baseline
                    norm_energy = energy
                    norm_mean = mean_val
                    norm_std = std_val
                    norm_max = max_val
            
                layer_feature_tensor = torch.cat([norm_energy, norm_mean, norm_std, norm_max], dim=1)

            layer_feature_tensor = torch.nan_to_num(layer_feature_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            all_layer_features.append(layer_feature_tensor)
            feature_dims.append(layer_feature_tensor.shape[1])
        
        if not all_layer_features:
            return None, []
            
        return all_layer_features, feature_dims


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

