"""
Enhanced Feature Extractor for Restorer

This module provides richer feature extraction methods that capture
more subtle differences in feature maps caused by different fault patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor that captures more subtle differences
    in feature maps caused by fault injection.
    """
    
    def __init__(self, sensitive_info: Dict, baseline_stats: Optional[Dict] = None):
        self.sensitive_info = sensitive_info
        self.baseline_stats = baseline_stats
    
    def extract_rich_features(
        self,
        activations: Dict[str, torch.Tensor],
        clean_activations: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Extract rich features from activations
        
        Args:
            activations: Dict of layer_name -> activation tensor
            clean_activations: Optional clean activations for relative features
            device: Device for computation
        
        Returns:
            features: [B, feature_dim] - rich feature vector
        """
        all_features = []
        
        for name, info in self.sensitive_info.items():
            if name not in activations:
                continue
            
            act = activations[name]
            idx = info.get("indices", [])
            
            if len(idx) == 0:
                # Auto-select top channels by energy
                if act.dim() == 4:
                    energy = act.pow(2).mean(dim=[2, 3]).mean(dim=0)
                else:
                    energy = act.pow(2).mean(dim=0)
                k = min(10, act.size(1))  # Select top 10 channels
                idx = torch.topk(energy, k, largest=True).indices.tolist()
            
            act_sel = act[:, idx]
            
            # Extract multiple types of features
            layer_features = []
            
            # 1. Basic statistics (existing)
            basic_stats = self._extract_basic_statistics(act_sel)
            layer_features.append(basic_stats)
            
            # 2. Distribution features (capture subtle differences)
            dist_features = self._extract_distribution_features(act_sel)
            layer_features.append(dist_features)
            
            # 3. Spatial features (capture spatial patterns)
            if act_sel.dim() == 4:
                spatial_features = self._extract_spatial_features(act_sel)
                layer_features.append(spatial_features)
            
            # 4. Frequency domain features (capture frequency patterns)
            if act_sel.dim() == 4:
                freq_features = self._extract_frequency_features(act_sel)
                layer_features.append(freq_features)
            
            # 5. Relative features (if clean activations available)
            if clean_activations is not None and name in clean_activations:
                clean_act = clean_activations[name][:, idx]
                relative_features = self._extract_relative_features(act_sel, clean_act)
                layer_features.append(relative_features)
            
            # 6. Gradient-like features (capture local changes)
            if act_sel.dim() == 4:
                gradient_features = self._extract_gradient_features(act_sel)
                layer_features.append(gradient_features)
            
            # Concatenate all features for this layer
            layer_feat = torch.cat(layer_features, dim=1)
            
            # Normalize if baseline available
            if self.baseline_stats and name in self.baseline_stats:
                layer_feat = self._normalize_features(layer_feat, name, device)
            
            all_features.append(layer_feat)
        
        if not all_features:
            return None
        
        return torch.cat(all_features, dim=1)
    
    def _extract_basic_statistics(self, act: torch.Tensor) -> torch.Tensor:
        """
        Extract basic statistics: energy, mean, std, max
        """
        if act.dim() == 4:  # Conv layer
            energy = act.pow(2).mean(dim=[2, 3])  # [B, C]
            mean_val = act.mean(dim=[2, 3])
            std_val = act.std(dim=[2, 3])
            max_val = act.flatten(2).max(dim=2)[0]
            min_val = act.flatten(2).min(dim=2)[0]
        else:  # Linear layer
            energy = act.pow(2)
            mean_val = act
            std_val = torch.zeros_like(mean_val)
            max_val = act
            min_val = act
        
        return torch.cat([energy, mean_val, std_val, max_val, min_val], dim=1)
    
    def _extract_distribution_features(self, act: torch.Tensor) -> torch.Tensor:
        """
        Extract distribution features: percentiles, skewness, kurtosis
        These capture subtle differences in value distributions
        """
        if act.dim() == 4:
            # Flatten spatial dimensions
            act_flat = act.flatten(2)  # [B, C, H*W]
            
            # Percentiles (capture distribution shape)
            percentiles = [10, 25, 50, 75, 90]
            percentile_features = []
            for p in percentiles:
                p_val = torch.quantile(act_flat, p / 100.0, dim=2)  # [B, C]
                percentile_features.append(p_val)
            percentile_feat = torch.cat(percentile_features, dim=1)  # [B, C*5]
            
            # Skewness (asymmetry)
            mean = act_flat.mean(dim=2, keepdim=True)  # [B, C, 1]
            std = act_flat.std(dim=2, keepdim=True) + 1e-6
            centered = act_flat - mean
            skewness = (centered.pow(3).mean(dim=2) / std.squeeze(2).pow(3))  # [B, C]
            
            # Kurtosis (tail heaviness)
            kurtosis = (centered.pow(4).mean(dim=2) / std.squeeze(2).pow(4)) - 3.0  # [B, C]
            
            return torch.cat([percentile_feat, skewness, kurtosis], dim=1)
        else:
            # For linear layers, use simpler features
            mean = act.mean(dim=1, keepdim=True)
            std = act.std(dim=1, keepdim=True) + 1e-6
            centered = act - mean
            skewness = (centered.pow(3).mean(dim=1, keepdim=True) / std.pow(3))
            kurtosis = (centered.pow(4).mean(dim=1, keepdim=True) / std.pow(4)) - 3.0
            return torch.cat([skewness, kurtosis], dim=1)
    
    def _extract_spatial_features(self, act: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features: capture spatial patterns and correlations
        """
        B, C, H, W = act.shape
        
        # Multi-scale spatial statistics (different grid sizes)
        spatial_features = []
        
        for grid_size in [1, 2, 4]:  # Global, 2x2, 4x4
            if H < grid_size or W < grid_size:
                continue
            
            h_step = H // grid_size
            w_step = W // grid_size
            
            grid_stats = []
            for r in range(grid_size):
                for c in range(grid_size):
                    grid = act[:, :, r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
                    grid_mean = grid.mean(dim=[2, 3])  # [B, C]
                    grid_std = grid.std(dim=[2, 3])
                    grid_stats.extend([grid_mean, grid_std])
            
            if grid_stats:
                spatial_features.append(torch.cat(grid_stats, dim=1))
        
        # Spatial correlation (capture relationships between regions)
        # Compute correlation between different spatial regions
        if H >= 2 and W >= 2:
            # Split into 4 quadrants
            h_mid, w_mid = H // 2, W // 2
            quadrants = [
                act[:, :, :h_mid, :w_mid].mean(dim=[2, 3]),      # Top-left
                act[:, :, :h_mid, w_mid:].mean(dim=[2, 3]),      # Top-right
                act[:, :, h_mid:, :w_mid].mean(dim=[2, 3]),      # Bottom-left
                act[:, :, h_mid:, w_mid:].mean(dim=[2, 3]),      # Bottom-right
            ]
            
            # Compute pairwise correlations
            corr_features = []
            for i in range(len(quadrants)):
                for j in range(i+1, len(quadrants)):
                    # Cosine similarity between quadrants
                    q1 = quadrants[i]  # [B, C]
                    q2 = quadrants[j]
                    corr = F.cosine_similarity(q1, q2, dim=1).unsqueeze(1)  # [B, 1]
                    corr_features.append(corr)
            
            if corr_features:
                spatial_features.append(torch.cat(corr_features, dim=1))
        
        return torch.cat(spatial_features, dim=1) if spatial_features else torch.zeros(B, 1, device=act.device)
    
    def _extract_frequency_features(self, act: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency domain features using FFT
        Captures frequency patterns that might be affected by faults
        """
        B, C, H, W = act.shape
        
        # Apply 2D FFT
        act_fft = torch.fft.fft2(act, norm='ortho')
        act_fft_abs = torch.abs(act_fft)  # [B, C, H, W]
        
        # Low-frequency energy (DC and low frequencies)
        h_low, w_low = H // 4, W // 4
        low_freq = act_fft_abs[:, :, :h_low, :w_low].mean(dim=[2, 3])  # [B, C]
        
        # High-frequency energy
        high_freq = act_fft_abs[:, :, h_low:, w_low:].mean(dim=[2, 3])  # [B, C]
        
        # Frequency distribution
        freq_flat = act_fft_abs.flatten(2)  # [B, C, H*W]
        freq_mean = freq_flat.mean(dim=2)
        freq_std = freq_flat.std(dim=2)
        freq_max = freq_flat.max(dim=2)[0]
        
        return torch.cat([low_freq, high_freq, freq_mean, freq_std, freq_max], dim=1)
    
    def _extract_relative_features(
        self,
        faulted_act: torch.Tensor,
        clean_act: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract relative features: differences between faulted and clean
        This is crucial for capturing fault-induced changes
        """
        # Absolute difference
        diff = faulted_act - clean_act
        abs_diff = torch.abs(diff)
        
        if faulted_act.dim() == 4:
            # Statistics of differences
            diff_mean = diff.mean(dim=[2, 3])  # [B, C]
            diff_std = diff.std(dim=[2, 3])
            abs_diff_mean = abs_diff.mean(dim=[2, 3])
            abs_diff_max = abs_diff.flatten(2).max(dim=2)[0]
            
            # Relative difference (normalized by clean magnitude)
            clean_mag = torch.abs(clean_act).mean(dim=[2, 3]) + 1e-6
            rel_diff = abs_diff_mean / clean_mag  # [B, C]
            
            # Correlation between faulted and clean
            faulted_flat = faulted_act.flatten(2).mean(dim=2)  # [B, C]
            clean_flat = clean_act.flatten(2).mean(dim=2)
            corr = F.cosine_similarity(faulted_flat, clean_flat, dim=1).unsqueeze(1)  # [B, 1]
            corr = corr.expand(-1, faulted_act.size(1))  # [B, C]
        else:
            diff_mean = diff
            diff_std = torch.zeros_like(diff_mean)
            abs_diff_mean = abs_diff
            abs_diff_max = abs_diff
            clean_mag = torch.abs(clean_act) + 1e-6
            rel_diff = abs_diff_mean / clean_mag
            corr = F.cosine_similarity(faulted_act, clean_act, dim=1).unsqueeze(1)
            corr = corr.expand(-1, faulted_act.size(1))
        
        return torch.cat([diff_mean, diff_std, abs_diff_mean, abs_diff_max, rel_diff, corr], dim=1)
    
    def _extract_gradient_features(self, act: torch.Tensor) -> torch.Tensor:
        """
        Extract gradient-like features: capture local changes and edges
        """
        B, C, H, W = act.shape
        
        # Compute gradients using Sobel-like filters
        # Horizontal gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=act.dtype, device=act.device).view(1, 1, 3, 3)
        sobel_x = sobel_x.expand(C, 1, 3, 3)
        
        # Vertical gradient
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=act.dtype, device=act.device).view(1, 1, 3, 3)
        sobel_y = sobel_y.expand(C, 1, 3, 3)
        
        # Apply filters
        grad_x = F.conv2d(act, sobel_x, groups=C, padding=1)  # [B, C, H, W]
        grad_y = F.conv2d(act, sobel_y, groups=C, padding=1)
        
        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        
        # Statistics of gradients
        grad_mag_mean = grad_mag.mean(dim=[2, 3])  # [B, C]
        grad_mag_std = grad_mag.std(dim=[2, 3])
        grad_mag_max = grad_mag.flatten(2).max(dim=2)[0]
        
        # Gradient direction (angle)
        grad_angle = torch.atan2(grad_y, grad_x)  # [B, C, H, W]
        grad_angle_mean = grad_angle.mean(dim=[2, 3])
        grad_angle_std = grad_angle.std(dim=[2, 3])
        
        return torch.cat([grad_mag_mean, grad_mag_std, grad_mag_max, grad_angle_mean, grad_angle_std], dim=1)
    
    def _normalize_features(
        self,
        features: torch.Tensor,
        layer_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalize features using baseline statistics
        """
        if self.baseline_stats is None or layer_name not in self.baseline_stats:
            return features
        
        baseline = self.baseline_stats[layer_name]
        
        # Use baseline mean and std if available
        if 'feature_mean' in baseline and 'feature_std' in baseline:
            mean = baseline['feature_mean'].to(device)
            std = baseline['feature_std'].to(device) + 1e-6
            
            # Ensure dimensions match
            if mean.dim() == 1:
                mean = mean.unsqueeze(0)
            if std.dim() == 1:
                std = std.unsqueeze(0)
            
            # Normalize
            features = (features - mean) / std
        
        # Handle NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features


class EnhancedSensitiveActivationCollector:
    """
    Enhanced version of SensitiveActivationCollector with rich feature extraction
    """
    
    def __init__(self, model, sensitive_info: Dict, baseline_stats: Optional[Dict] = None):
        self.model = model.module if hasattr(model, "module") else model
        self.sensitive_info = sensitive_info
        self.baseline_stats = baseline_stats
        self.buffers = {}
        self.handles = []
        self.feature_extractor = EnhancedFeatureExtractor(sensitive_info, baseline_stats)
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to collect activations"""
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
        """Clear collected activations"""
        self.buffers.clear()
    
    def remove(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def build_enhanced_feature_vector(
        self,
        device: torch.device,
        clean_activations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build enhanced feature vector with rich features
        
        Args:
            device: Device for computation
            clean_activations: Optional clean activations for relative features
        
        Returns:
            features: [B, feature_dim] - enhanced feature vector
        """
        if not self.buffers:
            return None
        
        return self.feature_extractor.extract_rich_features(
            self.buffers,
            clean_activations=clean_activations,
            device=device,
        )
    
    def build_layer_features(self, device: torch.device) -> Optional[tuple]:
        """
        Build layer-wise enhanced features (one tensor per layer)
        
        Args:
            device: Device for computation
        
        Returns:
            (layer_features, feature_dims): List of [B, feat_dim_i] tensors, and list of feature dimensions
        """
        if not self.buffers:
            return None
        
        all_layer_features = []
        feature_dims = []
        
        for name, info in self.sensitive_info.items():
            if name not in self.buffers:
                continue
            
            activations = self.buffers[name]
            idx = info.get("indices", [])
            
            if len(idx) == 0:
                # Auto-select top channels by energy
                act = activations
                if act.dim() == 4:
                    energy = act.pow(2).mean(dim=[2, 3]).mean(dim=0)
                else:
                    energy = act.pow(2).mean(dim=0)
                k = min(10, act.size(1))
                idx = torch.topk(energy, k, largest=True).indices.tolist()
            
            act_sel = activations[:, idx]
            
            # Extract enhanced features for this layer directly
            layer_features_list = []
            
            # 1. Basic statistics
            basic_stats = self.feature_extractor._extract_basic_statistics(act_sel)
            layer_features_list.append(basic_stats)
            
            # 2. Distribution features
            dist_features = self.feature_extractor._extract_distribution_features(act_sel)
            layer_features_list.append(dist_features)
            
            # 3. Spatial features (if conv layer)
            if act_sel.dim() == 4:
                spatial_features = self.feature_extractor._extract_spatial_features(act_sel)
                layer_features_list.append(spatial_features)
            
            # 4. Frequency domain features (if conv layer)
            if act_sel.dim() == 4:
                freq_features = self.feature_extractor._extract_frequency_features(act_sel)
                layer_features_list.append(freq_features)
            
            # 5. Gradient-like features (if conv layer)
            if act_sel.dim() == 4:
                gradient_features = self.feature_extractor._extract_gradient_features(act_sel)
                layer_features_list.append(gradient_features)
            
            # Concatenate all features for this layer
            layer_feat = torch.cat(layer_features_list, dim=1)
            
            # Normalize if baseline available
            if self.baseline_stats and name in self.baseline_stats:
                layer_feat = self.feature_extractor._normalize_features(layer_feat, name, device)
            
            all_layer_features.append(layer_feat)
            feature_dims.append(layer_feat.shape[1])
        
        if not all_layer_features:
            return None
        
        return (all_layer_features, feature_dims)

