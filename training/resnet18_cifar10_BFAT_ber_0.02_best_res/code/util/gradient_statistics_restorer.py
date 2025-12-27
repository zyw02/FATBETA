"""
Gradient Statistics-based Error Restorer

Implements the error resilience method from:
"Error Resilience in Deep Neural Networks Using Neuron Gradient Statistics"
IEEE TCAD, Vol. 43, No. 4, April 2024

Key components:
1. Gradient calculation (successive differences) for dense and conv layers
2. Statistical threshold computation using Chebyshev's inequality
3. Error localization and suppression (setting erroneous outputs to zero)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import numpy as np
from quan.func import QuanConv2d, QuanLinear


class WelfordStatistics:
    """
    Online statistics computation using Welford's algorithm
    for computing mean and variance incrementally
    """
    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None  # Sum of squared differences from mean
    
    def update(self, x: torch.Tensor):
        """
        Update statistics with new batch of data using vectorized Welford's algorithm
        
        Args:
            x: Tensor of shape [B, ...] or [B, C, H, W] or [B, C]
        """
        # Flatten to 1D for statistics
        if x.dim() == 2:  # [B, C]
            x_flat = x.view(-1)  # [B*C]
        elif x.dim() == 4:  # [B, C, H, W]
            x_flat = x.view(-1)  # [B*C*H*W]
        else:
            x_flat = x.view(-1)
        
        x_flat = x_flat.detach().cpu()
        
        # Vectorized Welford's algorithm for batch update
        n_new = x_flat.numel()
        if n_new == 0:
            return
        
        # Compute mean of new batch
        mean_new = x_flat.float().mean().item()
        
        if self.mean is None:
            # First update
            self.mean = mean_new
            self.M2 = x_flat.float().var(unbiased=False).item() * n_new
            self.count = n_new
        else:
            # Update with new batch using parallel algorithm
            # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            n_old = self.count
            n_total = n_old + n_new
            
            # Combined mean
            delta = mean_new - self.mean
            mean_combined = self.mean + delta * (n_new / n_total)
            
            # Combined variance (using parallel algorithm)
            # M2_new = variance_new * n_new
            var_new = x_flat.float().var(unbiased=False).item()
            M2_new = var_new * n_new
            
            # Update M2 using parallel formula
            self.M2 = self.M2 + M2_new + delta * delta * (n_old * n_new / n_total)
            
            # Update mean and count
            self.mean = mean_combined
            self.count = n_total
    
    def get_mean_std(self) -> Tuple[float, float]:
        """
        Get mean and standard deviation
        
        Returns:
            (mean, std): Mean and standard deviation
        """
        if self.count < 2:
            return (0.0, 1.0)
        variance = self.M2 / (self.count - 1)
        std = np.sqrt(max(variance, 1e-8))
        return (self.mean, std)


class GradientStatisticsCollector:
    """
    Collects gradient statistics from training data for threshold computation
    """
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        self.model = model
        self.layer_names = layer_names
        self.hooks = []
        self.statistics = defaultdict(WelfordStatistics)
        self.activations = {}
        
    def _make_hook(self, name: str):
        def hook(module, input, output):
            if name not in self.activations:
                self.activations[name] = []
            # Store activation for gradient computation
            self.activations[name].append(output.detach().clone())
        return hook
    
    def register_hooks(self):
        """Register forward hooks to collect activations"""
        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            if self.layer_names is None or name in self.layer_names:
                if isinstance(module, (nn.Conv2d, nn.Linear, QuanConv2d, QuanLinear)):
                    hook = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append((name, hook))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for name, hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def compute_gradients(self, activations: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute gradients (successive differences) for collected activations
        
        Args:
            activations: Dict mapping layer names to lists of activation tensors
            
        Returns:
            gradients: Dict mapping layer names to gradient tensors
        """
        gradients = {}
        
        for name, act_list in activations.items():
            if not act_list:
                continue
            
            # Process each activation in the list
            for act in act_list:
                grad = self._compute_gradient_for_activation(act)
                if name not in gradients:
                    gradients[name] = []
                gradients[name].append(grad)
        
        # Average gradients across batches for statistics
        avg_gradients = {}
        for name, grad_list in gradients.items():
            if grad_list:
                avg_grad = torch.stack(grad_list).mean(dim=0)
                avg_gradients[name] = avg_grad
                # Update statistics
                self.statistics[name].update(avg_grad)
        
        return avg_gradients
    
    def _compute_gradient_for_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient (successive differences) for a single activation tensor
        
        Args:
            activation: [B, C, H, W] for conv or [B, C] for linear
            
        Returns:
            gradient: Gradient tensor with same shape as activation
        """
        if activation.dim() == 2:  # Dense layer: [B, C]
            # Compute differences along channel dimension
            # For dense layer, we compute differences between adjacent neurons
            B, C = activation.shape
            if C < 2:
                return torch.zeros_like(activation)
            
            # Circular shift and compute difference
            shifted = torch.roll(activation, shifts=1, dims=1)
            gradient = activation - shifted
            return gradient
        
        elif activation.dim() == 4:  # Conv layer: [B, C, H, W]
            # For conv layers, compute gradients along channel dimension
            # using 1D convolution (as per paper)
            B, C, H, W = activation.shape
            if C < 2:
                return torch.zeros_like(activation)
            
            # Reshape to [B*H*W, C] for channel-wise processing
            act_reshaped = activation.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            act_reshaped = act_reshaped.view(-1, C)  # [B*H*W, C]
            
            # Compute differences along channel dimension
            shifted = torch.roll(act_reshaped, shifts=1, dims=1)
            gradient_reshaped = act_reshaped - shifted
            
            # Reshape back to [B, C, H, W]
            gradient = gradient_reshaped.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return gradient
        
        else:
            return torch.zeros_like(activation)
    
    def collect_statistics(self, data_loader, num_batches: Optional[int] = None):
        """
        Collect gradient statistics from training data
        
        Args:
            data_loader: DataLoader for training data
            num_batches: Number of batches to process (None for all)
        """
        self.model.eval()
        self.register_hooks()
        
        try:
            batch_count = 0
            total_batches = num_batches if num_batches is not None else len(data_loader)
            
            with torch.no_grad():
                for inputs, _ in data_loader:
                    if num_batches is not None and batch_count >= num_batches:
                        break
                    
                    inputs = inputs.to(next(self.model.parameters()).device)
                    _ = self.model(inputs)
                    
                    # Process gradients immediately for this batch to save memory
                    batch_gradients = {}
                    for name, act_list in self.activations.items():
                        if act_list:
                            # Process the last activation (current batch)
                            act = act_list[-1]
                            grad = self._compute_gradient_for_activation(act)
                            if name not in batch_gradients:
                                batch_gradients[name] = []
                            batch_gradients[name].append(grad)
                    
                    # Update statistics immediately
                    for name, grad_list in batch_gradients.items():
                        if grad_list:
                            # Average gradients across batch if multiple
                            avg_grad = torch.stack(grad_list).mean(dim=0) if len(grad_list) > 1 else grad_list[0]
                            self.statistics[name].update(avg_grad)
                    
                    # Clear activations to save memory
                    self.activations.clear()
                    batch_count += 1
                    
                    # Print progress every 10 batches
                    if batch_count % 10 == 0:
                        print(f"  已处理 {batch_count}/{total_batches} batches...")
            
            print(f"  统计收集完成: 共处理 {batch_count} batches, {len(self.statistics)} 层")
            
        finally:
            self.remove_hooks()
            self.activations.clear()
    
    def get_thresholds(self, k: float = 4.0) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get statistical thresholds using Chebyshev's inequality
        
        Args:
            k: Chebyshev parameter (kσ bounds)
            
        Returns:
            thresholds: Dict mapping layer names to (lower_bound, upper_bound) tensors
        """
        thresholds = {}
        for name, stats in self.statistics.items():
            mean, std = stats.get_mean_std()
            lower = mean - k * std
            upper = mean + k * std
            thresholds[name] = (torch.tensor(lower), torch.tensor(upper))
        return thresholds


class GradientStatisticsRestorer:
    """
    Gradient Statistics-based Error Restorer
    
    Implements error localization and suppression using gradient statistics
    """
    def __init__(
        self,
        model: nn.Module,
        thresholds: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        kernel_size: int = 3,
        k: float = 4.0,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: The model to apply error restoration to
            thresholds: Dict mapping layer names to (lower, upper) threshold tensors
            kernel_size: Kernel size for gradient refinement (3 or 9)
            k: Chebyshev parameter (if thresholds not provided, will compute with this k)
            layer_names: List of layer names to apply restoration to (None for all)
        """
        self.model = model
        self.thresholds = thresholds
        self.kernel_size = kernel_size
        self.k = k
        self.layer_names = layer_names
        self.hooks = []
        self.restore_enabled = False
        
        # Determine offset 'a' based on kernel size
        # For kernel_size=3: a=1, for kernel_size=9: a=4
        self.offset_a = (kernel_size - 1) // 2
    
    def enable(self):
        """Enable error restoration"""
        if self.restore_enabled:
            return
        self.restore_enabled = True
        self._register_hooks()
    
    def disable(self):
        """Disable error restoration"""
        if not self.restore_enabled:
            return
        self.restore_enabled = False
        self._remove_hooks()
    
    def _register_hooks(self):
        """Register forward hooks for error suppression"""
        modules = dict(self.model.named_modules())
        for name, module in modules.items():
            if self.layer_names is not None and name not in self.layer_names:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear, QuanConv2d, QuanLinear)):
                hook = module.register_forward_hook(self._make_restore_hook(name))
                self.hooks.append((name, hook))
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for name, hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _make_restore_hook(self, name: str):
        def hook(module, input, output):
            if not self.restore_enabled:
                return output
            
            if name not in self.thresholds:
                return output
            
            # Apply error suppression
            suppressed_output = self._suppress_errors(output, name)
            return suppressed_output
        
        return hook
    
    def _compute_gradient(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient (successive differences) for activation
        
        Args:
            activation: [B, C, H, W] or [B, C]
            
        Returns:
            gradient: Gradient tensor
        """
        if activation.dim() == 2:  # Dense: [B, C]
            if activation.size(1) < 2:
                return torch.zeros_like(activation)
            shifted = torch.roll(activation, shifts=1, dims=1)
            return activation - shifted
        
        elif activation.dim() == 4:  # Conv: [B, C, H, W]
            B, C, H, W = activation.shape
            if C < 2:
                return torch.zeros_like(activation)
            
            # Reshape for channel-wise processing
            act_reshaped = activation.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            act_reshaped = act_reshaped.view(-1, C)  # [B*H*W, C]
            
            # Compute differences
            shifted = torch.roll(act_reshaped, shifts=1, dims=1)
            gradient_reshaped = act_reshaped - shifted
            
            # Reshape back
            gradient = gradient_reshaped.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return gradient
        
        else:
            return torch.zeros_like(activation)
    
    def _refine_mask(self, mask: torch.Tensor, is_conv: bool) -> torch.Tensor:
        """
        Refine error mask using kernel-based refinement
        
        Equation (10): I' = shift(I & shift(I, a), -a)
        
        Args:
            mask: Binary mask [B, C, H, W] or [B, C] (can be float or bool)
            is_conv: Whether this is a convolutional layer
            
        Returns:
            refined_mask: Refined binary mask (bool type)
        """
        # Convert to bool for bitwise operations
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        if is_conv and mask.dim() == 4:
            # For conv layers, shift along channel dimension
            # Reshape to [B*H*W, C] for channel-wise processing
            B, C, H, W = mask.shape
            mask_reshaped = mask.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            mask_reshaped = mask_reshaped.view(-1, C)  # [B*H*W, C]
            
            # Apply refinement (bitwise AND on bool tensors)
            shifted1 = torch.roll(mask_reshaped, shifts=self.offset_a, dims=1)
            and_result = mask_reshaped & shifted1
            refined_reshaped = torch.roll(and_result, shifts=-self.offset_a, dims=1)
            
            # Reshape back
            refined = refined_reshaped.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            return refined
        
        elif not is_conv and mask.dim() == 2:
            # For dense layers, shift along channel dimension
            shifted1 = torch.roll(mask, shifts=self.offset_a, dims=1)
            and_result = mask & shifted1
            refined = torch.roll(and_result, shifts=-self.offset_a, dims=1)
            return refined
        
        else:
            return mask
    
    def _suppress_errors(self, output: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Suppress errors in output using gradient statistics
        
        Args:
            output: Layer output [B, C, H, W] or [B, C]
            layer_name: Name of the layer
            
        Returns:
            suppressed_output: Output with errors set to zero
        """
        lower, upper = self.thresholds[layer_name]
        lower = lower.to(output.device)
        upper = upper.to(output.device)
        
        # Compute gradient
        gradient = self._compute_gradient(output)
        
        # Create initial mask: I = (G > U) || (G < L)
        # Equation (9)
        mask_upper = gradient > upper
        mask_lower = gradient < lower
        mask = mask_upper | mask_lower
        
        # Refine mask: I' = shift(I & shift(I, a), -a)
        # Equation (10)
        is_conv = output.dim() == 4
        refined_mask = self._refine_mask(mask, is_conv)  # mask is already bool
        
        # Suppress errors: Ys = Y o (!I')
        # Equation (11)
        suppressed_output = output * (~refined_mask).float()
        
        return suppressed_output


def create_gradient_statistics_restorer(
    model: nn.Module,
    data_loader,
    k: float = 4.0,
    kernel_size: int = 3,
    num_statistics_batches: int = 50,
    layer_names: Optional[List[str]] = None,
) -> GradientStatisticsRestorer:
    """
    Create and initialize a Gradient Statistics Restorer
    
    Args:
        model: The model to apply restoration to
        data_loader: DataLoader for collecting statistics
        k: Chebyshev parameter (default 4.0)
        kernel_size: Kernel size for refinement (3 or 9, default 3)
        num_statistics_batches: Number of batches to collect statistics from
        layer_names: List of layer names to apply restoration to (None for all)
        
    Returns:
        restorer: Initialized GradientStatisticsRestorer
    """
    # Collect statistics
    collector = GradientStatisticsCollector(model, layer_names)
    collector.collect_statistics(data_loader, num_statistics_batches)
    
    # Get thresholds
    thresholds = collector.get_thresholds(k=k)
    
    # Create restorer
    restorer = GradientStatisticsRestorer(
        model=model,
        thresholds=thresholds,
        kernel_size=kernel_size,
        k=k,
        layer_names=layer_names,
    )
    
    return restorer

