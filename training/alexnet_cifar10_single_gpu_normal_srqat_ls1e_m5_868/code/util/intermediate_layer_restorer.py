"""
Intermediate Layer Feature Restorer

This module implements feature restoration at intermediate layers (activations)
rather than only at the output layer (logits). This approach:

1. Collects faulted activations from sensitive layers
2. Restores them using learned restoration networks
3. Injects restored activations back into the forward pass
4. Allows restored features to propagate through remaining layers

This is inspired by adversarial defense methods that perform feature denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .rl_restorer import RLRestorer, ActorNetwork, CriticNetwork
from .layerwise_rl_restorer import LayerwiseRLRestorer


class LayerActivationRestorer(nn.Module):
    """
    Restores activations for a single layer
    
    Takes faulted activations and restores them to be closer to clean activations.
    """
    def __init__(
        self,
        activation_shape: Tuple[int, ...],  # e.g., (C, H, W) for conv, (C,) for linear
        is_conv: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.activation_shape = activation_shape
        self.is_conv = is_conv
        
        if is_conv:
            # For convolutional activations: [B, C, H, W]
            C, H, W = activation_shape
            # Use 1x1 convolutions to process spatial features
            self.restorer = nn.Sequential(
                nn.Conv2d(C, hidden_dim, kernel_size=1),
                nn.LayerNorm([hidden_dim, H, W]),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.LayerNorm([hidden_dim, H, W]),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, C, kernel_size=1),
            )
        else:
            # For linear activations: [B, C]
            C = activation_shape[0]
            self.restorer = nn.Sequential(
                nn.Linear(C, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, C),
            )
    
    def forward(
        self,
        faulted_activation: torch.Tensor,
        clean_activation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Restore faulted activation
        
        Args:
            faulted_activation: [B, C, H, W] or [B, C] - faulted activation
            clean_activation: [B, C, H, W] or [B, C] - clean activation (for training only)
        
        Returns:
            restored_activation: [B, C, H, W] or [B, C] - restored activation
        """
        # Compute restoration delta
        delta = self.restorer(faulted_activation)
        
        # Apply restoration (residual connection)
        restored = faulted_activation + delta
        
        return restored


class IntermediateLayerRestorer(nn.Module):
    """
    Restores activations at multiple intermediate layers
    
    This is the main class that coordinates restoration across all sensitive layers.
    """
    def __init__(
        self,
        sensitive_info: Dict[str, Dict],
        model: nn.Module,
        hidden_dim: int = 256,
        use_rl: bool = False,  # Whether to use RL-based restoration
    ):
        super().__init__()
        self.sensitive_info = sensitive_info
        self.model = model.module if hasattr(model, "module") else model
        self.use_rl = use_rl
        
        # Create a restorer for each sensitive layer
        self.layer_restorers = nn.ModuleDict()
        self.activation_shapes = {}
        
        # Determine activation shapes by doing a dummy forward pass
        self._determine_activation_shapes()
        
        # Create restorers
        for layer_name, info in sensitive_info.items():
            if layer_name not in self.activation_shapes:
                continue
            
            shape = self.activation_shapes[layer_name]
            is_conv = len(shape) == 3  # (C, H, W) for conv, (C,) for linear
            
            self.layer_restorers[layer_name] = LayerActivationRestorer(
                activation_shape=shape,
                is_conv=is_conv,
                hidden_dim=hidden_dim,
            )
        
        # Hooks for intercepting and replacing activations
        self.hooks = []
        self.faulted_activations = {}
        self.clean_activations = {}
        self.restore_enabled = False
    
    def _determine_activation_shapes(self):
        """Determine activation shapes for each sensitive layer"""
        modules = dict(self.model.named_modules())
        
        # Create temporary hooks to capture shapes
        temp_hooks = []
        temp_activations = {}
        
        def make_temp_hook(name):
            def hook(module, input, output):
                if name not in temp_activations:
                    # Store shape (excluding batch dimension)
                    if output.dim() == 4:  # Conv: [B, C, H, W]
                        temp_activations[name] = output.shape[1:]
                    elif output.dim() == 2:  # Linear: [B, C]
                        temp_activations[name] = output.shape[1:]
            return hook
        
        for name in self.sensitive_info.keys():
            if name in modules:
                hook = modules[name].register_forward_hook(make_temp_hook(name))
                temp_hooks.append(hook)
        
        # Dummy forward pass
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Store shapes
        self.activation_shapes = temp_activations.copy()
        
        # Remove temp hooks
        for hook in temp_hooks:
            hook.remove()
    
    def register_hooks(self):
        """Register hooks to intercept activations and replace with restored ones"""
        if self.hooks:
            # Already registered
            return
        
        modules = dict(self.model.named_modules())
        
        def make_hook(name, restorer):
            def hook(module, input, output):
                if not self.restore_enabled:
                    return output
                
                # Get faulted activation (clone to avoid in-place modification)
                faulted = output.clone()
                
                # Restore if we have clean activation for training, or just restore faulted
                if name in self.clean_activations:
                    clean = self.clean_activations[name]
                    restored = restorer(faulted, clean)
                else:
                    restored = restorer(faulted, None)
                
                # Replace output with restored activation
                # Note: We need to return a tensor that requires grad if output requires grad
                if output.requires_grad:
                    restored = restored.requires_grad_(True)
                
                return restored
            
            return hook
        
        for name, restorer in self.layer_restorers.items():
            if name in modules:
                hook = modules[name].register_forward_hook(make_hook(name, restorer))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def enable_restoration(self):
        """Enable activation restoration"""
        self.restore_enabled = True
    
    def disable_restoration(self):
        """Disable activation restoration"""
        self.restore_enabled = False
    
    def set_clean_activations(self, clean_activations: Dict[str, torch.Tensor]):
        """Set clean activations for training"""
        self.clean_activations = clean_activations
    
    def clear_activations(self):
        """Clear stored activations"""
        self.faulted_activations.clear()
        self.clean_activations.clear()
    
    def forward(
        self,
        inputs: torch.Tensor,
        clean_activations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with intermediate layer restoration
        
        Args:
            inputs: Input tensor
            clean_activations: Clean activations for training (optional)
        
        Returns:
            output: Model output (logits)
            restored_activations: Dict of restored activations for each layer
        """
        if clean_activations is not None:
            self.set_clean_activations(clean_activations)
        
        # Enable restoration
        self.enable_restoration()
        
        # Forward pass (hooks will intercept and restore activations)
        output = self.model(inputs)
        
        # Disable restoration
        self.disable_restoration()
        
        # Return output and any additional info
        return output, {}


class HybridRestorer(nn.Module):
    """
    Hybrid Restorer: Combines intermediate layer restoration with output layer restoration
    
    This uses a two-stage approach:
    1. Restore activations at intermediate layers
    2. Further restore the final logits using output-layer restorer
    """
    def __init__(
        self,
        intermediate_restorer: IntermediateLayerRestorer,
        output_restorer: RLRestorer,  # or LayerwiseRLRestorer
        use_intermediate: bool = True,
        use_output: bool = True,
    ):
        super().__init__()
        self.intermediate_restorer = intermediate_restorer
        self.output_restorer = output_restorer
        self.use_intermediate = use_intermediate
        self.use_output = use_output
    
    def forward(
        self,
        inputs: torch.Tensor,
        logits_faulted: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        layer_features: Optional[List[torch.Tensor]] = None,
        clean_activations: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Two-stage restoration
        
        Args:
            inputs: Input tensor
            logits_faulted: Faulted logits (if already computed)
            features: Features for output restorer (if using concatenated features)
            layer_features: Layer features for output restorer (if using layerwise)
            clean_activations: Clean activations for intermediate restoration
            training: Training mode
        
        Returns:
            logits_restored: Restored logits
            info: Additional information
        """
        # Stage 1: Intermediate layer restoration
        if self.use_intermediate:
            logits_after_intermediate, _ = self.intermediate_restorer(
                inputs,
                clean_activations=clean_activations,
            )
        else:
            logits_after_intermediate = logits_faulted
        
        # Stage 2: Output layer restoration
        if self.use_output:
            if isinstance(self.output_restorer, LayerwiseRLRestorer):
                # Layer-wise restorer
                logits_restored, info = self.output_restorer(
                    logits_after_intermediate,
                    layer_features=layer_features or [],
                    training=training,
                )
            else:
                # Standard RL restorer
                logits_restored, info = self.output_restorer(
                    logits_after_intermediate,
                    features=features,
                    training=training,
                )
        else:
            logits_restored = logits_after_intermediate
            info = {}
        
        return logits_restored, info

