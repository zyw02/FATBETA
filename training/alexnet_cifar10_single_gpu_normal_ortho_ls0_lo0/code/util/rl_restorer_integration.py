"""
Integration utilities for RL Restorer with enhanced features

This module provides helper functions to integrate RL Restorer with
enhanced feature extraction into existing training pipelines.

Supports two modes:
1. Concatenated features: All layer features concatenated into a single vector (original)
2. Layer-wise features: Each layer's features kept separate, using attention to combine (new)
"""

import torch
from typing import Optional, Dict, Tuple, List
from .rl_restorer import RLRestorer
from .rl_restorer_trainer import RLRestorerTrainer
from .layerwise_rl_restorer import LayerwiseRLRestorer
from .enhanced_feature_extractor import EnhancedSensitiveActivationCollector
from .sensitive_restorer import SensitiveActivationCollector


def create_rl_restorer_with_features(
    model,
    sensitive_info: Dict,
    baseline_stats: Optional[Dict],
    num_classes: int,
    use_enhanced_features: bool = False,
    use_layerwise: bool = True,  # 新增：是否使用层级感知版本（推荐）
    feature_dim: Optional[int] = None,
    state_dim: int = 128,
    hidden_dim: int = 256,
    max_steps: int = 3,
    device: torch.device = None,
) -> Tuple[RLRestorer, object]:
    """
    Create RL Restorer with appropriate feature collector
    
    Args:
        model: The main model
        sensitive_info: Sensitive layer information
        baseline_stats: Baseline statistics for normalization
        num_classes: Number of classes
        use_enhanced_features: Whether to use enhanced feature extraction
        feature_dim: Feature dimension (if None, will be determined automatically)
        state_dim: State encoding dimension
        hidden_dim: Hidden layer dimension
        max_steps: Maximum correction steps
        device: Device for computation
    
    Returns:
        restorer: RLRestorer instance
        collector: Feature collector (EnhancedSensitiveActivationCollector or SensitiveActivationCollector)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create appropriate collector
    if use_enhanced_features:
        collector = EnhancedSensitiveActivationCollector(
            model, sensitive_info, baseline_stats
        )
        # Determine feature_dim by doing a dummy forward pass
        if feature_dim is None:
            feature_dim = _determine_enhanced_feature_dim(
                model, collector, sensitive_info, device
            )
    else:
        collector = SensitiveActivationCollector(
            model, sensitive_info, baseline_stats
        )
        # Determine feature_dim from standard features
        if feature_dim is None:
            feature_dim = _determine_standard_feature_dim(
                model, collector, sensitive_info, device
            )
    
    # Create RL Restorer (layer-wise or concatenated)
    if use_layerwise:
        # Layer-wise version: preserves hierarchical structure
        # First determine feature_dims_per_layer using the same collector type
        if use_enhanced_features:
            # Use EnhancedSensitiveActivationCollector to determine dimensions
            collector_temp = EnhancedSensitiveActivationCollector(model, sensitive_info, baseline_stats)
        else:
            # Use standard SensitiveActivationCollector
            collector_temp = SensitiveActivationCollector(model, sensitive_info, baseline_stats)
        
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        collector_temp.clear()
        with torch.no_grad():
            _ = model(dummy_input)
        layer_features_temp, feature_dims_per_layer = collector_temp.build_layer_features(device)
        collector_temp.remove()
        
        if layer_features_temp is None or len(layer_features_temp) == 0:
            raise RuntimeError("Failed to determine layer feature dimensions")
        
        restorer = LayerwiseRLRestorer(
            feature_dims_per_layer=feature_dims_per_layer,
            num_classes=num_classes,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            use_attention=True,  # Use attention to combine layer features
        ).to(device)
    else:
        # Original concatenated version
        restorer = RLRestorer(
            feature_dim=feature_dim,
            num_classes=num_classes,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            use_enhanced_features=use_enhanced_features,
        ).to(device)
    
    return restorer, collector


def _determine_standard_feature_dim(
    model,
    collector: SensitiveActivationCollector,
    sensitive_info: Dict,
    device: torch.device,
) -> int:
    """Determine feature dimension for standard features"""
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Adjust size as needed
    
    # Forward pass to collect features
    collector.clear()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Extract features
    features = collector.build_feature_vector(device)
    
    if features is None:
        raise ValueError("Could not determine feature dimension")
    
    return features.size(1)


def _determine_enhanced_feature_dim(
    model,
    collector: EnhancedSensitiveActivationCollector,
    sensitive_info: Dict,
    device: torch.device,
) -> int:
    """Determine feature dimension for enhanced features"""
    # Create dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Adjust size as needed
    
    # Forward pass to collect clean activations
    collector.clear()
    with torch.no_grad():
        _ = model(dummy_input)
    clean_activations = collector.buffers.copy()
    
    # Forward pass again to collect faulted activations (or reuse clean)
    collector.clear()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Extract enhanced features
    features = collector.build_enhanced_feature_vector(
        device=device,
        clean_activations=clean_activations,
    )
    
    if features is None:
        raise ValueError("Could not determine enhanced feature dimension")
    
    return features.size(1)


def extract_features_for_rl_restorer(
    collector,
    model,
    inputs: torch.Tensor,
    fault_injector,
    use_enhanced_features: bool = False,
    use_layerwise: bool = True,  # 新增：是否使用层级感知版本
    device: torch.device = None,
) -> Tuple[Optional[torch.Tensor], Optional[Dict], Optional[List[torch.Tensor]]]:
    """
    Extract features for RL Restorer
    
    Args:
        collector: Feature collector (EnhancedSensitiveActivationCollector or SensitiveActivationCollector)
        model: The main model
        inputs: Input tensors
        fault_injector: Fault injector
        use_enhanced_features: Whether using enhanced features
        use_layerwise: Whether using layer-wise features (preserves hierarchy)
        device: Device for computation
    
    Returns:
        features: Extracted features [B, feature_dim] (if not layerwise) or None (if layerwise)
        clean_activations: Clean activations (only for enhanced features, None otherwise)
        layer_features: List of [B, feat_dim_i] per layer (if layerwise) or None (if not layerwise)
    """
    if device is None:
        device = inputs.device
    
    clean_activations = None
    layer_features = None
    
    if use_layerwise:
        # Layer-wise features: preserve hierarchical structure
        collector.clear()
        fault_injector.enable()
        with torch.no_grad():
            _ = model(inputs)
        fault_injector.disable()
        
        layer_features, _ = collector.build_layer_features(device)
        features = None  # Not used in layer-wise mode
    elif use_enhanced_features:
        # Enhanced features require clean activations
        collector.clear()
        fault_injector.disable()
        with torch.no_grad():
            _ = model(inputs)
        clean_activations = collector.buffers.copy()
        
        # Collect faulted activations
        collector.clear()
        fault_injector.enable()
        with torch.no_grad():
            _ = model(inputs)
        fault_injector.disable()
        
        # Extract enhanced features
        features = collector.build_enhanced_feature_vector(
            device=device,
            clean_activations=clean_activations,
        )
    else:
        # Standard features (concatenated)
        collector.clear()
        fault_injector.enable()
        with torch.no_grad():
            _ = model(inputs)
        fault_injector.disable()
        
        features = collector.build_feature_vector(device)
    
    return features, clean_activations, layer_features

