"""
Integration utilities for Intermediate Layer Restorer

This module provides helper functions to integrate intermediate layer restoration
into existing training pipelines.
"""

import torch
from typing import Dict, Optional, Tuple
from .intermediate_layer_restorer import (
    IntermediateLayerRestorer,
    HybridRestorer,
)
from .intermediate_restorer_trainer import IntermediateLayerRestorerTrainer
from .sensitive_restorer import SensitiveActivationCollector


def create_intermediate_layer_restorer(
    model,
    sensitive_info: Dict,
    baseline_stats: Optional[Dict] = None,
    hidden_dim: int = 256,
) -> Tuple[IntermediateLayerRestorer, SensitiveActivationCollector]:
    """
    Create intermediate layer restorer with feature collector
    
    Args:
        model: The main model
        sensitive_info: Sensitive layer information
        baseline_stats: Baseline statistics (optional)
        hidden_dim: Hidden dimension for restoration networks
    
    Returns:
        restorer: IntermediateLayerRestorer instance
        collector: SensitiveActivationCollector for collecting activations
    """
    # Create restorer
    restorer = IntermediateLayerRestorer(
        sensitive_info=sensitive_info,
        model=model,
        hidden_dim=hidden_dim,
        use_rl=False,  # Currently using MLP-based restoration
    )
    
    # Register hooks for activation interception
    restorer.register_hooks()
    
    # Create collector for collecting activations
    collector = SensitiveActivationCollector(
        model=model,
        sensitive_info=sensitive_info,
        baseline_stats=baseline_stats,
    )
    
    return restorer, collector


def create_hybrid_restorer(
    intermediate_restorer: IntermediateLayerRestorer,
    output_restorer,  # RLRestorer or LayerwiseRLRestorer
    use_intermediate: bool = True,
    use_output: bool = True,
) -> HybridRestorer:
    """
    Create hybrid restorer combining intermediate and output layer restoration
    
    Args:
        intermediate_restorer: Intermediate layer restorer
        output_restorer: Output layer restorer (RL-based)
        use_intermediate: Whether to use intermediate layer restoration
        use_output: Whether to use output layer restoration
    
    Returns:
        hybrid_restorer: HybridRestorer instance
    """
    return HybridRestorer(
        intermediate_restorer=intermediate_restorer,
        output_restorer=output_restorer,
        use_intermediate=use_intermediate,
        use_output=use_output,
    )


def collect_activations_for_training(
    collector: SensitiveActivationCollector,
    model,
    inputs: torch.Tensor,
    fault_injector,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collect clean and faulted activations for training intermediate restorer
    
    Args:
        collector: Feature collector
        model: The main model
        inputs: Input tensor
        fault_injector: Fault injector
        device: Device for computation
    
    Returns:
        clean_activations: Dict of clean activations {layer_name: activation}
        faulted_activations: Dict of faulted activations {layer_name: activation}
    """
    # Collect clean activations
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
    faulted_activations = collector.buffers.copy()
    fault_injector.disable()
    
    # Filter to only sensitive layers and selected channels
    clean_filtered = {}
    faulted_filtered = {}
    
    for layer_name, info in collector.sensitive_info.items():
        if layer_name not in clean_activations:
            continue
        
        idx = info.get('indices', [])
        if not idx:
            continue
        
        clean_act = clean_activations[layer_name]
        faulted_act = faulted_activations[layer_name]
        
        # Select only sensitive channels
        if clean_act.dim() == 4:  # Conv: [B, C, H, W]
            clean_filtered[layer_name] = clean_act[:, idx, ...]
            faulted_filtered[layer_name] = faulted_act[:, idx, ...]
        elif clean_act.dim() == 2:  # Linear: [B, C]
            clean_filtered[layer_name] = clean_act[:, idx]
            faulted_filtered[layer_name] = faulted_act[:, idx]
    
    return clean_filtered, faulted_filtered


