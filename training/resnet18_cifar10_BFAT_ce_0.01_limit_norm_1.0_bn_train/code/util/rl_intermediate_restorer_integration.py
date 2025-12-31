"""
Integration utilities for RL-based Intermediate Layer Restorer

Provides helper functions to integrate RL intermediate layer restoration
into existing training pipelines.
"""

import torch
from typing import Dict, Optional, Tuple
from .rl_intermediate_layer_restorer import (
    RLIntermediateLayerRestorer,
    RLLayerRestorerTrainer,
)
from .rl_restorer_integration import (
    create_rl_restorer_with_features,
    extract_features_for_rl_restorer,
)
from .intermediate_restorer_integration import collect_activations_for_training
from .sensitive_restorer import SensitiveActivationCollector


def create_rl_intermediate_layer_restorer(
    model,
    sensitive_info: Dict,
    baseline_stats: Optional[Dict] = None,
    state_dim: int = 128,
    hidden_dim: int = 256,
    max_steps_per_layer: int = 2,
) -> Tuple[RLIntermediateLayerRestorer, SensitiveActivationCollector]:
    """
    Create RL-based intermediate layer restorer
    
    Args:
        model: The main model
        sensitive_info: Sensitive layer information
        baseline_stats: Baseline statistics (optional)
        state_dim: State dimension for RL
        hidden_dim: Hidden dimension
        max_steps_per_layer: Maximum correction steps per layer
    
    Returns:
        restorer: RLIntermediateLayerRestorer instance
        collector: SensitiveActivationCollector for collecting activations
    """
    # Create RL intermediate layer restorer
    restorer = RLIntermediateLayerRestorer(
        sensitive_info=sensitive_info,
        model=model,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        max_steps_per_layer=max_steps_per_layer,
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


def create_hybrid_rl_restorer(
    intermediate_restorer: RLIntermediateLayerRestorer,
    output_restorer,  # RLRestorer or LayerwiseRLRestorer
    use_intermediate: bool = True,
    use_output: bool = True,
):
    """
    Create hybrid RL restorer combining intermediate and output layer restoration
    
    Args:
        intermediate_restorer: RL intermediate layer restorer
        output_restorer: RL output layer restorer
        use_intermediate: Whether to use intermediate layer restoration
        use_output: Whether to use output layer restoration
    
    Returns:
        Hybrid restorer wrapper
    """
    class HybridRLRestorer:
        def __init__(self, intermediate, output, use_int, use_out):
            self.intermediate = intermediate
            self.output = output
            self.use_intermediate = use_int
            self.use_output = use_out
        
        def forward(
            self,
            inputs: torch.Tensor,
            logits_faulted: Optional[torch.Tensor] = None,
            features: Optional[torch.Tensor] = None,
            layer_features: Optional[list] = None,
            clean_activations: Optional[Dict[str, torch.Tensor]] = None,
            training: bool = False,
            ber_level: Optional[float] = None,
        ) -> Tuple[torch.Tensor, Dict]:
            """
            Two-stage RL restoration
            
            Args:
                inputs: Input tensor
                logits_faulted: Faulted logits (if already computed)
                features: Features for output restorer
                layer_features: Layer features for output restorer
                clean_activations: Clean activations for intermediate restoration
                training: Training mode
                ber_level: BER level for reward weighting
            
            Returns:
                logits_restored: Restored logits
                info: Additional information
            """
            # Stage 1: Intermediate layer restoration
            if self.use_intermediate:
                if clean_activations is not None:
                    self.intermediate.set_clean_activations(clean_activations)
                
                self.intermediate.enable_restoration()
                logits_after_intermediate = self.intermediate.model(inputs)
                self.intermediate.disable_restoration()
            else:
                logits_after_intermediate = logits_faulted
            
            # Stage 2: Output layer restoration
            if self.use_output:
                if isinstance(self.output, type):  # Check if it's LayerwiseRLRestorer
                    from .layerwise_rl_restorer import LayerwiseRLRestorer
                    if isinstance(self.output, LayerwiseRLRestorer):
                        logits_restored, info = self.output(
                            logits_after_intermediate,
                            layer_features=layer_features or [],
                            training=training,
                            ber_level=ber_level,
                        )
                    else:
                        logits_restored, info = self.output(
                            logits_after_intermediate,
                            features=features,
                            training=training,
                            ber_level=ber_level,
                        )
                else:
                    # Standard RL restorer
                    logits_restored, info = self.output(
                        logits_after_intermediate,
                        features=features,
                        training=training,
                        ber_level=ber_level,
                    )
            else:
                logits_restored = logits_after_intermediate
                info = {}
            
            return logits_restored, info
        
        def train(self):
            """Set to training mode"""
            self.intermediate.train()
            self.output.train()
        
        def eval(self):
            """Set to eval mode"""
            self.intermediate.eval()
            self.output.eval()
    
    return HybridRLRestorer(intermediate_restorer, output_restorer, use_intermediate, use_output)


def train_hybrid_rl_restorer_step(
    intermediate_trainer: RLLayerRestorerTrainer,
    output_trainer,  # RLRestorerTrainer
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    fault_injector,
    intermediate_collector: SensitiveActivationCollector,
    output_collector,  # For output restorer features
    use_intermediate: bool = True,
    use_output: bool = True,
    use_enhanced_features: bool = False,
    use_layerwise: bool = True,
    ber_level: Optional[float] = None,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Perform one training step for hybrid RL restorer
    
    Args:
        intermediate_trainer: Trainer for intermediate layer restorer
        output_trainer: Trainer for output layer restorer
        model: The main model
        inputs: Input tensor
        targets: Target labels
        fault_injector: Fault injector
        intermediate_collector: Collector for intermediate activations
        output_collector: Collector for output restorer features
        use_intermediate: Whether to train intermediate restorer
        use_output: Whether to train output restorer
        use_enhanced_features: Whether to use enhanced features for output restorer
        use_layerwise: Whether to use layerwise features for output restorer
        ber_level: BER level for reward weighting
        device: Device for computation
    
    Returns:
        metrics: Training metrics
    """
    if device is None:
        device = inputs.device
    
    all_metrics = {}
    
    # Collect clean logits
    fault_injector.disable()
    with torch.no_grad():
        logits_clean = model(inputs)
    
    # Collect faulted logits
    fault_injector.enable()
    logits_faulted = model(inputs)
    fault_injector.disable()
    
    # Train intermediate layer restorer
    if use_intermediate:
        # Collect clean and faulted activations
        clean_activations, faulted_activations = collect_activations_for_training(
            collector=intermediate_collector,
            model=model,
            inputs=inputs,
            fault_injector=fault_injector,
            device=device,
        )
        
        # Train intermediate restorer
        intermediate_metrics = intermediate_trainer.train_step(
            inputs=inputs,
            clean_activations=clean_activations,
            faulted_activations=faulted_activations,
        )
        
        # Add prefix to metrics
        for key, value in intermediate_metrics.items():
            all_metrics[f'intermediate_{key}'] = value
    
    # Train output layer restorer
    if use_output:
        # Extract features for output restorer
        features, _, layer_features = extract_features_for_rl_restorer(
            collector=output_collector,
            model=model,
            inputs=inputs,
            fault_injector=fault_injector,
            use_enhanced_features=use_enhanced_features,
            use_layerwise=use_layerwise,
            device=device,
        )
        
        # Train output restorer
        if use_layerwise and layer_features is not None:
            # Layerwise restorer
            # Note: Need to check if output_trainer supports layerwise
            # For now, assume it's standard RL restorer
            output_metrics = output_trainer.train_step(
                logits_faulted=logits_faulted,
                features=features,  # May be None for layerwise
                logits_clean=logits_clean,
                ber_level=ber_level,
            )
        else:
            output_metrics = output_trainer.train_step(
                logits_faulted=logits_faulted,
                features=features,
                logits_clean=logits_clean,
                ber_level=ber_level,
            )
        
        # Add prefix to metrics
        for key, value in output_metrics.items():
            all_metrics[f'output_{key}'] = value
    
    return all_metrics


