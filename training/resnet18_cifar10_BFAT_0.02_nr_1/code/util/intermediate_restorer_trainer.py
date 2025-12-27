"""
Training utilities for Intermediate Layer Restorer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from .intermediate_layer_restorer import IntermediateLayerRestorer


class IntermediateLayerRestorerTrainer:
    """
    Trainer for Intermediate Layer Restorer
    
    Uses supervised learning to train activation restoration networks.
    """
    def __init__(
        self,
        restorer: IntermediateLayerRestorer,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.restorer = restorer
        self.max_grad_norm = max_grad_norm
        
        # Optimizer for all layer restorers
        self.optimizer = optim.Adam(
            restorer.layer_restorers.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )
    
    def train_step(
        self,
        inputs: torch.Tensor,
        clean_activations: Dict[str, torch.Tensor],
        faulted_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Train on a batch
        
        Args:
            inputs: Input tensor
            clean_activations: Clean activations from each sensitive layer
            faulted_activations: Faulted activations from each sensitive layer
        
        Returns:
            metrics: Training metrics
        """
        self.restorer.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        layer_losses = {}
        
        # Train each layer restorer
        for layer_name, restorer in self.restorer.layer_restorers.items():
            if layer_name not in faulted_activations:
                continue
            
            faulted = faulted_activations[layer_name]
            clean = clean_activations.get(layer_name)
            
            if clean is None:
                continue
            
            # Forward pass
            restored = restorer(faulted, clean)
            
            # Compute loss: MSE between restored and clean
            mse_loss = nn.functional.mse_loss(restored, clean)
            
            # Additional loss: encourage small corrections (sparsity)
            correction = restored - faulted
            sparsity_loss = torch.abs(correction).mean() * 0.01
            
            # Total loss for this layer
            layer_loss = mse_loss + sparsity_loss
            total_loss += layer_loss
            layer_losses[layer_name] = layer_loss.item()
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.restorer.layer_restorers.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()
        
        metrics = {
            'total_loss': total_loss.item() if total_loss > 0 else 0.0,
            'layer_losses': layer_losses,
        }
        
        return metrics
    
    def eval_step(
        self,
        inputs: torch.Tensor,
        clean_activations: Dict[str, torch.Tensor],
        faulted_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Evaluate on a batch"""
        self.restorer.eval()
        
        total_loss = 0.0
        layer_losses = {}
        
        with torch.no_grad():
            for layer_name, restorer in self.restorer.layer_restorers.items():
                if layer_name not in faulted_activations:
                    continue
                
                faulted = faulted_activations[layer_name]
                clean = clean_activations.get(layer_name)
                
                if clean is None:
                    continue
                
                restored = restorer(faulted, clean)
                mse_loss = nn.functional.mse_loss(restored, clean)
                
                total_loss += mse_loss
                layer_losses[layer_name] = mse_loss.item()
        
        metrics = {
            'total_loss': total_loss.item() if total_loss > 0 else 0.0,
            'layer_losses': layer_losses,
        }
        
        return metrics


