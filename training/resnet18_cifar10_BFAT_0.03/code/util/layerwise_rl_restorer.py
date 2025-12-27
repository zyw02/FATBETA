"""
Layer-wise RL Restorer

Instead of concatenating all layer features into a single vector,
this version processes each layer's features separately and uses
attention or aggregation mechanisms to combine them.

This preserves the hierarchical structure of the network and allows
the model to learn layer-specific correction strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .rl_restorer import ActorNetwork, CriticNetwork, compute_gae


class LayerwiseStateEncoder(nn.Module):
    """
    Encodes state from logits and layer-wise features
    
    Uses attention mechanism to combine features from different layers,
    preserving the hierarchical structure.
    """
    def __init__(
        self,
        feature_dims_per_layer: List[int],
        num_classes: int,
        state_dim: int,
        hidden_dim: int = 256,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_layers = len(feature_dims_per_layer)
        self.use_attention = use_attention
        
        # Layer-specific feature encoders
        self.layer_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
            ) for feat_dim in feature_dims_per_layer
        ])
        
        # Logits encoder
        self.logits_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        if use_attention:
            # Multi-head attention for layer feature fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim // 2,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )
            
            # Layer position embeddings
            self.layer_pos_embed = nn.Embedding(self.num_layers, hidden_dim // 2)
            
            # Aggregation after attention
            self.aggregator = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
            )
        else:
            # Simple weighted sum
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
            self.aggregator = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
            )
        
        # Final state combiner
        self.state_combiner = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        layer_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] - current logits
            layer_features: List of [B, feat_dim_i] - features from each layer
        
        Returns:
            state: [B, state_dim] - encoded state
        """
        B = logits.size(0)
        
        # Encode each layer's features independently
        encoded_layers = []
        for i, (feat, encoder) in enumerate(zip(layer_features, self.layer_encoders)):
            encoded = encoder(feat)  # [B, hidden_dim // 2]
            encoded_layers.append(encoded)
        
        # Combine layer features
        if self.use_attention:
            # Stack layer features: [B, num_layers, hidden_dim // 2]
            layer_stack = torch.stack(encoded_layers, dim=1)
            
            # Add positional embeddings
            layer_ids = torch.arange(self.num_layers, device=logits.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.layer_pos_embed(layer_ids)  # [B, num_layers, hidden_dim // 2]
            layer_stack = layer_stack + pos_emb
            
            # Self-attention: each layer attends to all layers
            attended, _ = self.attention(layer_stack, layer_stack, layer_stack)  # [B, num_layers, hidden_dim // 2]
            
            # Aggregate: mean pooling over layers
            aggregated = attended.mean(dim=1)  # [B, hidden_dim // 2]
            aggregated = self.aggregator(aggregated)  # [B, hidden_dim // 2]
        else:
            # Weighted sum of layer features
            layer_stack = torch.stack(encoded_layers, dim=0)  # [num_layers, B, hidden_dim // 2]
            weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]
            aggregated = torch.sum(
                layer_stack * weights.view(-1, 1, 1),
                dim=0
            )  # [B, hidden_dim // 2]
            aggregated = self.aggregator(aggregated)  # [B, hidden_dim // 2]
        
        # Encode logits
        logits_embed = self.logits_encoder(logits)  # [B, hidden_dim // 2]
        
        # Combine features and logits
        combined = torch.cat([aggregated, logits_embed, logits], dim=1)  # [B, hidden_dim + num_classes]
        state = self.state_combiner(combined)  # [B, state_dim]
        
        return state


class LayerwiseRLRestorer(nn.Module):
    """
    Layer-wise RL Restorer that processes features from each layer separately
    
    This preserves the hierarchical structure and allows the model to learn
    layer-specific correction strategies.
    """
    def __init__(
        self,
        feature_dims_per_layer: List[int],
        num_classes: int,
        state_dim: int = 128,
        action_dim: Optional[int] = None,
        hidden_dim: int = 256,
        max_steps: int = 3,
        gamma: float = 0.99,
        use_statistical_preprocessing: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        
        if action_dim is None:
            action_dim = num_classes
        
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.gamma = gamma
        self.use_statistical_preprocessing = use_statistical_preprocessing
        
        # Layer-wise state encoder
        self.state_encoder = LayerwiseStateEncoder(
            feature_dims_per_layer=feature_dims_per_layer,
            num_classes=num_classes,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
        )
        
        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        
        # Critic network
        self.critic = CriticNetwork(state_dim, hidden_dim)
        
        # Statistical preprocessing (optional)
        if use_statistical_preprocessing:
            self.logits_stats = None
    
    def forward(
        self,
        logits_faulted: torch.Tensor,
        layer_features: List[torch.Tensor],
        logits_clean: Optional[torch.Tensor] = None,
        training: bool = False,
        deterministic: bool = False,
        ber_level: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            logits_faulted: [B, num_classes] - faulted logits
            layer_features: List of [B, feat_dim_i] - features from each sensitive layer
            logits_clean: [B, num_classes] - clean logits (for training only)
            training: If True, collect trajectories for training
            deterministic: If True, use deterministic policy
            ber_level: Current BER level (for adaptive reward weighting)
        
        Returns:
            logits_restored: [B, num_classes] - restored logits
            info: Dict containing training information
        """
        batch_size = logits_faulted.size(0)
        device = logits_faulted.device
        
        # Statistical preprocessing (optional)
        if self.use_statistical_preprocessing and self.logits_stats is not None:
            logits_current = self._statistical_preprocess(logits_faulted)
        else:
            logits_current = logits_faulted.clone()
        
        # Store trajectories for training
        trajectories = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'entropies': [],
            'values': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        # Multi-step correction
        for step in range(self.max_steps):
            # Encode state from layer-wise features
            state = self.state_encoder(logits_current, layer_features)  # [B, state_dim]
            
            # Get action from policy
            action, log_prob, entropy = self.actor.sample(state, deterministic=deterministic)
            
            # Get value estimate
            value = self.critic(state)  # [B, 1]
            
            # Apply correction
            logits_next = logits_current + action  # [B, num_classes]
            
            # Compute reward (if training)
            if training and logits_clean is not None:
                reward = self._compute_reward(
                    logits_next, logits_clean, logits_faulted, step,
                    is_final=(step == self.max_steps - 1),
                    ber_level=ber_level,
                )  # [B]
                
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                reward = torch.zeros(batch_size, device=device)
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Store trajectory
            if training:
                # Store states with computation graph for actor training
                # (will be detached in trainer if needed for critic)
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['entropies'].append(entropy)
                # Detach values to avoid keeping computation graph (values are only used for GAE)
                trajectories['values'].append(value.squeeze(-1).detach())
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                
                # Next state (for GAE computation)
                if step < self.max_steps - 1:
                    next_state = self.state_encoder(logits_next, layer_features)
                else:
                    next_state = None
                trajectories['next_states'].append(next_state)
            
            # Update current logits
            logits_current = logits_next
        
        # Prepare info dict
        info = {
            'trajectories': trajectories,
            'final_reward': reward.mean().item() if training else 0.0,
        }
        
        return logits_current, info
    
    def _statistical_preprocess(self, logits: torch.Tensor) -> torch.Tensor:
        """Statistical preprocessing (same as RLRestorer)"""
        if self.logits_stats is None:
            return logits
        
        mean = self.logits_stats['mean'].to(logits.device)
        std = self.logits_stats['std'].to(logits.device) + 1e-6
        
        # Normalize
        logits_normalized = (logits - mean) / std
        
        # Clip outliers
        threshold = 3.0
        outliers = torch.abs(logits_normalized) > threshold
        
        # Replace outliers with mean
        logits_corrected = logits_normalized.clone()
        logits_corrected[outliers] = 0.0  # Mean of normalized is 0
        
        # Denormalize
        logits_corrected = logits_corrected * std + mean
        
        return logits_corrected
    
    def _compute_reward(
        self,
        logits_restored: torch.Tensor,
        logits_clean: torch.Tensor,
        logits_faulted: torch.Tensor,
        step: int,
        is_final: bool,
        ber_level: Optional[float] = None,
    ) -> torch.Tensor:
        """Same reward function as RLRestorer"""
        batch_size = logits_restored.size(0)
        device = logits_restored.device
        
        # Accuracy improvement
        pred_restored = logits_restored.argmax(dim=1)
        pred_clean = logits_clean.argmax(dim=1)
        pred_faulted = logits_faulted.argmax(dim=1)
        
        acc_restored = (pred_restored == pred_clean).float()
        acc_faulted = (pred_faulted == pred_clean).float()
        acc_improvement = acc_restored - acc_faulted
        
        # BER weighting
        if ber_level is not None:
            ber_weight = 1.0 + ber_level * 10.0
        else:
            ber_weight = 1.0
        acc_reward = acc_improvement * 2.0 * ber_weight
        
        # MSE reduction
        mse_faulted = F.mse_loss(logits_faulted, logits_clean, reduction='none').mean(dim=1)
        mse_restored = F.mse_loss(logits_restored, logits_clean, reduction='none').mean(dim=1)
        mse_reduction = mse_faulted - mse_restored
        mse_reward = torch.clamp(mse_reduction / 10.0, -1.0, 1.0)
        
        # Confidence improvement
        conf_restored = F.softmax(logits_restored, dim=1).max(dim=1)[0]
        conf_faulted = F.softmax(logits_faulted, dim=1).max(dim=1)[0]
        conf_improvement = conf_restored - conf_faulted
        conf_reward = conf_improvement * 0.5
        
        # Direction correctness
        correction = logits_restored - logits_faulted
        target_correction = logits_clean - logits_faulted
        direction_cosine = F.cosine_similarity(correction, target_correction, dim=1)
        direction_reward = direction_cosine * 0.5
        
        # Step penalty
        step_penalty = torch.full((batch_size,), -0.05 * step, device=device)
        
        # Overcorrection penalty
        overcorrection_penalty = torch.where(
            acc_restored < acc_faulted,
            torch.full((batch_size,), -0.5, device=device),
            torch.zeros(batch_size, device=device)
        )
        
        # Sparsity reward
        correction_magnitude = torch.abs(correction).mean(dim=1)
        sparsity_reward = -correction_magnitude * 0.05
        
        # Final bonus
        if is_final:
            final_bonus = torch.where(
                acc_restored > 0.9,
                torch.ones(batch_size, device=device) * 1.0,
                torch.zeros(batch_size, device=device)
            )
        else:
            final_bonus = torch.zeros(batch_size, device=device)
        
        total_reward = (
            acc_reward +
            mse_reward +
            conf_reward +
            direction_reward +
            step_penalty +
            overcorrection_penalty +
            sparsity_reward +
            final_bonus
        )
        
        return total_reward
    
    def update_statistics(self, logits_clean: torch.Tensor):
        """Update statistical preprocessing parameters"""
        self.logits_stats = {
            'mean': logits_clean.mean(dim=0),
            'std': logits_clean.std(dim=0),
        }

