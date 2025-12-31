"""
Lightweight RL Restorer - Reduced parameter version

This version uses smaller networks and fewer parameters while maintaining
the core RL functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from .rl_restorer import compute_gae


class LightweightActorNetwork(nn.Module):
    """
    Lightweight Actor network with fewer parameters
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        
        # Smaller policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and std
        )
        
        # Initialize weights
        nn.init.orthogonal_(self.policy_net[0].weight, gain=0.01)
        nn.init.constant_(self.policy_net[0].bias, 0.0)
        nn.init.orthogonal_(self.policy_net[2].weight, gain=0.01)
        nn.init.constant_(self.policy_net[2].bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.policy_net(state)
        mean = output[:, :self.action_dim]
        std = F.softplus(output[:, self.action_dim:]) + 1e-6
        return mean, std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(state.size(0), device=state.device)
            entropy = torch.zeros(state.size(0), device=state.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
        
        return action, log_prob, entropy


class LightweightCriticNetwork(nn.Module):
    """
    Lightweight Critic network with fewer parameters
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize weights
        nn.init.orthogonal_(self.value_net[0].weight, gain=1.0)
        nn.init.constant_(self.value_net[0].bias, 0.0)
        nn.init.orthogonal_(self.value_net[2].weight, gain=1.0)
        nn.init.constant_(self.value_net[2].bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_net(state)


class LightweightStateEncoder(nn.Module):
    """
    Lightweight state encoder with fewer parameters
    """
    def __init__(self, feature_dim: int, num_classes: int, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Simple feature encoder
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        
        # Simple logits encoder
        self.logits_encoder = nn.Linear(num_classes, hidden_dim)
        
        # State combiner (directly to state_dim)
        self.state_combiner = nn.Linear(hidden_dim * 2 + num_classes, state_dim)
    
    def forward(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        feat_embed = F.relu(self.feature_encoder(features))
        logits_embed = F.relu(self.logits_encoder(logits))
        combined = torch.cat([feat_embed, logits_embed, logits], dim=1)
        state = F.relu(self.state_combiner(combined))
        return state


class LightweightRLRestorer(nn.Module):
    """
    Lightweight RL Restorer with reduced parameters
    
    Uses smaller networks to reduce parameter count while maintaining
    core RL functionality.
    """
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        state_dim: int = 64,  # Reduced from 128
        action_dim: Optional[int] = None,
        hidden_dim: int = 64,  # Reduced from 256
        max_steps: int = 3,
        gamma: float = 0.99,
        use_statistical_preprocessing: bool = True,
    ):
        super().__init__()
        
        if action_dim is None:
            action_dim = num_classes
        
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.gamma = gamma
        self.use_statistical_preprocessing = use_statistical_preprocessing
        
        # Lightweight state encoder
        self.state_encoder = LightweightStateEncoder(feature_dim, num_classes, state_dim, hidden_dim)
        
        # Lightweight actor
        self.actor = LightweightActorNetwork(state_dim, action_dim, hidden_dim)
        
        # Lightweight critic
        self.critic = LightweightCriticNetwork(state_dim, hidden_dim)
        
        # Statistical preprocessing (optional)
        if use_statistical_preprocessing:
            self.logits_stats = None
    
    def forward(
        self,
        logits_faulted: torch.Tensor,
        features: torch.Tensor,
        logits_clean: Optional[torch.Tensor] = None,
        training: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """Same interface as RLRestorer"""
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
            # Encode state
            state = self.state_encoder(logits_current, features)
            
            # Get action from policy
            action, log_prob, entropy = self.actor.sample(state, deterministic=deterministic)
            
            # Get value estimate
            value = self.critic(state).squeeze(-1)
            
            # Apply correction
            logits_next = logits_current + action
            
            # Compute reward (if training)
            if training and logits_clean is not None:
                reward = self._compute_reward(
                    logits_next, logits_clean, logits_faulted, step, 
                    is_final=(step == self.max_steps - 1)
                )
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                reward = torch.zeros(batch_size, device=device)
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Store trajectory
            if training:
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['entropies'].append(entropy)
                trajectories['values'].append(value)
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                
                # Next state
                if step < self.max_steps - 1:
                    next_state = self.state_encoder(logits_next, features)
                    trajectories['next_states'].append(next_state)
                else:
                    trajectories['next_states'].append(None)
            
            # Update current logits
            logits_current = logits_next
            
            # Early stopping (inference only)
            if not training and deterministic:
                improvement = self._compute_improvement(logits_next, logits_clean, logits_faulted)
                done = improvement > 0.95
                if done.all():
                    break
        
        # Prepare return info
        info = {}
        if training:
            info['trajectories'] = trajectories
            info['final_reward'] = trajectories['rewards'][-1].mean().item() if trajectories['rewards'] else 0.0
            info['num_steps'] = step + 1
        
        return logits_current, info
    
    def _statistical_preprocess(self, logits: torch.Tensor) -> torch.Tensor:
        """Statistical preprocessing"""
        if self.logits_stats is None:
            return logits
        
        logits_smoothed = F.avg_pool1d(
            logits.unsqueeze(1), kernel_size=3, padding=1, stride=1
        ).squeeze(1)
        
        mean = self.logits_stats['mean'].to(logits.device)
        std = self.logits_stats['std'].to(logits.device)
        z_scores = (logits_smoothed - mean) / (std + 1e-6)
        outliers = torch.abs(z_scores) > 3.0
        
        logits_corrected = logits_smoothed.clone()
        logits_corrected[outliers] = mean.expand_as(logits_corrected)[outliers]
        
        return logits_corrected
    
    def _compute_reward(
        self,
        logits_restored: torch.Tensor,
        logits_clean: torch.Tensor,
        logits_faulted: torch.Tensor,
        step: int,
        is_final: bool,
    ) -> torch.Tensor:
        """Same reward function as RLRestorer"""
        batch_size = logits_restored.size(0)
        device = logits_restored.device
        
        pred_restored = logits_restored.argmax(dim=1)
        pred_clean = logits_clean.argmax(dim=1)
        pred_faulted = logits_faulted.argmax(dim=1)
        
        acc_restored = (pred_restored == pred_clean).float()
        acc_faulted = (pred_faulted == pred_clean).float()
        acc_improvement = acc_restored - acc_faulted
        
        mse_faulted = F.mse_loss(logits_faulted, logits_clean, reduction='none').mean(dim=1)
        mse_restored = F.mse_loss(logits_restored, logits_clean, reduction='none').mean(dim=1)
        mse_reduction = mse_faulted - mse_restored
        mse_reward = torch.clamp(mse_reduction / 10.0, -1.0, 1.0)
        
        correction = logits_restored - logits_faulted
        target_correction = logits_clean - logits_faulted
        direction_cosine = F.cosine_similarity(correction, target_correction, dim=1)
        direction_reward = direction_cosine * 0.5
        
        step_penalty = torch.full((batch_size,), -0.05 * step, device=device)
        
        overcorrection_penalty = torch.where(
            acc_restored < acc_faulted,
            torch.full((batch_size,), -0.5, device=device),
            torch.zeros(batch_size, device=device)
        )
        
        if is_final:
            final_bonus = torch.where(
                acc_restored > 0.9,
                torch.ones(batch_size, device=device),
                torch.zeros(batch_size, device=device)
            )
        else:
            final_bonus = torch.zeros(batch_size, device=device)
        
        total_reward = (
            acc_improvement * 2.0 +
            mse_reward +
            direction_reward +
            step_penalty +
            overcorrection_penalty +
            final_bonus
        )
        
        return total_reward
    
    def _compute_improvement(
        self,
        logits_restored: torch.Tensor,
        logits_clean: torch.Tensor,
        logits_faulted: torch.Tensor,
    ) -> torch.Tensor:
        """Compute improvement ratio"""
        mse_faulted = F.mse_loss(logits_faulted, logits_clean, reduction='none').mean(dim=1)
        mse_restored = F.mse_loss(logits_restored, logits_clean, reduction='none').mean(dim=1)
        improvement = 1.0 - (mse_restored / (mse_faulted + 1e-6))
        return improvement.clamp(0.0, 1.0)
    
    def update_statistics(self, logits_clean: torch.Tensor):
        """Update statistical preprocessing parameters"""
        self.logits_stats = {
            'mean': logits_clean.mean(dim=0),
            'std': logits_clean.std(dim=0),
        }


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_models(feature_dim: int = 512, num_classes: int = 10):
    """Compare parameter counts between standard and lightweight versions"""
    from .rl_restorer import RLRestorer
    
    # Standard version
    standard = RLRestorer(
        feature_dim=feature_dim,
        num_classes=num_classes,
        state_dim=128,
        hidden_dim=256,
    )
    
    # Lightweight version
    lightweight = LightweightRLRestorer(
        feature_dim=feature_dim,
        num_classes=num_classes,
        state_dim=64,
        hidden_dim=64,
    )
    
    standard_params = count_parameters(standard)
    lightweight_params = count_parameters(lightweight)
    
    print(f"Feature Dim: {feature_dim}, Num Classes: {num_classes}")
    print(f"Standard RL Restorer: {standard_params:,} parameters")
    print(f"Lightweight RL Restorer: {lightweight_params:,} parameters")
    print(f"Reduction: {standard_params - lightweight_params:,} parameters ({100 * (1 - lightweight_params/standard_params):.1f}% reduction)")
    
    return standard_params, lightweight_params


