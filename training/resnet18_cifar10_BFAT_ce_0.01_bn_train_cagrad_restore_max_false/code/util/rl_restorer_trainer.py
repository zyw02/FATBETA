"""
Training utilities for RL Restorer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from .rl_restorer import RLRestorer, compute_gae


class RLRestorerTrainer:
    """
    Trainer for RL Restorer using Actor-Critic algorithm
    """
    def __init__(
        self,
        restorer: RLRestorer,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        clip_epsilon: float = 0.2,  # For PPO-style clipping
        use_ppo: bool = True,
    ):
        self.restorer = restorer
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.use_ppo = use_ppo
        
        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(
            list(restorer.actor.parameters()) + list(restorer.state_encoder.parameters()),
            lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            restorer.critic.parameters(),
            lr=critic_lr
        )
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=100, eta_min=1e-6
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=100, eta_min=1e-6
        )
    
    def train_step(
        self,
        logits_faulted: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        logits_clean: torch.Tensor = None,
        ber_level: Optional[float] = None,  # 新增：当前 BER 级别
        layer_features: Optional[List[torch.Tensor]] = None,  # For LayerwiseRLRestorer
    ) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            logits_faulted: [B, num_classes] - faulted logits
            features: [B, feature_dim] - features (for standard RLRestorer)
            logits_clean: [B, num_classes] - clean logits (target)
            ber_level: Current BER level (for adaptive reward weighting)
            layer_features: List of [B, feat_dim_i] - layer-wise features (for LayerwiseRLRestorer)
        
        Returns:
            metrics: Dict of training metrics
        """
        self.restorer.train()
        
        # Forward pass: collect trajectories
        # Check if restorer is LayerwiseRLRestorer (has layer_features parameter)
        if layer_features is not None:
            logits_restored, info = self.restorer(
                logits_faulted,
                layer_features,
                logits_clean,
                training=True,
                deterministic=False,
                ber_level=ber_level,
            )
        else:
            logits_restored, info = self.restorer(
                logits_faulted,
                features,
                logits_clean,
                training=True,
                deterministic=False,
                ber_level=ber_level,  # 传递 BER 级别
            )
        
        trajectories = info['trajectories']
        
        # Extract trajectories
        states = torch.stack(trajectories['states'], dim=0)  # [T, B, state_dim]
        actions = torch.stack(trajectories['actions'], dim=0)  # [T, B, action_dim]
        old_log_probs = torch.stack(trajectories['log_probs'], dim=0)  # [T, B]
        entropies = torch.stack(trajectories['entropies'], dim=0)  # [T, B]
        values = torch.stack(trajectories['values'], dim=0)  # [T, B]
        rewards = torch.stack(trajectories['rewards'], dim=0)  # [T, B]
        dones = torch.stack(trajectories['dones'], dim=0)  # [T, B]
        next_states = trajectories['next_states']
        
        # Compute next values
        # Detach next_states to avoid computation graph issues
        next_values = []
        for i, ns in enumerate(next_states):
            if ns is not None:
                with torch.no_grad():
                    # Use detached next_state for value estimation
                    next_val = self.restorer.critic(ns.detach()).squeeze(-1)  # [B]
            else:
                next_val = None
            next_values.append(next_val)
        
        # Compute GAE
        advantages, returns = compute_gae(
            [rewards[t] for t in range(len(rewards))],
            [values[t] for t in range(len(values))],
            next_values,
            [dones[t] for t in range(len(dones))],
            self.gamma,
            self.lambda_,
        )
        
        # Detach advantages and returns to avoid computation graph issues
        advantages = advantages.detach()
        returns = returns.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten for processing
        T, B = advantages.shape
        states_flat = states.view(T * B, -1)  # [T*B, state_dim]
        actions_flat = actions.view(T * B, -1)  # [T*B, action_dim]
        old_log_probs_flat = old_log_probs.view(T * B)  # [T*B]
        advantages_flat = advantages.view(T * B)  # [T*B]
        returns_flat = returns.view(T * B)  # [T*B]
        values_flat = values.view(T * B)  # [T*B]
        entropies_flat = entropies.view(T * B)  # [T*B]
        
        # Actor loss
        _, new_log_probs, new_entropies = self.restorer.actor.sample(states_flat, deterministic=False)
        
        if self.use_ppo:
            # PPO-style clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs_flat)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            actor_loss = -torch.min(
                ratio * advantages_flat,
                clipped_ratio * advantages_flat
            ).mean()
        else:
            # Standard policy gradient
            actor_loss = -(new_log_probs * advantages_flat).mean()
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -new_entropies.mean()
        
        # Total actor loss
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
        
        # Critic loss
        # Use detached states for critic to avoid sharing computation graph with actor
        new_values = self.restorer.critic(states_flat.detach()).squeeze(-1)  # [T*B]
        critic_loss = nn.functional.mse_loss(new_values, returns_flat)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.restorer.actor.parameters()) + list(self.restorer.state_encoder.parameters()),
            self.max_grad_norm
        )
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.restorer.critic.parameters(),
            self.max_grad_norm
        )
        self.critic_optimizer.step()
        
        # Compute metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': new_entropies.mean().item(),
            'total_actor_loss': total_actor_loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages_flat.mean().item(),
            'mean_value': new_values.mean().item(),
            'final_reward': info['final_reward'],
        }
        
        return metrics
    
    def update_schedulers(self):
        """Update learning rate schedulers"""
        self.actor_scheduler.step()
        self.critic_scheduler.step()
    
    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates"""
        return {
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],
        }

