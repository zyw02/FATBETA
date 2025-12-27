"""
Reinforcement Learning based Restorer

This module implements a restorer that uses reinforcement learning to learn
how to correct faulted logits. It uses Actor-Critic architecture with
multi-step correction capability.

Supports both standard and enhanced feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np


class ActorNetwork(nn.Module):
    """
    Actor network: outputs correction delta (action)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and std
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [B, state_dim]
        
        Returns:
            mean: [B, action_dim] - mean of action distribution
            std: [B, action_dim] - std of action distribution
        """
        output = self.policy_net(state)
        mean = output[:, :self.action_dim]
        std = F.softplus(output[:, self.action_dim:]) + 1e-6  # Ensure positive
        
        return mean, std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: [B, state_dim]
            deterministic: If True, return mean action
        
        Returns:
            action: [B, action_dim]
            log_prob: [B] - log probability of action
            entropy: [B] - entropy of action distribution
        """
        mean, std = self.forward(state)
        
        if deterministic:
            action = mean
            # For deterministic, log_prob is not meaningful, return 0
            log_prob = torch.zeros(state.size(0), device=state.device)
            entropy = torch.zeros(state.size(0), device=state.device)
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)  # Sum over action dimensions
            entropy = dist.entropy().sum(dim=1)
        
        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Critic network: estimates state value
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, state_dim]
        
        Returns:
            value: [B, 1] - estimated state value
        """
        return self.value_net(state)


class StateEncoder(nn.Module):
    """
    Encodes state from logits and features
    """
    def __init__(self, feature_dim: int, num_classes: int, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Logits encoder
        self.logits_encoder = nn.Sequential(
            nn.Linear(num_classes, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # State combiner
        self.state_combiner = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
        )
    
    def forward(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] - current logits (faulted or partially corrected)
            features: [B, feature_dim] - features from sensitive layers
        
        Returns:
            state: [B, state_dim] - encoded state
        """
        # Encode features
        feat_embed = self.feature_encoder(features)  # [B, hidden_dim // 2]
        
        # Encode logits
        logits_embed = self.logits_encoder(logits)  # [B, hidden_dim // 2]
        
        # Combine
        combined = torch.cat([feat_embed, logits_embed, logits], dim=1)  # [B, hidden_dim + num_classes]
        state = self.state_combiner(combined)  # [B, state_dim]
        
        return state


class RLRestorer(nn.Module):
    """
    Reinforcement Learning based Restorer
    
    Uses Actor-Critic architecture to learn how to correct faulted logits.
    Supports multi-step correction.
    
    Supports both standard and enhanced feature extraction:
    - Standard: Uses simple features (energy, mean, std, max)
    - Enhanced: Uses rich features (distribution, spatial, frequency, relative, gradient)
    """
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        state_dim: int = 128,
        action_dim: Optional[int] = None,
        hidden_dim: int = 256,
        max_steps: int = 3,
        gamma: float = 0.99,
        use_statistical_preprocessing: bool = True,
        use_enhanced_features: bool = False,
    ):
        super().__init__()
        
        if action_dim is None:
            action_dim = num_classes
        
        self.num_classes = num_classes
        self.max_steps = max_steps
        self.gamma = gamma
        self.use_statistical_preprocessing = use_statistical_preprocessing
        self.use_enhanced_features = use_enhanced_features
        
        # State encoder
        # For enhanced features, feature_dim will be much larger
        self.state_encoder = StateEncoder(feature_dim, num_classes, state_dim, hidden_dim)
        
        # Actor network
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        
        # Critic network
        self.critic = CriticNetwork(state_dim, hidden_dim)
        
        # Statistical preprocessing (optional)
        if use_statistical_preprocessing:
            self.logits_stats = None  # Will be set during training
    
    def forward(
        self,
        logits_faulted: torch.Tensor,
        features: torch.Tensor,
        logits_clean: Optional[torch.Tensor] = None,
        training: bool = False,
        deterministic: bool = False,
        ber_level: Optional[float] = None,  # 新增：当前 BER 级别
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Restore faulted logits using RL policy
        
        Args:
            logits_faulted: [B, num_classes] - faulted logits
            features: [B, feature_dim] - features from sensitive layers
            logits_clean: [B, num_classes] - clean logits (for training only)
            training: If True, collect trajectories for training
            deterministic: If True, use deterministic policy (mean action)
        
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
            # Encode state
            state = self.state_encoder(logits_current, features)  # [B, state_dim]
            
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
                    ber_level=ber_level  # 传递 BER 级别
                )  # [B]
                
                # Check if correction is good enough (early stopping)
                if not deterministic:
                    # For training, always do max_steps to collect full trajectories
                    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
                else:
                    # For inference, early stop if good enough
                    improvement = self._compute_improvement(logits_next, logits_clean, logits_faulted)
                    done = improvement > 0.95  # 95% improvement threshold
            else:
                reward = torch.zeros(batch_size, device=device)
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Store trajectory
            if training:
                # Store states and values with computation graph for actor training
                # (will be detached in trainer if needed for critic)
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['entropies'].append(entropy)
                # Detach values to avoid keeping computation graph (values are only used for GAE)
                trajectories['values'].append(value.squeeze(-1).detach())  # [B]
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                
                # Next state (for value estimation)
                if step < self.max_steps - 1:
                    next_state = self.state_encoder(logits_next, features)
                    trajectories['next_states'].append(next_state)
                else:
                    trajectories['next_states'].append(None)
            
            # Update current logits
            logits_current = logits_next
            
            # Early stopping (inference only)
            if not training and deterministic:
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
        """
        Statistical preprocessing: smooth and correct obvious anomalies
        """
        if self.logits_stats is None:
            return logits
        
        # Smooth logits (1D average pooling)
        logits_smoothed = F.avg_pool1d(
            logits.unsqueeze(1), kernel_size=3, padding=1, stride=1
        ).squeeze(1)
        
        # Correct extreme outliers (3-sigma rule)
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
        ber_level: Optional[float] = None,  # 新增：当前 BER 级别
    ) -> torch.Tensor:
        """
        Compute reward for RL training (改进版，借鉴对抗防御思路)
        
        Args:
            logits_restored: [B, num_classes] - current restored logits
            logits_clean: [B, num_classes] - target clean logits
            logits_faulted: [B, num_classes] - original faulted logits
            step: current correction step
            is_final: whether this is the final step
            ber_level: current BER level (for adaptive reward weighting)
        
        Returns:
            reward: [B] - reward for each sample
        """
        batch_size = logits_restored.size(0)
        device = logits_restored.device
        
        # 1. Accuracy improvement (main reward) - 根据 BER 加权
        pred_restored = logits_restored.argmax(dim=1)
        pred_clean = logits_clean.argmax(dim=1)
        pred_faulted = logits_faulted.argmax(dim=1)
        
        acc_restored = (pred_restored == pred_clean).float()
        acc_faulted = (pred_faulted == pred_clean).float()
        acc_improvement = acc_restored - acc_faulted  # [-1, 1]
        
        # BER 加权：BER 越高，准确率提升的奖励越大
        if ber_level is not None:
            ber_weight = 1.0 + ber_level * 10.0  # BER 越高，权重越大
        else:
            ber_weight = 1.0
        acc_reward = acc_improvement * 2.0 * ber_weight
        
        # 2. Logits similarity (dense reward)
        mse_faulted = F.mse_loss(logits_faulted, logits_clean, reduction='none').mean(dim=1)
        mse_restored = F.mse_loss(logits_restored, logits_clean, reduction='none').mean(dim=1)
        mse_reduction = mse_faulted - mse_restored  # Positive if improved
        mse_reward = torch.clamp(mse_reduction / 10.0, -1.0, 1.0)  # Normalize
        
        # 3. Confidence improvement (借鉴对抗防御中的置信度提升)
        # 高置信度意味着模型更确定，这在对抗防御中很重要
        conf_clean = F.softmax(logits_clean, dim=1).max(dim=1)[0]  # [B]
        conf_restored = F.softmax(logits_restored, dim=1).max(dim=1)[0]  # [B]
        conf_faulted = F.softmax(logits_faulted, dim=1).max(dim=1)[0]  # [B]
        conf_improvement = conf_restored - conf_faulted  # 恢复后置信度提升
        conf_reward = conf_improvement * 0.5  # 置信度奖励
        
        # 4. Correction direction correctness
        correction = logits_restored - logits_faulted
        target_correction = logits_clean - logits_faulted
        direction_cosine = F.cosine_similarity(correction, target_correction, dim=1)
        direction_reward = direction_cosine * 0.5  # [0, 0.5]
        
        # 5. Step penalty (encourage fewer steps)
        step_penalty = torch.full((batch_size,), -0.05 * step, device=device)
        
        # 6. Overcorrection penalty (如果恢复后比故障前还差)
        overcorrection_penalty = torch.where(
            acc_restored < acc_faulted,
            torch.full((batch_size,), -0.5, device=device),
            torch.zeros(batch_size, device=device)
        )
        
        # 7. Sparsity reward (鼓励最小修正，借鉴对抗防御中的稀疏性)
        # 修正幅度应该尽可能小，只修正必要的部分
        correction_magnitude = torch.abs(correction).mean(dim=1)  # [B]
        sparsity_reward = -correction_magnitude * 0.05  # 修正越小，奖励越大
        
        # 8. Final bonus (if reached target)
        if is_final:
            # 最终奖励：如果恢复后准确率很高，给予额外奖励
            final_bonus = torch.where(
                acc_restored > 0.9,
                torch.ones(batch_size, device=device) * 1.0,
                torch.zeros(batch_size, device=device)
            )
        else:
            final_bonus = torch.zeros(batch_size, device=device)
        
        # Total reward (改进版)
        total_reward = (
            acc_reward +                      # 准确率提升（BER 加权）
            mse_reward +                      # MSE 减少
            conf_reward +                     # 置信度提升（新增）
            direction_reward +                # 方向正确性
            step_penalty +                    # 步数惩罚
            overcorrection_penalty +          # 过度修正惩罚
            sparsity_reward +                 # 稀疏性奖励（新增）
            final_bonus                       # 最终奖励
        )
        
        return total_reward
    
    def _compute_improvement(
        self,
        logits_restored: torch.Tensor,
        logits_clean: torch.Tensor,
        logits_faulted: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute improvement ratio for early stopping
        """
        mse_faulted = F.mse_loss(logits_faulted, logits_clean, reduction='none').mean(dim=1)
        mse_restored = F.mse_loss(logits_restored, logits_clean, reduction='none').mean(dim=1)
        
        # Avoid division by zero
        improvement = 1.0 - (mse_restored / (mse_faulted + 1e-6))
        return improvement.clamp(0.0, 1.0)
    
    def update_statistics(self, logits_clean: torch.Tensor):
        """
        Update statistical preprocessing parameters
        
        Args:
            logits_clean: [N, num_classes] - clean logits for statistics
        """
        self.logits_stats = {
            'mean': logits_clean.mean(dim=0),
            'std': logits_clean.std(dim=0),
        }


def compute_gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    next_values: List[Optional[torch.Tensor]],
    dones: List[torch.Tensor],
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: List of [B] reward tensors
        values: List of [B] value tensors
        next_values: List of [B] or None for next state values
        dones: List of [B] done flags
        gamma: discount factor
        lambda_: GAE lambda parameter
    
    Returns:
        advantages: [T, B] - computed advantages
        returns: [T, B] - computed returns
    """
    T = len(rewards)
    B = rewards[0].size(0)
    device = rewards[0].device
    
    advantages = torch.zeros(T, B, device=device)
    returns = torch.zeros(T, B, device=device)
    
    # Compute next values (use 0 for terminal states)
    next_vals = []
    for i, nv in enumerate(next_values):
        if nv is not None:
            next_vals.append(nv)
        else:
            # Terminal state, value is 0
            next_vals.append(torch.zeros(B, device=device))
    
    # Compute GAE backwards
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            # Last step: next value is 0 (terminal state)
            next_value = torch.zeros(B, device=rewards[0].device)
        else:
            next_value = next_vals[t]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t].float()) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns

