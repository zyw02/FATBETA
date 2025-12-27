"""
RL-based Intermediate Layer Feature Restorer

Uses Reinforcement Learning to learn how to restore activations at intermediate layers.
This is more flexible than supervised learning as it can learn complex restoration strategies
that optimize for the final output quality rather than just matching clean activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .rl_restorer import ActorNetwork, CriticNetwork, compute_gae


class RLLayerActivationRestorer(nn.Module):
    """
    RL-based activation restorer for a single layer
    
    Uses Actor-Critic to learn restoration actions (corrections to activations).
    """
    def __init__(
        self,
        activation_shape: Tuple[int, ...],  # e.g., (C, H, W) for conv, (C,) for linear
        is_conv: bool = True,
        state_dim: int = 128,
        hidden_dim: int = 256,
        max_steps: int = 2,  # Multi-step correction for each layer
    ):
        super().__init__()
        self.activation_shape = activation_shape
        self.is_conv = is_conv
        self.max_steps = max_steps
        
        # Determine action dimension
        if is_conv:
            C, H, W = activation_shape
            # For conv, we can restore per-channel or per-spatial-location
            # Here we use per-channel restoration (simpler and more efficient)
            self.action_dim = C
            self.activation_size = C * H * W
        else:
            C = activation_shape[0]
            self.action_dim = C
            self.activation_size = C
        
        # State encoder: encodes current activation state
        # State = [flattened_activation_features, layer_embedding]
        feature_dim = self.activation_size
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Actor: outputs restoration action (correction per channel)
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )
        
        # Critic: estimates value of current state
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(
        self,
        faulted_activation: torch.Tensor,
        clean_activation: Optional[torch.Tensor] = None,
        training: bool = False,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Restore activation using RL policy
        
        Args:
            faulted_activation: [B, C, H, W] or [B, C] - faulted activation
            clean_activation: [B, C, H, W] or [B, C] - clean activation (for training)
            training: If True, collect trajectories
            deterministic: If True, use deterministic policy
        
        Returns:
            restored_activation: [B, C, H, W] or [B, C] - restored activation
            info: Dict with trajectories and other info
        """
        batch_size = faulted_activation.size(0)
        # Get device from model parameters (more reliable than input tensor device)
        model_device = next(self.state_encoder.parameters()).device
        # Ensure input is on the same device as the model
        faulted_activation = faulted_activation.to(model_device)
        if clean_activation is not None:
            clean_activation = clean_activation.to(model_device)
        device = model_device
        
        # Flatten activation for state encoding
        if self.is_conv:
            B, C, H, W = faulted_activation.shape
            activation_flat = faulted_activation.view(B, -1)  # [B, C*H*W]
        else:
            activation_flat = faulted_activation  # [B, C]
        
        # Store trajectories
        trajectories = {
            'activations': [],  # Store activations to recompute states during training
            'states': [],
            'actions': [],
            'log_probs': [],
            'entropies': [],
            'values': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        current_activation = faulted_activation.clone()
        
        # Multi-step correction
        for step in range(self.max_steps):
            # Store activation before encoding (for recomputing states during training)
            if training:
                trajectories['activations'].append(activation_flat.clone())
            
            # Encode state
            state = self.state_encoder(activation_flat)  # [B, state_dim]
            
            # Get action from policy
            action, log_prob, entropy = self.actor.sample(state, deterministic=deterministic)
            
            # Get value estimate
            value = self.critic(state)  # [B, 1]
            
            # Apply correction: action is per-channel correction
            # For conv: broadcast to spatial dimensions
            # For linear: directly apply
            if self.is_conv:
                # action: [B, C], need to broadcast to [B, C, H, W]
                correction = action.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                correction = correction.expand_as(current_activation)  # [B, C, H, W]
            else:
                correction = action  # [B, C]
            
            next_activation = current_activation + correction
            
            # Compute reward (if training)
            if training and clean_activation is not None:
                reward = self._compute_reward(
                    next_activation,
                    clean_activation,
                    faulted_activation,
                    step,
                    is_final=(step == self.max_steps - 1),
                )  # [B]
                
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:
                reward = torch.zeros(batch_size, device=device)
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # Store trajectory
            if training:
                # Store detached state for reference (we'll recompute from activations during training)
                trajectories['states'].append(state.detach())
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['entropies'].append(entropy)
                # Detach value to avoid keeping computation graph
                trajectories['values'].append(value.squeeze(-1).detach())
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                
                # Next state
                if step < self.max_steps - 1:
                    if self.is_conv:
                        next_flat = next_activation.view(B, -1)
                    else:
                        next_flat = next_activation
                    next_state = self.state_encoder(next_flat)
                else:
                    next_state = None
                trajectories['next_states'].append(next_state)
            
            # Update current activation
            current_activation = next_activation
            activation_flat = current_activation.view(B, -1) if self.is_conv else current_activation
        
        info = {
            'trajectories': trajectories,
            'final_reward': reward.mean().item() if training else 0.0,
        }
        
        return current_activation, info
    
    def _compute_reward(
        self,
        restored_activation: torch.Tensor,
        clean_activation: torch.Tensor,
        faulted_activation: torch.Tensor,
        step: int,
        is_final: bool,
    ) -> torch.Tensor:
        """
        Compute reward for restoration action
        
        Reward is based on:
        1. How close restored activation is to clean activation
        2. Improvement over faulted activation
        3. Sparsity (smaller corrections are better)
        """
        batch_size = restored_activation.size(0)
        device = restored_activation.device
        
        # 1. MSE reduction (main reward)
        mse_faulted = F.mse_loss(faulted_activation, clean_activation, reduction='none')
        mse_restored = F.mse_loss(restored_activation, clean_activation, reduction='none')
        
        if self.is_conv:
            mse_faulted = mse_faulted.mean(dim=[1, 2, 3])  # [B]
            mse_restored = mse_restored.mean(dim=[1, 2, 3])  # [B]
        else:
            mse_faulted = mse_faulted.mean(dim=1)  # [B]
            mse_restored = mse_restored.mean(dim=1)  # [B]
        
        mse_reduction = mse_faulted - mse_restored  # Positive if improved
        mse_reward = torch.clamp(mse_reduction * 10.0, -1.0, 1.0)  # Normalize
        
        # 2. Cosine similarity improvement
        if self.is_conv:
            restored_flat = restored_activation.view(batch_size, -1)
            clean_flat = clean_activation.view(batch_size, -1)
            faulted_flat = faulted_activation.view(batch_size, -1)
        else:
            restored_flat = restored_activation
            clean_flat = clean_activation
            faulted_flat = faulted_activation
        
        cos_sim_restored = F.cosine_similarity(restored_flat, clean_flat, dim=1)
        cos_sim_faulted = F.cosine_similarity(faulted_flat, clean_flat, dim=1)
        cos_improvement = cos_sim_restored - cos_sim_faulted
        cos_reward = cos_improvement * 0.5
        
        # 3. Sparsity reward (encourage small corrections)
        correction = restored_activation - faulted_activation
        if self.is_conv:
            correction_mag = torch.abs(correction).mean(dim=[1, 2, 3])  # [B]
        else:
            correction_mag = torch.abs(correction).mean(dim=1)  # [B]
        sparsity_reward = -correction_mag * 0.1
        
        # 4. Step penalty
        step_penalty = torch.full((batch_size,), -0.05 * step, device=device)
        
        # Total reward
        total_reward = (
            mse_reward * 2.0 +      # Main reward
            cos_reward +             # Direction reward
            sparsity_reward +        # Sparsity reward
            step_penalty             # Step penalty
        )
        
        return total_reward


class RLIntermediateLayerRestorer(nn.Module):
    """
    RL-based intermediate layer restorer
    
    Uses RL to learn restoration strategies for multiple layers.
    Each layer has its own RL agent (Actor-Critic).
    """
    def __init__(
        self,
        sensitive_info: Dict[str, Dict],
        model: nn.Module,
        state_dim: int = 128,
        hidden_dim: int = 256,
        max_steps_per_layer: int = 2,
    ):
        super().__init__()
        self.sensitive_info = sensitive_info
        self.model = model.module if hasattr(model, "module") else model
        self.max_steps_per_layer = max_steps_per_layer
        
        # Create RL restorer for each sensitive layer
        self.layer_restorers = nn.ModuleDict()
        self.activation_shapes = {}
        
        # Map from original layer names (may contain ".") to safe module names
        self.layer_name_to_module_name = {}
        self.module_name_to_layer_name = {}
        
        # Determine activation shapes
        self._determine_activation_shapes()
        
        # Create RL restorers
        for layer_name, info in sensitive_info.items():
            if layer_name not in self.activation_shapes:
                continue
            
            # Convert layer name to safe module name (replace "." with "_")
            module_name = layer_name.replace(".", "_")
            self.layer_name_to_module_name[layer_name] = module_name
            self.module_name_to_layer_name[module_name] = layer_name
            
            shape = self.activation_shapes[layer_name]
            is_conv = len(shape) == 3
            
            self.layer_restorers[module_name] = RLLayerActivationRestorer(
                activation_shape=shape,
                is_conv=is_conv,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                max_steps=max_steps_per_layer,
            )
        
        # Hooks for intercepting activations
        self.hooks = []
        self.clean_activations = {}
        self.restore_enabled = False
    
    def _determine_activation_shapes(self):
        """Determine activation shapes for each sensitive layer (using sensitive channels only)"""
        modules = dict(self.model.named_modules())
        temp_hooks = []
        temp_activations = {}
        
        def make_temp_hook(name):
            def hook(module, input, output):
                if name not in temp_activations:
                    if output.dim() == 4:
                        temp_activations[name] = output.shape[1:]
                    elif output.dim() == 2:
                        temp_activations[name] = output.shape[1:]
            return hook
        
        for name in self.sensitive_info.keys():
            if name in modules:
                hook = modules[name].register_forward_hook(make_temp_hook(name))
                temp_hooks.append(hook)
        
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Adjust shapes to use only sensitive channels
        self.activation_shapes = {}
        for name, full_shape in temp_activations.items():
            if name not in self.sensitive_info:
                continue
            
            idx = self.sensitive_info[name].get("indices", [])
            if not idx:
                continue
            
            num_sensitive_channels = len(idx)
            
            if len(full_shape) == 3:  # Conv: (C, H, W)
                C, H, W = full_shape
                # Use number of sensitive channels instead of full channels
                self.activation_shapes[name] = (num_sensitive_channels, H, W)
            elif len(full_shape) == 1:  # Linear: (C,)
                # Use number of sensitive channels instead of full channels
                self.activation_shapes[name] = (num_sensitive_channels,)
            else:
                # Fallback: use full shape
                self.activation_shapes[name] = full_shape
        
        for hook in temp_hooks:
            hook.remove()
    
    def register_hooks(self):
        """Register hooks to intercept and restore activations"""
        if self.hooks:
            return
        
        modules = dict(self.model.named_modules())
        
        def make_hook(name, restorer):
            def hook(module, input, output):
                if not self.restore_enabled:
                    return output
                
                # Get sensitive channel indices for this layer
                idx = self.sensitive_info.get(name, {}).get("indices", [])
                if not idx:
                    # No sensitive channels, return original output
                    return output
                
                # Select only sensitive channels
                if output.dim() == 4:  # Conv: [B, C, H, W]
                    faulted = output[:, idx, :, :].clone()  # [B, num_sensitive, H, W]
                elif output.dim() == 2:  # Linear: [B, C]
                    faulted = output[:, idx].clone()  # [B, num_sensitive]
                else:
                    return output
                
                # Get clean activation for sensitive channels only
                clean = None
                if name in self.clean_activations:
                    clean_act = self.clean_activations[name]
                    if clean_act.dim() == 4:  # Conv
                        clean = clean_act[:, idx, :, :]
                    elif clean_act.dim() == 2:  # Linear
                        clean = clean_act[:, idx]
                
                # Use RL to restore
                restored_sensitive, _ = restorer(
                    faulted,
                    clean,
                    training=False,  # Inference mode
                    deterministic=True,
                )
                
                # Replace sensitive channels in original output
                restored = output.clone()
                if output.dim() == 4:  # Conv
                    restored[:, idx, :, :] = restored_sensitive
                elif output.dim() == 2:  # Linear
                    restored[:, idx] = restored_sensitive
                
                if output.requires_grad:
                    restored = restored.requires_grad_(True)
                
                return restored
            
            return hook
        
        for module_name, restorer in self.layer_restorers.items():
            # Convert module name back to original layer name for hook registration
            layer_name = self.module_name_to_layer_name.get(module_name, module_name)
            if layer_name in modules:
                hook = modules[layer_name].register_forward_hook(make_hook(layer_name, restorer))
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
        self.clean_activations.clear()
    
    def forward(
        self,
        inputs: torch.Tensor,
        clean_activations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with RL-based intermediate layer restoration
        
        Args:
            inputs: Input tensor
            clean_activations: Clean activations for training (optional)
        
        Returns:
            output: Model output (logits)
            info: Additional information
        """
        if clean_activations is not None:
            self.set_clean_activations(clean_activations)
        
        self.enable_restoration()
        output = self.model(inputs)
        self.disable_restoration()
        
        return output, {}


class RLLayerRestorerTrainer:
    """
    Trainer for RL-based layer restorer
    
    Uses Actor-Critic algorithm (similar to RLRestorerTrainer).
    """
    def __init__(
        self,
        restorer: RLIntermediateLayerRestorer,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        use_ppo: bool = True,
        clip_epsilon: float = 0.2,
    ):
        self.restorer = restorer
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_ppo = use_ppo
        self.clip_epsilon = clip_epsilon
        
        # Separate optimizers for each layer's actor and critic
        self.layer_optimizers = {}
        for module_name, layer_restorer in restorer.layer_restorers.items():
            # Use module_name as key for optimizers (consistent with layer_restorers keys)
            self.layer_optimizers[module_name] = {
                'actor': torch.optim.Adam(
                    list(layer_restorer.actor.parameters()) + 
                    list(layer_restorer.state_encoder.parameters()),
                    lr=actor_lr,
                ),
                'critic': torch.optim.Adam(
                    layer_restorer.critic.parameters(),
                    lr=critic_lr,
                ),
            }
    
    def train_step(
        self,
        inputs: torch.Tensor,
        clean_activations: Dict[str, torch.Tensor],
        faulted_activations: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Train all layer restorers
        
        Args:
            inputs: Input tensor
            clean_activations: Clean activations for each layer
            faulted_activations: Faulted activations for each layer
        
        Returns:
            metrics: Training metrics
        """
        self.restorer.train()
        
        all_metrics = {}
        
        # Train each layer restorer independently
        for module_name, layer_restorer in self.restorer.layer_restorers.items():
            # Convert module name back to original layer name for accessing activations
            layer_name = self.restorer.module_name_to_layer_name.get(module_name, module_name)
            
            if layer_name not in faulted_activations:
                continue
            
            faulted = faulted_activations[layer_name]
            clean = clean_activations.get(layer_name)
            
            if clean is None:
                continue
            
            # Forward pass: collect trajectories
            restored, info = layer_restorer(
                faulted,
                clean,
                training=True,
                deterministic=False,
            )
            
            trajectories = info['trajectories']
            
            # Extract trajectories
            actions = torch.stack(trajectories['actions'], dim=0)  # [T, B, action_dim]
            old_log_probs = torch.stack(trajectories['log_probs'], dim=0)  # [T, B]
            entropies = torch.stack(trajectories['entropies'], dim=0)  # [T, B]
            values = torch.stack(trajectories['values'], dim=0)  # [T, B]
            rewards = torch.stack(trajectories['rewards'], dim=0)  # [T, B]
            dones = torch.stack(trajectories['dones'], dim=0)  # [T, B]
            next_states = trajectories['next_states']
            
            # Recompute states from activations (to get gradients through state_encoder)
            activations = trajectories['activations']
            states_list = []
            for act_flat in activations:
                state = layer_restorer.state_encoder(act_flat)  # [B, state_dim]
                states_list.append(state)
            states = torch.stack(states_list, dim=0)  # [T, B, state_dim]
            
            # Compute next values
            next_values = []
            for i, ns in enumerate(next_states):
                if ns is not None:
                    with torch.no_grad():
                        next_val = layer_restorer.critic(ns).squeeze(-1)  # [B]
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
            
            # Flatten
            T, B = advantages.shape
            states_flat = states.view(T * B, -1)
            actions_flat = actions.view(T * B, -1)
            old_log_probs_flat = old_log_probs.view(T * B)
            advantages_flat = advantages.view(T * B)
            returns_flat = returns.view(T * B)
            values_flat = values.view(T * B)
            entropies_flat = entropies.view(T * B)
            
            # Actor loss
            _, new_log_probs, new_entropies = layer_restorer.actor.sample(
                states_flat, deterministic=False
            )
            
            if self.use_ppo:
                ratio = torch.exp(new_log_probs - old_log_probs_flat)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                )
                actor_loss = -torch.min(
                    ratio * advantages_flat,
                    clipped_ratio * advantages_flat
                ).mean()
            else:
                actor_loss = -(new_log_probs * advantages_flat).mean()
            
            entropy_loss = -new_entropies.mean()
            total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
            
            # Critic loss
            # Use detached states for critic to avoid sharing computation graph with actor
            # Critic doesn't need gradients through state_encoder (only actor does)
            # returns_flat is already detached from compute_gae
            new_values = layer_restorer.critic(states_flat.detach()).squeeze(-1)
            critic_loss = F.mse_loss(new_values, returns_flat)
            
            # Update (use module_name for optimizers, as that's what we used as key)
            optimizers = self.layer_optimizers[module_name]
            
            # Actor update
            optimizers['actor'].zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(layer_restorer.actor.parameters()) + 
                list(layer_restorer.state_encoder.parameters()),
                self.max_grad_norm,
            )
            optimizers['actor'].step()
            
            # Critic update
            optimizers['critic'].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                layer_restorer.critic.parameters(),
                self.max_grad_norm,
            )
            optimizers['critic'].step()
            
            # Store metrics
            all_metrics[f'{layer_name}_actor_loss'] = actor_loss.item()
            all_metrics[f'{layer_name}_critic_loss'] = critic_loss.item()
            all_metrics[f'{layer_name}_entropy'] = new_entropies.mean().item()
            all_metrics[f'{layer_name}_mean_reward'] = rewards.mean().item()
        
        return all_metrics

