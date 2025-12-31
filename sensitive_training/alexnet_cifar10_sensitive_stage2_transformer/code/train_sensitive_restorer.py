import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to Python path to ensure util module can be imported
# This needs to be done before any util imports
_script_file = Path(__file__)
_project_root = _script_file.parent.parent.parent.parent  # Go up to project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model import create_model
from util import get_config, init_logger, set_global_seed, preprocess_model, init_dataloader
from util import ProgressMonitor, TensorBoardMonitor
from util.dist import logger_info, is_master
from util.utils import copy_code, create_optimizer_and_lr_scheduler
from util.model_ema import ModelEma
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint, save_checkpoint
from util.sensitive_features import collect_gradient_sensitivity, compute_activation_baseline
from util.sensitive_restorer import SensitiveActivationCollector, SensitiveChannelRestorer
from util.sensitive_restorer_transformer import LayerwiseRestorer
from util.fault_injector import FaultInjector
from util.progressive_ber_scheduler import ProgressiveBERScheduler

# Import RL Restorer components
try:
    from util.rl_restorer_integration import (
        create_rl_restorer_with_features,
        extract_features_for_rl_restorer,
    )
    from util.rl_restorer_trainer import RLRestorerTrainer
    RL_RESTORER_AVAILABLE = True
except ImportError:
    RL_RESTORER_AVAILABLE = False
    logger_info(logging.getLogger(), "RL Restorer not available, using standard restorer")

# Import RL Intermediate Layer Restorer components
try:
    from util.rl_intermediate_restorer_integration import (
        create_rl_intermediate_layer_restorer,
        train_hybrid_rl_restorer_step,
    )
    from util.rl_intermediate_layer_restorer import RLLayerRestorerTrainer
    from util.intermediate_restorer_integration import collect_activations_for_training
    RL_INTERMEDIATE_AVAILABLE = True
except ImportError:
    RL_INTERMEDIATE_AVAILABLE = False
    logger_info(logging.getLogger(), "RL Intermediate Layer Restorer not available")


class SchedulerConfig:
    """A simple config class for timm scheduler creation."""
    def __init__(self, sched=None, epochs=None, min_lr=None, warmup_lr=None, warmup_epochs=None, 
                 cooldown_epochs=None, decay_rate=None, decay_epochs=None, configs=None, stage2_epochs=None):
        # Support two initialization modes:
        # 1. Direct parameters (used in main function)
        # 2. From configs object (used in train_restorer function)
        if configs is not None:
            # Mode 2: Initialize from configs object
            self.sched = getattr(configs, 'sched', 'cosine')
            self.epochs = stage2_epochs
            self.min_lr = getattr(configs, 'min_lr', 0.0)
            self.warmup_epochs = getattr(configs, 'warmup_epochs', 0)
            self.warmup_lr = getattr(configs, 'warmup_lr', 0.0)
            self.decay_rate = getattr(configs, 'decay_rate', 0.1)
            self.decay_epochs = getattr(configs, 'decay_epochs', 30)
            self.cooldown_epochs = getattr(configs, 'cooldown_epochs', 0)
        else:
            # Mode 1: Direct parameters
            self.sched = sched
            self.epochs = epochs
            self.min_lr = min_lr
            self.warmup_lr = warmup_lr
            self.warmup_epochs = warmup_epochs
            self.cooldown_epochs = cooldown_epochs
            self.decay_rate = decay_rate
            self.decay_epochs = decay_epochs
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 training for sensitive channel restorer")
    parser.add_argument("--config", required=True, help="Path to stage2 config yaml")
    parser.add_argument("--stage1_ckpt", required=True, help="Checkpoint from stage1 training")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    return parser.parse_args()


def init_logging(configs, script_dir, output_dir_override=None):
    output_dir = script_dir / (output_dir_override or configs.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Try to find logging.conf, fallback to None if not found
    log_conf_file = script_dir / "logging.conf"
    if not log_conf_file.exists():
        # Try parent directory
        log_conf_file = script_dir.parent / "logging.conf"
        if not log_conf_file.exists():
            # Try root directory
            log_conf_file = Path(__file__).parent.parent.parent.parent / "logging.conf"
            if not log_conf_file.exists():
                log_conf_file = None
    log_dir = init_logger(configs.name, output_dir, log_conf_file if log_conf_file and log_conf_file.exists() else None)
    logger = logging.getLogger()
    pymonitor = ProgressMonitor(logger)
    tbmonitor = TensorBoardMonitor(logger, log_dir)
    return logger, log_dir, pymonitor, tbmonitor


def prepare_stats_for_device(raw_stats, device):
    """准备统计信息到设备，支持新的多特征格式和向后兼容"""
    prepared = {}
    feature_dim = 0
    for name, entry in raw_stats.items():
        indices = entry.get("indices", [])
        if not indices:
            continue
        
        # Check if new format (multiple statistics) or old format (only energy)
        if "energy_mean" in entry:
            # New format: multiple statistics
            prepared[name] = {
                "indices": indices,
                "energy_mean": entry["energy_mean"].to(device),
                "energy_std": entry["energy_std"].to(device),
                "mean_mean": entry["mean_mean"].to(device),
                "mean_std": entry["mean_std"].to(device),
                "std_mean": entry["std_mean"].to(device),
                "std_std": entry["std_std"].to(device),
                "max_mean": entry["max_mean"].to(device),
                "max_std": entry["max_std"].to(device),
            }
            # Each channel contributes 4 features: energy, mean, std, max
            feature_dim += len(indices) * 4
        else:
            # Old format: only energy (backward compatibility)
            prepared[name] = {
                "indices": indices,
                "mean": entry["mean"].to(device),
                "std": entry["std"].to(device),
            }
            feature_dim += len(indices)
    return prepared, feature_dim


def train_restorer(model, train_loader, criterion, prepared_stats, feature_dim, device, configs, baseline_stats=None):
    training_model = model.module if hasattr(model, "module") else model
    # Set the main model to evaluation mode. This is crucial for disabling dropout, etc.
    training_model.eval()
    
    # CRITICAL FIX: DO NOT set requires_grad=False on the entire model.
    # This breaks the computation graph for fault injection and feature collection.
    # The optimizer will only update the restorer's parameters, effectively
    # freezing the main model's weights.
    # for param in training_model.parameters():
    #     param.requires_grad_(False)

    # Architecture selection: RL Restorer, Transformer, or MLP
    use_rl_restorer = configs.sensitive_restorer.get("use_rl_restorer", False)
    use_enhanced_features = configs.sensitive_restorer.get("use_enhanced_features", False)
    use_transformer = configs.sensitive_restorer.get("use_transformer", False)
    
    if use_rl_restorer and RL_RESTORER_AVAILABLE:
        # RL Restorer with enhanced features
        logger_info(logging.getLogger(), f"[Stage2] Using RL Restorer with enhanced_features={use_enhanced_features}")
        
        from util.rl_restorer_integration import create_rl_restorer_with_features
        
        use_layerwise = configs.sensitive_restorer.get("use_layerwise", True)
        restorer, collector = create_rl_restorer_with_features(
            model=training_model,
            sensitive_info=prepared_stats,
            baseline_stats=prepared_stats,
            num_classes=configs.dataloader.num_classes,
            use_enhanced_features=use_enhanced_features,
            use_layerwise=use_layerwise,  # 使用层级感知版本
            feature_dim=None,  # Auto-detect
            state_dim=configs.sensitive_restorer.get("rl_state_dim", 128),
            hidden_dim=configs.sensitive_restorer.get("rl_hidden_dim", 256),
            max_steps=configs.sensitive_restorer.get("rl_max_steps", 3),
            device=device,
        )
        
        total_params = sum(p.numel() for p in restorer.parameters())
        
        # Get feature dimension info based on encoder type
        if hasattr(restorer.state_encoder, 'layer_encoders'):
            # LayerwiseStateEncoder
            feature_dims = [enc[0].in_features for enc in restorer.state_encoder.layer_encoders]
            feature_dim_str = f"Layerwise: {feature_dims}"
        elif hasattr(restorer.state_encoder, 'feature_encoder'):
            # Standard StateEncoder
            feature_dim_str = f"{restorer.state_encoder.feature_encoder[0].in_features}"
        else:
            feature_dim_str = "Unknown"
        
        logger_info(logging.getLogger(), 
                   f"[RL Restorer] Feature dimension: {feature_dim_str}")
        logger_info(logging.getLogger(), 
                   f"[RL Restorer] Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Create RL trainer
        from util.rl_restorer_trainer import RLRestorerTrainer
        rl_trainer = RLRestorerTrainer(
            restorer=restorer,
            actor_lr=float(configs.sensitive_restorer.get("rl_actor_lr", 3e-4)),
            critic_lr=float(configs.sensitive_restorer.get("rl_critic_lr", 3e-4)),
            gamma=float(configs.sensitive_restorer.get("rl_gamma", 0.99)),
            lambda_=float(configs.sensitive_restorer.get("rl_lambda", 0.95)),
            entropy_coef=float(configs.sensitive_restorer.get("rl_entropy_coef", 0.01)),
            value_coef=float(configs.sensitive_restorer.get("rl_value_coef", 0.5)),
            max_grad_norm=float(configs.sensitive_restorer.get("max_grad_norm", 1.0)),
            use_ppo=configs.sensitive_restorer.get("rl_use_ppo", True),
        )
        
        # Update statistics for statistical preprocessing (optional)
        if restorer.use_statistical_preprocessing:
            logger_info(logging.getLogger(), "[RL Restorer] Collecting statistics for preprocessing...")
            logits_clean_list = []
            training_model.eval()
            with torch.no_grad():
                for inputs, _ in train_loader:
                    inputs = inputs.to(device)
                    logits_clean = training_model(inputs)
                    logits_clean_list.append(logits_clean)
                    if len(logits_clean_list) * inputs.size(0) > 1000:  # Collect 1000 samples
                        break
            logits_clean_all = torch.cat(logits_clean_list, dim=0)
            restorer.update_statistics(logits_clean_all)
            logger_info(logging.getLogger(), "[RL Restorer] Statistics collected!")
        
        # Create RL intermediate layer restorer if enabled
        intermediate_restorer = None
        intermediate_trainer = None
        intermediate_collector = None
        
        use_rl_intermediate = configs.sensitive_restorer.get("use_rl_intermediate", False)
        if use_rl_intermediate and RL_INTERMEDIATE_AVAILABLE:
            logger_info(logging.getLogger(), "[RL Intermediate] Creating RL intermediate layer restorer...")
            intermediate_restorer, intermediate_collector = create_rl_intermediate_layer_restorer(
                model=training_model,
                sensitive_info=prepared_stats,
                baseline_stats=baseline_stats,
                state_dim=configs.sensitive_restorer.get("rl_state_dim", 128),
                hidden_dim=configs.sensitive_restorer.get("rl_hidden_dim", 256),
                max_steps_per_layer=configs.sensitive_restorer.get("rl_intermediate_max_steps", 2),
            )
            
            intermediate_trainer = RLLayerRestorerTrainer(
                restorer=intermediate_restorer,
                actor_lr=float(configs.sensitive_restorer.get("rl_actor_lr", 3e-4)),
                critic_lr=float(configs.sensitive_restorer.get("rl_critic_lr", 3e-4)),
                gamma=float(configs.sensitive_restorer.get("rl_gamma", 0.99)),
                lambda_=float(configs.sensitive_restorer.get("rl_lambda", 0.95)),
                entropy_coef=float(configs.sensitive_restorer.get("rl_entropy_coef", 0.01)),
                value_coef=float(configs.sensitive_restorer.get("rl_value_coef", 0.5)),
                max_grad_norm=float(configs.sensitive_restorer.get("max_grad_norm", 1.0)),
                use_ppo=configs.sensitive_restorer.get("rl_use_ppo", True),
            )
            
            total_intermediate_params = sum(p.numel() for p in intermediate_restorer.parameters())
            logger_info(logging.getLogger(), f"[RL Intermediate] Parameters: {total_intermediate_params:,} ({total_intermediate_params/1e6:.2f}M)")
            logger_info(logging.getLogger(), "[RL Intermediate] ✓ Created successfully!")
        
        # RL Restorer uses its own trainer, so set optimizer and scheduler to None
        restorer_optimizer = None
        restorer_lr_scheduler = None
        
        # Continue to training loop (don't return here)
    
    elif use_transformer:
        # Transformer architecture
        # First, determine actual feature dimensions for each layer
        collector_temp = SensitiveActivationCollector(training_model, prepared_stats)
        sample_inputs, _ = next(iter(train_loader))
        sample_inputs = sample_inputs.to(device)
        with torch.no_grad():
            _ = training_model(sample_inputs)
        layer_features_temp = collector_temp.build_layer_features(device)
        collector_temp.remove()
        
        if layer_features_temp:
            # Get actual feature dimensions for each layer
            feature_dims_per_layer = [feat.shape[1] for feat in layer_features_temp]
            num_layers = len(feature_dims_per_layer)
            logger_info(logging.getLogger(), 
                       f"Detected layer feature dimensions: {feature_dims_per_layer}")
        else:
            # Fallback: use expected dimensions
            num_layers = len([k for k in prepared_stats.keys() if prepared_stats[k].get("indices")])
            feature_dim_per_layer = configs.sensitive_restorer.get("topk_per_layer", 12) * 4
            feature_dims_per_layer = [feature_dim_per_layer] * num_layers
            logger_info(logging.getLogger(), 
                       f"Using expected feature dimensions: {feature_dims_per_layer}")
        
        embed_dim = configs.sensitive_restorer.get("transformer_embed_dim", 256)
        num_transformer_layers = configs.sensitive_restorer.get("num_transformer_layers", 4)
        num_heads = configs.sensitive_restorer.get("num_heads", 8)
        
        restorer = LayerwiseRestorer(
            num_layers=num_layers,
            feature_dims_per_layer=feature_dims_per_layer,
            num_classes=configs.dataloader.num_classes,
            embed_dim=embed_dim,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
        ).to(device)
        logger_info(logging.getLogger(), 
                   f"Using Transformer architecture: {num_layers} layers, embed_dim={embed_dim}, "
                   f"transformer_layers={num_transformer_layers}, num_heads={num_heads}")
    else:
        # MLP architecture
        restorer = SensitiveChannelRestorer(
            feature_dim,
            configs.dataloader.num_classes,
            hidden_dim=configs.sensitive_restorer.get("restorer_hidden_dim", 128),
        ).to(device)
        logger_info(logging.getLogger(), f"Using MLP architecture: hidden_dim={configs.sensitive_restorer.get('restorer_hidden_dim', 128)}")
    
    # Create collector for standard restorer (Transformer or MLP)
    if not (use_rl_restorer and RL_RESTORER_AVAILABLE):
        # For standard restorer, create collector with baseline_stats if available
        collector = SensitiveActivationCollector(training_model, prepared_stats, baseline_stats)
    
    total_params = sum(p.numel() for p in restorer.parameters())
    logger_info(logging.getLogger(), f"Restorer parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Get stage2_epochs first (needed for scheduler)
    stage2_epochs = configs.sensitive_restorer.get("stage2_epochs", 0)
    
    # Switch optimizer to SGD with momentum, which is often more stable for noisy gradients
    # Use smaller initial LR for more stable training with progressive BER
    default_lr = configs.sensitive_restorer.get("restorer_lr", 5e-3)  # Reduced from 1e-2 to 5e-3
    restorer_optimizer = torch.optim.SGD(
        restorer.parameters(),
        lr=default_lr,
        momentum=0.9,
        weight_decay=configs.sensitive_restorer.get("restorer_weight_decay", 1e-4),  # Small weight decay for regularization
    )
    logger_info(logging.getLogger(), f"Restorer optimizer: SGD with momentum=0.9, lr={default_lr:.2e}, weight_decay={restorer_optimizer.param_groups[0]['weight_decay']:.2e}")
    
    # Create learning rate scheduler for restorer (using same scheduler config as main model)
    from timm.scheduler import create_scheduler
    # Use the module-level SchedulerConfig class
    scheduler_config = SchedulerConfig(configs=configs, stage2_epochs=stage2_epochs)
    restorer_lr_scheduler, _ = create_scheduler(scheduler_config, restorer_optimizer)
    logger_info(logging.getLogger(), f"Restorer LR scheduler: {scheduler_config.sched}, initial_lr={restorer_optimizer.param_groups[0]['lr']:.6f}")

    kl_div_weight = configs.sensitive_restorer.get("kl_div_weight", 0.0)
    temperature = configs.sensitive_restorer.get("temperature", 4.0)
    stage2_ber = float(configs.sensitive_restorer.get("stage2_ber", 4e-2))
    stage2_seed = configs.sensitive_restorer.get("stage2_seed", getattr(configs, "seed", 42))

    # Multi-BER training setup
    stage2_mix_bers = configs.sensitive_restorer.get("stage2_mix_bers", [stage2_ber])
    stage2_mix_prob = configs.sensitive_restorer.get("stage2_mix_prob", 0.0)
    rl_gain = configs.sensitive_restorer.get("rl_gain", 0.7)
    rl_penalty = configs.sensitive_restorer.get("rl_penalty", 0.4)
    direction_weight = configs.sensitive_restorer.get("direction_weight", 0.5)
    use_rl_weighted = configs.sensitive_restorer.get("use_rl_weighted", True)
    
    fault_injector = FaultInjector(
        model=training_model,
        mode="ber",
        ber=stage2_ber,
        enable_in_training=False,  # Model is in eval mode, so use inference mode
        enable_in_inference=True,  # Enable fault injection in inference mode
        seed=stage2_seed,
        skip_first_last=configs.sensitive_restorer.get("skip_first_last", False),
        use_random_flip_in_training=True,  # Use completely random bit-flip during restorer training
    )
    fault_injector.disable()

    # Enable dynamic bit training in Stage 2 (same as Stage 1)
    # This allows restorer to learn fault correction under different bit configurations
    use_dynamic_bit = configs.enable_dynamic_bit_training
    target_bits = configs.target_bits
    max_bit = max(target_bits)  # Highest bit (e.g., 6-bit for Mixed 0)
    
    if use_dynamic_bit:
        from util.mpq import sample_max_cands
        logger_info(logging.getLogger(), f"Stage 2 will use dynamic bit training (target_bits: {target_bits})")
        logger_info(logging.getLogger(), f"Initializing with highest bit configuration: {max_bit}-bit (Mixed 0)")
        sample_max_cands(training_model, configs)  # Start with Mixed 0
    else:
        from util.mpq import switch_bit_width
        logger_info(logging.getLogger(), f"Setting model to fixed highest bit configuration: {max_bit}-bit")
        switch_bit_width(training_model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
    logger_info(logging.getLogger(), f"✓ Model bit-width configured")

    epoch_times = []
    start_time = time.time()
    trigger_index = 0
    
    # BER configuration: fixed BER, Beta distribution sampling, or progressive BER
    use_fixed_ber = configs.sensitive_restorer.get("use_fixed_ber", False)  # Use fixed BER for debugging convergence
    use_progressive_ber = configs.sensitive_restorer.get("use_progressive_ber", False)  # Use progressive BER (curriculum learning)
    fixed_ber = float(configs.sensitive_restorer.get("fixed_ber", 3e-2))  # Fixed BER value (default: 3e-2, moderate value)
    
    # Progressive BER scheduler (借鉴对抗训练中的渐进式攻击强度)
    progressive_ber_scheduler = None
    if use_progressive_ber:
        ber_min = float(configs.sensitive_restorer.get("ber_min", 2e-2))  # Minimum BER (至少 2e-2)
        ber_max = float(configs.sensitive_restorer.get("ber_max", 1e-1))  # Maximum BER
        schedule_type = configs.sensitive_restorer.get("progressive_schedule_type", "cosine")  # 'linear', 'cosine', 'exponential', 'step'
        warmup_epochs = configs.sensitive_restorer.get("progressive_warmup_epochs", 10)
        progressive_ber_scheduler = ProgressiveBERScheduler(
            ber_min=ber_min,
            ber_max=ber_max,
            total_epochs=stage2_epochs,
            schedule_type=schedule_type,
            warmup_epochs=warmup_epochs,
        )
        logger_info(logging.getLogger(), f"[Stage2] Using PROGRESSIVE BER: {schedule_type} schedule from {ber_min:.2e} to {ber_max:.2e} (warmup: {warmup_epochs} epochs)")
        logger_info(logging.getLogger(), f"[Stage2] Progressive BER schedule: {progressive_ber_scheduler.get_schedule_info()}")
    
    if use_fixed_ber:
        # Fixed BER mode: use the same BER for all batches (good for debugging convergence)
        logger_info(logging.getLogger(), f"[Stage2] Using FIXED BER: {fixed_ber:.2e} (for convergence testing)")
        # Set ber_min and ber_max to fixed_ber for consistency (though they won't be used in fixed mode)
        ber_min = fixed_ber
        ber_max = fixed_ber
        beta_alpha = 2.0
        beta_beta = 2.0
    elif not use_progressive_ber:
        # Beta distribution sampling mode (中间胖，两头瘦)
        ber_min = float(configs.sensitive_restorer.get("ber_min", 2e-2))  # Minimum BER (至少 2e-2)
        ber_max = float(configs.sensitive_restorer.get("ber_max", 1e-1))  # Maximum BER
        beta_alpha = float(configs.sensitive_restorer.get("beta_alpha", 2.0))  # Beta distribution shape parameter (alpha)
        beta_beta = float(configs.sensitive_restorer.get("beta_beta", 2.0))  # Beta distribution shape parameter (beta)
        # Beta(2, 2) gives a bell-shaped distribution (中间胖，两头瘦)
        logger_info(logging.getLogger(), f"[Stage2] BER sampling: Beta({beta_alpha}, {beta_beta}) distribution in [{ber_min:.2e}, {ber_max:.2e}]")
    
    # Gradient clipping for stability
    max_grad_norm = configs.sensitive_restorer.get("max_grad_norm", 1.0)
    logger_info(logging.getLogger(), f"[Stage2] Gradient clipping: max_norm={max_grad_norm}")
    
    # Hard sample mining configuration
    use_hard_sample_mining = configs.sensitive_restorer.get("use_hard_sample_mining", False)
    hard_sample_ratio = configs.sensitive_restorer.get("hard_sample_ratio", 1.0)
    
    for epoch in range(stage2_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kl_loss = 0.0
        running_dir_loss = 0.0
        sample_count = 0
        
        # Accuracy counters
        clean_correct = 0
        faulted_correct = 0
        restored_correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Dynamic bit training: randomly sample bit configuration for each batch
            # This allows restorer to learn fault correction under different bit configurations
            if use_dynamic_bit:
                from util.mpq import sample_one_mixed_policy, sample_max_cands
                # Randomly choose: Mixed 0 (max) with probability 1/3, or random mixed policy
                import random
                if random.random() < 0.33:  # 33% chance to use Mixed 0 (highest bit)
                    sample_max_cands(training_model, configs)
                    current_bit_mode = "Mixed 0 (max)"
                else:  # 67% chance to use random mixed policy
                    w_conf, a_conf, _ = sample_one_mixed_policy(training_model, configs)
                    current_bit_mode = f"Mixed (random)"
            # If not using dynamic bit, model stays at fixed max_bit configuration

            # Sample BER: fixed, progressive, or Beta distribution
            if use_fixed_ber:
                # Fixed BER mode: use the same BER for all batches
                effective_ber = fixed_ber
            elif use_progressive_ber and progressive_ber_scheduler is not None:
                # Progressive BER mode: use scheduler to get BER for current epoch
                effective_ber = progressive_ber_scheduler.get_ber(epoch)
            else:
                # Beta distribution sampling (中间胖，两头瘦)
                # Beta distribution samples in [0, 1], then map to [ber_min, ber_max]
                beta_sample = np.random.beta(beta_alpha, beta_beta)
                effective_ber = ber_min + (ber_max - ber_min) * beta_sample
            trigger_index += 1
            
            # Update fault injector BER
            fault_injector.ber = effective_ber

            collector.clear()
            fault_injector.disable()
            with torch.no_grad():
                logits_clean = training_model(inputs)
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += (pred_clean == targets).sum().item()

            collector.clear()
            fault_injector.enable()
            logits_faulted = training_model(inputs)
            fault_injector.disable()
            
            # Check for NaN/Inf in faulted logits (critical for restorer training)
            if torch.isnan(logits_faulted).any() or torch.isinf(logits_faulted).any():
                logger_info(logging.getLogger(), f"WARNING: NaN/Inf in faulted logits, skipping batch")
                continue
            
            pred_faulted = logits_faulted.argmax(dim=1)
            faulted_correct += (pred_faulted == targets).sum().item()
            
            # Identify hard samples (faulted and wrong)
            is_faulted_wrong = (pred_faulted != targets)
            is_faulted_correct = (pred_faulted == targets)

            # Extract features based on architecture
            if use_rl_restorer and RL_RESTORER_AVAILABLE:
                # RL Restorer: use integration function
                from util.rl_restorer_integration import extract_features_for_rl_restorer
                use_layerwise = configs.sensitive_restorer.get("use_layerwise", True)
                features, _, layer_features = extract_features_for_rl_restorer(
                    collector=collector,
                    model=training_model,
                    inputs=inputs,
                    fault_injector=fault_injector,
                    use_enhanced_features=use_enhanced_features,
                    use_layerwise=use_layerwise,
                    device=device,
                )
                if features is None and layer_features is None:
                    continue
                # Check for NaN/Inf in features
                if features is not None:
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        logger_info(logging.getLogger(), f"WARNING: NaN/Inf in features, skipping batch")
                        continue
                if layer_features is not None:
                    for feat in layer_features:
                        if torch.isnan(feat).any() or torch.isinf(feat).any():
                            logger_info(logging.getLogger(), f"WARNING: NaN/Inf in layer features, skipping batch")
                            continue
            elif use_transformer:
                layer_features = collector.build_layer_features(device)
                if not layer_features:
                    continue
                # Check for NaN/Inf in features
                for feat in layer_features:
                    if torch.isnan(feat).any() or torch.isinf(feat).any():
                        logger_info(logging.getLogger(), f"WARNING: NaN/Inf in layer features, skipping batch")
                        continue
            else: # MLP
                features = collector.build_feature_vector(device)
                if features is None:
                    continue
                # Check for NaN/Inf in features
                if torch.isnan(features).any() or torch.isinf(features).any():
                    logger_info(logging.getLogger(), f"WARNING: NaN/Inf in features, skipping batch")
                    continue
            
            # Train restorer
            if use_rl_restorer and RL_RESTORER_AVAILABLE:
                # Check if we have intermediate restorer
                use_rl_intermediate = configs.sensitive_restorer.get("use_rl_intermediate", False)
                if use_rl_intermediate and RL_INTERMEDIATE_AVAILABLE and intermediate_restorer is not None:
                    # Hybrid RL training: intermediate + output layer
                    # Collect clean and faulted activations for intermediate restorer
                    clean_activations, faulted_activations = collect_activations_for_training(
                        collector=intermediate_collector,
                        model=training_model,
                        inputs=inputs,
                        fault_injector=fault_injector,
                        device=device,
                    )
                    
                    # Train intermediate layer restorer
                    intermediate_restorer.train()
                    intermediate_metrics = intermediate_trainer.train_step(
                        inputs=inputs,
                        clean_activations=clean_activations,
                        faulted_activations=faulted_activations,
                    )
                    
                    # Train output layer restorer
                    restorer.train()
                    # Pass layer_features if use_layerwise, otherwise pass features
                    if use_layerwise and layer_features is not None:
                        output_metrics = rl_trainer.train_step(
                            logits_faulted=logits_faulted,
                            layer_features=layer_features,
                            logits_clean=logits_clean,
                            ber_level=effective_ber,
                        )
                    else:
                        output_metrics = rl_trainer.train_step(
                            logits_faulted=logits_faulted,
                            features=features,
                            logits_clean=logits_clean,
                            ber_level=effective_ber,
                        )
                    
                    # Evaluate restored logits (using intermediate restoration)
                    intermediate_restorer.eval()
                    restorer.eval()
                    with torch.no_grad():
                        # First apply intermediate restoration
                        intermediate_restorer.set_clean_activations(clean_activations)
                        intermediate_restorer.enable_restoration()
                        logits_after_intermediate = training_model(inputs)
                        intermediate_restorer.disable_restoration()
                        
                        # Then apply output layer restoration
                        if use_layerwise and layer_features is not None:
                            logits_restored, info = restorer(
                                logits_faulted=logits_after_intermediate,
                                layer_features=layer_features,
                                logits_clean=logits_clean,
                                training=False,
                                deterministic=True,
                                ber_level=effective_ber,
                            )
                        else:
                            logits_restored, info = restorer(
                                logits_faulted=logits_after_intermediate,
                                features=features,
                                logits_clean=logits_clean,
                                training=False,
                                deterministic=True,
                                ber_level=effective_ber,
                            )
                        pred_restored = logits_restored.argmax(dim=1)
                        restored_correct += (pred_restored == targets).sum().item()
                    
                    # Update running statistics (combined metrics)
                    # Intermediate metrics: {layer_name}_actor_loss, {layer_name}_critic_loss, etc.
                    intermediate_total_loss = sum(v for k, v in intermediate_metrics.items() if 'actor_loss' in k or 'critic_loss' in k)
                    output_total_loss = output_metrics.get('total_actor_loss', output_metrics.get('actor_loss', 0.0))
                    
                    running_loss += (output_total_loss + intermediate_total_loss) * inputs.size(0)
                    running_ce_loss += output_metrics.get('actor_loss', 0.0) * inputs.size(0)
                    running_kl_loss += output_metrics.get('critic_loss', 0.0) * inputs.size(0)
                    running_dir_loss += intermediate_total_loss * inputs.size(0)
                    sample_count += inputs.size(0)
                else:
                    # Standard RL Restorer training (output layer only)
                    restorer.train()
                    # Pass layer_features if use_layerwise, otherwise pass features
                    if use_layerwise and layer_features is not None:
                        metrics = rl_trainer.train_step(
                            logits_faulted=logits_faulted,
                            layer_features=layer_features,
                            logits_clean=logits_clean,
                            ber_level=effective_ber,  # 传递当前 BER 级别用于奖励函数加权
                        )
                    else:
                        metrics = rl_trainer.train_step(
                            logits_faulted=logits_faulted,
                            features=features,
                            logits_clean=logits_clean,
                            ber_level=effective_ber,  # 传递当前 BER 级别用于奖励函数加权
                        )
                    
                    # Evaluate restored logits
                    restorer.eval()
                    with torch.no_grad():
                        if use_layerwise and layer_features is not None:
                            logits_restored, info = restorer(
                                logits_faulted=logits_faulted,
                                layer_features=layer_features,
                                logits_clean=logits_clean,
                                training=False,
                                deterministic=True,
                                ber_level=effective_ber,
                            )
                        else:
                            logits_restored, info = restorer(
                                logits_faulted=logits_faulted,
                                features=features,
                                logits_clean=logits_clean,
                                training=False,
                                deterministic=True,
                                ber_level=effective_ber,
                            )
                        pred_restored = logits_restored.argmax(dim=1)
                        restored_correct += (pred_restored == targets).sum().item()
                    
                    # Update running statistics (RL metrics)
                    running_loss += metrics['total_actor_loss'] * inputs.size(0)
                    running_ce_loss += metrics['actor_loss'] * inputs.size(0)
                    running_kl_loss += metrics['critic_loss'] * inputs.size(0)
                    running_dir_loss += 0.0  # Not used for RL
                    sample_count += inputs.size(0)
                
                # Skip the rest of the loss computation (RL handles it)
                continue
            else:
                # Standard restorer training
                restorer.train()
                restorer_optimizer.zero_grad()
                
                if use_transformer:
                    logits_restored, gate = restorer(logits_faulted, layer_features)
                else:
                    logits_restored, gate = restorer(logits_faulted, features)
            
            # Debug: Check for NaN or Inf
            if torch.isnan(logits_restored).any() or torch.isinf(logits_restored).any():
                logger_info(logging.getLogger(), f"WARNING: NaN or Inf detected in logits_restored!")
                continue
            
            pred_restored = logits_restored.argmax(dim=1)
            restored_correct += (pred_restored == targets).sum().item()
            
            # High BER training strategy: Hard Sample Mining
            # Focus on samples where faulted prediction is wrong (hard samples)
            hard_mask = None  # Initialize for later use
            if use_hard_sample_mining:
                # Only train on hard samples (faulted and wrong)
                hard_mask = is_faulted_wrong
                if hard_mask.sum() == 0:
                    # No hard samples in this batch, skip
                    continue
                
                # Optionally subsample hard samples if too many
                if hard_sample_ratio < 1.0 and hard_mask.sum() > 0:
                    num_hard = int(hard_mask.sum() * hard_sample_ratio)
                    hard_indices = torch.where(hard_mask)[0]
                    selected_indices = hard_indices[torch.randperm(len(hard_indices))[:num_hard]]
                    hard_mask = torch.zeros_like(hard_mask)
                    hard_mask[selected_indices] = True
                
                # Filter to hard samples only
                logits_restored_hard = logits_restored[hard_mask]
                logits_clean_hard = logits_clean[hard_mask]
                logits_faulted_hard = logits_faulted[hard_mask]
                targets_hard = targets[hard_mask]
                
                # Use hard samples for loss computation
                ce_loss_vec = F.cross_entropy(logits_restored_hard, targets_hard, reduction='none')
                ce_loss = ce_loss_vec.mean()
                
                # Also compute MSE loss for hard samples (stronger signal)
                mse_weight = configs.sensitive_restorer.get("hard_sample_mse_weight", 0.3)
                mse_loss = F.mse_loss(logits_restored_hard, logits_clean_hard) if mse_weight > 0 else torch.tensor(0.0, device=device)
            else:
                # Original RL-weighted loss
                if use_rl_weighted:
                    sample_reward = torch.ones_like(targets, dtype=torch.float, device=device)
                    sample_reward = sample_reward + rl_gain * is_faulted_wrong.float()
                    sample_reward = sample_reward - rl_penalty * is_faulted_correct.float()
                    sample_reward = sample_reward.clamp_min(0.1).detach()
                else:
                    sample_reward = torch.ones_like(targets, dtype=torch.float, device=device)
                
                # CE loss with RL weighting
                ce_loss_vec = F.cross_entropy(logits_restored, targets, reduction='none')
                ce_loss = (ce_loss_vec * sample_reward).mean()
                mse_loss = torch.tensor(0.0, device=device)
            
            # KL-Divergence loss for knowledge distillation
            if use_hard_sample_mining:
                # For hard samples, use simplified loss
                if kl_div_weight > 0:
                    T = temperature
                    kl_loss = F.kl_div(
                        F.log_softmax(logits_restored_hard / T, dim=1),
                        F.softmax(logits_clean_hard.detach() / T, dim=1),
                        reduction='batchmean'
                    )
                else:
                    kl_loss = torch.tensor(0.0, device=device, requires_grad=False)
                direction_loss = torch.tensor(0.0, device=device, requires_grad=False)
                
                # Combined loss for hard samples
                loss = ce_loss + kl_div_weight * kl_loss + mse_weight * mse_loss
            else:
                # Original loss computation
                if kl_div_weight > 0:
                    T = temperature
                    kl_loss_vec = F.kl_div(
                        F.log_softmax(logits_restored / T, dim=1),
                        F.softmax(logits_clean.detach() / T, dim=1),
                        reduction='none'
                    ).sum(dim=1)
                    kl_loss = (kl_loss_vec * sample_reward).mean()
                else:
                    kl_loss = torch.tensor(0.0, device=device, requires_grad=False)
                
                # Direction loss (correction direction should match clean-faulted direction)
                # Use cosine similarity or normalized direction to avoid large absolute values
                correction_pred = logits_restored - logits_faulted.detach()
                correction_target = logits_clean.detach() - logits_faulted.detach()
                if direction_weight > 0:
                    # Focus on target class direction
                    target_delta_pred = correction_pred.gather(1, targets.unsqueeze(1)).squeeze(1)
                    target_delta_target = correction_target.gather(1, targets.unsqueeze(1)).squeeze(1)
                    direction_mask = target_delta_target.abs() > 0.15  # Only when correction is needed
                    if direction_mask.any():
                        # Normalize the direction loss to avoid large absolute values
                        # Use relative error instead of absolute squared error
                        target_delta_target_masked = target_delta_target[direction_mask]
                        target_delta_pred_masked = target_delta_pred[direction_mask]
                        # Normalize by target magnitude to get relative error
                        normalized_error = ((target_delta_pred_masked - target_delta_target_masked) / (target_delta_target_masked.abs() + 1e-6)) ** 2
                        dir_loss_vec = torch.zeros_like(target_delta_pred)
                        dir_loss_vec[direction_mask] = normalized_error
                        direction_loss = (dir_loss_vec * sample_reward).mean()
                    else:
                        direction_loss = torch.tensor(0.0, device=device, requires_grad=False)
                else:
                    direction_loss = torch.tensor(0.0, device=device, requires_grad=False)
                
                # Combined loss
                loss = ce_loss + kl_div_weight * F.kl_div(F.log_softmax(logits_restored / temperature, dim=-1), F.softmax(logits_clean / temperature, dim=-1), reduction='none').mean() + kl_loss + direction_weight * direction_loss
            
            loss.backward()
            
            # Gradient clipping for stability
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(restorer.parameters(), max_grad_norm)
            
            restorer_optimizer.step()

            # Update running statistics
            if use_hard_sample_mining:
                # Only count hard samples
                num_samples = hard_mask.sum().item() if hard_mask.sum() > 0 else 0
                if num_samples > 0:
                    running_loss += loss.item() * num_samples
                    running_ce_loss += ce_loss.item() * num_samples
                    running_kl_loss += kl_loss.item() * num_samples
                    running_dir_loss += 0.0  # Direction loss not used for hard samples
                    sample_count += num_samples
            else:
                running_loss += loss.item() * inputs.size(0)
                running_ce_loss += ce_loss.item() * inputs.size(0)
                running_kl_loss += kl_loss.item() * inputs.size(0)
                running_dir_loss += direction_loss.item() * inputs.size(0)
                sample_count += inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = running_loss / max(1, sample_count)
        avg_ce_loss = running_ce_loss / max(1, sample_count)
        avg_kl_loss = running_kl_loss / max(1, sample_count)
        avg_dir_loss = running_dir_loss / max(1, sample_count)
        clean_acc = 100.0 * clean_correct / max(1, total_samples)
        faulted_acc = 100.0 * faulted_correct / max(1, total_samples)
        restored_acc = 100.0 * restored_correct / max(1, total_samples)
        improvement = restored_acc - faulted_acc
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        recent_epochs = min(5, len(epoch_times))
        avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs
        remaining_epochs = stage2_epochs - epoch - 1
        remaining_time = avg_epoch_time * remaining_epochs
        eta = datetime.now() + timedelta(seconds=remaining_time)

        # Step learning rate scheduler
        if use_rl_restorer and RL_RESTORER_AVAILABLE and rl_trainer is not None:
            rl_trainer.update_schedulers()
            current_lr = rl_trainer.get_lr()['actor_lr']
        elif restorer_lr_scheduler is not None:
            # Check if scheduler is plateau type (needs metric) or step type (needs epoch)
            # Check the actual scheduler type by inspecting its class name
            scheduler_type = configs.sched if hasattr(configs, 'sched') else 'cosine'
            scheduler_class_name = restorer_lr_scheduler.__class__.__name__.lower()
            
            if 'plateau' in scheduler_class_name:
                # Plateau scheduler needs a metric (loss value, lower is better)
                # step(metrics, epoch=None) - epoch is optional positional argument
                restorer_lr_scheduler.step(avg_loss, epoch)
            else:
                # Other schedulers (cosine, step, etc.) use epoch
                restorer_lr_scheduler.step(epoch)
            current_lr = restorer_optimizer.param_groups[0]['lr']
        else:
            current_lr = 0.0
        
        logger_info(logging.getLogger(), f"[Stage2] Epoch {epoch}: total_loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, KL={avg_kl_loss:.4f}, Dir={avg_dir_loss:.4f})")
        logger_info(
            logging.getLogger(),
            f"  Acc - Clean: {clean_acc:.2f}% | Faulted: {faulted_acc:.2f}% | Restored: {restored_acc:.2f}% | Improvement: {improvement:+.2f}% | LR: {current_lr:.6f}"
        )
        if epoch == 0:
            logger_info(logging.getLogger(), f"  [Debug] Fault injection enabled: {fault_injector._enabled if hasattr(fault_injector, '_enabled') else 'N/A'}")
            if use_fixed_ber:
                logger_info(logging.getLogger(), f"  [Debug] Using FIXED BER: {fixed_ber:.2e}")
            else:
                logger_info(logging.getLogger(), f"  [Debug] BER sampling: Beta({beta_alpha}, {beta_beta}) in [{ber_min:.2e}, {ber_max:.2e}]")
            logger_info(logging.getLogger(), f"  [Debug] Faulted acc drop: {clean_acc - faulted_acc:.2f}% (should be > 0 if fault injection works)")
        logger_info(
            logging.getLogger(),
            f"  ⏱️ Epoch耗时: {epoch_time:.1f}s | 平均Epoch耗时: {avg_epoch_time:.1f}s | 剩余Epoch: {remaining_epochs}",
        )
        if remaining_epochs > 0:
            logger_info(
                logging.getLogger(),
                f"  📅 预估剩余时间: {remaining_time/60:.1f}分钟 | 预估完成时间: {eta.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            logger_info(logging.getLogger(), f"  ✅ Stage2完成！总耗时: {(time.time()-start_time)/60:.1f}分钟")

    collector.remove()
    fault_injector.disable()
    
    # Return appropriate values based on restorer type
    if use_rl_restorer and RL_RESTORER_AVAILABLE:
        # RL Restorer: return (restorer, rl_trainer, collector, None, intermediate_restorer, intermediate_trainer, intermediate_collector)
        return restorer, rl_trainer, collector, None, intermediate_restorer, intermediate_trainer, intermediate_collector
    else:
        # Standard restorer: return (restorer, optimizer, lr_scheduler)
        return restorer, restorer_optimizer, restorer_lr_scheduler


def main():
    import sys
    # Parse our custom arguments first
    args = parse_args()
    script_dir = Path.cwd()
    
    # Add project root to Python path to ensure util module can be imported
    project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Temporarily modify sys.argv to pass config file to get_config as positional arg
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], args.config]
    try:
        configs = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    configs.output_dir = args.output_dir or configs.output_dir

    set_global_seed(seed=getattr(configs, "seed", 42))
    logger, log_dir, pymonitor, tbmonitor = init_logging(configs, script_dir)

    copy_code(logger, src=str(script_dir), dst=os.path.join(log_dir, "code"))

    # Initialize dataloader first (needed for sensitivity analysis)
    # Normalize dataset path to absolute path (relative to project root)
    if not os.path.isabs(configs.dataloader.path):
        # If relative path, make it relative to project root
        project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
        configs.dataloader.path = str(project_root / configs.dataloader.path)
    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)

    # --- CRITICAL FIX: Correct Initialization and Loading Order ---

    # 1. Build and load the Stage 1 main model first. DO NOT pass the optimizer yet.
    logger_info(logging.getLogger(), "Loading Stage 1 checkpoint for analysis...")
    training_model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    training_model = preprocess_model(training_model, configs)
    training_model = replace_module_by_names(training_model, find_modules_to_quantize(training_model, configs))
    training_model.to(args.device)
    load_checkpoint(training_model, args.stage1_ckpt) # optimizer=None
    logger_info(logging.getLogger(), "Stage 1 model loaded successfully.")

    # 2. Now that the model is loaded, perform gradient sensitivity analysis.
    logger_info(logging.getLogger(), "Collecting gradient sensitivity...")
    # Use save_path from config if available, otherwise use log_dir
    save_path = getattr(configs, 'save_path', None)
    if save_path is None:
        save_path = log_dir
    else:
        # If save_path is relative, make it relative to output_dir
        if not os.path.isabs(save_path):
            save_path = os.path.join(configs.output_dir, save_path)
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
    sensitive_channels_path = os.path.join(save_path, "sensitive_channels.pth")
    sensitive_channels = collect_gradient_sensitivity(
        model=training_model,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss().to(args.device),
        device=args.device,
        topk_per_layer=configs.sensitive_restorer.get("topk_per_layer", 8),
        max_batches=configs.sensitive_restorer.get("sensitivity_batches", 50),
        output_path=sensitive_channels_path,
    )
    if not sensitive_channels:
        raise RuntimeError("No sensitive channels identified for Stage2 training.")
    logger_info(logging.getLogger(), f"Identified {sum(len(v['indices']) for v in sensitive_channels.values())} sensitive channels across {len(sensitive_channels)} layers.")

    # 3. Compute the activation baseline for the identified channels.
    logger_info(logging.getLogger(), "Computing activation baseline...")
    baseline_path = os.path.join(save_path, "sensitive_baseline.pth")
    baseline_stats = compute_activation_baseline(
        model=training_model,
        data_loader=train_loader,
        sensitive_channels=sensitive_channels,
        device=args.device,
        max_batches=configs.sensitive_restorer.get("baseline_batches", 50),
        output_path=baseline_path,
    )
    logger_info(logging.getLogger(), "Activation baseline computed.")

    # 4. Initialize the Restorer model.
    # We still need to do a dummy pass to get feature dimensions, as this depends on the model and sensitive channels.
    logger_info(logging.getLogger(), "Running dummy forward pass for Restorer initialization...")
    collector_for_dim = SensitiveActivationCollector(training_model, sensitive_channels, baseline_stats)
    dummy_input = torch.randn(2, 3, 32, 32).to(args.device)
    training_model(dummy_input)
    layer_feats, feature_dims_per_layer = collector_for_dim.build_layer_features(dummy_input.device)
    if layer_feats is None or len(layer_feats) == 0:
        raise RuntimeError("Failed to extract features from dummy forward pass")
    if not feature_dims_per_layer:
        # Fallback: compute from layer_feats if feature_dims is empty
        feature_dims_per_layer = [feat.shape[1] for feat in layer_feats]
    collector_for_dim.remove()
    logger_info(logging.getLogger(), f"Determined feature dims: {feature_dims_per_layer}")

    restorer = LayerwiseRestorer(
        num_layers=len(sensitive_channels),
        feature_dims_per_layer=feature_dims_per_layer,
        num_classes=configs.dataloader.num_classes,
        expert_hidden_dim=configs.sensitive_restorer.get("expert_hidden_dim", 128)
    )
    restorer.to(args.device)
    total_params = sum(p.numel() for p in restorer.parameters() if p.requires_grad)
    logger_info(logging.getLogger(), f"LayerwiseRestorer initialized. Trainable params: {total_params / 1e3:.1f}K")

    # 5. NOW, create the optimizer that will ONLY manage the restorer's parameters.
    logger_info(logging.getLogger(), "Creating optimizer for the Restorer...")
    restorer_optimizer = torch.optim.SGD(
        restorer.parameters(),
        lr=configs.sensitive_restorer.get("restorer_lr", 1e-3),
        momentum=0.9,
        weight_decay=configs.sensitive_restorer.get("restorer_weight_decay", 0.0),
    )
    
    # 6. Create the LR scheduler for this new optimizer.
    from timm.scheduler import create_scheduler
    stage2_epochs = configs.sensitive_restorer.get("stage2_epochs", 0)
    scheduler_configs = SchedulerConfig(
        sched=configs.sched,
        epochs=stage2_epochs,
        min_lr=configs.min_lr,
        warmup_lr=configs.warmup_lr,
        warmup_epochs=configs.warmup_epochs,
        cooldown_epochs=configs.cooldown_epochs,
        decay_rate=getattr(configs, 'decay_rate', 0.1),
        decay_epochs=getattr(configs, 'decay_epochs', 30),
    )
    restorer_lr_scheduler, _ = create_scheduler(scheduler_configs, restorer_optimizer)

    # 7. Optionally, resume training for Stage 2 by loading a restorer checkpoint.
    # This will correctly load state into the restorer, its optimizer, and its scheduler.
    if configs.sensitive_restorer.get("resume", None):
        logger_info(logging.getLogger(), f"Resuming Stage 2 from: {configs.sensitive_restorer.resume}")
        load_checkpoint(restorer, configs.sensitive_restorer.resume, optimizer=restorer_optimizer, lr_scheduler=restorer_lr_scheduler)

    # The main training loop starts here
    # train_loader already initialized above for sensitivity analysis
    # Note: We don't need to create optimizer/lr_scheduler for the main model in Stage 2,
    # as we're only training the restorer. The main model stays in eval mode.
    # If you need to train the main model as well, uncomment the following lines:
    # optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(training_model, configs)
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # load_checkpoint(
    #     training_model,
    #     args.stage1_ckpt,
    #     'cuda',
    #     strict=False,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     lr_scheduler_q=lr_scheduler_q,
    #     optimizer_q=optimizer_q,
    # )
    
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Prepare stats for training (convert baseline_stats to the format expected by train_restorer)
    # The train_restorer function expects prepared_stats which is the same as sensitive_channels
    # but with baseline stats integrated. For now, we'll use sensitive_channels as prepared_stats.
    prepared_stats = sensitive_channels
    # Calculate total feature dimension
    feature_dim = sum(len(v['indices']) * 4 for v in sensitive_channels.values())  # 4 stats per channel
    
    # Note: baseline_stats is already computed above and will be passed to train_restorer
    # through the function signature, but we need to make sure it's available for collector creation
    
    # Now call train_restorer to actually train the restorer
    restorer_result = train_restorer(
        training_model,
        train_loader,
        criterion,
        prepared_stats,
        feature_dim,
        args.device,
        configs,
        baseline_stats=baseline_stats,  # Pass baseline_stats to train_restorer
    )
    
    # Handle different return types: RL Restorer vs standard restorer
    use_rl_restorer = configs.sensitive_restorer.get("use_rl_restorer", False)
    use_rl_intermediate = configs.sensitive_restorer.get("use_rl_intermediate", False)
    intermediate_restorer = None
    intermediate_trainer = None
    intermediate_collector = None
    
    if use_rl_restorer and RL_RESTORER_AVAILABLE:
        if len(restorer_result) == 7:
            # RL Restorer with Intermediate: (restorer, rl_trainer, collector, None, intermediate_restorer, intermediate_trainer, intermediate_collector)
            restorer, rl_trainer, collector, _, intermediate_restorer, intermediate_trainer, intermediate_collector = restorer_result
            restorer_optimizer = None  # RL uses its own trainer
            restorer_lr_scheduler = None  # RL uses its own scheduler
            logger_info(logging.getLogger(), "[Stage2] ✓ RL Restorer with Intermediate Layer Restoration enabled!")
        elif len(restorer_result) == 4 and restorer_result[3] is None:
            # RL Restorer: (restorer, rl_trainer, collector, None)
            restorer, rl_trainer, collector, _ = restorer_result
            intermediate_restorer = None
            intermediate_trainer = None
            intermediate_collector = None
            restorer_optimizer = None  # RL uses its own trainer
            restorer_lr_scheduler = None  # RL uses its own scheduler
        elif len(restorer_result) == 3:
            # RL Restorer: (restorer, rl_trainer, None)
            restorer, rl_trainer, _ = restorer_result
            restorer_optimizer = None
            restorer_lr_scheduler = None
            collector = None  # Should have been created in train_restorer
        else:
            raise ValueError(f"Unexpected return value from train_restorer for RL: {len(restorer_result)} items")
    else:
        # Standard restorer: (restorer, optimizer, lr_scheduler)
        restorer, restorer_optimizer, restorer_lr_scheduler = restorer_result
        rl_trainer = None
        collector = None

    # Note: For Stage 2, we only save the restorer, not the main model
    # The main model should already be saved from Stage 1
    save_checkpoint(
        configs.sensitive_restorer.get('stage2_epochs', 0),
        configs.arch,
        training_model,  # Use training_model instead of model
        training_model,  # target_model is the same as training_model for Stage 2
        None,  # No optimizer for main model in Stage 2
        {'stage': 'sensitive_restorer'},
        False,
        'sensitive_stage2',
        log_dir,
        lr_scheduler=None,  # No lr_scheduler for main model in Stage 2
        lr_scheduler_q=None,
        optimizer_q=None,
        sensitive_restorer=restorer,
        sensitive_optimizer=restorer_optimizer if restorer_optimizer is not None else None,
        sensitive_lr_scheduler=restorer_lr_scheduler if restorer_lr_scheduler is not None else None,
    )

    if is_master():
        tbmonitor.writer.close()


if __name__ == "__main__":
    main()
