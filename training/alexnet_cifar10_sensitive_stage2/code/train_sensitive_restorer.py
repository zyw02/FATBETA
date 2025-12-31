import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    log_dir = init_logger(configs.name, output_dir, script_dir / "logging.conf")
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


def train_restorer(model, train_loader, criterion, prepared_stats, feature_dim, device, configs):
    training_model = model.module if hasattr(model, "module") else model
    # Set the main model to evaluation mode. This is crucial for disabling dropout, etc.
    training_model.eval()
    
    # CRITICAL FIX: DO NOT set requires_grad=False on the entire model.
    # This breaks the computation graph for fault injection and feature collection.
    # The optimizer will only update the restorer's parameters, effectively
    # freezing the main model's weights.
    # for param in training_model.parameters():
    #     param.requires_grad_(False)

    # Architecture selection
    use_transformer = configs.sensitive_restorer.get("use_transformer", False)
    
    if use_transformer:
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
    
    total_params = sum(p.numel() for p in restorer.parameters())
    logger_info(logging.getLogger(), f"Restorer parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Get stage2_epochs first (needed for scheduler)
    stage2_epochs = configs.sensitive_restorer.get("stage2_epochs", 0)
    
    # Switch optimizer to SGD with momentum, which is often more stable for noisy gradients
    restorer_optimizer = torch.optim.SGD(
        restorer.parameters(),
        lr=configs.sensitive_restorer.get("restorer_lr", 1e-2), # Default to 0.01 for SGD
        momentum=0.9,
        weight_decay=configs.sensitive_restorer.get("restorer_weight_decay", 0.0),
    )
    logger_info(logging.getLogger(), "Switched restorer optimizer to SGD with momentum=0.9")
    
    # Create learning rate scheduler for restorer (using same scheduler config as main model)
    from timm.scheduler import create_scheduler
    # Create a temporary config object for scheduler
    class SchedulerConfig:
        def __init__(self, configs, stage2_epochs):
            self.sched = getattr(configs, 'sched', 'cosine')
            self.epochs = stage2_epochs
            self.min_lr = getattr(configs, 'min_lr', 0.0)
            self.warmup_epochs = getattr(configs, 'warmup_epochs', 0)
            self.warmup_lr = getattr(configs, 'warmup_lr', 0.0)
            self.decay_rate = getattr(configs, 'decay_rate', 0.1)
            self.decay_epochs = getattr(configs, 'decay_epochs', 30)
            self.cooldown_epochs = getattr(configs, 'cooldown_epochs', 0)
    
    scheduler_config = SchedulerConfig(configs, stage2_epochs)
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

    collector = SensitiveActivationCollector(training_model, prepared_stats)

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
    
    # BER configuration: fixed BER or Beta distribution sampling
    use_fixed_ber = configs.sensitive_restorer.get("use_fixed_ber", False)  # Use fixed BER for debugging convergence
    fixed_ber = configs.sensitive_restorer.get("fixed_ber", 3e-2)  # Fixed BER value (default: 3e-2, moderate value)
    
    if use_fixed_ber:
        # Fixed BER mode: use the same BER for all batches (good for debugging convergence)
        logger_info(logging.getLogger(), f"[Stage2] Using FIXED BER: {fixed_ber:.2e} (for convergence testing)")
    else:
        # Beta distribution sampling mode (中间胖，两头瘦)
        ber_min = configs.sensitive_restorer.get("ber_min", 1e-2)  # Minimum BER
        ber_max = configs.sensitive_restorer.get("ber_max", 1e-1)  # Maximum BER
        beta_alpha = configs.sensitive_restorer.get("beta_alpha", 2.0)  # Beta distribution shape parameter (alpha)
        beta_beta = configs.sensitive_restorer.get("beta_beta", 2.0)  # Beta distribution shape parameter (beta)
        # Beta(2, 2) gives a bell-shaped distribution (中间胖，两头瘦)
        logger_info(logging.getLogger(), f"[Stage2] BER sampling: Beta({beta_alpha}, {beta_beta}) distribution in [{ber_min:.2e}, {ber_max:.2e}]")
    
    # Gradient clipping for stability
    max_grad_norm = configs.sensitive_restorer.get("max_grad_norm", 1.0)
    logger_info(logging.getLogger(), f"[Stage2] Gradient clipping: max_norm={max_grad_norm}")
    
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

            # Sample BER: fixed or Beta distribution
            if use_fixed_ber:
                # Fixed BER mode: use the same BER for all batches
                effective_ber = fixed_ber
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
            pred_faulted = logits_faulted.argmax(dim=1)
            faulted_correct += (pred_faulted == targets).sum().item()
            
            # Identify hard samples (faulted and wrong)
            is_faulted_wrong = (pred_faulted != targets)
            is_faulted_correct = (pred_faulted == targets)

            # Extract features based on architecture
            if use_transformer:
                layer_features = collector.build_layer_features(device)
                if not layer_features:
                    continue
            else: # MLP
                features = collector.build_feature_vector(device)
                if features is None:
                    continue
            
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
            
            # RL-weighted loss
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
            
            # KL-Divergence loss for knowledge distillation
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

        # Step learning rate scheduler (timm scheduler requires epoch parameter)
        restorer_lr_scheduler.step(epoch)
        current_lr = restorer_optimizer.param_groups[0]['lr']
        
        logger_info(logging.getLogger(), f"[Stage2] Epoch {epoch}: total_loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, KL={avg_kl_loss:.4f}, Dir={avg_dir_loss:.4f})")
        logger_info(
            logging.getLogger(),
            f"  Acc - Clean: {clean_acc:.2f}% | Faulted: {faulted_acc:.2f}% | Restored: {restored_acc:.2f}% | Improvement: {improvement:+.2f}% | LR: {current_lr:.6f}"
        )
        if epoch == 0:
            logger_info(logging.getLogger(), f"  [Debug] Fault injection enabled: {fault_injector._enabled if hasattr(fault_injector, '_enabled') else 'N/A'}")
            logger_info(logging.getLogger(), f"  [Debug] Effective BER: {effective_ber:.2e}, Stage2 BER: {stage2_ber:.2e}")
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
    return restorer, restorer_optimizer, restorer_lr_scheduler


def main():
    import sys
    # Parse our custom arguments first
    args = parse_args()
    script_dir = Path.cwd()
    
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
    sensitive_channels_path = os.path.join(configs.save_path, "sensitive_channels.pth")
    sensitive_channels = collect_gradient_sensitivity(
        model=training_model,
        data_loader=train_loader,
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
    baseline_path = os.path.join(configs.save_path, "sensitive_baseline.pth")
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
    collector_for_dim.register_hooks()
    dummy_input = torch.randn(2, 3, 32, 32).to(args.device)
    training_model(dummy_input)
    _, feature_dims_per_layer, _ = collector_for_dim.build_layer_features(dummy_input.device)
    collector_for_dim.remove_hooks()
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
    stage2_epochs = configs.sensitive_restorer.get("stage2_epochs", 0)
    scheduler_configs = SchedulerConfig(
        sched=configs.sched,
        epochs=stage2_epochs,
        min_lr=configs.min_lr,
        warmup_lr=configs.warmup_lr,
        warmup_epochs=configs.warmup_epochs,
        cooldown_epochs=configs.cooldown_epochs,
    )
    restorer_lr_scheduler, _ = create_scheduler(restorer_optimizer, **scheduler_configs.to_dict())

    # 7. Optionally, resume training for Stage 2 by loading a restorer checkpoint.
    # This will correctly load state into the restorer, its optimizer, and its scheduler.
    if configs.sensitive_restorer.get("resume", None):
        logger_info(logging.getLogger(), f"Resuming Stage 2 from: {configs.sensitive_restorer.resume}")
        load_checkpoint(restorer, configs.sensitive_restorer.resume, optimizer=restorer_optimizer, lr_scheduler=restorer_lr_scheduler)

    # The main training loop starts here
    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)

    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(model, configs)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    load_checkpoint(
        model,
        args.stage1_ckpt,
        'cuda',
        strict=False,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_q=lr_scheduler_q,
        optimizer_q=optimizer_q,
    )

    logger_info(logger, "Collecting gradient sensitivity (Stage2)...")
    sensitive_channels = collect_gradient_sensitivity(
        model,
        train_loader,
        criterion,
        device=torch.device('cuda'),
        topk_per_layer=configs.sensitive_restorer.get('topk_per_layer', 8),
        max_batches=configs.sensitive_restorer.get('sensitivity_batches', 50),
        output_path=os.path.join(log_dir, 'sensitive_channels.pth'),
    )
    logger_info(logger, "Computing activation baseline for sensitive channels...")
    baseline_stats = compute_activation_baseline(
        model,
        train_loader,
        sensitive_channels,
        device=torch.device('cuda'),
        max_batches=configs.sensitive_restorer.get('baseline_batches', 50),
        output_path=os.path.join(log_dir, 'sensitive_baseline.pth'),
    )

    prepared_stats, feature_dim = prepare_stats_for_device(baseline_stats, torch.device('cuda'))
    if feature_dim == 0:
        raise RuntimeError("No sensitive channels identified for Stage2 training.")

    restorer, restorer_optimizer, restorer_lr_scheduler = train_restorer(
        model,
        train_loader,
        criterion,
        prepared_stats,
        feature_dim,
        torch.device('cuda'),
        configs,
    )

    save_checkpoint(
        configs.sensitive_restorer.get('stage2_epochs', 0),
        configs.arch,
        model,
        target_model,
        optimizer,
        {'stage': 'sensitive_restorer'},
        False,
        'sensitive_stage2',
        log_dir,
        lr_scheduler=lr_scheduler,
        lr_scheduler_q=lr_scheduler_q,
        optimizer_q=optimizer_q,
        sensitive_restorer=restorer,
        sensitive_optimizer=restorer_optimizer,
    )

    if is_master():
        tbmonitor.writer.close()


if __name__ == "__main__":
    main()
