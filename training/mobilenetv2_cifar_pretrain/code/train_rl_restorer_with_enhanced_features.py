"""
Training script for RL Restorer with Enhanced Features

This script demonstrates how to integrate RL Restorer with enhanced features
into the training pipeline. It can be used as a template or directly.
"""

import argparse
import logging
import os
import time
from datetime import datetime
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
from util.fault_injector import FaultInjector

# Import RL Restorer components
from util.rl_restorer_integration import (
    create_rl_restorer_with_features,
    extract_features_for_rl_restorer,
)
from util.rl_restorer_trainer import RLRestorerTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="RL Restorer training with enhanced features")
    parser.add_argument("--config", required=True, help="Path to config yaml")
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
    """Prepare statistics for device"""
    prepared = {}
    feature_dim = 0
    for name, entry in raw_stats.items():
        indices = entry.get("indices", [])
        if not indices:
            continue
        
        if "energy_mean" in entry:
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
            feature_dim += len(indices) * 4
        else:
            prepared[name] = {
                "indices": indices,
                "mean": entry["mean"].to(device),
                "std": entry["std"].to(device),
            }
            feature_dim += len(indices)
    return prepared, feature_dim


def train_rl_restorer(model, train_loader, prepared_stats, device, configs):
    """
    Train RL Restorer with enhanced features
    """
    training_model = model.module if hasattr(model, "module") else model
    training_model.eval()
    
    # Configuration
    use_enhanced_features = configs.sensitive_restorer.get("use_enhanced_features", False)
    num_classes = configs.dataloader.num_classes
    
    logger_info(logging.getLogger(), 
                f"[RL Restorer] Using enhanced features: {use_enhanced_features}")
    
    # Create RL Restorer with appropriate feature collector
    restorer, collector = create_rl_restorer_with_features(
        model=training_model,
        sensitive_info=prepared_stats,
        baseline_stats=prepared_stats,  # Use prepared_stats as baseline
        num_classes=num_classes,
        use_enhanced_features=use_enhanced_features,
        feature_dim=None,  # Auto-detect
        state_dim=configs.sensitive_restorer.get("rl_state_dim", 128),
        hidden_dim=configs.sensitive_restorer.get("rl_hidden_dim", 256),
        max_steps=configs.sensitive_restorer.get("rl_max_steps", 3),
        device=device,
    )
    
    # Log feature dimension
    logger_info(logging.getLogger(), 
                f"[RL Restorer] Feature dimension: {restorer.state_encoder.feature_encoder[0].in_features}")
    logger_info(logging.getLogger(), 
                f"[RL Restorer] Total parameters: {sum(p.numel() for p in restorer.parameters()):,}")
    
    # Create trainer
    trainer = RLRestorerTrainer(
        restorer=restorer,
        actor_lr=configs.sensitive_restorer.get("rl_actor_lr", 3e-4),
        critic_lr=configs.sensitive_restorer.get("rl_critic_lr", 3e-4),
        gamma=configs.sensitive_restorer.get("rl_gamma", 0.99),
        lambda_=configs.sensitive_restorer.get("rl_lambda", 0.95),
        entropy_coef=configs.sensitive_restorer.get("rl_entropy_coef", 0.01),
        value_coef=configs.sensitive_restorer.get("rl_value_coef", 0.5),
        max_grad_norm=configs.sensitive_restorer.get("max_grad_norm", 1.0),
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
    
    # Fault injector setup
    stage2_ber = float(configs.sensitive_restorer.get("stage2_ber", 4e-2))
    stage2_seed = configs.sensitive_restorer.get("stage2_seed", getattr(configs, "seed", 42))
    
    fault_injector = FaultInjector(
        model=training_model,
        mode="ber",
        ber=stage2_ber,
        enable_in_training=False,
        enable_in_inference=True,
        seed=stage2_seed,
        skip_first_last=configs.sensitive_restorer.get("skip_first_last", False),
        use_random_flip_in_training=True,
    )
    
    # BER configuration
    use_fixed_ber = configs.sensitive_restorer.get("use_fixed_ber", False)
    fixed_ber = configs.sensitive_restorer.get("fixed_ber", 3e-2)
    
    if use_fixed_ber:
        logger_info(logging.getLogger(), 
                   f"[RL Restorer] Using FIXED BER: {fixed_ber:.2e}")
    else:
        ber_min = configs.sensitive_restorer.get("ber_min", 1e-2)
        ber_max = configs.sensitive_restorer.get("ber_max", 1e-1)
        beta_alpha = configs.sensitive_restorer.get("beta_alpha", 2.0)
        beta_beta = configs.sensitive_restorer.get("beta_beta", 2.0)
        logger_info(logging.getLogger(), 
                   f"[RL Restorer] BER sampling: Beta({beta_alpha}, {beta_beta}) in [{ber_min:.2e}, {ber_max:.2e}]")
    
    # Training loop
    num_epochs = configs.sensitive_restorer.get("num_epochs", 50)
    
    for epoch in range(num_epochs):
        training_model.eval()
        restorer.train()
        
        epoch_start_time = time.time()
        total_samples = 0
        clean_correct = 0
        faulted_correct = 0
        restored_correct = 0
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_reward = 0.0
        total_entropy = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Sample BER
            if use_fixed_ber:
                effective_ber = fixed_ber
            else:
                beta_sample = np.random.beta(beta_alpha, beta_beta)
                effective_ber = ber_min + (ber_max - ber_min) * beta_sample
            
            fault_injector.ber = effective_ber
            
            # Get clean logits
            collector.clear()
            fault_injector.disable()
            with torch.no_grad():
                logits_clean = training_model(inputs)
                pred_clean = logits_clean.argmax(dim=1)
                clean_correct += (pred_clean == targets).sum().item()
            
            # Get faulted logits
            collector.clear()
            fault_injector.enable()
            with torch.no_grad():
                logits_faulted = training_model(inputs)
            fault_injector.disable()
            
            # Check for NaN/Inf in faulted logits
            if torch.isnan(logits_faulted).any() or torch.isinf(logits_faulted).any():
                logger_info(logging.getLogger(), "WARNING: NaN/Inf in faulted logits, skipping batch")
                continue
            
            pred_faulted = logits_faulted.argmax(dim=1)
            faulted_correct += (pred_faulted == targets).sum().item()
            
            # Extract features (automatically handles enhanced vs standard)
            features, _ = extract_features_for_rl_restorer(
                collector=collector,
                model=training_model,
                inputs=inputs,
                fault_injector=fault_injector,
                use_enhanced_features=use_enhanced_features,
                device=device,
            )
            
            if features is None:
                continue
            
            # Check for NaN/Inf in features
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger_info(logging.getLogger(), "WARNING: NaN/Inf in features, skipping batch")
                continue
            
            # Train RL Restorer
            metrics = trainer.train_step(
                logits_faulted=logits_faulted,
                features=features,
                logits_clean=logits_clean,
            )
            
            # Evaluate restored logits
            restorer.eval()
            with torch.no_grad():
                logits_restored, info = restorer(
                    logits_faulted=logits_faulted,
                    features=features,
                    training=False,
                    deterministic=True,
                )
                pred_restored = logits_restored.argmax(dim=1)
                restored_correct += (pred_restored == targets).sum().item()
            
            # Accumulate metrics
            total_actor_loss += metrics['actor_loss']
            total_critic_loss += metrics['critic_loss']
            total_reward += metrics['mean_reward']
            total_entropy += metrics['entropy']
            num_batches += 1
            total_samples += inputs.size(0)
            
            # Print progress
            if batch_idx % 100 == 0:
                logger_info(logging.getLogger(),
                           f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Actor Loss: {metrics['actor_loss']:.4f}, "
                           f"Critic Loss: {metrics['critic_loss']:.4f}, "
                           f"Reward: {metrics['mean_reward']:.4f}, "
                           f"Entropy: {metrics['entropy']:.4f}")
        
        # Update learning rate
        trainer.update_schedulers()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        clean_acc = 100.0 * clean_correct / total_samples
        faulted_acc = 100.0 * faulted_correct / total_samples
        restored_acc = 100.0 * restored_correct / total_samples
        
        avg_actor_loss = total_actor_loss / max(num_batches, 1)
        avg_critic_loss = total_critic_loss / max(num_batches, 1)
        avg_reward = total_reward / max(num_batches, 1)
        avg_entropy = total_entropy / max(num_batches, 1)
        
        logger_info(logging.getLogger(),
                   f"\nEpoch {epoch} Summary:")
        logger_info(logging.getLogger(),
                   f"  Clean Acc: {clean_acc:.2f}%")
        logger_info(logging.getLogger(),
                   f"  Faulted Acc: {faulted_acc:.2f}%")
        logger_info(logging.getLogger(),
                   f"  Restored Acc: {restored_acc:.2f}%")
        logger_info(logging.getLogger(),
                   f"  Improvement: {restored_acc - faulted_acc:.2f}%")
        logger_info(logging.getLogger(),
                   f"  Avg Actor Loss: {avg_actor_loss:.4f}")
        logger_info(logging.getLogger(),
                   f"  Avg Critic Loss: {avg_critic_loss:.4f}")
        logger_info(logging.getLogger(),
                   f"  Avg Reward: {avg_reward:.4f}")
        logger_info(logging.getLogger(),
                   f"  Avg Entropy: {avg_entropy:.4f}")
        logger_info(logging.getLogger(),
                   f"  Learning Rates: {trainer.get_lr()}")
        logger_info(logging.getLogger(),
                   f"  Time: {epoch_time:.2f}s")
    
    # Cleanup
    collector.remove()
    fault_injector.disable()
    
    return restorer, trainer


def main():
    args = parse_args()
    configs = get_config(args.config)
    set_global_seed(getattr(configs, "seed", 42))
    
    script_dir = Path(__file__).parent
    logger, log_dir, pymonitor, tbmonitor = init_logging(configs, script_dir, args.output_dir)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger_info(logger, f"Using device: {device}")
    
    # Load model
    model = create_model(configs)
    model = model.to(device)
    
    # Load stage1 checkpoint
    checkpoint = load_checkpoint(args.stage1_ckpt, device)
    model.load_state_dict(checkpoint["model"])
    logger_info(logger, f"Loaded stage1 checkpoint from {args.stage1_ckpt}")
    
    # Preprocess model (quantization, etc.)
    model = preprocess_model(model, configs)
    
    # Collect sensitive features
    train_loader, _ = init_dataloader(configs)
    sensitive_info = collect_gradient_sensitivity(model, train_loader, configs, device)
    baseline_stats = compute_activation_baseline(model, train_loader, sensitive_info, configs, device)
    
    # Prepare statistics
    prepared_stats, feature_dim = prepare_stats_for_device(baseline_stats, device)
    
    # Train RL Restorer
    restorer, trainer = train_rl_restorer(
        model, train_loader, prepared_stats, device, configs
    )
    
    # Save checkpoint
    checkpoint_path = log_dir / "rl_restorer_final.pth"
    torch.save({
        'restorer': restorer.state_dict(),
        'config': configs,
    }, checkpoint_path)
    logger_info(logger, f"Saved RL Restorer to {checkpoint_path}")


if __name__ == '__main__':
    main()


