"""
Knowledge Distillation for Sensitive Channel Restorer
Train a small student restorer by distilling knowledge from a large teacher restorer
"""
import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

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
from util.fault_injector import FaultInjector


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge distillation for sensitive channel restorer")
    parser.add_argument("--config", required=True, help="Path to stage2 config yaml")
    parser.add_argument("--stage1_ckpt", required=True, help="Checkpoint from stage1 training")
    parser.add_argument("--teacher_ckpt", required=True, help="Checkpoint with trained teacher restorer")
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
    prepared = {}
    feature_dim = 0
    for name, entry in raw_stats.items():
        indices = entry.get("indices", [])
        if not indices:
            continue
        prepared[name] = {
            "indices": indices,
            "mean": entry["mean"].to(device),
            "std": entry["std"].to(device),
        }
        feature_dim += len(indices)
    return prepared, feature_dim


def train_student_restorer(
    model, train_loader, criterion, prepared_stats, feature_dim, device, configs, teacher_restorer
):
    training_model = model.module if hasattr(model, "module") else model
    training_model.eval()  # Freeze main model
    
    # Student restorer (small)
    student_hidden_dim = configs.sensitive_restorer.get("student_hidden_dim", 120)
    student_restorer = SensitiveChannelRestorer(
        feature_dim=feature_dim,
        num_classes=configs.dataloader.num_classes,
        hidden_dim=student_hidden_dim
    ).to(device)
    
    # Teacher restorer (large, frozen)
    teacher_restorer.eval()
    for param in teacher_restorer.parameters():
        param.requires_grad = False
    
    restorer_lr = configs.sensitive_restorer.get("restorer_lr", 0.001)
    restorer_optimizer = torch.optim.Adam(student_restorer.parameters(), lr=restorer_lr)
    
    stage2_epochs = configs.sensitive_restorer.get("distill_epochs", 100)
    stage2_ber = float(configs.sensitive_restorer.get("stage2_ber", 4e-2))
    stage2_seed = configs.sensitive_restorer.get("stage2_seed", getattr(configs, "seed", 42))
    
    # Distillation parameters
    distill_weight = configs.sensitive_restorer.get("distill_weight", 0.5)
    temperature = configs.sensitive_restorer.get("temperature", 4.0)
    mse_weight = configs.sensitive_restorer.get("mse_weight", 0.5)
    direction_weight = configs.sensitive_restorer.get("direction_weight", 0.5)
    
    # Multi-BER training setup
    stage2_mix_bers = configs.sensitive_restorer.get("stage2_mix_bers", [stage2_ber])
    stage2_mix_prob = configs.sensitive_restorer.get("stage2_mix_prob", 0.0)
    
    fault_injector = FaultInjector(
        model=training_model,
        mode="ber",
        ber=stage2_ber,
        enable_in_training=True,
        enable_in_inference=False,
        seed=stage2_seed,
        skip_first_last=configs.sensitive_restorer.get("skip_first_last", False),
        use_random_flip_in_training=True,  # Use completely random bit-flip during restorer training
    )
    fault_injector.disable()
    
    collector = SensitiveActivationCollector(training_model, prepared_stats)
    
    epoch_times = []
    start_time = time.time()
    trigger_index = 0
    
    logger_info(logging.getLogger(), f"Starting knowledge distillation:")
    logger_info(logging.getLogger(), f"  Teacher params: {sum(p.numel() for p in teacher_restorer.parameters()):,}")
    logger_info(logging.getLogger(), f"  Student params: {sum(p.numel() for p in student_restorer.parameters()):,}")
    logger_info(logging.getLogger(), f"  Distill weight: {distill_weight}, Temperature: {temperature}")
    
    for epoch in range(stage2_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_distill_loss = 0.0
        running_mse_loss = 0.0
        running_dir_loss = 0.0
        sample_count = 0
        
        clean_correct = 0
        faulted_correct = 0
        student_correct = 0
        teacher_correct = 0
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Randomly sample BER for each batch in range [1e-2, 1e-1]
            import random
            effective_ber = random.uniform(1e-2, 1e-1)
            trigger_index += 1
            
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
            
            features = collector.build_feature_vector(device)
            if features is None:
                continue
            
            # Teacher forward (frozen)
            with torch.no_grad():
                logits_teacher, _ = teacher_restorer(logits_faulted, features)
                pred_teacher = logits_teacher.argmax(dim=1)
                teacher_correct += (pred_teacher == targets).sum().item()
            
            # Student forward
            student_restorer.train()
            restorer_optimizer.zero_grad()
            logits_student, _ = student_restorer(logits_faulted, features)
            pred_student = logits_student.argmax(dim=1)
            student_correct += (pred_student == targets).sum().item()
            
            # Loss computation
            # 1. CE loss
            ce_loss = F.cross_entropy(logits_student, targets)
            
            # 2. Knowledge distillation loss (KL divergence)
            student_log_probs = F.log_softmax(logits_student / temperature, dim=1)
            teacher_probs = F.softmax(logits_teacher / temperature, dim=1)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            
            # 3. MSE loss (student vs clean)
            mse_loss = 0.0
            if mse_weight > 0:
                mse_loss = F.mse_loss(logits_student, logits_clean.detach())
            
            # 4. Direction loss
            direction_loss = 0.0
            if direction_weight > 0:
                correction_pred = logits_student - logits_faulted.detach()
                correction_target = logits_clean.detach() - logits_faulted.detach()
                target_delta_pred = correction_pred.gather(1, targets.unsqueeze(1)).squeeze(1)
                target_delta_target = correction_target.gather(1, targets.unsqueeze(1)).squeeze(1)
                direction_mask = target_delta_target.abs() > 0.15
                if direction_mask.any():
                    dir_loss_vec = ((target_delta_pred - target_delta_target) ** 2) * direction_mask.float()
                    direction_loss = dir_loss_vec.mean()
            
            # Combined loss
            loss = (1 - distill_weight) * ce_loss + distill_weight * distill_loss + \
                   mse_weight * mse_loss + direction_weight * direction_loss
            
            loss.backward()
            restorer_optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_ce_loss += ce_loss.item() * inputs.size(0)
            running_distill_loss += distill_loss.item() * inputs.size(0)
            running_mse_loss += mse_loss.item() * inputs.size(0) if mse_weight > 0 else 0
            running_dir_loss += direction_loss.item() * inputs.size(0) if direction_weight > 0 else 0
            sample_count += inputs.size(0)
            total_samples += inputs.size(0)
        
        avg_loss = running_loss / max(1, sample_count)
        avg_ce_loss = running_ce_loss / max(1, sample_count)
        avg_distill_loss = running_distill_loss / max(1, sample_count)
        avg_mse_loss = running_mse_loss / max(1, sample_count) if mse_weight > 0 else 0.0
        avg_dir_loss = running_dir_loss / max(1, sample_count) if direction_weight > 0 else 0.0
        
        clean_acc = 100.0 * clean_correct / max(1, total_samples)
        faulted_acc = 100.0 * faulted_correct / max(1, total_samples)
        student_acc = 100.0 * student_correct / max(1, total_samples)
        teacher_acc = 100.0 * teacher_correct / max(1, total_samples)
        student_improvement = student_acc - faulted_acc
        teacher_improvement = teacher_acc - faulted_acc
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        recent_epochs = min(5, len(epoch_times))
        avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs
        remaining_epochs = stage2_epochs - epoch - 1
        remaining_time = avg_epoch_time * remaining_epochs
        eta = datetime.now() + timedelta(seconds=remaining_time)
        
        logger_info(logging.getLogger(), 
                   f"[Distill] Epoch {epoch}: total_loss={avg_loss:.4f} "
                   f"(CE={avg_ce_loss:.4f}, Distill={avg_distill_loss:.4f}, "
                   f"MSE={avg_mse_loss:.4f}, Dir={avg_dir_loss:.4f})")
        logger_info(logging.getLogger(),
                   f"  Acc - Clean: {clean_acc:.2f}% | Faulted: {faulted_acc:.2f}% | "
                   f"Teacher: {teacher_acc:.2f}% (+{teacher_improvement:+.2f}%) | "
                   f"Student: {student_acc:.2f}% (+{student_improvement:+.2f}%)")
        logger_info(logging.getLogger(),
                   f"  ⏱️ Epoch耗时: {epoch_time:.1f}s | 剩余Epoch: {remaining_epochs}")
        if remaining_epochs > 0:
            logger_info(logging.getLogger(),
                       f"  📅 预估剩余时间: {remaining_time/60:.1f}分钟")
    
    collector.remove()
    fault_injector.disable()
    return student_restorer, restorer_optimizer


def main():
    import sys
    args = parse_args()
    script_dir = Path.cwd()
    
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
    
    device = torch.device(args.device)
    
    # Load main model
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    
    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Load stage1 checkpoint
    load_checkpoint(model, args.stage1_ckpt, device, strict=False)
    
    # Collect sensitivity and baseline (same as stage2)
    logger_info(logger, "Collecting gradient sensitivity...")
    sensitive_channels = collect_gradient_sensitivity(
        model, train_loader, criterion, device=device,
        topk_per_layer=configs.sensitive_restorer.get('topk_per_layer', 8),
        max_batches=configs.sensitive_restorer.get('sensitivity_batches', 50),
    )
    
    logger_info(logger, "Computing activation baseline...")
    baseline_stats = compute_activation_baseline(
        model, train_loader, sensitive_channels, device=device,
        max_batches=configs.sensitive_restorer.get('baseline_batches', 50),
    )
    
    prepared_stats, feature_dim = prepare_stats_for_device(baseline_stats, device)
    
    # Load teacher restorer from checkpoint
    logger_info(logger, f"Loading teacher restorer from {args.teacher_ckpt}")
    teacher_checkpoint = torch.load(args.teacher_ckpt, map_location=device)
    if 'sensitive_restorer' not in teacher_checkpoint:
        raise ValueError("Teacher checkpoint does not contain sensitive_restorer!")
    
    teacher_hidden_dim = configs.sensitive_restorer.get("teacher_hidden_dim", 256)
    teacher_restorer = SensitiveChannelRestorer(
        feature_dim=feature_dim,
        num_classes=configs.dataloader.num_classes,
        hidden_dim=teacher_hidden_dim
    ).to(device)
    teacher_restorer.load_state_dict(teacher_checkpoint['sensitive_restorer'])
    logger_info(logger, f"✓ Teacher restorer loaded ({sum(p.numel() for p in teacher_restorer.parameters()):,} params)")
    
    # Train student restorer
    student_restorer, student_optimizer = train_student_restorer(
        model, train_loader, criterion, prepared_stats, feature_dim, device, configs, teacher_restorer
    )
    
    # Save checkpoint
    checkpoint_path = os.path.join(log_dir, "distilled_student_checkpoint.pth.tar")
    save_checkpoint(
        model, student_optimizer, None, None, None, None,
        epoch=configs.sensitive_restorer.get("distill_epochs", 100),
        is_best=False,
        checkpoint_path=checkpoint_path,
        sensitive_restorer=student_restorer,
        sensitive_optimizer=student_optimizer,
    )
    logger_info(logger, f"✓ Student restorer saved to {checkpoint_path}")


if __name__ == "__main__":
    main()

