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
from util.qat import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint, save_checkpoint
from util.sensitive_features import collect_gradient_sensitivity, compute_activation_baseline
from util.sensitive_restorer import SensitiveActivationCollector, SensitiveChannelRestorer
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


def train_restorer(model, train_loader, criterion, prepared_stats, feature_dim, device, configs):
    training_model = model.module if hasattr(model, "module") else model
    training_model.eval()
    for param in training_model.parameters():
        param.requires_grad_(False)

    restorer = SensitiveChannelRestorer(
        feature_dim,
        configs.dataloader.num_classes,
        hidden_dim=configs.sensitive_restorer.get("restorer_hidden_dim", 128),
    ).to(device)
    restorer_optimizer = torch.optim.Adam(
        restorer.parameters(),
        lr=configs.sensitive_restorer.get("restorer_lr", 1e-3),
        weight_decay=configs.sensitive_restorer.get("restorer_weight_decay", 0.0),
    )

    mse_weight = configs.sensitive_restorer.get("mse_weight", 0.5)
    stage2_epochs = configs.sensitive_restorer.get("stage2_epochs", 0)
    stage2_ber = float(configs.sensitive_restorer.get("stage2_ber", 4e-2))
    stage2_seed = configs.sensitive_restorer.get("stage2_seed", getattr(configs, "seed", 42))

    fault_injector = FaultInjector(
        model=training_model,
        mode="ber",
        ber=stage2_ber,
        enable_in_training=True,
        enable_in_inference=False,
        seed=stage2_seed,
        skip_first_last=configs.sensitive_restorer.get("skip_first_last", False),
    )
    fault_injector.disable()

    collector = SensitiveActivationCollector(training_model, prepared_stats)

    epoch_times = []
    start_time = time.time()
    for epoch in range(stage2_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        sample_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            collector.clear()
            fault_injector.disable()
            with torch.no_grad():
                logits_clean = training_model(inputs)

            collector.clear()
            fault_injector.enable()
            logits_faulted = training_model(inputs)
            fault_injector.disable()

            features = collector.build_feature_vector(device)
            if features is None:
                continue

            restorer.train()
            restorer_optimizer.zero_grad()
            logits_restored, gate = restorer(logits_faulted, features)
            loss = criterion(logits_restored, targets)
            if mse_weight > 0:
                loss = loss + mse_weight * F.mse_loss(logits_restored, logits_clean.detach())
            loss.backward()
            restorer_optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

        avg_loss = running_loss / max(1, sample_count)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        recent_epochs = min(5, len(epoch_times))
        avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs
        remaining_epochs = stage2_epochs - epoch - 1
        remaining_time = avg_epoch_time * remaining_epochs
        eta = datetime.now() + timedelta(seconds=remaining_time)

        logger_info(logging.getLogger(), f"[Stage2] Epoch {epoch}: loss={avg_loss:.4f}")
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
    return restorer, restorer_optimizer


def main():
    args = parse_args()
    script_dir = Path.cwd()
    configs = get_config(default_file=args.config)
    configs.output_dir = args.output_dir or configs.output_dir

    set_global_seed(seed=getattr(configs, "seed", 42))
    logger, log_dir, pymonitor, tbmonitor = init_logging(configs, script_dir)

    copy_code(logger, src=str(script_dir), dst=os.path.join(log_dir, "code"))

    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.cuda()
    target_model = ModelEma(model, decay=configs.ema_decay)

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

    restorer, restorer_optimizer = train_restorer(
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
