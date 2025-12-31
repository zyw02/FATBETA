import logging
import torch
import yaml
import os
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn.parallel import DistributedDataParallel

from model import create_model
from util import (
    ProgressMonitor,
    TensorBoardMonitor,
    get_config,
    init_logger,
    set_global_seed,
    setup_print,
    load_checkpoint,
    save_checkpoint,
    preprocess_model,
    init_dataloader,
)
from util.utils import copy_code, create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend
from util.model_ema import ModelEma
from quan import find_modules_to_quantize, replace_module_by_names
from util.mpq import switch_bit_width
from process_nude import train, validate, PerformanceScoreboard


def init_logger_and_monitor(configs, script_dir):
    if is_master():
        output_dir = script_dir / configs.output_dir
        output_dir.mkdir(exist_ok=True)

        log_dir = init_logger(configs.name, output_dir, script_dir / "logging.conf")
        logger = logging.getLogger()

        with open(log_dir / "configs.yaml", "w") as yaml_file:
            yaml.safe_dump(configs, yaml_file)

        pymonitor = ProgressMonitor(logger)
        tbmonitor = TensorBoardMonitor(logger, log_dir)
        return logger, log_dir, pymonitor, tbmonitor
    else:
        return None, None, None, None


def _max_target_bit(configs) -> int:
    target_bits = getattr(configs, "target_bits", [6, 5, 4, 3, 2])
    if isinstance(target_bits, (list, tuple)) and len(target_bits) > 0:
        return int(max(target_bits))
    return int(target_bits) if target_bits is not None else 6


def main():
    script_dir = Path.cwd()
    configs = get_config(default_file=script_dir / "template.yaml")

    assert configs.training_device == "gpu", "NOT SUPPORT CPU TRAINING NOW"
    init_dist_nccl_backend(configs)

    logger, log_dir, pymonitor, tbmonitor = init_logger_and_monitor(configs, script_dir)
    monitors = [pymonitor, tbmonitor]
    setup_print(is_master=(configs.local_rank == 0))

    # Backup code for reproducibility
    if not configs.eval and not configs.search:
        code_dst = os.path.join(log_dir, "code")
        copy_code(logger, src=str(script_dir), dst=code_dst)

    # NUDE: keep training deterministic
    set_global_seed(seed=0)

    # NUDE+SRQAT: allow SR-QAT if explicitly enabled in config, otherwise disable
    # Check if scale_penalty should be enabled
    srqat_enabled = False
    if hasattr(configs, "scale_penalty"):
        srqat_enabled = getattr(configs.scale_penalty, "enabled", False)
        if not srqat_enabled:
            configs.scale_penalty.enabled = False
            configs.scale_penalty.lambda_scale = 0.0
        else:
            logger_info(logger, f"[NUDE+SRQAT] SR-QAT enabled with lambda_scale={configs.scale_penalty.lambda_scale}")
    
    # Orthogonality is still disabled in nude mode
    if hasattr(configs, "orthogonality_penalty"):
        configs.orthogonality_penalty.enabled = False
        configs.orthogonality_penalty.lambda_ortho = 0.0
    if hasattr(configs, "fault_aware_training"):
        configs.fault_aware_training.enabled = False

    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model.eval()

    # DDP if distributed
    if configs.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[configs.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    # Data
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)

    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(model, configs)

    # Warm-up forward for output_size init etc.
    input_size = 32 if configs.dataloader.dataset in ["cifar10", "cifar100"] else 224
    model(torch.randn((1, 3, input_size, input_size)).cuda())

    target_model = ModelEma(model, decay=configs.ema_decay)
    start_epoch = 0

    if configs.resume.path and os.path.exists(configs.resume.path):
        model, start_epoch, _ = load_checkpoint(
            model,
            configs.resume.path,
            "cuda",
            lean=configs.resume.lean,
            optimizer=optimizer,
            override_optim=configs.eval,
            lr_scheduler=lr_scheduler,
            lr_scheduler_q=lr_scheduler_q,
            optimizer_q=optimizer_q,
        )

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0.0 else torch.nn.CrossEntropyLoss().cuda()

    max_bit = _max_target_bit(configs)
    logger_info(logger, f"[NUDE] Force training only max(target_bits)={max_bit} (no nr_random_sample)")
    switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
    switch_bit_width(target_model.ema, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

    perf = PerformanceScoreboard(configs.log.num_best_scores)

    if configs.eval:
        acc = validate(test_loader, target_model.ema, criterion, -1, monitors, configs, nr_random_sample=0)
        logger_info(logger, f"[NUDE][EVAL] Top1: {acc:.3f}")
        return

    for epoch in range(start_epoch, configs.epochs):
        if configs.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_top1, t_top5, t_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            monitors,
            configs,
            model_ema=target_model,
            nr_random_sample=0,
            optimizer_q=optimizer_q,
        )

        # Validate on EMA model
        v_top1 = validate(test_loader, target_model.ema, criterion, epoch, monitors, configs, nr_random_sample=0)
        perf.update(v_top1, 0.0, epoch)

        logger_info(
            logger,
            f"[NUDE][EPOCH] {epoch}/{configs.epochs} "
            f"Train: Top1={t_top1:.2f} Top5={t_top5:.2f} Loss={t_loss:.4f} | "
            f"Val(EMA): Top1={v_top1:.2f}",
        )

        # Save only ONE checkpoint (overwrite) to save disk space.
        # Use repo's canonical save_checkpoint() signature (util/checkpoint.py).
        # NOTE: We do NOT save a separate "best" checkpoint to avoid extra files.
        save_checkpoint(
            epoch=epoch + 1,
            arch=configs.arch,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            extras={},
            is_best=False,
            name=configs.name,
            output_dir=str(log_dir),
            lr_scheduler=lr_scheduler,
            lr_scheduler_q=lr_scheduler_q,
            optimizer_q=optimizer_q,
        )

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)


if __name__ == "__main__":
    main()


