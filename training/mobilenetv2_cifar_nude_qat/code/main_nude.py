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
from util.fault_injector import FaultInjector
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
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
    except Exception:
        pass

    # Backup code for reproducibility
    if is_master() and not configs.eval and not configs.search:
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

    bfat_cfg = getattr(configs, 'bfat', None)
    use_bfat = bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False)

    # Create model
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model.cuda()

    # Create optimizer and scheduler (before DDP wrapping)
    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(model, configs)

    # 1. Load Checkpoint (into plain model)
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

    # 2. Switch to MAX bit-width BEFORE any forward pass or EMA creation
    max_bit = _max_target_bit(configs)
    logger_info(logger, f"[NUDE] Initializing model at max target bit-width: {max_bit}")
    switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

    # 3. Warm-up forward (initializes LSQ step size 's' based on real weights at correct bit-width)
    # This also handles output_size init for some models
    input_size = 32 if configs.dataloader.dataset in ["cifar10", "cifar100"] else 224
    with torch.no_grad():
        # 🌟 CRITICAL: Use SAME seed on all ranks for warm-up to ensure 
        # identical LsqQuan 's' initialization across GPUs
        torch.manual_seed(0)
        model(torch.randn((1, 3, input_size, input_size)).cuda())
    
    # 🌟 CRITICAL: Broadcast Rank 0's initialized state (including 's' and buffers)
    # to all other ranks to ensure 100% consistency before EMA creation.
    if configs.distributed:
        logger_info(logger, "[NUDE] Broadcasting initialized parameters and buffers to all ranks...")
        for p in model.parameters():
            torch.distributed.broadcast(p, src=0)
        for b in model.buffers():
            torch.distributed.broadcast(b, src=0)

    # 4. Initialize EMA AFTER loading and warm-up
    # Now EMA starts with correct weights AND correctly initialized quantization parameters
    ema_resume = configs.resume.path if (configs.resume.path and os.path.exists(configs.resume.path) and not configs.resume.lean) else ''
    target_model = ModelEma(model, decay=configs.ema_decay, resume=ema_resume)
    # Ensure EMA is also at max bit-width
    switch_bit_width(target_model.ema, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

    # 5. FaultInjector Setup for BFAT (needs unwrapped model)
    fault_injector = None
    if use_bfat:
        ber_raw = getattr(bfat_cfg, 'ber', 1e-2)
        ber_msb = getattr(bfat_cfg, 'ber_msb', None)
        ber_secondary_msb = getattr(bfat_cfg, 'ber_secondary_msb', None)
        skip_msb = getattr(bfat_cfg, 'skip_msb', False)
        only_msb = getattr(bfat_cfg, 'only_msb', False)
        all_bits = getattr(bfat_cfg, 'all_bits', False)
        bfat_bit_index = getattr(bfat_cfg, 'bit_index', None)
        bfat_dual_bit = getattr(bfat_cfg, 'dual_bit', False)
        exclude_layers = getattr(bfat_cfg, 'exclude_layers', None)

        ber = float(ber_raw)
        
        fault_injector = FaultInjector(
            model=model,
            mode="ber",
            ber=ber,
            enable_in_training=True,
            enable_in_inference=False,
            seed=getattr(configs, 'seed', 42),
            exclude_layers=exclude_layers,
            skip_msb=skip_msb,
            only_msb=only_msb,
            all_bits=all_bits,
            bfat_bit_index=bfat_bit_index,
            bfat_dual_bit=bfat_dual_bit,
            ber_msb=ber_msb,
            ber_secondary_msb=ber_secondary_msb
        )
        logger_info(logger, '=' * 80)
        logger_info(logger, f'🚀 [NUDE] BFAT ENABLED')
        logger_info(logger, '=' * 80)
        logger_info(logger, f'  ✅ FaultInjector initialized for BFAT')
    
    # 6. DDP wrapping (LAST STEP)
    if configs.distributed:
        model = DistributedDataParallel(model, device_ids=[configs.local_rank], find_unused_parameters=True)

    # Data
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)
    logger_info(logger, f'[DEBUG] Dataloaders initialized: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}')

    # Build a master-only eval loader once to avoid per-epoch worker spawn overhead
    eval_loader_master = test_loader

    enable_linear_scaling_rule = False
    if enable_linear_scaling_rule and configs.distributed:
        configs.lr = configs.lr * configs.world_size * configs.dataloader.batch_size / 512
        configs.min_lr = configs.min_lr * \
            configs.world_size * configs.dataloader.batch_size / 512
        configs.warmup_lr = configs.warmup_lr * \
            configs.world_size * configs.dataloader.batch_size / 512

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0.0 else torch.nn.CrossEntropyLoss().cuda()

    max_bit = _max_target_bit(configs)
    logger_info(logger, f"[NUDE] Force training only max(target_bits)={max_bit} (no nr_random_sample)")
    switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
    switch_bit_width(target_model.ema, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

    perf = PerformanceScoreboard(configs.log.num_best_scores)

    if configs.eval:
        eval_model = target_model.ema.module if hasattr(target_model.ema, 'module') else target_model.ema
        acc = validate(eval_loader_master, eval_model, criterion, -1, monitors, configs, nr_random_sample=0)
        if is_master():
            logger_info(logger, f"[NUDE][EVAL] Top1: {acc:.3f}")
        if configs.distributed:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        return

    # 训练时间预估相关变量
    import time
    from datetime import datetime, timedelta
    epoch_times = []  # 记录每个epoch的时间
    training_start_time = time.time()  # 训练开始时间

    for epoch in range(start_epoch, configs.epochs):
        epoch_start_time = time.time()  # 当前epoch开始时间
        if configs.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 旋转故障注入器的种子，增加训练多样性 (同步 main_normal.py 的做法)
        if fault_injector is not None:
            initial_seed = getattr(configs, 'seed', 42)
            fault_injector.seed = initial_seed + epoch
            logger_info(logger, f'🎲 Epoch {epoch}: FaultInjector seed rotated to {fault_injector.seed}')

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
            fault_injector=fault_injector,
        )

        # Validate on EMA model (all ranks run; master updates scoreboard)
        eval_model = target_model.ema.module if hasattr(target_model.ema, 'module') else target_model.ema
        v_top1 = validate(eval_loader_master, eval_model, criterion, epoch, monitors, configs, nr_random_sample=0)
        if is_master():
            perf.update(v_top1, 0.0, epoch)
        if configs.distributed:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        # 计算epoch时间并记录
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # 计算平均epoch时间（使用最近5个epoch的平均值）
        recent_epochs = min(5, len(epoch_times))
        avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs

        # 计算剩余epoch数和预估完成时间
        remaining_epochs = configs.epochs - epoch - 1
        estimated_remaining_time = avg_epoch_time * remaining_epochs

        def format_time(seconds):
            """将秒数格式化为易读的时间字符串"""
            if seconds < 60:
                return f"{seconds:.1f}秒"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}分{secs}秒"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours}小时{minutes}分{secs}秒"

        estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        estimated_completion_str = estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S")

        if is_master():
            curr_lr = optimizer.param_groups[0]['lr']
            curr_qlr = optimizer_q.param_groups[0]['lr'] if optimizer_q is not None else 0.0
            logger_info(
                logger,
                f"[NUDE][EPOCH] {epoch}/{configs.epochs} "
                f"Train: Top1={t_top1:.2f} Top5={t_top5:.2f} Loss={t_loss:.4f} | "
                f"Val(EMA): Top1={v_top1:.2f} | "
                f"LR: {curr_lr:.6f} QLR: {curr_qlr:.6f}",
            )
        logger_info(logger, f'  ⏱️  本Epoch耗时: {format_time(epoch_time)} | 平均Epoch耗时: {format_time(avg_epoch_time)} | 剩余Epoch数: {remaining_epochs}')
        if remaining_epochs > 0:
            logger_info(logger, f'  📅 预估剩余时间: {format_time(estimated_remaining_time)} | 预估完成时间: {estimated_completion_str}')
        else:
            logger_info(logger, f'  ✅ 训练完成！总耗时: {format_time(time.time() - training_start_time)}')

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
