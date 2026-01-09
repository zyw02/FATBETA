import logging
import torch
import yaml
import os
import time
from datetime import datetime, timedelta
import sys
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
from util.utils import copy_code, create_optimizer_and_lr_scheduler, analyze_gradient_alignment
from util.dist import logger_info, is_master, init_dist_nccl_backend
from util.model_ema import ModelEma
from util.weight_schd import CosineSched
from quan import find_modules_to_quantize, replace_module_by_names
from util.mpq import switch_bit_width
from process_g import train, validate, PerformanceScoreboard

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

def main():
    script_dir = Path.cwd()
    configs = get_config(default_file=script_dir / "template.yaml")
    init_dist_nccl_backend(configs)

    logger, log_dir, pymonitor, tbmonitor = init_logger_and_monitor(configs, script_dir)
    monitors = [pymonitor, tbmonitor]
    setup_print(is_master=(configs.local_rank == 0))

    if not configs.eval and not configs.search:
        copy_code(logger, src=str(script_dir), dst=os.path.join(log_dir, "code"))

    set_global_seed(seed=0)

    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model.eval()

    if configs.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[configs.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)
    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(model, configs)

    # Warm-up
    input_size = 32 if configs.dataloader.dataset in ["cifar10", "cifar100"] else 224
    model(torch.randn((1, 3, input_size, input_size)).cuda())

    target_model = ModelEma(model, decay=configs.ema_decay)
    start_epoch = 0

    if configs.resume.path and os.path.exists(configs.resume.path):
        # 使用 override_optim=True 确保不加载旧的优化器状态
        # 并显式将 start_epoch 重置为 0，因为这是全新的 QAT 实验
        model, _, _ = load_checkpoint(
            model, configs.resume.path, "cuda", strict=False, 
            optimizer=optimizer, override_optim=True,
            lr_scheduler=lr_scheduler, optimizer_q=optimizer_q
        )
        start_epoch = 0
        logger_info(logger, f"Loaded pre-trained weights from {configs.resume.path}. Starting fresh experiment from Epoch 0.")

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0.0 else torch.nn.CrossEntropyLoss().cuda()
    perf = PerformanceScoreboard(configs.log.num_best_scores)

    # 持久化相似度分析结果
    current_similarities = None

    # 初始化辅助损失退火策略 (同步 main_normal.py)
    annealing_schedule = CosineSched(
        start_step=len(train_loader) * 40,
        max_step=len(train_loader) * configs.epochs,
        eta_start=0,
        eta_end=0.1
    )
    logger_info(logger, "[G-PIPELINE] Annealing schedule created (Starts at Epoch 40)")

    # 训练时间预估相关变量
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(start_epoch, configs.epochs):
        epoch_start_time = time.time()
        if configs.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 前 5 轮 (epoch < 5) 不启用相似度指导采样，强制为 None 以触发随机采样
        training_similarities = current_similarities if epoch >= 5 else None
        if epoch < 5:
            logger_info(logger, f"Epoch {epoch}: Using standard random sampling (Warm-up phase)")
        else:
            logger_info(logger, f"Epoch {epoch}: Using gradient-alignment aware sampling")

        # 训练：传入指导数据
        t_top1, t_top5, t_loss = train(
            train_loader, model, criterion, optimizer, epoch, monitors, configs,
            similarity_results=training_similarities,
            nr_random_sample=getattr(configs, 'num_random_path', 2),
            optimizer_q=optimizer_q,
            annealing_schedule=annealing_schedule
        )

        # 验证
        v_top1 = validate(test_loader, target_model.ema, criterion, epoch, monitors, configs)
        perf.update(v_top1, 0.0, epoch)

        # 时间预估计算
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        avg_epoch_time = sum(epoch_times[-min(5, len(epoch_times)):]) / min(5, len(epoch_times))
        remaining_epochs = configs.epochs - epoch - 1
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        def format_time(seconds):
            if seconds < 60: return f"{seconds:.1f}s"
            elif seconds < 3600: return f"{int(seconds // 60)}m{int(seconds % 60)}s"
            else: return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"

        estimated_completion_str = (datetime.now() + timedelta(seconds=estimated_remaining_time)).strftime("%Y-%m-%d %H:%M:%S")

        logger_info(logger, f"Epoch {epoch} Summary - Top1: {v_top1:.2f}% | Time: {format_time(epoch_time)} | Avg: {format_time(avg_epoch_time)}")
        if remaining_epochs > 0:
            logger_info(logger, f"  ⏱️  Remaining: {format_time(estimated_remaining_time)} | ETA: {estimated_completion_str}")
        else:
            logger_info(logger, f"  ✅ Training Finished! Total Time: {format_time(time.time() - training_start_time)}")

        # 核心：在每轮训练结束后，计算新的相似度数据，用于指导下一轮
        logger_info(logger, f"Evaluating gradient alignment for Epoch {epoch}...")
        current_similarities = analyze_gradient_alignment(
            model=model,
            loader=train_loader,
            criterion=criterion,
            target_bits=configs.target_bits,
            configs=configs,
            logger=logger,
            num_batches=4 # 使用 4 个 batch 累积
        )

        save_checkpoint(
            epoch=epoch + 1, arch=configs.arch, model=model, target_model=target_model,
            optimizer=optimizer, extras={}, is_best=False, name=configs.name,
            output_dir=str(log_dir), lr_scheduler=lr_scheduler, optimizer_q=optimizer_q
        )

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

if __name__ == "__main__":
    main()
