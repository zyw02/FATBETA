import logging
import torch
import yaml
import os
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from model import create_model
from util import (ProgressMonitor, TensorBoardMonitor, 
                  get_config, init_logger, set_global_seed, setup_print, load_checkpoint, save_checkpoint, preprocess_model, init_dataloader)
from util.utils import copy_code, create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend, tbmonitor_add_scalars
from process_pretrain import train, validate, PerformanceScoreboard

def init_logger_and_monitor(configs, script_dir):
    if is_master():
        output_dir = script_dir / configs.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir = init_logger(configs.name, output_dir,
                              script_dir / 'logging.conf')
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
    configs = get_config(default_file=script_dir / 'template.yaml')

    assert configs.training_device == 'gpu', 'NOT SUPPORT CPU TRAINING NOW'

    init_dist_nccl_backend(configs)

    logger, log_dir, pymonitor, tbmonitor = init_logger_and_monitor(
        configs, script_dir)
    monitors = [pymonitor, tbmonitor]

    setup_print(is_master=(configs.local_rank == 0))
    
    if is_master() and not configs.eval:
        code_dst = os.path.join(log_dir, "code")
        copy_code(logger, src=str(script_dir), dst=code_dst)
    
    set_global_seed(seed=getattr(configs, 'seed', 42))

    # Create model (Pure Floating Point, no quantizer insertion)
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained) 
    model = preprocess_model(model, configs)

    model.cuda()
    if configs.distributed:
        model = DistributedDataParallel(model, device_ids=[configs.local_rank])

    # ------------- data --------------
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)

    # Simplified optimizer and scheduler for floating-point pre-training
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=configs.lr,
        momentum=configs.momentum,
        weight_decay=configs.weight_decay,
        nesterov=True
    )
    
    from timm.scheduler import create_scheduler
    lr_scheduler, _ = create_scheduler(configs, optimizer)
    
    start_epoch = 0
    if configs.resume.path and os.path.exists(configs.resume.path):
        model, start_epoch, _ = load_checkpoint(model, configs.resume.path, 'cuda', lean=configs.resume.lean, optimizer=optimizer, lr_scheduler=lr_scheduler)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    perf_scoreboard = PerformanceScoreboard(configs.log.num_best_scores)
    
    if configs.eval:
        acc = validate(test_loader, model, criterion, -1, monitors, configs)
        print(f"Test Accuracy: {acc:.2f}%")
        return

    logger_info(logger, f'Start Floating-Point Pre-training: {configs.arch} on {configs.dataloader.dataset}')
    for epoch in range(start_epoch, configs.epochs):
        if configs.distributed:
            train_sampler.set_epoch(epoch)

        logger_info(logger, '>>>>>>>> Epoch %3d' % epoch)
        t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer, epoch, monitors, configs)
        
        v_top1, v_top5, v_loss = validate(val_loader, model, criterion, epoch, monitors, configs)
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
        tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)

        perf_scoreboard.update(v_top1, v_top5, epoch)
        is_best = perf_scoreboard.is_best(epoch)

        save_checkpoint(epoch, configs.arch, model, None, optimizer,
                        {'top1': v_top1, 'top5': v_top5},
                        is_best, configs.name, log_dir, lr_scheduler=lr_scheduler)

    if configs.local_rank == 0:
        tbmonitor.writer.close()

if __name__ == "__main__":
    main()

