"""
Main entry point for Gradient Surgery Mixed-Precision QAT Training.

Usage:
    torchrun --nproc_per_node=8 main_gs.py configs/training/r20.yaml
"""

import logging
import torch
import yaml
import os
import time
from datetime import timedelta, datetime
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from model import create_model
from util import (ProgressMonitor, TensorBoardMonitor, 
                  get_config, init_logger, set_global_seed, setup_print, 
                  load_checkpoint, save_checkpoint, preprocess_model, init_dataloader)
from util.mpq import sample_min_cands, switch_bit_width
from util.greedy_search import search, reset_bit_cands
from util.model_ema import ModelEma
from util.qat import get_quantized_layers
from util.loss_ops import DistributionLoss
from util.utils import create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend, tbmonitor_add_scalars
from util.weight_schd import CosineSched
from quan import find_modules_to_quantize, replace_module_by_names
from policy import BITS
from process_gs import train, validate, PerformanceScoreboard  # Use gradient surgery process
from evolution_search import EvolutionSearcher


def init_logger_and_monitor(configs, script_dir):
    if is_master():
        output_dir = script_dir / configs.output_dir
        output_dir.mkdir(exist_ok=True)

        log_dir = init_logger(configs.name, output_dir, script_dir / 'logging.conf')
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

    assert configs.rank >= 0, 'ERROR IN RANK'
    # assert configs.distributed

    logger, log_dir, pymonitor, tbmonitor = init_logger_and_monitor(configs, script_dir)
    monitors = [pymonitor, tbmonitor]

    setup_print(is_master=(configs.local_rank == 0))
    set_global_seed(seed=0)

    teacher_model = None
    using_distillation = configs.kd
    if using_distillation:
        teacher_model = create_model('resnet101', dataset=configs.dataloader.dataset)
        teacher_model.eval()

    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained) 
    model = preprocess_model(model, configs)

    logger_info(logger, 'Inserted quantizers into the original model')
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))

    model.cuda()
    model.eval()

    # ------------- optimizer --------------
    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(model, configs)

    start_epoch = 0
    if configs.resume.path and os.path.exists(configs.resume.path):
        model, start_epoch, _ = load_checkpoint(
            model, configs.resume.path, 'cuda', 
            lean=configs.resume.lean, 
            optimizer=optimizer, 
            override_optim=configs.eval,
            lr_scheduler=lr_scheduler, 
            lr_scheduler_q=lr_scheduler_q, 
            optimizer_q=optimizer_q
        )
        # Ensure training starts from epoch 0 if it's a pre-trained FP model
        if configs.resume.lean:
            start_epoch = 0
            logger_info(logger, f"Loaded pre-trained weights from {configs.resume.path}. Starting fresh experiment from Epoch 0.")

    # ------------- warmup for quantizer initialization --------------
    # Use correct input size based on dataset
    input_size = 32 if configs.dataloader.dataset in ['cifar10', 'cifar100'] else 224
    logger_info(logger, f'Performing quantizer initialization warmup with input size {input_size}x{input_size}...')
    with torch.no_grad():
        model(torch.randn((8, 3, input_size, input_size)).cuda())

    # ------------- EMA --------------
    target_model = ModelEma(model, decay=configs.ema_decay)
    # Force EMA sync with main model after potential checkpoint loading and warmup
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in target_model.ema.state_dict().items():
            if k in msd:
                v.copy_(msd[k])
    
    # Load EMA bit candidates if available
    if configs.resume.path and os.path.exists(configs.resume.path):
        w_cands, a_cands = target_model._load_checkpoint(configs.resume.path)
        if len(w_cands) > 0 and len(a_cands) > 0:
            q_layers_ema, _ = get_quantized_layers(target_model.ema)
            for idx, layer in enumerate(q_layers_ema):
                layer.set_bit_cands(w_cands[idx], a_cands[idx])

    # ------------- DDP --------------
    if configs.distributed:
        model = DistributedDataParallel(
            model, device_ids=[configs.local_rank], find_unused_parameters=True
        )
        if teacher_model is not None:
            teacher_model = DistributedDataParallel(
                teacher_model.cuda(), device_ids=[configs.local_rank], find_unused_parameters=True
            )

    # ------------- data --------------
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(
        configs.dataloader, arch=configs.arch
    )

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0. else \
        torch.nn.CrossEntropyLoss().cuda()

    soft_criterion = DistributionLoss() if teacher_model is not None else None

    mode = 'training' 
    target_bit_width = configs.target_bits
    max_bit_width_cand = max(target_bit_width)

    perf_scoreboard = PerformanceScoreboard(configs.log.num_best_scores)
    print(model)
    
    switch_bit_width(model, quan_scheduler=configs.quan, wbit=target_bit_width, abits=target_bit_width)
    switch_bit_width(target_model.ema, quan_scheduler=configs.quan, wbit=target_bit_width, abits=target_bit_width)

    annealing_schedule = CosineSched(
        start_step=len(train_loader) * 40,
        max_step=len(train_loader) * configs.epochs,
        eta_start=0,
        eta_end=0.1
    )

    lr_scheduler.step(start_epoch)

    freezing_annealing_schedule = None
    if configs.enable_dynamic_bit_training:
        logger_info(logger, 'Start dynamic bit-width training with Gradient Surgery...')
        freezing_annealing_schedule = CosineSched(
            start_step=0,
            max_step=configs.epochs // 2,
            eta_start=0.5,
            eta_end=0.2
        )

    # Log gradient surgery config
    gs_config = getattr(configs, 'gradient_surgery', None)
    if gs_config and getattr(gs_config, 'enabled', False):
        logger_info(logger, '=' * 60)
        logger_info(logger, '🔧 [GS-MPQ] Gradient Surgery for Mixed-Precision QAT')
        logger_info(logger, f'   Projection Mode: {getattr(gs_config, "projection_mode", "direction")}')
        logger_info(logger, f'   Limit Norm: {getattr(gs_config, "limit_norm", False)}')
        logger_info(logger, '=' * 60)

    if configs.eval:
        bitwidth_policies = BITS[configs.arch]
        bops_limit = []
        ret = validate(
            test_loader, target_model.ema, criterion, -1, monitors, configs, 
            train_loader=train_loader, eval_predefined_arch=bitwidth_policies, 
            nr_random_sample=300, bops_limit=bops_limit
        )
        print(ret)

    elif configs.search:
        searcher = 'bid_search'
        assert searcher in ['bid_search', 'random_search', 'evolution_searcher']

        if searcher == 'evolution_searcher':
            q_layers, _ = get_quantized_layers(target_model.ema)
            searcher = EvolutionSearcher(
                configs, 'cuda', train_loader, target_model.ema, 
                val_loader, test_loader, 
                output_dir=f'./evolution_searcher/{configs.arch}/{configs.bops_limits}_bops', 
                quantized_layers=q_layers
            )
            searcher.search()

        elif searcher == 'bid_search':
            reset_bit_cands(model=target_model.ema, reset=False)
            switch_bit_width(
                target_model.ema, quan_scheduler=configs.quan, 
                wbit=max_bit_width_cand - 1, abit=max_bit_width_cand
            )
            
            conf = search(
                loader=train_loader, model=target_model.ema, criterion=criterion, 
                metrics=('bitops', [configs.bops_limits]), epoch=0, cfgs=configs, 
                start_bits=configs.start_bit_width
            )
            
            acc = validate(
                test_loader, target_model.ema, criterion, -1, monitors,
                configs, train_loader=train_loader, eval_predefined_arch=conf
            )
            print(conf)

        elif searcher == 'random_search':
            from util.random_search import do_random_search
            conf = do_random_search(
                train_loader, model, criterion=criterion, 
                metrics=configs.bops_limits, quan_scheduler=configs.quan
            )
            print(conf)

    else:  # training
        logger_info(logger, ('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
        logger_info(logger, 'Total epoch: %d, Start epoch %d', configs.epochs, start_epoch)
        
        v_top1, v_top5, v_loss = 0, 0, 0
        training_start_time = time.time()

        for epoch in range(start_epoch, configs.epochs):
            epoch_start_time = time.time()
            if configs.distributed:
                train_sampler.set_epoch(epoch)

            logger_info(logger, '>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = train(
                train_loader, model, criterion, optimizer,
                epoch, monitors, configs, 
                model_ema=target_model, 
                nr_random_sample=getattr(configs, 'num_random_path', 3),
                soft_criterion=soft_criterion, 
                teacher_model=teacher_model,
                optimizer_q=optimizer_q, 
                mode=mode, 
                annealing_schedule=annealing_schedule,
                freezing_annealing_schedule=freezing_annealing_schedule
            )
            
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            if lr_scheduler_q is not None:
                lr_scheduler_q.step()

            tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)

            # save main model
            save_checkpoint(
                epoch, configs.arch, model, target_model, optimizer,
                {'top1': v_top1, 'top5': v_top5},
                False, configs.name, log_dir, 
                lr_scheduler=lr_scheduler, 
                lr_scheduler_q=lr_scheduler_q, 
                optimizer_q=optimizer_q
            )

            # Time monitoring
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            elapsed_time = epoch_end_time - training_start_time
            epochs_done = epoch - start_epoch + 1
            avg_epoch_time = elapsed_time / epochs_done
            remaining_epochs = configs.epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            
            etc_time = datetime.now() + timedelta(seconds=eta_seconds)
            
            logger_info(logger, 'Epoch %d duration: %s', epoch, str(timedelta(seconds=int(epoch_duration))))
            logger_info(logger, 'Estimated completion time (ETC): %s', etc_time.strftime('%Y-%m-%d %H:%M:%S'))
            logger_info(logger, 'Time remaining: %s', str(timedelta(seconds=int(eta_seconds))))

            # if epoch % 20 == 0:
            #     save_checkpoint(
            #         epoch, configs.arch, model, target_model, optimizer, 
            #         {'top1': v_top1, 'top5': v_top5}, 
            #         False, f'epoch_{str(epoch)}_checkpoint.pth.tar', log_dir, 
            #         lr_scheduler=lr_scheduler, 
            #         lr_scheduler_q=lr_scheduler_q, 
            #         optimizer_q=optimizer_q
            #     )

    if configs.local_rank == 0:
        tbmonitor.writer.close()


if __name__ == "__main__":
    main()
