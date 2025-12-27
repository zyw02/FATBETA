import logging
import torch
import yaml
import os
import json
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from model import create_model
from util import (ProgressMonitor, TensorBoardMonitor, 
                  get_config, init_logger, set_global_seed, setup_print, load_checkpoint, save_checkpoint, preprocess_model, init_dataloader)
from util.utils import copy_code
from util.mpq import sample_min_cands, switch_bit_width
from util.greedy_search import search, reset_bit_cands
from util.model_ema import ModelEma
from util.qat import get_quantized_layers
# DISABLED: Distribution Loss disabled for fault tolerance study
# from util.loss_ops import DistributionLoss
from util.utils import create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend, tbmonitor_add_scalars
from util.weight_schd import CosineSched
from quan import find_modules_to_quantize, replace_module_by_names
from policy import BITS
from process_normal import train, validate, PerformanceScoreboard
from evolution_search import EvolutionSearcher
from util.fault_injector import FaultInjector
from util.output_corrector import create_output_corrector


def init_logger_and_monitor(configs, script_dir):
    if is_master():
        output_dir = script_dir / configs.output_dir
        output_dir.mkdir(exist_ok=True)

        log_dir = init_logger(configs.name, output_dir,
                              script_dir / 'logging.conf')
        logger = logging.getLogger()

        with open(log_dir / "configs.yaml", "w") as yaml_file:  # dump experiment config
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
    # Allow single GPU training (distributed=False)
    # assert configs.distributed

    logger, log_dir, pymonitor, tbmonitor = init_logger_and_monitor(
        configs, script_dir)
    monitors = [pymonitor, tbmonitor]

    setup_print(is_master=(configs.local_rank == 0))
    
    # Backup code for experiment reproducibility (similar to SAQ)
    if not configs.eval and not configs.search:
        code_dst = os.path.join(log_dir, "code")
        copy_code(logger, src=str(script_dir), dst=code_dst)
    
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

    model.eval()

    # Wrap model with DDP only if distributed training is enabled
    if configs.distributed:
        wrap_the_model_with_ddp = lambda x: DistributedDataParallel(x.cuda(), device_ids=[configs.local_rank], find_unused_parameters=True)
        model = wrap_the_model_with_ddp(model)
        if using_distillation:
            teacher_model = wrap_the_model_with_ddp(teacher_model)
    else:
        # Single GPU training - just move model to GPU
        model = model.cuda()
        if using_distillation:
            teacher_model = teacher_model.cuda()

    # ------------- data --------------
    logger_info(logger, '[DEBUG] Initializing dataloaders...')
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)
    logger_info(logger, f'[DEBUG] Dataloaders initialized: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}')

    enable_linear_scaling_rule = False
    if enable_linear_scaling_rule:
        configs.lr = configs.lr * dist.get_world_size() * configs.dataloader.batch_size / 512
        configs.min_lr = configs.min_lr * \
            dist.get_world_size() * configs.dataloader.batch_size / 512
        configs.warmup_lr = configs.warmup_lr * \
            dist.get_world_size() * configs.dataloader.batch_size / 512

    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(
        model, configs)
    
    # 初始化输出修正器（如果启用故障感知训练）
    output_corrector = None
    if not configs.eval and not configs.search:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None and getattr(fault_aware_training_config, 'enabled', False):
            trades_config = getattr(fault_aware_training_config, 'trades', {})
            use_corrector = getattr(trades_config, 'use_corrector', False)
            if use_corrector:
                num_classes = configs.dataloader.num_classes
                wmid_tau = getattr(trades_config, 'corrector_wmid_tau', 2.0)
                wmid_beta = getattr(trades_config, 'corrector_wmid_beta', 2.0)
                logits_feature_dim = getattr(trades_config, 'corrector_logits_feature_dim', 32)
                fusion_hidden_dim = getattr(trades_config, 'corrector_fusion_hidden_dim', 128)
                gap_stats_momentum = getattr(trades_config, 'corrector_gap_stats_momentum', 0.95)
                direction_deadzone = getattr(trades_config, 'corrector_direction_deadzone', 0.05)
                anchor_ber = getattr(trades_config, 'corrector_anchor_ber', getattr(fault_aware_training_config, 'ber', 2e-2))
                ber_bucket_centers = getattr(trades_config, 'corrector_ber_buckets', None)
                # AlexNet的中间层数量（根据模型架构确定）
                num_layers = 8  # AlexNet约8层（conv1-5 + fc6-8）
                output_corrector = create_output_corrector(
                    num_classes=num_classes,
                    num_layers=num_layers,
                    device=torch.device('cuda'),
                    wmid_tau=wmid_tau,
                    wmid_beta=wmid_beta,
                    logits_feature_dim=logits_feature_dim,
                    fusion_hidden_dim=fusion_hidden_dim,
                    max_correction=getattr(trades_config, 'corrector_max_correction', 3.0),
                    gap_stats_momentum=gap_stats_momentum,
                    direction_deadzone=direction_deadzone,
                    anchor_ber=anchor_ber,
                    ber_bucket_centers=ber_bucket_centers
                )
                output_corrector.set_runtime_context(ber=0.0, stage='train')
                # 创建独立的corrector optimizer（完全独立于模型训练）
                corrector_params = list(output_corrector.parameters())
                corrector_optimizer = torch.optim.Adam(
                    corrector_params,
                    lr=getattr(trades_config, 'corrector_lr', configs.lr * 0.1),  # 可以使用不同的学习率
                    weight_decay=configs.weight_decay
                )
                corrector_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    corrector_optimizer, T_max=configs.epochs, eta_min=0
                )
                logger_info(logger, '=' * 80)
                logger_info(logger, '🔧 OUTPUT CORRECTOR - ENABLED (INDEPENDENT TRAINING)')
                logger_info(logger, '=' * 80)
                logger_info(logger, f'  ✅ Corrector initialized: {num_classes} classes')
                logger_info(logger, f'  ✅ Parameters: {output_corrector.get_num_parameters()} total')
                logger_info(logger, f'  ✅ Independent optimizer: Adam (lr={corrector_optimizer.param_groups[0]["lr"]})')
                logger_info(logger, '  ✅ Corrector training is completely independent from model training')
                logger_info(logger, '=' * 80)
            else:
                corrector_optimizer = None
                corrector_lr_scheduler = None
        else:
            corrector_optimizer = None
            corrector_lr_scheduler = None
    else:
        corrector_optimizer = None
        corrector_lr_scheduler = None

    start_epoch = 0

    # Determine input size based on dataset
    input_size = 32 if configs.dataloader.dataset in ['cifar10', 'cifar100'] else 224
    logger_info(logger, f'[DEBUG] Testing model forward with input size {input_size}x{input_size}...')
    model(torch.randn((1, 3, input_size, input_size)).cuda())
    logger_info(logger, '[DEBUG] Model forward test completed')

    logger_info(logger, '[DEBUG] Creating ModelEma...')
    target_model = ModelEma(model, decay=configs.ema_decay)
    logger_info(logger, '[DEBUG] ModelEma created')
    
    if configs.resume.path and os.path.exists(configs.resume.path):
        model, start_epoch, _ = load_checkpoint(model, configs.resume.path, 'cuda', lean=configs.resume.lean, optimizer=optimizer, override_optim=configs.eval,
                                                lr_scheduler=lr_scheduler, lr_scheduler_q=lr_scheduler_q, optimizer_q=optimizer_q,
                                                output_corrector=output_corrector, corrector_optimizer=corrector_optimizer)
        reset_bn_cands = not (getattr(configs, "eval", False) or getattr(configs, "search", False))
        
        w_cands, a_cands = target_model._load_checkpoint(configs.resume.path, )
        q_layers_ema, _ = get_quantized_layers(target_model.ema)
        for idx, layer in enumerate(q_layers_ema):
            layer.set_bit_cands(w_cands[idx], a_cands[idx])

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0. else \
        torch.nn.CrossEntropyLoss().cuda()

    # DISABLED: Distribution Loss disabled for fault tolerance study
    soft_criterion = None  # DistributionLoss() if teacher_model is not None else None
    logger_info(logger, '=' * 80)
    logger_info(logger, '⚠️  DISTRIBUTION LOSS DISABLED - This version is for fault tolerance study')
    logger_info(logger, '=' * 80)

    mode = 'training' 
    target_bit_width = configs.target_bits
    max_bit_width_cand = max(target_bit_width)

    perf_scoreboard = PerformanceScoreboard(configs.log.num_best_scores)
    logger_info(logger, '[DEBUG] Printing model structure (this may print quantizer info)...')
    print(model)
    logger_info(logger, '[DEBUG] Model structure printed')
    
    logger_info(logger, f'[DEBUG] Switching bit width for model to {target_bit_width}...')
    switch_bit_width(model, quan_scheduler=configs.quan, 
                     wbit=target_bit_width, abits=target_bit_width)
    logger_info(logger, '[DEBUG] Model bit width switched')
    
    logger_info(logger, f'[DEBUG] Switching bit width for EMA model to {target_bit_width}...')
    switch_bit_width(target_model.ema, quan_scheduler=configs.quan, 
                     wbit=target_bit_width, abits=target_bit_width)
    logger_info(logger, '[DEBUG] EMA model bit width switched')

    # 初始化故障注入器（支持 FAT 和 BFAT）
    fault_injector = None
    if not configs.eval and not configs.search:
        fat_cfg = getattr(configs, 'fault_aware_training', None)
        bfat_cfg = getattr(configs, 'bfat', None)
        
        fat_enabled = fat_cfg is not None and getattr(fat_cfg, 'enabled', False)
        bfat_enabled = bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False)

        if fat_enabled or bfat_enabled:
            # 获取 BER（优先从 FAT 配置获取，否则从 BFAT 获取）
            if fat_enabled:
                ber = float(getattr(fat_cfg, 'ber', 1e-2))
            else:
                ber = float(getattr(bfat_cfg, 'ber', 1e-2))
                
            training_model = model.module if configs.distributed else model
            
            seed_list = getattr(fat_cfg, 'seed_list', None) if fat_enabled else None
            if seed_list is not None:
                if isinstance(seed_list, (list, tuple)):
                    seed_list = list(seed_list)
                else:
                    seed_list = [int(seed_list)]
            
            skip_msb = getattr(fat_cfg, 'skip_msb', False) if fat_enabled else False
            only_msb = getattr(fat_cfg, 'only_msb', False) if fat_enabled else False
            bfat_bit_index = getattr(bfat_cfg, 'bit_index', None) if bfat_enabled else None

            fault_injector = FaultInjector(
                model=training_model,
                mode="ber",
                ber=ber,
                enable_in_training=True,
                enable_in_inference=False,
                seed=getattr(configs, 'seed', 42),
                seed_list=seed_list,
                skip_msb=skip_msb,
                only_msb=only_msb,
                bfat_bit_index=bfat_bit_index
            )
            
            logger_info(logger, '=' * 80)
            logger_info(logger, f'🚀 FAULT INJECTOR - INITIALIZED (FAT: {fat_enabled}, BFAT: {bfat_enabled})')
            logger_info(logger, f'  ✅ Initial BER: {ber}')
            logger_info(logger, '=' * 80)
        else:
            logger_info(logger, '=' * 80)
            logger_info(logger, '⚠️  FAULT INJECTION - DISABLED')
            logger_info(logger, '=' * 80)
            if fat_cfg is None and bfat_cfg is None:
                logger_info(logger, '  Reason: No fault configuration found in YAML')
            else:
                logger_info(logger, f'  Reason: fat.enabled={fat_enabled}, bfat.enabled={bfat_enabled}')
            logger_info(logger, '=' * 80)

    logger_info(logger, f'[DEBUG] Creating annealing schedule (train_loader length: {len(train_loader)})...')
    annealing_schedule = CosineSched(
        start_step=len(train_loader) * 40,
        max_step=len(train_loader) * configs.epochs,
        eta_start=0,
        eta_end=0.1
    )
    logger_info(logger, '[DEBUG] Annealing schedule created')

    logger_info(logger, '[DEBUG] Stepping lr_scheduler...')
    lr_scheduler.step(start_epoch)
    logger_info(logger, '[DEBUG] lr_scheduler stepped')

    # freezing_annealing_schedule = None
    freezing_annealing_schedule = None
    if configs.enable_dynamic_bit_training:
        logger_info(logger, '[DEBUG] Creating freezing annealing schedule...')
        logger_info(logger, 'Start dynamic bit-width training...')
        freezing_annealing_schedule = CosineSched(
            start_step=0,
            max_step=configs.epochs//2,
            eta_start=0.5,
            eta_end=0.2
        )
        logger_info(logger, '[DEBUG] Freezing annealing schedule created')

    if configs.eval:
        # Check if bit width config file is specified
        bitwidth_config_path = getattr(configs, 'bit_width_config_path', None)
        if bitwidth_config_path is not None and os.path.exists(bitwidth_config_path):
            logger_info(logger, f"Loading bit width configuration from: {bitwidth_config_path}")
            with open(bitwidth_config_path, 'r') as f:
                config_data = json.load(f)
            
            # Reconstruct conf format: [(bops_limit, [weight_bits], [act_bits]), ...]
            bitwidth_policies = []
            for cfg in config_data['configurations']:
                bitwidth_policies.append((
                    cfg['bops_limit'],
                    cfg['weight_bits'],
                    cfg['act_bits']
                ))
            logger_info(logger, f"Loaded {len(bitwidth_policies)} bit width configurations")
        else:
            # Use predefined policies if no config file specified
            bitwidth_policies = BITS[configs.arch]
            if bitwidth_config_path is not None:
                logger_info(logger, f"Warning: bit_width_config_path '{bitwidth_config_path}' not found, using predefined policies")

        bops_limit = []
        ret = validate(test_loader, target_model.ema, criterion, -1, monitors, configs, train_loader=train_loader,
                       eval_predefined_arch=bitwidth_policies, nr_random_sample=300, bops_limit=bops_limit)

        print(ret)

    elif configs.search:
        searcher = 'bid_search'

        assert searcher in ['bid_search', 'random_search', 'evolution_searcher']

        if searcher == 'evolution_searcher':
            q_layers, _ = get_quantized_layers(target_model.ema)
            searcher = EvolutionSearcher(configs, 'cuda', train_loader, target_model.ema, val_loader, test_loader, output_dir=f'./evolution_searcher/{configs.arch}/{configs.bops_limits}_bops', quantized_layers=q_layers)
            searcher.search()

        elif searcher == 'bid_search':
            reset_bit_cands(model=target_model.ema, reset=False)
            switch_bit_width(target_model.ema,
                            quan_scheduler=configs.quan, wbit=max_bit_width_cand-1, abits=max_bit_width_cand)
            
            # Pass fault_injector=None to search function (it will check configs.fault_aware_search internally)
            conf = search(loader=train_loader, model=target_model.ema, criterion=criterion, metrics=('bitops', [configs.bops_limits]), epoch=0, cfgs=configs, start_bits=configs.start_bit_width, fault_injector=None)
            
            acc = validate(test_loader, target_model.ema, criterion, -1, monitors,
                        configs, train_loader=train_loader, eval_predefined_arch=conf)
            print(conf)
            
            # Save bit width configuration to file
            if is_master():
                output_dir = script_dir / configs.output_dir
                output_dir.mkdir(exist_ok=True)
                search_config_file = output_dir / f"{configs.name}_bit_width_config.json"
                
                # Convert conf to JSON-serializable format
                # conf format: [(bops_limit, [weight_bits], [act_bits]), ...]
                config_data = {
                    'arch': configs.arch,
                    'bops_limits': configs.bops_limits,
                    'target_bits': configs.target_bits,
                    'configurations': []
                }
                for item in conf:
                    if len(item) == 3:
                        bops_limit, weight_bits, act_bits = item
                        # Convert numpy/torch types to Python native types
                        weight_bits_list = [int(x) if isinstance(x, (int, float, torch.Tensor)) else int(x.item()) if hasattr(x, 'item') else x for x in weight_bits]
                        act_bits_list = [int(x) if isinstance(x, (int, float, torch.Tensor)) else int(x.item()) if hasattr(x, 'item') else x for x in act_bits]
                        config_data['configurations'].append({
                            'bops_limit': float(bops_limit),
                            'weight_bits': weight_bits_list,
                            'act_bits': act_bits_list
                        })
                
                with open(search_config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                logger_info(logger, f"Saved bit width configuration to: {search_config_file}")
                logger_info(logger, f"To use this configuration for evaluation, set 'bit_width_config_path' in eval config or pass it via command line")

        elif searcher == 'random_search':
            from util.random_search import do_random_search
            conf = do_random_search(train_loader, model, criterion=criterion, metrics=configs.bops_limits, quan_scheduler=configs.quan)
            print(conf)

    else:  # training
        logger_info(logger, ('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
        logger_info(logger, 'Total epoch: %d, Start epoch %d', configs.epochs, start_epoch)
        
        v_top1, v_top5, v_loss = 0, 0, 0
        
        # 训练时间预估相关变量
        import time
        from datetime import datetime, timedelta
        epoch_times = []  # 记录每个epoch的时间
        training_start_time = time.time()  # 训练开始时间

        for epoch in range(start_epoch, configs.epochs):
            epoch_start_time = time.time()  # 当前epoch开始时间
            if configs.distributed:
                train_sampler.set_epoch(epoch)

            # 旋转故障注入器的种子，增加训练多样性
            if fault_injector is not None:
                initial_seed = getattr(configs, 'seed', 42)
                fault_injector.seed = initial_seed + epoch
                # 如果有 seed_list，我们也希望它能随 epoch 漂移（可选，目前主要针对单 seed 情况）
                logger_info(logger, f'🎲 Epoch {epoch}: FaultInjector seed rotated to {fault_injector.seed}')

            logger_info(logger, '>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                           epoch, monitors, configs, model_ema=target_model, nr_random_sample=getattr(
                                               configs, 'num_random_path', 3),
                                           soft_criterion=soft_criterion, teacher_model=teacher_model,
                                           optimizer_q=optimizer_q, mode=mode, 
                                           annealing_schedule=annealing_schedule,
                                           freezing_annealing_schedule=freezing_annealing_schedule,
                                           fault_injector=fault_injector,
                                           output_corrector=output_corrector,
                                           corrector_optimizer=corrector_optimizer
                                           )
            
            # 如果有验证阶段，也需要记录验证时间
            # 注意：这里只记录训练时间，验证时间会在validate函数中单独处理
            # 如果需要更准确的预估，可以在validate调用前后也记录时间
            
            # 计算epoch时间并记录（包括训练和验证）
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)
            
            # 计算平均epoch时间（使用最近5个epoch的平均值，更准确）
            recent_epochs = min(5, len(epoch_times))
            avg_epoch_time = sum(epoch_times[-recent_epochs:]) / recent_epochs
            
            # 计算剩余epoch数和预估完成时间
            remaining_epochs = configs.epochs - epoch - 1
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            
            # 格式化时间显示
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
            
            # 计算预估完成时间（当前时间 + 剩余时间）
            estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
            estimated_completion_str = estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Print epoch summary with time estimation
            logger_info(logger, 'Epoch %3d Summary - Train Top1: %.3f%%  Top5: %.3f%%  Loss: %.4f' % 
                       (epoch, t_top1, t_top5, t_loss))
            logger_info(logger, '  ⏱️  本Epoch耗时: %s | 平均Epoch耗时: %s | 剩余Epoch数: %d' % 
                       (format_time(epoch_time), format_time(avg_epoch_time), remaining_epochs))
            if remaining_epochs > 0:
                logger_info(logger, '  📅 预估剩余时间: %s | 预估完成时间: %s' % 
                           (format_time(estimated_remaining_time), estimated_completion_str))
            else:
                logger_info(logger, '  ✅ 训练完成！总耗时: %s' % format_time(time.time() - training_start_time))
            
            if lr_scheduler is not None:
                lr_scheduler.step(epoch+1)

            if lr_scheduler_q is not None:
                lr_scheduler_q.step()
            
            # 更新corrector的学习率（独立于模型）
            if corrector_lr_scheduler is not None:
                corrector_lr_scheduler.step()

            tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor_add_scalars(tbmonitor, 'Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)

            # save main model
            save_checkpoint(epoch, configs.arch, model, target_model, optimizer,
                            {
                                'top1': v_top1, 'top5': v_top5
                            },
                            False, configs.name, log_dir, lr_scheduler=lr_scheduler, lr_scheduler_q=lr_scheduler_q, optimizer_q=optimizer_q,
                            output_corrector=output_corrector, corrector_optimizer=corrector_optimizer)

            # NOTE: Disabled periodic extra checkpoint saving to reduce disk usage.
            # We keep only the rolling checkpoint: `{configs.name}_checkpoint.pth.tar` (overwritten each epoch),
            # which is sufficient for evaluation and sweep scripts.
    
    # 训练结束后打印seed使用统计
    if fault_injector is not None and hasattr(fault_injector, 'print_seed_usage_stats'):
        fault_injector.print_seed_usage_stats(logger)

    if configs.local_rank == 0:
        tbmonitor.writer.close()  # close the TensorBoard


if __name__ == "__main__":
    main()

