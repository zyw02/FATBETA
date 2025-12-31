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
from util.loss_ops import DistributionLoss
from util.utils import create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend, tbmonitor_add_scalars
from util.weight_schd import CosineSched
from quan import find_modules_to_quantize, replace_module_by_names
from policy import BITS
from process_olm_fat import train, validate, PerformanceScoreboard
from evolution_search import EvolutionSearcher
from util.fault_injector import FaultInjector
from util.output_corrector import create_output_corrector
from util.learnable_olm import LearnableOLMManager
from util.search_olm_manager import SearchOLMManager


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
                                                output_corrector=output_corrector, corrector_optimizer=corrector_optimizer,
                                                learnable_olm_manager=learnable_olm_manager, olm_optimizer=olm_optimizer,
                                                search_olm_manager=search_olm_manager)
        
        # 如果从checkpoint加载了search_olm_manager的映射，需要传递给FaultInjector
        if search_olm_manager is not None and fault_injector is not None:
            loaded_mappings = search_olm_manager.get_olm_mappings()
            loaded_code_to_value = search_olm_manager.get_olm_code_to_value()
            if loaded_mappings:
                fault_injector.update_olm_mappings(loaded_mappings, loaded_code_to_value)
                logger_info(logger, f'  ✅ 已从checkpoint恢复OLM映射并传递给FaultInjector（{len(loaded_mappings)}层）')
        
        reset_bn_cands = not (getattr(configs, "eval", False) or getattr(configs, "search", False))
        
        w_cands, a_cands = target_model._load_checkpoint(configs.resume.path, )
        q_layers_ema, _ = get_quantized_layers(target_model.ema)
        for idx, layer in enumerate(q_layers_ema):
            layer.set_bit_cands(w_cands[idx], a_cands[idx])

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0. else \
        torch.nn.CrossEntropyLoss().cuda()

    soft_criterion = DistributionLoss() if teacher_model is not None else None

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

    # 初始化可学习OLM管理器（仅在训练模式且启用learnable OLM时）
    # 不再使用 Learnable OLM，只使用传统 OLM 搜索
    learnable_olm_manager = None
    olm_optimizer = None
    # 初始化搜索OLM管理器（用于FAT训练中的OLM搜索）
    search_olm_manager = None
    if not configs.eval and not configs.search:
        # 跳过 Learnable OLM 初始化（已禁用，只使用传统 OLM 搜索）
        learnable_olm_config = getattr(configs, 'learnable_olm', None)
        if False:  # 禁用 Learnable OLM，不再初始化
            # 获取训练用的模型
            training_model = model.module if configs.distributed else model
            
            # 获取要应用OLM的层列表
            layer_names = getattr(learnable_olm_config, 'layer_names', [])
            if not layer_names:
                # 如果没有指定，默认对所有量化层应用OLM（包括动态位宽层和fixed_bits层）
                from util.qat import get_quantized_layers
                from quan.func import QuanConv2d, QuanLinear
                q_layers_temp, _ = get_quantized_layers(training_model)
                # 构建层名称列表（包括动态位宽层）
                layer_names = []
                for name, module in training_model.named_modules():
                    if module in q_layers_temp:
                        layer_names.append(name)
                
                # 同时添加 fixed_bits 层（这些层不在 get_quantized_layers 中）
                for name, module in training_model.named_modules():
                    if isinstance(module, (QuanConv2d, QuanLinear)):
                        if module not in q_layers_temp:
                            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                                layer_names.append(name)
                
                logger_info(logger, f'No layer_names specified, using all quantized layers (including fixed_bits): {layer_names}')
            
            # 获取每层的位宽（参考 tools/train_olm_encoder.py 的写法）
            # 对于动态位宽层，使用 target_bits 中的最大位宽以确保覆盖所有可能的二进制值
            target_bit_width = getattr(configs, 'target_bits', [8])
            max_target_bit = max(target_bit_width) if target_bit_width else 8
            
            bit_widths = {}
            from util.qat import get_quantized_layers
            from quan.func import QuanConv2d, QuanLinear
            q_layers, _ = get_quantized_layers(training_model)
            
            # 构建层名称映射：从模块对象到层名称（包括动态位宽层和fixed_bits层）
            layer_name_map = {}
            for name, module in training_model.named_modules():
                if isinstance(module, (QuanConv2d, QuanLinear)):
                    layer_name_map[module] = name
            
            # 处理所有层（动态位宽层和fixed_bits层）
            for name, module in training_model.named_modules():
                if not isinstance(module, (QuanConv2d, QuanLinear)):
                    continue
                
                if name not in layer_names:
                    continue
                
                # 检查是否有量化配置（支持 bits 和 fixed_bits）
                wbits = None
                if hasattr(module, 'bits') and module.bits is not None:
                    # 动态位宽层：使用 target_bits 中的最大位宽以确保覆盖所有可能的二进制值
                    wbits = max_target_bit
                    logger_info(logger, f'   Dynamic bit-width layer {name}: using max target_bits={max_target_bit} for OLM')
                elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    # fixed_bits 层使用 8bit 以确保覆盖所有可能的二进制值
                    wbits = 8
                    logger_info(logger, f'   Fixed-bits layer {name}: using 8-bit for OLM')
                else:
                    wbits = 8  # 默认8位
                
                if wbits is not None:
                    # 转换为int
                    if isinstance(wbits, torch.Tensor):
                        wbits = int(wbits.item())
                    else:
                        wbits = int(wbits)
                    bit_widths[name] = wbits
            
            # 创建LearnableOLMManager
            learnable_olm_manager = LearnableOLMManager(
                model=training_model,
                layer_names=layer_names,
                bit_widths=bit_widths,
                device=torch.device('cuda'),
                init_method=getattr(learnable_olm_config, 'init_method', 'identity'),
                temperature=getattr(learnable_olm_config, 'temperature', 1.0),
                use_straight_through=getattr(learnable_olm_config, 'use_straight_through', True),
            )
            
            # 创建OLM优化器
            olm_params = learnable_olm_manager.get_parameters()
            if olm_params:
                olm_lr = getattr(learnable_olm_config, 'lr', configs.lr * 0.1)  # 默认使用模型学习率的10%
                olm_optimizer = torch.optim.Adam(
                    olm_params,
                    lr=olm_lr,
                    weight_decay=getattr(learnable_olm_config, 'weight_decay', configs.weight_decay)
                )
                logger_info(logger, '=' * 80)
                logger_info(logger, '🔧 LEARNABLE OLM - ENABLED')
                logger_info(logger, '=' * 80)
                logger_info(logger, f'  ✅ LearnableOLMManager initialized')
                logger_info(logger, f'  ✅ Layers: {layer_names}')
                logger_info(logger, f'  ✅ Bit widths: {bit_widths}')
                logger_info(logger, f'  ✅ OLM Optimizer: Adam (lr={olm_lr})')
                logger_info(logger, f'  ✅ Initialization: Will use traditional OLM at epoch 40')
                logger_info(logger, '=' * 80)
            else:
                logger_info(logger, '⚠️  No OLM parameters found, disabling learnable OLM')
                learnable_olm_manager = None
    
    # 初始化故障注入器（仅在训练模式且启用故障感知训练时）
    fault_injector = None
    if not configs.eval and not configs.search:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None and getattr(fault_aware_training_config, 'enabled', False):
            # 获取BER值（支持渐进式调度）
            # 确保ber是浮点数（YAML可能解析为字符串，如"1e-2"）
            ber_raw = getattr(fault_aware_training_config, 'ber', 1e-2)
            ber = float(ber_raw)  # float()可以处理字符串和数字
            trades_config = getattr(fault_aware_training_config, 'trades', {})
            use_kl = getattr(trades_config, 'use_kl', False)
            alpha = getattr(trades_config, 'alpha', 0.6)
            beta = getattr(trades_config, 'beta', 1.0)
            
            # 获取训练用的模型（用于故障注入）
            # 注意：故障注入器应该作用于训练时的模型，而不是EMA模型
            training_model = model.module if configs.distributed else model
            
            # 获取seed_list配置（可选）
            seed_list = getattr(fault_aware_training_config, 'seed_list', None)
            if seed_list is not None:
                # 确保seed_list是列表格式
                if isinstance(seed_list, (list, tuple)):
                    seed_list = list(seed_list)
                else:
                    # 如果是单个值，转换为列表
                    seed_list = [int(seed_list)]
            
            # 不使用 Learnable OLM，只使用传统 OLM 搜索
            # 不传递 learnable_olm_manager 给 FaultInjector
            fault_injector = FaultInjector(
                model=training_model,
                mode="ber",
                ber=ber,
                enable_in_training=True,
                enable_in_inference=False,
                seed=getattr(configs, 'seed', 42),
                seed_list=seed_list,  # 传递seed_list（如果提供）
                learnable_olm_manager=None  # 不使用 Learnable OLM
            )
            # 醒目的日志输出
            logger_info(logger, '=' * 80)
            logger_info(logger, '🚀 FAULT-AWARE TRAINING (FAT) - ENABLED')
            logger_info(logger, '=' * 80)
            logger_info(logger, f'  ✅ FaultInjector initialized')
            logger_info(logger, f'  ✅ BER (Bit-Error-Rate): {ber}')
            logger_info(logger, f'  ✅ TRADES Loss Method: {"KL Divergence" if use_kl else "Simple Combination"}')
            if not use_kl:
                logger_info(logger, f'  ✅ TRADES Weights: alpha={alpha}, beta={beta}')
            else:
                logger_info(logger, f'  ✅ TRADES KL Weight: beta={beta}')
            logger_info(logger, f'  ✅ Training mode: Enabled (Inference mode: Disabled)')
            if seed_list is not None:
                logger_info(logger, f'  ✅ Seed List: {seed_list} (训练时每次forward随机选择，验证时从中采样)')
            else:
                logger_info(logger, f'  ✅ Seed: {getattr(configs, "seed", 42)} (固定seed)')
            logger_info(logger, '=' * 80)
        else:
            logger_info(logger, '=' * 80)
            logger_info(logger, '⚠️  FAULT-AWARE TRAINING (FAT) - DISABLED')
            logger_info(logger, '=' * 80)
            if fault_aware_training_config is None:
                logger_info(logger, '  Reason: fault_aware_training config not found in YAML')
            else:
                enabled_status = getattr(fault_aware_training_config, 'enabled', False)
                logger_info(logger, f'  Reason: fault_aware_training.enabled = {enabled_status}')
            logger_info(logger, '=' * 80)
        
        # 初始化搜索OLM管理器（仅在启用FAT时）
        if fault_injector is not None:
            # 检查是否启用了FAT
            fault_aware_training_config_check = getattr(configs, 'fault_aware_training', None)
            use_fat_enabled = fault_aware_training_config_check is not None and getattr(fault_aware_training_config_check, 'enabled', False)
            if use_fat_enabled:
                # 获取要应用OLM的层列表（与learnable_olm_manager一致）
                search_olm_config = getattr(fault_aware_training_config_check, 'search_olm', {})
            layer_names = getattr(search_olm_config, 'layer_names', [])
            if not layer_names:
                # 如果没有指定，默认对所有量化层应用OLM
                from util.qat import get_quantized_layers
                from quan.func import QuanConv2d, QuanLinear
                q_layers_temp, _ = get_quantized_layers(training_model)
                layer_names = []
                for name, module in training_model.named_modules():
                    if module in q_layers_temp:
                        layer_names.append(name)
                # 同时添加 fixed_bits 层
                for name, module in training_model.named_modules():
                    if isinstance(module, (QuanConv2d, QuanLinear)):
                        if module not in q_layers_temp:
                            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                                layer_names.append(name)
                logger_info(logger, f'No search_olm.layer_names specified, using all quantized layers: {layer_names}')
                
                # 获取每层的位宽
                # 对于动态位宽层，直接使用target_bits的最大值（与sample_max一致）
                # 对于fixed_bits层，使用yaml配置中的值（通常是8）
                target_bit_width = getattr(configs, 'target_bits', [8])
                max_target_bit = max(target_bit_width) if target_bit_width else 8
                bit_widths = {}
                for name, module in training_model.named_modules():
                    if not isinstance(module, (QuanConv2d, QuanLinear)):
                        continue
                    if name not in layer_names:
                        continue
                    
                    # 判断是动态位宽层还是fixed_bits层
                    if hasattr(module, 'bits') and module.bits is not None:
                        # 动态位宽层：直接使用target_bits的最大值
                        wbits = max_target_bit
                    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                        # fixed_bits层：从yaml配置中获取（通常是8）
                        # 处理可能的列表/元组情况
                        fixed_bits_raw = module.fixed_bits
                        if isinstance(fixed_bits_raw, (list, tuple)):
                            wbits = fixed_bits_raw[0] if len(fixed_bits_raw) > 0 else 8
                        elif isinstance(fixed_bits_raw, torch.Tensor):
                            wbits = int(fixed_bits_raw.item())
                        else:
                            wbits = fixed_bits_raw
                        wbits = int(wbits) if not isinstance(wbits, int) else wbits
                    else:
                        # 默认8位
                        wbits = 8
                    
                    bit_widths[name] = wbits
                
                # 创建SearchOLMManager
                search_olm_manager = SearchOLMManager(
                    model=training_model,
                    layer_names=layer_names,
                    bit_widths=bit_widths,
                    update_freq=getattr(search_olm_config, 'update_freq', 10),  # 每10个epoch更新一次
                    method=getattr(search_olm_config, 'method', 'simulated_annealing'),
                    num_samples=getattr(search_olm_config, 'num_samples', 1000),
                    max_iterations=getattr(search_olm_config, 'max_iterations', 3000),
                    device=torch.device('cuda')
                )
                
                # 立即将初始映射传递给FaultInjector（这样从第一个batch就可以使用OLM）
                if fault_injector is not None:
                    initial_mappings = search_olm_manager.get_olm_mappings()
                    initial_code_to_value = search_olm_manager.get_olm_code_to_value()
                    if initial_mappings:
                        fault_injector.update_olm_mappings(initial_mappings, initial_code_to_value)
                        logger_info(logger, f'  ✅ 已将初始OLM映射传递给FaultInjector（{len(initial_mappings)}层）')
                
                logger_info(logger, '=' * 80)
                logger_info(logger, '🔍 SEARCH OLM MANAGER - ENABLED')
                logger_info(logger, '=' * 80)
                logger_info(logger, f'  ✅ SearchOLMManager initialized')
                logger_info(logger, f'  ✅ Layers: {layer_names}')
                logger_info(logger, f'  ✅ Update frequency: {search_olm_manager.update_freq} epochs')
                logger_info(logger, f'  ✅ Method: {search_olm_manager.method}')
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
                                           corrector_optimizer=corrector_optimizer,
                                           learnable_olm_manager=learnable_olm_manager,
                                           olm_optimizer=olm_optimizer,
                                           search_olm_manager=search_olm_manager
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
                            output_corrector=output_corrector, corrector_optimizer=corrector_optimizer,
                            learnable_olm_manager=learnable_olm_manager, olm_optimizer=olm_optimizer,
                            search_olm_manager=search_olm_manager)

            if epoch % 20 == 0:
                save_checkpoint(epoch, configs.arch, model, target_model, optimizer, {
                    'top1': v_top1, 'top5': v_top5}, False, f'epoch_{str(epoch)}_checkpoint.pth.tar', log_dir, lr_scheduler=lr_scheduler, lr_scheduler_q=lr_scheduler_q, optimizer_q=optimizer_q,
                    output_corrector=output_corrector, corrector_optimizer=corrector_optimizer,
                    learnable_olm_manager=learnable_olm_manager, olm_optimizer=olm_optimizer,
                    search_olm_manager=search_olm_manager)
    
    # 训练结束后打印seed使用统计
    if fault_injector is not None and hasattr(fault_injector, 'print_seed_usage_stats'):
        fault_injector.print_seed_usage_stats(logger)

    if configs.local_rank == 0:
        tbmonitor.writer.close()  # close the TensorBoard


if __name__ == "__main__":
    main()
