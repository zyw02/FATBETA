import logging
import math
import operator
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from quan.func import SwithableBatchNorm
from util import AverageMeter
from util.utils import model_profiling, calibrate_batchnorm_state, accuracy, update_meter, set_global_seed
from util.qat import profile_layerwise_quantization_metric, freeze_layers, set_bit_width, auxiliary_quantized_loss, remove_hook_for_quantized_layers, set_forward_hook_for_quantized_layers
from util.mpq import sample_one_mixed_policy, sample_max_cands, sample_min_cands
from util.dist import master_only, logger_info

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def compute_overall_loss(outputs, teacher_outputs, targets, criterion, model, quantization_error_minimization=False, QE_loss_weight=.5, disable_smallest_regularization=True, configs=None):
    task_loss = loss_forward(outputs, teacher_outputs, targets, criterion)

    if quantization_error_minimization or disable_smallest_regularization:
        QE_loss, distribution_loss = auxiliary_quantized_loss(model, 
                                                           quantization_error_minimization=quantization_error_minimization, 
                                                           fairness_regularization=disable_smallest_regularization
                                                           )
    else:
        QE_loss, distribution_loss = 0, 0

    QE_loss *= QE_loss_weight

    adaptive_region_weight_decay = getattr(configs, 'adaptive_region_weight_decay', configs.weight_decay)
    distribution_loss *= (adaptive_region_weight_decay - configs.weight_decay)

    return task_loss + QE_loss + distribution_loss, QE_loss, distribution_loss


@master_only
def show_training_info(meters, target_bits, nr_random_sample, mode):
    iters = len(meters) if mode == 'training' else 1
    for i in range(iters):
            logger.info('==> %s Top1: %.3f    Top5: %.3f    Loss: %.3f', meters[i]['name'],
                        meters[i]['top1'].avg, meters[i]['top5'].avg, meters[i]['loss'].avg)


@master_only
def update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, nr_random_sample, optimizer, optimizer_q, mode='training', corrector_stats=None):
    iters = len(meters) if mode == 'training' else 1
    for m in monitors:
        for i in range(iters):
            # if meters[i]['top1'].avg == 0.:
            #     continue
            p = meters[i]['name'] + ' '
            
            # 基础训练信息
            update_dict = {
                'Loss': meters[i]['loss'],
                'QE Loss': meters[i]['QE_loss'], 
                'Distribution Loss': meters[i]['dist_loss'], 
                'IDM Loss': meters[i]['IDM_loss'], 
                'Top1': meters[i]['top1'],
                'Top5': meters[i]['top5'],
                'LR': optimizer.param_groups[0]['lr'],
                'QLR': optimizer_q.param_groups[0]['lr'] if optimizer_q is not None else 0
            }
            
            # 如果有corrector统计信息，添加到日志中
            if corrector_stats is not None:
                update_dict['CorrLoss'] = corrector_stats['loss']  # Corrector Loss
                update_dict['FltAcc'] = corrector_stats['acc_faulted']  # Faulted Acc
                update_dict['CorrAcc'] = corrector_stats['acc_corrected']  # Corrected Acc
                update_dict['AccGain'] = corrector_stats['acc_improvement']  # Acc Improvement
                if corrector_stats['rl_weight'] is not None:
                    update_dict['RLWeight'] = corrector_stats['rl_weight']  # RL Weight
            
            m.update(epoch, batch_idx + 1, steps_per_epoch, p + 'Training', update_dict)
        
        if mode == 'finetuning':
            continue

def compute_entropy(probs):
    """
    计算概率分布的信息熵: H(p) = -Σ p_i * log(p_i)
    
    Args:
        probs: 概率分布，shape [batch_size, num_classes]
    
    Returns:
        entropy: 每个样本的熵，shape [batch_size]
    """
    # 避免log(0)，添加小的epsilon
    eps = 1e-8
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


def compute_entropy_loss(probs_normal, probs_faulted, mode='difference'):
    """
    计算基于信息熵的损失项，用于约束故障下的模型行为。
    
    支持三种模式：
    1. 'difference': 最小化正常和故障输出的熵差异
       L_entropy = |H(p_normal) - H(p_faulted)|
    2. 'constraint': 约束故障下的熵不要太大（避免过度不确定）
       L_entropy = max(0, H(p_faulted) - H(p_normal))
    3. 'balance': 平衡正常和故障下的熵，同时约束故障熵
       L_entropy = |H(p_normal) - H(p_faulted)| + λ * max(0, H(p_faulted) - H_target)
    
    Args:
        probs_normal: 正常输出的概率分布，shape [batch_size, num_classes]
        probs_faulted: 故障输出的概率分布，shape [batch_size, num_classes]
        mode: 熵损失模式，'difference', 'constraint', 或 'balance'
    
    Returns:
        entropy_loss: 熵损失标量
    """
    entropy_normal = compute_entropy(probs_normal)  # [batch_size]
    entropy_faulted = compute_entropy(probs_faulted)  # [batch_size]
    
    if mode == 'difference':
        # 最小化熵差异：希望故障下的熵与正常时接近
        entropy_diff = torch.abs(entropy_normal - entropy_faulted)
        entropy_loss = entropy_diff.mean()
    
    elif mode == 'constraint':
        # 约束故障熵：希望故障下的熵不要比正常时大太多
        entropy_excess = torch.clamp(entropy_faulted - entropy_normal, min=0.0)
        entropy_loss = entropy_excess.mean()
    
    elif mode == 'balance':
        # 平衡模式：同时最小化熵差异和约束故障熵
        entropy_diff = torch.abs(entropy_normal - entropy_faulted)
        # 目标熵：正常熵的1.2倍（允许适度增加，但不允许过度不确定）
        entropy_target = entropy_normal * 1.2
        entropy_excess = torch.clamp(entropy_faulted - entropy_target, min=0.0)
        entropy_loss = entropy_diff.mean() + 0.5 * entropy_excess.mean()
    
    else:
        raise ValueError(f"Unknown entropy mode: {mode}. Must be 'difference', 'constraint', or 'balance'")
    
    return entropy_loss


def loss_forward(outputs, teacher_outputs, targets, criterion):
    loss = criterion(outputs, targets)

    if teacher_outputs is not None:
        loss = 1/2 * loss + 1/2 * F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(teacher_outputs, dim=-1), reduction='batchmean')
    
    return loss

def get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min):
    if mode == 'training':
        if not sample_current_max and not sample_current_min:
            num_fixed_sample = len(target_bits)
            fixed_name = [f'Bits [{target_bits[i]}]' for i in range(num_fixed_sample)]
            num_fixed_sample = 0
        else:
            num_fixed_sample = sample_current_min + sample_current_max
            if num_fixed_sample == 2:
                fixed_name = ['Max', 'Min']
            else:
                fixed_name = ['Max'] if not sample_current_min else ['Min']
        meters = [{
            'name': fixed_name[i] if i < num_fixed_sample else f'Mixed {i - num_fixed_sample}', 
            'loss': AverageMeter(),
            'QE_loss': AverageMeter(),
            'dist_loss': AverageMeter(),
            'IDM_loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'batch_time': AverageMeter()
        } for i in range(num_fixed_sample + nr_random_sample)]
    else:
        meters = [{
            'name': 'Finetune',
            'loss': AverageMeter(),
            'QE_loss': AverageMeter(),
            'dist_loss': AverageMeter(),
            'IDM_loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'batch_time': AverageMeter()
        }]

        num_fixed_sample = 1
    
    return meters, num_fixed_sample

def train(train_loader, model, criterion, optimizer, epoch, monitors, configs, model_ema=None, nr_random_sample=2, mode='training', soft_criterion=None, teacher_model=None, optimizer_q=None, annealing_schedule=None, freezing_annealing_schedule=None, IDM_weight=0.01, scaler=None, fault_injector=None, output_corrector=None, corrector_optimizer=None):
    assert mode in ['finetuning', 'training']

    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    if getattr(configs, 'sandwich_training', False):
        sample_current_max, sample_current_min = True, True
    else:
        sample_current_max, sample_current_min = False, False
    
    sample_current_max = True
    
    print(f"[DEBUG] train() called for epoch {epoch}, mode={mode}")
    print("[TRAIN] Bit-width candidates:", target_bits)
    
    meters, num_fixed_sample = get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min)
    
    # ========== 三阶段训练：检查corrector_start_epoch ==========
    corrector_start_epoch = None
    should_train_corrector = False
    should_freeze_model = False
    stage2_mix_prob = 0.0
    stage2_mix_bers = []
    stage3_anchor_prob = 0.0
    stage3_anchor_ber = 2e-2
    stage_label = 'stage1'
    if output_corrector is not None:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None:
            trades_config = getattr(fault_aware_training_config, 'trades', {})
            stage2_mix_prob = float(getattr(trades_config, 'stage2_mix_prob', stage2_mix_prob))
            stage2_mix_bers_cfg = getattr(trades_config, 'stage2_mix_bers', stage2_mix_bers)
            stage2_mix_bers = [float(b) for b in stage2_mix_bers_cfg] if stage2_mix_bers_cfg else []
            stage3_anchor_prob = float(getattr(trades_config, 'stage3_anchor_prob', stage3_anchor_prob))
            stage3_anchor_ber = float(getattr(trades_config, 'stage3_anchor_ber', stage3_anchor_ber))
            use_corrector = getattr(trades_config, 'use_corrector', False)
            if use_corrector:
                corrector_start_epoch = getattr(trades_config, 'corrector_start_epoch', None)
                if corrector_start_epoch is not None:
                    should_train_corrector = (epoch >= corrector_start_epoch)  # 第三阶段才训练corrector
                    should_freeze_model = (epoch >= corrector_start_epoch)  # 第三阶段冻结主模型
                    
                    # 在corrector_start_epoch时固定主模型参数
                    if epoch == corrector_start_epoch:
                        logger_info(logging.getLogger(__name__), 
                                  f"🔒 [Three-Stage Training] Epoch {epoch}: Freezing main model parameters, starting corrector training (Stage 3)")
                        for param in model.parameters():
                            param.requires_grad = False
                        # 确保optimizer也停止更新
                        if optimizer is not None:
                            optimizer.zero_grad()
                    elif epoch < corrector_start_epoch:
                        # 确保主模型参数可训练（第一阶段和第二阶段）
                        for param in model.parameters():
                            param.requires_grad = True
                    
                    # 日志输出
                    schedule_config = getattr(fault_aware_training_config, 'schedule', None)
                    fat_start_epoch = getattr(schedule_config, 'start_epoch', 50) if schedule_config else 50
                    if epoch < fat_start_epoch:
                        logger_info(logging.getLogger(__name__), 
                                  f"📊 [Three-Stage Training] Epoch {epoch}: Stage 1 - Clean training (no fault injection, corrector not trained)")
                    elif epoch < corrector_start_epoch:
                        logger_info(logging.getLogger(__name__), 
                                  f"📊 [Three-Stage Training] Epoch {epoch}: Stage 2 - Main model FAT training (corrector baseline updated, not trained)")
                    else:
                        logger_info(logging.getLogger(__name__), 
                                  f"🔧 [Three-Stage Training] Epoch {epoch}: Stage 3 - Corrector training only (main model frozen)")
    
    # Corrector训练统计信息（如果启用）
    corrector_stats = None
    if output_corrector is not None:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None:
            trades_config = getattr(fault_aware_training_config, 'trades', {})
            use_corrector = getattr(trades_config, 'use_corrector', False)
            if use_corrector:
                corrector_stats = {
                    'loss': AverageMeter(),
                    'acc_faulted': AverageMeter(),
                    'acc_corrected': AverageMeter(),
                    'acc_improvement': AverageMeter(),
                    'rl_weight': AverageMeter() if getattr(trades_config, 'corrector_use_rl_weighted', False) else None
                }

    # Handle single GPU mode where sampler might be None
    if train_loader.sampler is not None:
        total_sample = len(train_loader.sampler)
    else:
        total_sample = len(train_loader.dataset)
    batch_size = configs.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    
    print(f"[DEBUG] Total samples: {total_sample}, Batch size: {batch_size}, Steps per epoch: {steps_per_epoch}")

    information_distortion_mitigation = getattr(configs, 'information_distortion_mitigation', False)
    if information_distortion_mitigation:
        assert sample_current_max

    # 故障感知训练（TRADES风格）配置
    use_fault_aware_training = False
    fault_aware_training_config = None
    current_ber = None
    start_epoch = 0  # 初始化start_epoch，确保在函数作用域内
    if fault_injector is not None:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None:
            use_fault_aware_training = getattr(fault_aware_training_config, 'enabled', False)
            if use_fault_aware_training:
                trades_config = getattr(fault_aware_training_config, 'trades', {})
                use_kl = getattr(trades_config, 'use_kl', False)
                alpha = getattr(trades_config, 'alpha', 0.6)
                beta = getattr(trades_config, 'beta', 1.0)
                
                # ========== 三阶段训练：动态设置BER和seed_list ==========
                schedule_config = getattr(fault_aware_training_config, 'schedule', None)
                fat_start_epoch = getattr(schedule_config, 'start_epoch', 50) if schedule_config else 50
                corrector_start_epoch = getattr(trades_config, 'corrector_start_epoch', None)
                
                # 判断当前阶段
                is_stage1 = (epoch < fat_start_epoch)  # 第一阶段：干净训练，不使用fault_injector
                is_stage2 = (epoch >= fat_start_epoch and corrector_start_epoch is not None and epoch < corrector_start_epoch)  # 第二阶段：FAT训练
                is_stage3 = (corrector_start_epoch is not None and epoch >= corrector_start_epoch)  # 第三阶段：corrector训练
                stage_label = 'stage1'
                if is_stage3:
                    stage_label = 'stage3'
                elif is_stage2:
                    stage_label = 'stage2'
                
                # 第一阶段：不使用fault_injector
                if is_stage1:
                    use_fault_aware_training = False
                    current_ber = 0.0
                else:
                    # 第二阶段和第三阶段：动态设置seed_list
                    seed_list_stage2 = getattr(fault_aware_training_config, 'seed_list_stage2', None)
                    seed_list_stage3 = getattr(fault_aware_training_config, 'seed_list_stage3', None)
                    
                    if is_stage3 and seed_list_stage3 is not None:
                        # 第三阶段：多seed轮流使用
                        if hasattr(fault_injector, 'seed_list'):
                            seed_list = seed_list_stage3 if isinstance(seed_list_stage3, list) else [seed_list_stage3]
                            fault_injector.seed_list = seed_list
                            # 轮流使用seed：根据epoch选择seed
                            if len(seed_list) > 0:
                                seed_index = (epoch - corrector_start_epoch) % len(seed_list)
                                fault_injector.seed = seed_list[seed_index]
                    elif is_stage2 and seed_list_stage2 is not None:
                        # 第二阶段：使用单seed
                        if hasattr(fault_injector, 'seed_list'):
                            seed_list = seed_list_stage2 if isinstance(seed_list_stage2, list) else [seed_list_stage2]
                            fault_injector.seed_list = seed_list
                            if len(seed_list) > 0:
                                fault_injector.seed = seed_list[0]
                
                # 渐进式BER调度（仅在启用时使用，否则使用三阶段逻辑）
                if schedule_config is not None and getattr(schedule_config, 'enabled', False):
                    schedule_type = getattr(schedule_config, 'type', 'constant')
                    if schedule_type == 'progressive':
                        # 获取渐进式调度配置
                        progressive_config = getattr(schedule_config, 'progressive', {})
                        start_epoch_ratio = getattr(progressive_config, 'start_epoch_ratio', 0.0)  # 延迟启用FAT的epoch比例
                        # 支持多个阶段，从phase1到phase7（去掉1e-4，在1e-2和1e-1之间添加递进的BER）
                        phase1_epochs = getattr(progressive_config, 'phase1_epochs', 0.3)
                        phase2_epochs = getattr(progressive_config, 'phase2_epochs', 0.6)
                        phase3_epochs = getattr(progressive_config, 'phase3_epochs', 0.8)
                        phase4_epochs = getattr(progressive_config, 'phase4_epochs', 0.85)
                        phase5_epochs = getattr(progressive_config, 'phase5_epochs', 0.9)
                        phase6_epochs = getattr(progressive_config, 'phase6_epochs', 0.95)
                        phase7_epochs = getattr(progressive_config, 'phase7_epochs', 1.0)
                        
                        # 计算FAT启用的起始epoch
                        total_epochs = configs.epochs
                        start_epoch = int(total_epochs * start_epoch_ratio)
                        fat_epochs = total_epochs - start_epoch
                        
                        # 如果当前epoch在FAT启用之前，禁用FAT
                        if epoch < start_epoch:
                            use_fault_aware_training = False
                            current_ber = 0.0
                        else:
                            # 计算相对于整个训练进度的比例（不是FAT范围内的相对进度）
                            # phaseX_epochs配置的是整个训练进度的比例（如0.75表示75%）
                            progress = epoch / total_epochs if total_epochs > 0 else 0.0
                            
                            # 根据进度确定BER值（去掉1e-4，在1e-2和1e-1之间添加递进的故障率）
                            # 注意：如果phase6_epochs=1.0，则phase7被禁用，最高只到BER=5e-2
                            if progress < phase1_epochs:
                                current_ber = 1e-3  # 小故障，开始适应
                            elif progress < phase2_epochs:
                                current_ber = 1e-2  # 目标故障率
                            elif progress < phase3_epochs:
                                current_ber = 2e-2  # 逐步增加
                            elif progress < phase4_epochs:
                                current_ber = 3e-2  # 继续增加
                            elif progress < phase5_epochs:
                                current_ber = 4e-2  # 接近高故障率
                            elif progress < phase6_epochs:
                                current_ber = 5e-2  # 继续增加
                            elif phase6_epochs < 1.0 and progress < phase7_epochs:
                                # 只有当phase6_epochs < 1.0时，才使用phase7（BER=1e-1）
                                current_ber = 1e-1  # 极高故障率
                            else:
                                # 如果phase6_epochs=1.0，则最高只到BER=5e-2
                                current_ber = 5e-2 if phase6_epochs >= 1.0 else 1e-1
                            
                            # 更新fault_injector的BER值
                            fault_injector.ber = float(current_ber)
                    else:
                        # 固定BER策略
                        # 支持start_epoch参数（直接指定epoch数）或start_epoch_ratio（比例）
                        total_epochs = configs.epochs
                        start_epoch_direct = getattr(schedule_config, 'start_epoch', None)
                        if start_epoch_direct is not None:
                            # 直接指定epoch数
                            start_epoch = int(start_epoch_direct)
                        else:
                            # 使用比例（向后兼容）
                            start_epoch_ratio = getattr(schedule_config, 'start_epoch_ratio', 0.0)
                            start_epoch = int(total_epochs * start_epoch_ratio)
                        
                        # 如果当前epoch在FAT启用之前，禁用FAT
                        if epoch < start_epoch:
                            use_fault_aware_training = False
                            current_ber = 0.0
                        else:
                            # 使用固定BER
                            current_ber = getattr(fault_aware_training_config, 'ber', 1e-2)
                            # 确保current_ber是浮点数（YAML可能解析为字符串）
                            current_ber = float(current_ber)
                            fault_injector.ber = current_ber
                else:
                    # 没有启用调度，根据三阶段设置BER
                    if is_stage1:
                        # 第一阶段：不使用fault_injector，BER=0
                        current_ber = 0.0
                        use_fault_aware_training = False
                    elif is_stage2:
                        # 第二阶段：使用固定BER=2e-2
                        current_ber = getattr(fault_aware_training_config, 'ber', 2e-2)
                        current_ber = float(current_ber)
                        fault_injector.ber = current_ber
                    elif is_stage3:
                        # 第三阶段：BER递进（3e-2到8e-2）
                        # 计算第三阶段内的相对进度（0.0到1.0）
                        stage3_start = corrector_start_epoch
                        stage3_total = configs.epochs - stage3_start
                        stage3_progress = (epoch - stage3_start) / stage3_total if stage3_total > 0 else 0.0
                        
                        # BER从3e-2线性递进到8e-2
                        ber_start = 3e-2
                        ber_end = 8e-2
                        current_ber = ber_start + (ber_end - ber_start) * stage3_progress
                        current_ber = float(current_ber)
                        fault_injector.ber = current_ber
                
                if use_fault_aware_training:
                    use_entropy = getattr(trades_config, 'use_entropy', False)
                    entropy_weight = getattr(trades_config, 'entropy_weight', 0.1)
                    entropy_mode = getattr(trades_config, 'entropy_mode', 'difference')
                    use_self_compensation = getattr(trades_config, 'use_self_compensation', False)
                    compensation_weight = getattr(trades_config, 'compensation_weight', 0.1)
                    logger_info(logger, '=' * 80)
                    logger_info(logger, f'🔥 FAULT-AWARE TRAINING (FAT) - ACTIVE in train() function')
                    if is_stage1:
                        stage_info = "Stage 1 (Clean Training)"
                    elif is_stage2:
                        stage_info = "Stage 2 (Main Model FAT)"
                    else:
                        stage_info = "Stage 3 (Corrector Training)"
                    logger_info(logger, f'   Epoch {epoch}/{configs.epochs} (Progress: {epoch/configs.epochs*100:.1f}%), {stage_info}, TRADES Loss: {"KL Div" if use_kl else "Simple"}')
                    if is_stage1:
                        ber_schedule_info = "No Fault Injection"
                    elif is_stage2:
                        ber_schedule_info = "Fixed (2e-2)"
                    else:
                        ber_schedule_info = "Progressive (3e-2→8e-2)"
                    logger_info(logger, f'   Current BER: {current_ber:.2e} ({ber_schedule_info})')
                    if is_stage1:
                        seed_info = "N/A (No Fault Injection)"
                    elif is_stage2:
                        seed_info = "Single-seed (42)"
                    else:
                        seed_list = fault_injector.seed_list if hasattr(fault_injector, 'seed_list') and fault_injector.seed_list else []
                        current_seed = fault_injector.seed if hasattr(fault_injector, 'seed') else 'N/A'
                        seed_info = f"Multi-seed (Rotating, {len(seed_list)} seeds, current: {current_seed})"
                    logger_info(logger, f'   Seed Strategy: {seed_info}')
                    if not use_kl:
                        logger_info(logger, f'   Loss = {alpha} * loss_normal + {beta} * loss_faulted')
                    else:
                        logger_info(logger, f'   Loss = loss_normal + {beta} * KL(p_normal, p_faulted)')
                    if use_entropy:
                        logger_info(logger, f'   Entropy Regularization: Enabled (mode={entropy_mode}, weight={entropy_weight})')
                    if use_self_compensation:
                        logger_info(logger, f'   Self-Compensation: Enabled (weight={compensation_weight})')
                    logger_info(logger, '=' * 80)
                else:
                    logger_info(logger, f'⚠️  FAT is DISABLED for epoch {epoch} (will start at epoch {start_epoch if "start_epoch" in locals() else "N/A"})')

    logger_info(logger, 'Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    print(f'[DEBUG] Train loader length: {len(train_loader)}, Sampler: {train_loader.sampler}')
    
    num_updates = epoch * len(train_loader)
    seed = num_updates
    set_global_seed(seed + 1)
    print(f'[DEBUG] Setting global seed to {seed + 1}')
    
    print(f'[DEBUG] Setting model to train mode...')
    model.train()
    if model_ema:
        model_ema.ema.train()
    print(f'[DEBUG] Model set to train mode')

    T = 2 if epoch <= int(configs.epochs * 0.72) else 15

    if configs.enable_dynamic_bit_training and \
         epoch > 5 and (epoch + 1) % T == 0:
        print(f'[DEBUG] Processing dynamic bit training freeze logic...')
        freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
        freezing_metric = profile_layerwise_quantization_metric(model=model)
        freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio, 
                      progressive=False, logger=logger, org_cands=configs.target_bits
                      )
        logger_info(logger=logger, msg= f'Current freezing ratio: {freezing_ratio}')

    if teacher_model is not None:
        teacher_model.eval()
        print("Training with KD...")
    
    total_subnets = num_fixed_sample + nr_random_sample
    print(f'[DEBUG] Starting training loop, total_subnets={total_subnets}, train_loader batches={len(train_loader)}')
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx == 0:
            print(f'[DEBUG] Processing first batch, inputs shape: {inputs.shape}, targets shape: {targets.shape}')
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        external_teacher_outputs = None
        if teacher_model is not None and soft_criterion is not None:
            with torch.no_grad():
                external_teacher_outputs = teacher_model(inputs)

        QE_loss_weight = annealing_schedule(num_updates) # We use a scheduler for the weights of QE loss according to QAT Oscillations Overcoming [ICML'22]. 

        if sample_current_max:
            start_time = time.time()

            sample_max_cands(model, configs)

            if information_distortion_mitigation:
                target_features = []
                hooks = set_forward_hook_for_quantized_layers(model, target_features, is_max=True)

            max_outputs = model(inputs)

            loss, QE_loss, dist_loss = compute_overall_loss(max_outputs, external_teacher_outputs, targets, criterion, model, quantization_error_minimization=False, 
                                                                configs=configs, disable_smallest_regularization=True)

            # ========== 三阶段训练：第二阶段不更新主模型 ==========
            if not should_freeze_model:
                # 第一阶段和第二阶段（主模型FAT训练）：正常反向传播
                loss.backward()
            else:
                # 第三阶段（主模型已冻结）：不反向传播，只用于统计
                pass

            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

            teacher_outputs = max_outputs.clone().detach()
            
            # ========== 三阶段训练：第二阶段需要detach loss用于统计 ==========
            if should_freeze_model:
                # 第三阶段：主模型已冻结，loss需要detach用于统计
                loss_detached = loss.detach() if hasattr(loss, 'detach') and isinstance(loss, torch.Tensor) else loss
                QE_loss_detached = QE_loss.detach() if QE_loss is not None and hasattr(QE_loss, 'detach') and isinstance(QE_loss, torch.Tensor) else QE_loss
                dist_loss_detached = dist_loss.detach() if dist_loss is not None and hasattr(dist_loss, 'detach') and isinstance(dist_loss, torch.Tensor) else dist_loss
                loss = loss_detached
                QE_loss = QE_loss_detached
                dist_loss = dist_loss_detached
            
            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[0], loss, QE_loss, dist_loss, 0, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
                
        weight_conf_pool = []

        for iter_idx in range(nr_random_sample):
            start_time = time.time()
            active_ber_for_iter = float(current_ber if current_ber is not None else 0.0)

            w_conf, a_conf, min_w_index = sample_one_mixed_policy(model, configs)
            weight_conf_pool.append(w_conf)
            
            if information_distortion_mitigation:
                distorted_features = []
                hooks = set_forward_hook_for_quantized_layers(model, distorted_features, is_max=False)

            # === TRADES风格的故障感知训练 ===
            if use_fault_aware_training and fault_injector is not None:
                # 获取TRADES配置参数
                trades_config = getattr(fault_aware_training_config, 'trades', {})
                use_kl = getattr(trades_config, 'use_kl', False)
                alpha = getattr(trades_config, 'alpha', 0.6)
                beta = getattr(trades_config, 'beta', 1.0)
                
                # 第一次forward: 正常情况（无故障）
                fault_injector.disable()
                if batch_idx == 0 and iter_idx == 0:
                    logger_info(logger, f'[FAT] Batch {batch_idx}, Iter {iter_idx}: First forward (NORMAL, no fault)')
                
                # 提取正常运行的中间层激活值（用于corrector故障诊断）
                # 在作用域外定义，确保后续corrector训练可以访问
                normal_activations = []
                faulted_activations = []
                activation_hooks = None
                activation_hooks_faulted = None
                if use_corrector and output_corrector is not None:
                    from util.qat import set_forward_hook_for_conv_linear_layers, remove_hook_for_quantized_layers
                    activation_hooks = set_forward_hook_for_conv_linear_layers(model, normal_activations)
                
                outputs_normal = model(inputs)
                
                # 移除hook
                if activation_hooks is not None:
                    from util.qat import remove_hook_for_quantized_layers
                    remove_hook_for_quantized_layers(activation_hooks)
                loss_normal, QE_loss_normal, dist_loss_normal = compute_overall_loss(
                    outputs_normal, teacher_outputs, targets, criterion, model, 
                    quantization_error_minimization=epoch>40, 
                    QE_loss_weight=QE_loss_weight, 
                    disable_smallest_regularization=True, 
                    configs=configs
                )
                
                # 第二次forward: 故障注入
                effective_ber = active_ber_for_iter
                trigger_index = batch_idx * max(1, nr_random_sample) + iter_idx
                if is_stage3 and stage3_anchor_prob > 0.0:
                    anchor_period = max(int(round(1.0 / max(stage3_anchor_prob, 1e-6))), 1)
                    if anchor_period > 0 and trigger_index % anchor_period == 0:
                        effective_ber = stage3_anchor_ber
                elif is_stage2 and stage2_mix_prob > 0.0 and len(stage2_mix_bers) > 0:
                    mix_period = max(int(round(1.0 / max(stage2_mix_prob, 1e-6))), 1)
                    if mix_period > 0 and trigger_index % mix_period == 0:
                        mix_idx = (trigger_index // mix_period) % len(stage2_mix_bers)
                        effective_ber = stage2_mix_bers[mix_idx]
                active_ber_for_iter = float(effective_ber)
                fault_injector.ber = active_ber_for_iter
                fault_injector.enable()
                # Reset forward seed to ensure all layers in this forward use the same base_seed
                fault_injector.reset_forward_seed()
                if batch_idx == 0 and iter_idx == 0:
                    logger_info(logger, f'[FAT] Batch {batch_idx}, Iter {iter_idx}: Second forward (FAULTED, BER={active_ber_for_iter:.2e})')
                
                # 提取故障运行的中间层激活值（用于corrector故障诊断）
                if use_corrector and output_corrector is not None:
                    from util.qat import set_forward_hook_for_conv_linear_layers, remove_hook_for_quantized_layers
                    activation_hooks_faulted = set_forward_hook_for_conv_linear_layers(model, faulted_activations)
                
                outputs_faulted = model(inputs)
                
                # 移除hook
                if activation_hooks_faulted is not None:
                    from util.qat import remove_hook_for_quantized_layers
                    remove_hook_for_quantized_layers(activation_hooks_faulted)
                loss_faulted, QE_loss_faulted, dist_loss_faulted = compute_overall_loss(
                    outputs_faulted, teacher_outputs, targets, criterion, model, 
                    quantization_error_minimization=epoch>40, 
                    QE_loss_weight=QE_loss_weight, 
                    disable_smallest_regularization=True, 
                    configs=configs
                )
                
                # TRADES损失计算（支持信息熵正则化和自适应对齐补偿）
                use_entropy = getattr(trades_config, 'use_entropy', False)
                entropy_weight = getattr(trades_config, 'entropy_weight', 0.1)
                entropy_mode = getattr(trades_config, 'entropy_mode', 'difference')  # 'difference', 'constraint', 'balance'
                
                # 自适应对齐补偿配置
                use_self_compensation = getattr(trades_config, 'use_self_compensation', False)
                compensation_weight = getattr(trades_config, 'compensation_weight', 0.1)
                
                # 输出修正器配置
                use_corrector = getattr(trades_config, 'use_corrector', False)
                corrector_weight = getattr(trades_config, 'corrector_weight', 0.1)
                
                if use_kl:
                    # 使用KL散度: L = L(x_normal, y) + β * KL(p(x_normal), p(x_faulted))
                    probs_normal = F.softmax(outputs_normal, dim=1)
                    log_probs_faulted = F.log_softmax(outputs_faulted, dim=1)
                    kl_div = F.kl_div(log_probs_faulted, probs_normal, reduction='batchmean')
                    loss = loss_normal + beta * kl_div
                    
                    # 信息熵正则化（如果启用）
                    if use_entropy:
                        entropy_loss = compute_entropy_loss(probs_normal, F.softmax(outputs_faulted, dim=1), mode=entropy_mode)
                        loss = loss + entropy_weight * entropy_loss
                    
                    # 自适应对齐补偿（如果启用）
                    if use_self_compensation:
                        # 计算概率分布差异
                        probs_faulted = F.softmax(outputs_faulted, dim=1)
                        prob_diff = probs_normal - probs_faulted
                        
                        # 自适应权重：对差异大的样本给予更大权重
                        # adaptive_weight = torch.abs(prob_diff).sum(dim=1)  # 总差异
                        adaptive_weight = torch.abs(prob_diff).sum(dim=1)  # [batch_size]
                        
                        # 自补偿损失：让模型自动学习减少概率分布差异
                        # 对每个样本，计算概率差异的平方，然后乘以自适应权重
                        compensation_loss = (prob_diff ** 2).mean(dim=1) * adaptive_weight  # [batch_size]
                        compensation_loss = compensation_loss.mean()  # 标量
                        
                        loss = loss + compensation_weight * compensation_loss
                    
                    # 输出修正器训练（完全独立，不影响模型loss）
                    # Corrector是独立的模块，使用独立的optimizer单独训练
                    # 注意：corrector_loss不加入模型的loss中，保持完全独立
                    if use_corrector and output_corrector is not None and use_fault_aware_training:
                        # 使用修正器修正故障输出（使用detach确保不影响模型梯度）
                        outputs_faulted_detached = outputs_faulted.detach()
                        outputs_normal_detached = outputs_normal.detach()
                        
                        # 使用激活值进行故障诊断（B+A方案）
                        # normal_activations和faulted_activations在两次forward时已提取
                        normal_acts_for_corrector = [act.detach() for act in normal_activations] if len(normal_activations) > 0 else None
                        faulted_acts_for_corrector = [act.detach() for act in faulted_activations] if len(faulted_activations) > 0 else None
                        
                        # ========== 更新正常状态基准（EMA）==========
                        # B+A方案：更新概率原型p_c和能量原型e_c
                        # 三阶段训练：第二阶段（50-199）也更新baseline，为第三阶段训练做准备
                        output_corrector.update_baseline(
                            outputs_normal_detached,
                            activations=normal_acts_for_corrector,
                            targets=targets
                        )
                        
                        # ========== B+A混合修正：能量门控 + 概率修正 ==========
                        # B（能量）：WM-ID_e → 门控scale（要不要修）
                        # A（概率）：z_p → 修正方向（怎么修）
                        # 注意：即使在第二阶段（不训练corrector），也要计算FltAcc和CorrAcc用于监控
                        outputs_corrected = output_corrector(
                            outputs_faulted_detached,
                            activations=faulted_acts_for_corrector,
                            targets=targets
                        )
                        
                        # 预先计算预测结果（用于统计FltAcc和CorrAcc）
                        with torch.no_grad():
                            pred_faulted = outputs_faulted_detached.argmax(dim=1)
                            pred_normal = outputs_normal_detached.argmax(dim=1)
                            pred_corrected = outputs_corrected.argmax(dim=1)
                            is_faulted_wrong = (pred_faulted != targets)
                            is_faulted_correct = (pred_faulted == targets)
                            
                            # 计算FltAcc和CorrAcc（第二阶段和第三阶段都需要）
                            if corrector_stats is not None:
                                acc_faulted_val = (pred_faulted == targets).float().mean().item() * 100
                                acc_corrected_val = (pred_corrected == targets).float().mean().item() * 100
                                acc_improvement_val = acc_corrected_val - acc_faulted_val
                                
                                corrector_stats['acc_faulted'].update(acc_faulted_val, inputs.size(0))
                                corrector_stats['acc_corrected'].update(acc_corrected_val, inputs.size(0))
                                corrector_stats['acc_improvement'].update(acc_improvement_val, inputs.size(0))
                        
                        # ========== 三阶段训练：只在第三阶段（should_train_corrector=True）时训练corrector ==========
                        # 第一阶段和第二阶段：只更新baseline和统计FltAcc/CorrAcc，不训练corrector
                        if should_train_corrector:
                            # 确保corrector_optimizer存在
                            if corrector_optimizer is None:
                                logger_info(logging.getLogger(__name__), 
                                          f"[Three-Stage Training] Epoch {epoch}: corrector_optimizer is None, skipping corrector training")
                            else:
                                # ========== V8：自监督相对修正训练目标 ==========
                                correction_pred = outputs_corrected - outputs_faulted_detached
                                correction_target = outputs_normal_detached - outputs_faulted_detached
                                mse_loss = F.mse_loss(correction_pred, correction_target)
                                ce_loss = F.cross_entropy(outputs_corrected, targets)
                                sparse_loss = correction_pred.abs().mean()

                                eps = 1e-6
                                pred_norm = correction_pred.norm(dim=1, keepdim=True)
                                target_norm = correction_target.norm(dim=1, keepdim=True)
                                direction_loss = torch.tensor(0.0, device=outputs_corrected.device)
                                valid_mask = (target_norm > eps).float()
                                if torch.any(valid_mask > 0):
                                    direction_pred = correction_pred / (pred_norm + eps)
                                    direction_target = correction_target / (target_norm + eps)
                                    cosine = (direction_pred * direction_target).sum(dim=1, keepdim=True)
                                    direction_loss = (1 - cosine).mul(valid_mask).sum() / (valid_mask.sum() + eps)
                                magnitude_target = torch.clamp(target_norm, max=getattr(output_corrector, 'max_correction', 3.0))
                                magnitude_loss = F.l1_loss(pred_norm, magnitude_target)

                                # ========== 稳定性约束：不过度修正已正确的预测 ==========
                                stability_loss = torch.tensor(0.0, device=outputs_corrected.device)
                                if is_faulted_correct.any():
                                    correction_on_correct = outputs_corrected[is_faulted_correct] - outputs_faulted_detached[is_faulted_correct]
                                    stability_loss = (correction_on_correct ** 2).mean()
                                
                                normal_stability_loss = torch.tensor(0.0, device=outputs_corrected.device)
                                if normal_acts_for_corrector is not None:
                                    outputs_normal_corrected = output_corrector(
                                        outputs_normal_detached,
                                        activations=normal_acts_for_corrector,
                                        targets=targets
                                    )
                                    correction_on_normal = outputs_normal_corrected - outputs_normal_detached
                                    normal_stability_loss = (correction_on_normal ** 2).mean()
                                
                                # 合并稳定性损失
                                total_stability_loss = 0.5 * stability_loss + 0.5 * normal_stability_loss
                                
                                # ========== 修正成功率加权（可选 RL ）==========
                                reward_weight = torch.tensor(1.0, device=outputs_corrected.device)
                                use_rl_weighted = getattr(trades_config, 'corrector_use_rl_weighted', True)
                                if use_rl_weighted:
                                    with torch.no_grad():
                                        # 只统计"原本错误，修正后正确"的样本
                                        correction_success = (pred_corrected == targets) & is_faulted_wrong
                                        correction_rate = correction_success.float().mean()
                                        # sigmoid归一化：修正成功率越高，权重越大
                                        reward_weight = torch.sigmoid(correction_rate * 10)

                                # ========== 总损失 ==========
                                w_mse = getattr(trades_config, 'corrector_weight_mse', 1.0)
                                w_ce = getattr(trades_config, 'corrector_weight_ce', 0.5)
                                w_sparse = getattr(trades_config, 'corrector_weight_sparse', 0.05)
                                w_stability = getattr(trades_config, 'corrector_weight_stability', 0.1)

                                corrector_loss = (
                                    w_mse * mse_loss +
                                    w_ce * ce_loss +
                                    w_sparse * sparse_loss +
                                    w_stability * total_stability_loss
                                ) * reward_weight
                                
                                # 独立训练corrector（不影响模型loss）
                                corrector_optimizer.zero_grad()
                                corrector_loss.backward()
                                corrector_optimizer.step()
                                
                                # 累积corrector训练统计信息（第三阶段）
                                if corrector_stats is not None:
                                    corrector_stats['loss'].update(corrector_loss.item(), inputs.size(0))
                                    if use_rl_weighted and corrector_stats['rl_weight'] is not None:
                                        corrector_stats['rl_weight'].update(reward_weight.item(), inputs.size(0))
                    
                    # 使用normal的QE和dist loss
                    QE_loss = QE_loss_normal
                    dist_loss = dist_loss_normal
                else:
                    # 使用简单组合: L = α * L(x_normal, y) + β * L(x_faulted, y)
                    loss = alpha * loss_normal + beta * loss_faulted
                    
                    # 信息熵正则化（如果启用）
                    if use_entropy:
                        probs_normal = F.softmax(outputs_normal, dim=1)
                        probs_faulted = F.softmax(outputs_faulted, dim=1)
                        entropy_loss = compute_entropy_loss(probs_normal, probs_faulted, mode=entropy_mode)
                        loss = loss + entropy_weight * entropy_loss
                    
                    # 自适应对齐补偿（如果启用）
                    if use_self_compensation:
                        probs_normal = F.softmax(outputs_normal, dim=1)
                        probs_faulted = F.softmax(outputs_faulted, dim=1)
                        prob_diff = probs_normal - probs_faulted
                        adaptive_weight = torch.abs(prob_diff).sum(dim=1)
                        compensation_loss = (prob_diff ** 2).mean(dim=1) * adaptive_weight
                        compensation_loss = compensation_loss.mean()
                        loss = loss + compensation_weight * compensation_loss
                    
                    # 输出修正器训练（完全独立，不影响模型loss）
                    # Corrector是独立的模块，使用独立的optimizer单独训练
                    if use_corrector and output_corrector is not None and use_fault_aware_training:
                        # 使用修正器修正故障输出（使用detach确保不影响模型梯度）
                        outputs_faulted_detached = outputs_faulted.detach()
                        outputs_normal_detached = outputs_normal.detach()
                        
                        # 使用激活值进行故障诊断（B+A方案）
                        # normal_activations和faulted_activations在两次forward时已提取
                        normal_acts_for_corrector = [act.detach() for act in normal_activations] if len(normal_activations) > 0 else None
                        faulted_acts_for_corrector = [act.detach() for act in faulted_activations] if len(faulted_activations) > 0 else None
                        
                        # ========== 更新正常状态基准（EMA）==========
                        # B+A方案：更新概率原型p_c和能量原型e_c
                        output_corrector.update_baseline(
                            outputs_normal_detached,
                            activations=normal_acts_for_corrector,
                            targets=targets
                        )
                        
                        # ========== B+A混合修正：能量门控 + 概率修正 ==========
                        # B（能量）：WM-ID_e → 门控scale（要不要修）
                        # A（概率）：z_p → 修正方向（怎么修）
                        # 注意：即使在第二阶段（不训练corrector），也要计算FltAcc和CorrAcc用于监控
                        fault_context = {
                            'ber': active_ber_for_iter,
                            'stage': stage_label,
                            'update_stats': True
                        }
                        outputs_corrected, corrector_details = output_corrector(
                            outputs_faulted_detached,
                            activations=faulted_acts_for_corrector,
                            targets=targets,
                            fault_context=fault_context,
                            return_details=True
                        )
                        
                        # 预先计算预测结果（用于统计FltAcc和CorrAcc）
                        with torch.no_grad():
                            pred_faulted = outputs_faulted_detached.argmax(dim=1)
                            pred_normal = outputs_normal_detached.argmax(dim=1)
                            pred_corrected = outputs_corrected.argmax(dim=1)
                            is_faulted_wrong = (pred_faulted != targets)
                            is_faulted_correct = (pred_faulted == targets)
                            
                            # 计算FltAcc和CorrAcc（第二阶段和第三阶段都需要）
                            if corrector_stats is not None:
                                acc_faulted_val = (pred_faulted == targets).float().mean().item() * 100
                                acc_corrected_val = (pred_corrected == targets).float().mean().item() * 100
                                acc_improvement_val = acc_corrected_val - acc_faulted_val
                                
                                corrector_stats['acc_faulted'].update(acc_faulted_val, inputs.size(0))
                                corrector_stats['acc_corrected'].update(acc_corrected_val, inputs.size(0))
                                corrector_stats['acc_improvement'].update(acc_improvement_val, inputs.size(0))
                        pred_faulted_idx = pred_faulted.unsqueeze(1)
                        gap_for_stats = outputs_faulted_detached.gather(1, targets.unsqueeze(1)) - outputs_faulted_detached.gather(1, pred_faulted_idx)
                        output_corrector.update_gap_statistics(gap_for_stats.squeeze(1), active_ber_for_iter)
                        
                        # ========== V9：方向 + 相对幅度监督 ==========
                        correction_pred = outputs_corrected - outputs_faulted_detached
                        correction_target = outputs_normal_detached - outputs_faulted_detached
                        target_delta = correction_target.gather(1, targets.unsqueeze(1)).squeeze(1)
                        direction_mask = target_delta.abs() > output_corrector.direction_deadzone

                        use_rl_weighted = getattr(trades_config, 'corrector_use_rl_weighted', True)
                        if use_rl_weighted:
                            reward_gain = float(getattr(trades_config, 'corrector_rl_gain', 0.5))
                            reward_penalty = float(getattr(trades_config, 'corrector_rl_penalty', 0.3))
                            sample_reward = torch.ones_like(targets, dtype=torch.float, device=targets.device)
                            sample_reward = sample_reward + reward_gain * ((pred_corrected == targets) & is_faulted_wrong).float()
                            sample_reward = sample_reward - reward_penalty * ((pred_corrected != targets) & is_faulted_correct).float()
                            sample_reward = sample_reward.clamp_min(0.1).detach()
                        else:
                            sample_reward = torch.ones_like(targets, dtype=torch.float, device=targets.device)
                        avg_reward = sample_reward.mean().item()

                        mse_loss_vec = ((correction_pred - correction_target) ** 2).mean(dim=1)
                        mse_loss = (mse_loss_vec * sample_reward).mean()
                        ce_loss_vec = F.cross_entropy(outputs_corrected, targets, reduction='none')
                        ce_loss = (ce_loss_vec * sample_reward).mean()
                        sparse_loss_vec = correction_pred.abs().mean(dim=1)
                        sparse_loss = (sparse_loss_vec * sample_reward).mean()

                        # ========== 稳定性约束 ==========
                        stability_loss = torch.tensor(0.0, device=outputs_corrected.device)
                        if is_faulted_correct.any():
                            correction_on_correct = outputs_corrected[is_faulted_correct] - outputs_faulted_detached[is_faulted_correct]
                            stability_loss = (correction_on_correct ** 2).mean()
                        
                        normal_stability_loss = torch.tensor(0.0, device=outputs_corrected.device)
                        if normal_acts_for_corrector is not None:
                            outputs_normal_corrected = output_corrector(
                                outputs_normal_detached,
                                activations=normal_acts_for_corrector,
                                targets=targets,
                                fault_context={'ber': 0.0, 'stage': 'clean', 'update_stats': False}
                            )
                            correction_on_normal = outputs_normal_corrected - outputs_normal_detached
                            normal_stability_loss = (correction_on_normal ** 2).mean()
                        
                        total_stability_loss = 0.5 * stability_loss + 0.5 * normal_stability_loss

                        direction_loss = torch.tensor(0.0, device=outputs_corrected.device)
                        magnitude_loss = torch.tensor(0.0, device=outputs_corrected.device)
                        direction_logits = corrector_details['direction_logits']
                        magnitude_unit = corrector_details['magnitude_unit'].squeeze(1)
                        if direction_mask.any():
                            direction_targets = (target_delta > 0).float()
                            direction_logits_target = direction_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                            direction_loss_vec = F.binary_cross_entropy_with_logits(
                                direction_logits_target[direction_mask],
                                direction_targets[direction_mask],
                                reduction='none'
                            )
                            direction_loss = (direction_loss_vec * sample_reward[direction_mask]).mean()

                            magnitude_target = torch.clamp(target_delta.abs() / output_corrector.max_correction, max=1.0)
                            magnitude_loss_vec = F.smooth_l1_loss(
                                magnitude_unit[direction_mask],
                                magnitude_target[direction_mask],
                                reduction='none'
                            )
                            magnitude_loss = (magnitude_loss_vec * sample_reward[direction_mask]).mean()

                        # ========== 总损失 ==========
                        w_mse = getattr(trades_config, 'corrector_weight_mse', 1.0)
                        w_ce = getattr(trades_config, 'corrector_weight_ce', 0.5)
                        w_sparse = getattr(trades_config, 'corrector_weight_sparse', 0.05)
                        w_stability = getattr(trades_config, 'corrector_weight_stability', 0.1)
                        w_direction = getattr(trades_config, 'corrector_weight_direction', 0.5)
                        w_magnitude = getattr(trades_config, 'corrector_weight_magnitude', 0.5)

                        corrector_loss = (
                            w_mse * mse_loss +
                            w_ce * ce_loss +
                            w_sparse * sparse_loss +
                            w_stability * total_stability_loss +
                            w_direction * direction_loss +
                            w_magnitude * magnitude_loss
                        )
                        
                        if should_train_corrector:
                            if corrector_optimizer is None:
                                logger_info(logging.getLogger(__name__),
                                            f"[Three-Stage Training] Epoch {epoch}: corrector_optimizer is None, skipping corrector training")
                            else:
                                corrector_optimizer.zero_grad()
                                corrector_loss.backward()
                                corrector_optimizer.step()
                                if corrector_stats is not None:
                                    corrector_stats['loss'].update(corrector_loss.item(), inputs.size(0))
                        if corrector_stats is not None and corrector_stats['rl_weight'] is not None:
                            corrector_stats['rl_weight'].update(avg_reward, inputs.size(0))
                    
                    # 组合QE和dist loss（非KL分支）
                    QE_loss = alpha * QE_loss_normal + beta * QE_loss_faulted
                    dist_loss = alpha * dist_loss_normal + beta * dist_loss_faulted
                
                # 使用normal输出计算准确率（用于显示）
                outputs = outputs_normal
                
                # 计算并记录corrector的准确率（如果启用）
                if use_corrector and output_corrector is not None:
                    with torch.no_grad():
                        # 使用故障输出和激活值（24维输入）
                        normal_acts_for_corrector = [act.detach() for act in normal_activations] if len(normal_activations) > 0 else None
                        faulted_acts_for_corrector = [act.detach() for act in faulted_activations] if len(faulted_activations) > 0 else None
                        outputs_corrected = output_corrector(
                            outputs_faulted,
                            activations=faulted_acts_for_corrector,
                            targets=targets,
                            fault_context={'ber': active_ber_for_iter, 'stage': stage_label}
                        )
                        acc1_corrected, acc5_corrected = accuracy(outputs_corrected.data, targets.data, topk=(1, 5))
                        acc1_normal, acc5_normal = accuracy(outputs_normal.data, targets.data, topk=(1, 5))
                        if batch_idx == 0 and iter_idx == 0:
                            logger_info(logger, f'[Corrector] Batch {batch_idx}, Iter {iter_idx}: Model acc={acc1_normal.item():.2f}%, With Corrector acc={acc1_corrected.item():.2f}%')
                
                if batch_idx == 0 and iter_idx == 0:
                    entropy_info = ""
                    if use_entropy:
                        probs_n = F.softmax(outputs_normal, dim=1)
                        probs_f = F.softmax(outputs_faulted, dim=1)
                        entropy_n = compute_entropy(probs_n).mean().item()
                        entropy_f = compute_entropy(probs_f).mean().item()
                        entropy_info = f", entropy_normal={entropy_n:.4f}, entropy_faulted={entropy_f:.4f}"
                    compensation_info = ""
                    if use_self_compensation:
                        probs_n = F.softmax(outputs_normal, dim=1)
                        probs_f = F.softmax(outputs_faulted, dim=1)
                        prob_diff = probs_n - probs_f
                        prob_diff_norm = (prob_diff ** 2).mean().item()
                        compensation_info = f", compensation_loss={prob_diff_norm:.4f}"
                    corrector_info = ""
                    if use_corrector and output_corrector is not None:
                        # 使用故障输出和激活值（24维输入）
                        normal_acts_for_corrector = [act.detach() for act in normal_activations] if len(normal_activations) > 0 else None
                        faulted_acts_for_corrector = [act.detach() for act in faulted_activations] if len(faulted_activations) > 0 else None
                        outputs_c = output_corrector(
                            outputs_faulted,
                            normal_intermediate_activations=normal_acts_for_corrector,
                            faulted_intermediate_activations=faulted_acts_for_corrector,
                            fault_context={'ber': active_ber_for_iter, 'stage': stage_label}
                        )
                        corrector_loss_val = F.mse_loss(outputs_c, outputs_normal).item()
                        corrector_info = f", corrector_loss={corrector_loss_val:.4f}"
                    logger_info(logger, f'[FAT] Batch {batch_idx}, Iter {iter_idx}: TRADES loss computed, normal_loss={loss_normal.item():.4f}, faulted_loss={loss_faulted.item():.4f}{entropy_info}{compensation_info}{corrector_info}')
            else:
                # 原有训练流程（无故障感知训练）
                if batch_idx == 0 and iter_idx == 0:
                    logger_info(logger, f'[FAT] Batch {batch_idx}, Iter {iter_idx}: FAT is DISABLED, using standard training')
                outputs = model(inputs)
                loss, QE_loss, dist_loss = compute_overall_loss(
                    outputs, teacher_outputs, targets, criterion, model, 
                    quantization_error_minimization=epoch>40, 
                    QE_loss_weight=QE_loss_weight, 
                    disable_smallest_regularization=True, 
                    configs=configs
                )
                
                # 前50轮（FAT未启用时），corrector不训练
                # 理由：
                # 1. 前50轮没有故障样本，学习恒等映射没有意义
                # 2. 学习恒等映射可能让corrector陷入局部最优，难以在FAT启用后学习有用的修正模式
                # 3. 随机初始化的corrector在FAT启用后可以更自由地学习，不受恒等映射约束
                # 4. 更符合"corrector只在有故障时才需要"的设计理念
                # Corrector保持随机初始化，在FAT启用后（第50轮后）才开始学习
                if output_corrector is not None and batch_idx == 0 and iter_idx == 0:
                    trades_config = getattr(fault_aware_training_config, 'trades', {}) if fault_aware_training_config else {}
                    use_corrector = getattr(trades_config, 'use_corrector', False)
                    if use_corrector:
                        # 计算FAT启用的起始epoch
                        schedule_config = getattr(fault_aware_training_config, 'schedule', None) if fault_aware_training_config else None
                        if schedule_config and getattr(schedule_config, 'enabled', False):
                            progressive_config = getattr(schedule_config, 'progressive', {})
                            start_epoch_ratio = getattr(progressive_config, 'start_epoch_ratio', 0.25)
                            start_epoch = int(configs.epochs * start_epoch_ratio)
                        else:
                            start_epoch = 0
                        logger_info(logger, f'[Corrector] Pre-FAT phase (epoch < {start_epoch}): Corrector not trained, keeping random initialization. Training will start when FAT is enabled.')

            IDM_loss = 0
            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

                IDM_loss = sum([F.mse_loss(s, t).sum() if s is not None else 0 for s, t in zip(distorted_features, target_features)])
                loss += (IDM_loss * IDM_weight)
            
            # ========== 两阶段训练：第二阶段不更新主模型 ==========
            if not should_freeze_model:
                # 第一阶段：正常训练主模型
                loss.backward()
                acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                update_meter(meters[iter_idx+num_fixed_sample], loss, QE_loss, dist_loss, IDM_loss, 
                            acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
            else:
                # 第二阶段：主模型已冻结，只计算loss用于统计，不反向传播
                # 注意：loss需要在no_grad外先detach，否则在no_grad内无法访问
                # 检查loss是否被正确初始化
                if 'loss' not in locals() or loss is None:
                    logger.warning(f"[Stage 2] Loss not initialized for Mixed {iter_idx}, batch {batch_idx}")
                    loss = torch.tensor(0.0, device=inputs.device)
                
                loss_detached = loss.detach() if hasattr(loss, 'detach') and isinstance(loss, torch.Tensor) else (torch.tensor(float(loss), device=inputs.device) if not isinstance(loss, torch.Tensor) else loss)
                QE_loss_detached = QE_loss.detach() if QE_loss is not None and hasattr(QE_loss, 'detach') and isinstance(QE_loss, torch.Tensor) else (QE_loss if QE_loss is not None else torch.tensor(0.0, device=inputs.device))
                dist_loss_detached = dist_loss.detach() if dist_loss is not None and hasattr(dist_loss, 'detach') and isinstance(dist_loss, torch.Tensor) else (dist_loss if dist_loss is not None else torch.tensor(0.0, device=inputs.device))
                IDM_loss_detached = IDM_loss.detach() if IDM_loss is not None and hasattr(IDM_loss, 'detach') and isinstance(IDM_loss, torch.Tensor) else (IDM_loss if IDM_loss is not None else torch.tensor(0.0, device=inputs.device))
                
                with torch.no_grad():
                    acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                    # Debug: 打印loss值
                    if batch_idx == 0 and iter_idx == 0:
                        logger.info(f"[Stage 2 Debug] Mixed {iter_idx}, loss={loss_detached.item() if isinstance(loss_detached, torch.Tensor) else loss_detached:.6f}, QE_loss={QE_loss_detached.item() if isinstance(QE_loss_detached, torch.Tensor) else QE_loss_detached:.6f}, acc1={acc1.item():.2f}")
                    update_meter(meters[iter_idx+num_fixed_sample], loss_detached, QE_loss_detached, dist_loss_detached, IDM_loss_detached, 
                                acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        # ========== 两阶段训练：第二阶段不更新主模型optimizer ==========
        if not should_freeze_model:
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            if optimizer_q is not None:
                optimizer_q.step()
        else:
            # 第二阶段：主模型已冻结，不更新optimizer
            pass

        num_updates += 1

        if model_ema is not None:
            model_ema.update(model)
        
        if (batch_idx + 1) % configs.log.print_freq == 0:
            update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, nr_random_sample, optimizer, optimizer_q, mode=mode, corrector_stats=corrector_stats)
            logger_info(logger, "="*140)  # 加长分隔线以适应更多信息

    show_training_info(meters, target_bits, nr_random_sample, mode=mode)
    
    # 如果有corrector，在训练结束时也打印corrector的详细统计信息
    if corrector_stats is not None:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None:
            trades_config = getattr(fault_aware_training_config, 'trades', {})
            use_corrector = getattr(trades_config, 'use_corrector', False)
            if use_corrector:
                logger_info(logger, '=' * 80)
                logger_info(logger, f'🔧 CORRECTOR TRAINING SUMMARY - Epoch {epoch}')
                logger_info(logger, '=' * 80)
                logger_info(logger, f'  Model Top1: {meters[0]["top1"].avg:.2f}%')
                if use_fault_aware_training:
                    logger_info(logger, f'  Corrector Loss: {corrector_stats["loss"].avg:.4f}')
                    logger_info(logger, f'  Faulted Acc: {corrector_stats["acc_faulted"].avg:.2f}%')
                    logger_info(logger, f'  Corrected Acc: {corrector_stats["acc_corrected"].avg:.2f}%')
                    logger_info(logger, f'  Acc Improvement: {corrector_stats["acc_improvement"].avg:+.2f}%')
                    if corrector_stats['rl_weight'] is not None:
                        logger_info(logger, f'  RL Weight (avg): {corrector_stats["rl_weight"].avg:.4f}')
                    if corrector_optimizer is not None:
                        logger_info(logger, f'  Corrector LR: {corrector_optimizer.param_groups[0]["lr"]:.6f}')
                else:
                    logger_info(logger, f'  Status: Pre-FAT phase, Corrector not trained (will start at epoch {start_epoch if "start_epoch" in locals() else "N/A"})')
                logger_info(logger, '=' * 80)
    
    return meters[0]['top1'].avg, meters[0]['top5'].avg, meters[0]['loss'].avg


def validate(data_loader, model, criterion, epoch, monitors, configs, nr_random_sample=3, alpha=1, train_loader=None, eval_predefined_arch=None, bops_limit=1e10, train_mode=False, output_corrector=None):
    target_bits = configs.target_bits

    criterion = torch.nn.CrossEntropyLoss().cuda()

    meters = [{
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'QE_loss': AverageMeter(),
        'dist_loss': AverageMeter(),
        'IDM_loss': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(len(target_bits) + nr_random_sample)]

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size

    logger_info(logger, msg=f'Validation: {total_sample} samples ({batch_size} per mini-batch)')

    model.eval()

    def _eval(_loader, meter, meter_corrected=None):
        for batch_idx, (inputs, targets) in enumerate(_loader):
            inputs = inputs.to(configs.device)
            targets = targets.to(configs.device)
            start_time = time.time()

            # 如果启用corrector，在forward时提取激活值
            current_activations = []
            activation_hooks = None
            if batch_idx == 0:  # 第一个batch打印调试信息
                print(f"[DEBUG _eval] output_corrector is not None: {output_corrector is not None}")
                print(f"[DEBUG _eval] meter_corrected is not None: {meter_corrected is not None}")
            if output_corrector is not None and meter_corrected is not None:
                from util.qat import set_forward_hook_for_conv_linear_layers, remove_hook_for_quantized_layers
                activation_hooks = set_forward_hook_for_conv_linear_layers(model, current_activations)
                if batch_idx == 0:
                    print(f"[DEBUG _eval] Registered {len(activation_hooks)} activation hooks")
                if batch_idx == 0:
                    logger_info(logger, f"[DEBUG] Registered {len(activation_hooks)} activation hooks for corrector")
            
            outputs = model(inputs)
            
            # 移除hook
            if activation_hooks is not None:
                from util.qat import remove_hook_for_quantized_layers
                remove_hook_for_quantized_layers(activation_hooks)
                if batch_idx == 0:
                    logger_info(logger, f"[DEBUG] Extracted {len(current_activations)} activations, shapes: {[act.shape for act in current_activations[:3]]}")
            
            # 计算模型本体的准确率
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meter, loss, None, None, None, acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
            
            # 如果启用corrector，也计算修正后的准确率
            if output_corrector is not None and meter_corrected is not None:
                with torch.no_grad():
                    # B+A推理：提取当前激活值能量，计算WM-ID_e进行门控
                    current_acts_for_corrector = [act.detach() for act in current_activations] if len(current_activations) > 0 else None
                    if batch_idx == 0:
                        logger_info(logger, f"[DEBUG] Calling corrector with {len(current_acts_for_corrector) if current_acts_for_corrector else 0} activations")
                    outputs_corrected = output_corrector(
                        outputs,
                        activations=current_acts_for_corrector,
                        targets=None  # 推理时无targets，使用top-1预测
                    )
                    if batch_idx == 0:
                        logger_info(logger, f"[DEBUG] Corrector output diff from input: {torch.norm(outputs_corrected - outputs).item():.4f}")
                    
                    # ========== 详细的logits分析（第一个batch） ==========
                    if batch_idx == 0:
                        pred_faulted = outputs.argmax(dim=1)
                        pred_corrected = outputs_corrected.argmax(dim=1)
                        is_wrong = (pred_faulted != targets)
                        is_corrected = (pred_corrected == targets) & is_wrong  # 错误→正确
                        is_damaged = (pred_faulted == targets) & (pred_corrected != targets)  # 正确→错误
                        is_unchanged = (pred_faulted == pred_corrected)
                        
                        logger_info(logger, f"\n{'='*80}")
                        logger_info(logger, f"[LOGITS ANALYSIS] Batch 0 Detailed Analysis")
                        logger_info(logger, f"{'='*80}")
                        logger_info(logger, f"Total samples: {len(targets)}")
                        logger_info(logger, f"Faulted wrong: {is_wrong.sum().item()}/{len(targets)} ({is_wrong.float().mean().item()*100:.1f}%)")
                        logger_info(logger, f"Corrected (wrong→right): {is_corrected.sum().item()}/{len(targets)}")
                        logger_info(logger, f"Damaged (right→wrong): {is_damaged.sum().item()}/{len(targets)}")
                        logger_info(logger, f"Unchanged: {is_unchanged.sum().item()}/{len(targets)} ({is_unchanged.float().mean().item()*100:.1f}%)")
                        
                        # 分析错误样本的logits
                        if is_wrong.any():
                            wrong_indices = torch.where(is_wrong)[0][:5]  # 分析前5个错误样本
                            logger_info(logger, f"\n[Analysis of {len(wrong_indices)} wrong samples]")
                            for i, idx in enumerate(wrong_indices):
                                target_cls = targets[idx].item()
                                pred_f = pred_faulted[idx].item()
                                pred_c = pred_corrected[idx].item()
                                
                                logits_f = outputs[idx]
                                logits_c = outputs_corrected[idx]
                                correction = outputs_corrected[idx] - outputs[idx]
                                
                                # 计算需要的修正量（理想情况）
                                ideal_correction = torch.zeros_like(logits_f)
                                ideal_correction[target_cls] = (logits_f[pred_f] - logits_f[target_cls] + 1.0)  # 需要增加target_cls，使其超过pred_f
                                
                                logger_info(logger, f"\n  Sample {idx}: target={target_cls}, pred_faulted={pred_f}, pred_corrected={pred_c}")
                                logger_info(logger, f"    Faulted logits: {logits_f.cpu().numpy()}")
                                logger_info(logger, f"    Corrected logits: {logits_c.cpu().numpy()}")
                                logger_info(logger, f"    Actual correction: {correction.cpu().numpy()}")
                                logger_info(logger, f"    Ideal correction (to fix): {ideal_correction.cpu().numpy()}")
                                logger_info(logger, f"    Correction direction match: {torch.sign(correction[target_cls]) == torch.sign(ideal_correction[target_cls])}")
                                logger_info(logger, f"    Correction magnitude: target_cls={correction[target_cls].item():.4f} (ideal={ideal_correction[target_cls].item():.4f})")
                                logger_info(logger, f"    Gap to fix: {logits_f[pred_f] - logits_f[target_cls]:.4f}, correction on target: {correction[target_cls].item():.4f}")
                        
                        # 分析为什么修正没有改变预测
                        if is_wrong.any():
                            wrong_indices_all = torch.where(is_wrong)[0]
                            if len(wrong_indices_all) > 0:
                                # 计算gap（错误预测的logits - 正确类别的logits）
                                pred_faulted_wrong = pred_faulted[wrong_indices_all]
                                targets_wrong = targets[wrong_indices_all]
                                gaps = outputs[wrong_indices_all, pred_faulted_wrong] - outputs[wrong_indices_all, targets_wrong]
                                
                                # 计算在正确类别上的修正量
                                corrections_on_target = (outputs_corrected[wrong_indices_all, targets_wrong] - outputs[wrong_indices_all, targets_wrong])
                                
                                # 计算在错误预测类别上的修正量
                                corrections_on_wrong = (outputs_corrected[wrong_indices_all, pred_faulted_wrong] - outputs[wrong_indices_all, pred_faulted_wrong])
                                
                                logger_info(logger, f"\n[Why corrections didn't change predictions]")
                                logger_info(logger, f"  Wrong samples: {len(wrong_indices_all)}")
                                logger_info(logger, f"  Average gap to fix: {gaps.mean().item():.4f} (need to close this gap)")
                                logger_info(logger, f"  Average correction on target class: {corrections_on_target.mean().item():.4f}")
                                logger_info(logger, f"  Average correction on wrong class: {corrections_on_wrong.mean().item():.4f}")
                                logger_info(logger, f"  Net correction (target - wrong): {(corrections_on_target - corrections_on_wrong).mean().item():.4f}")
                                logger_info(logger, f"  Correction / Gap ratio: {(corrections_on_target / (gaps + 1e-6)).mean().item():.4f}")
                                logger_info(logger, f"  (If ratio < 1.0, correction is too small to fix the gap)")
                                logger_info(logger, f"  (If net correction < gap, correction won't change prediction)")
                        
                        logger_info(logger, f"{'='*80}\n")
                    
                    loss_corrected = criterion(outputs_corrected, targets)
                    acc1_corrected, acc5_corrected = accuracy(outputs_corrected.data, targets.data, topk=(1, 5))
                    if batch_idx == 0:
                        logger_info(logger, f"[DEBUG] Before update: meter_corrected['top1'].avg = {meter_corrected['top1'].avg:.4f}, acc1_corrected = {acc1_corrected.item():.4f}")
                    update_meter(meter_corrected, loss_corrected, None, None, None, acc1_corrected, acc5_corrected, inputs.size(0), time.time() - start_time, configs.world_size)
                    if batch_idx == 0:
                        logger_info(logger, f"[DEBUG] After update: meter_corrected['top1'].avg = {meter_corrected['top1'].avg:.4f}")
                    if batch_idx == 1:
                        logger_info(logger, f"[DEBUG] Batch 1: meter['top1'].avg = {meter['top1'].avg:.4f}, meter_corrected['top1'].avg = {meter_corrected['top1'].avg:.4f}, acc1 = {acc1.item():.4f}, acc1_corrected = {acc1_corrected.item():.4f}")
    
    if train_mode:
        logger_info(logger, msg='Using training mode...')
        model.train()

    if eval_predefined_arch == None:
        from policy import MIN_POLICY
        eval_predefined_arch = [
            MIN_POLICY
        ]
    
    meters = [{
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'QE_loss': AverageMeter(),
        'dist_loss': AverageMeter(),
        'IDM_loss': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(len(eval_predefined_arch))]
    
    # 如果启用corrector，创建额外的meter来记录修正后的准确率
    meters_corrected = None
    if output_corrector is not None:
        meters_corrected = [{
            'loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'QE_loss': AverageMeter(),
            'dist_loss': AverageMeter(),
            'IDM_loss': AverageMeter(),
            'batch_time': AverageMeter()
        } for _ in range(len(eval_predefined_arch))]

    for idx, arch in enumerate(eval_predefined_arch): 
        w_configs, a_configs = arch[-2], arch[-1]
        if arch[0] == -1:
            sample_min_cands(model, configs)
        elif arch[0] == 32:
            pass
        else:
            set_bit_width(model, w_configs, a_configs)
        
        with torch.no_grad():
            if configs.post_training_batchnorm_calibration:
                assert train_loader is not None

                world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                calibrate_batchnorm_state(model, loader=train_loader, reset=True, distributed_training=(world_size > 1), num_batch=7000//world_size//configs.dataloader.batch_size)
            
            meter_corrected = meters_corrected[idx] if meters_corrected is not None else None
            _eval(data_loader, meters[idx], meter_corrected)
            bops, size = model_profiling(model=model, return_layers=False)

            if meters_corrected is not None and meters_corrected[idx] is not None:
                logger_info(logger, f"[DEBUG] meters[{idx}]['top1'].avg = {meters[idx]['top1'].avg:.4f}, meters_corrected[{idx}]['top1'].avg = {meters_corrected[idx]['top1'].avg:.4f}")
                logger_info(logger, msg=f"Arch {idx}, BitOPs {round(bops, 2)} G, Size {round(size, 2)} MB, Top-1 Acc. {round(meters[idx]['top1'].avg, 2)}% (Model) | {round(meters_corrected[idx]['top1'].avg, 2)}% (With Corrector)")
            else:
                logger_info(logger, msg=f"Arch {idx}, BitOPs {round(bops, 2)} G, Size {round(size, 2)} MB, Top-1 Acc. {round(meters[idx]['top1'].avg, 2)}")
    
    # 如果使用corrector，返回两个列表：[model_accs, corrected_accs]
    # 否则只返回model_accs（保持向后兼容）
    if meters_corrected is not None:
        return ([meters[idx]['top1'].avg for idx in range(len(eval_predefined_arch))],
                [meters_corrected[idx]['top1'].avg for idx in range(len(eval_predefined_arch))])
    else:
        return [meters[idx]['top1'].avg for idx in range(len(eval_predefined_arch))]


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch