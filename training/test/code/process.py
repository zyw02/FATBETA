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
from util.dist import master_only, logger_info, is_master

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

def train(train_loader, model, criterion, optimizer, epoch, monitors, configs, model_ema=None, nr_random_sample=2, mode='training', soft_criterion=None, teacher_model=None, optimizer_q=None, annealing_schedule=None, freezing_annealing_schedule=None, IDM_weight=0.01, scaler=None, fault_injector=None, output_corrector=None, corrector_optimizer=None, device=None):
    # This is a special mode for Stage 2 restorer training
    is_restorer_training = (optimizer is None and output_corrector is not None and corrector_optimizer is not None)
    
    if is_restorer_training:
        if device is None:
            device = next(model.parameters()).device
            
        logger.info(f"Entered Stage 2 Restorer Training mode for epoch {epoch}.")
        model.eval() 
        output_corrector.train()

        meters = {
            'restorer_loss': AverageMeter(), 'clean_acc': AverageMeter(),
            'faulted_acc': AverageMeter(), 'restored_acc': AverageMeter(), 'improvement': AverageMeter(),
            'batch_time': AverageMeter()
        }
        
        # Attach the collector to the restorer if it's not already there
        if not hasattr(output_corrector, 'collector') or output_corrector.collector is None:
            # This assumes baseline_stats is loaded and available in some way.
            # This part of the logic might need adjustment if baseline_stats are not accessible here.
            # For now, we assume the collector is attached outside this function.
            pass

        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            corrector_optimizer.zero_grad()

            # 1. Get clean logits
            if fault_injector: fault_injector.disable()
            with torch.no_grad():
                logits_clean = model(inputs)

            # 2. Get faulted logits and features
            if fault_injector:
                fault_injector.enable()
                fault_injector.reset_forward_seed()
            
            collector = getattr(output_corrector, 'collector', None)
            if collector: collector.clear_features()

            with torch.no_grad():
                logits_faulted = model(inputs)
            
            if collector:
                layer_features, _ = collector.build_layer_features(inputs.device)
            else:
                layer_features = None
            
            logits_restored, gate = output_corrector(logits_faulted.detach(), layer_features)

            ce_loss = F.cross_entropy(logits_restored, targets)
            
            kl_loss = torch.tensor(0.0, device=device)
            if configs.sensitive_restorer.kl_div_weight > 0:
                T = configs.sensitive_restorer.temperature
                kl_loss = F.kl_div(
                    F.log_softmax(logits_restored / T, dim=1),
                    F.softmax(logits_clean.detach() / T, dim=1),
                    reduction='batchmean'
                )

            dir_loss = torch.tensor(0.0, device=device)
            if configs.sensitive_restorer.direction_weight > 0:
                pred_delta = logits_restored - logits_faulted.detach()
                target_delta = logits_clean.detach() - logits_faulted.detach()
                dir_loss = 1 - F.cosine_similarity(pred_delta, target_delta, dim=-1).mean()
            
            total_loss = (ce_loss + 
                          configs.sensitive_restorer.kl_div_weight * kl_loss +
                          configs.sensitive_restorer.direction_weight * dir_loss)

            total_loss.backward()
            corrector_optimizer.step()

            with torch.no_grad():
                clean_acc, _ = accuracy(logits_clean, targets, topk=(1, 5))
                faulted_acc, _ = accuracy(logits_faulted, targets, topk=(1, 5))
                restored_acc, _ = accuracy(logits_restored, targets, topk=(1, 5))
                
                meters['restorer_loss'].update(total_loss.item(), inputs.size(0))
                meters['clean_acc'].update(clean_acc.item(), inputs.size(0))
                meters['faulted_acc'].update(faulted_acc.item(), inputs.size(0))
                meters['restored_acc'].update(restored_acc.item(), inputs.size(0))
                meters['improvement'].update(restored_acc.item() - faulted_acc.item(), inputs.size(0))
            
            meters['batch_time'].update(time.time() - end)
            end = time.time()
            
            if (batch_idx + 1) % configs.log.print_freq == 0 and is_master():
                logger.info(
                    f"Epoch: [{epoch}][{batch_idx+1}/{len(train_loader)}] | "
                    f"Time {meters['batch_time'].val:.3f} ({meters['batch_time'].avg:.3f}) | "
                    f"Loss {meters['restorer_loss'].avg:.4f} | "
                    f"Accs(C/F/R): {meters['clean_acc'].avg:.2f}/{meters['faulted_acc'].avg:.2f}/{meters['restored_acc'].avg:.2f} | "
                    f"Gain {meters['improvement'].avg:+.2f}%"
                )
        return meters['restored_acc'].avg, 0, meters['restorer_loss'].avg

    # --- Original Stage 1 training logic starts here ---
    assert mode in ['finetuning', 'training']

    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    if getattr(configs, 'sandwich_training', False):
        sample_current_max, sample_current_min = True, True
    else:
        sample_current_max, sample_current_min = False, False
    
    sample_current_max = True # This seems to be hardcoded in the original file
    
    meters, num_fixed_sample = get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min)
    
    # This logic is from the original file, related to a three-stage corrector training not used by our new method
    # It is kept for full compatibility but will not be active for Stage 1.
    should_freeze_model = False
    if output_corrector is not None:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', {})
        trades_config = getattr(fault_aware_training_config, 'trades', {})
        use_corrector = getattr(trades_config, 'use_corrector', False)
        if use_corrector:
            corrector_start_epoch = getattr(trades_config, 'corrector_start_epoch', None)
            if corrector_start_epoch is not None and epoch >= corrector_start_epoch:
                should_freeze_model = True

    model.train()
    if model_ema:
        model_ema.ema.train()

    # Dynamic bit-width freezing logic from original file
    T = 2 if epoch <= int(configs.epochs * 0.72) else 15
    if configs.enable_dynamic_bit_training and epoch > 5 and (epoch + 1) % T == 0:
        freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
        freezing_metric = profile_layerwise_quantization_metric(model=model)
        freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio, progressive=False, logger=logger, org_cands=configs.target_bits)
        logger_info(logger=logger, msg=f'Current freezing ratio: {freezing_ratio}')

    if teacher_model is not None:
        teacher_model.eval()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        # Sandwich training: Max bit-width path
        if sample_current_max:
            start_time = time.time()
            sample_max_cands(model, configs)
            max_outputs = model(inputs)
            loss, QE_loss, dist_loss = compute_overall_loss(max_outputs, None, targets, criterion, model, configs=configs, disable_smallest_regularization=True)
            loss.backward()
            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[0], loss, QE_loss, dist_loss, 0, acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
        
        # Sandwich training: Random bit-width paths
        for iter_idx in range(nr_random_sample):
            start_time = time.time()
            sample_one_mixed_policy(model, configs)
            outputs = model(inputs)
            loss, QE_loss, dist_loss = compute_overall_loss(outputs, None, targets, criterion, model, configs=configs, quantization_error_minimization=epoch > 40, disable_smallest_regularization=True)
            loss.backward()
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[iter_idx + num_fixed_sample], loss, QE_loss, dist_loss, 0, acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        if not should_freeze_model:
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            if optimizer_q is not None:
                optimizer_q.step()
        
        if model_ema is not None:
            model_ema.update(model)
        
        if (batch_idx + 1) % configs.log.print_freq == 0:
            update_monitors(monitors, meters, target_bits, epoch, batch_idx, len(train_loader), nr_random_sample, optimizer, optimizer_q, mode=mode)
            logger_info(logger, "="*140)

    show_training_info(meters, target_bits, nr_random_sample, mode=mode)
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