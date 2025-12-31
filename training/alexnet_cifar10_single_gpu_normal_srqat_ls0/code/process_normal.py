import logging
import math
import operator
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from quan.func import SwithableBatchNorm
from quan.func import QuanConv2d, QuanLinear
from quan.quantizer.lsq import LsqQuan
from util import AverageMeter
from util.utils import model_profiling, calibrate_batchnorm_state, accuracy, update_meter, set_global_seed
from util.qat import profile_layerwise_quantization_metric, freeze_layers, set_bit_width, auxiliary_quantized_loss, remove_hook_for_quantized_layers, set_forward_hook_for_quantized_layers
from util.mpq import sample_one_mixed_policy, sample_max_cands, sample_min_cands
from util.dist import master_only, logger_info

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


# ---------------- SR-QAT: Penalty-based Scale-Constrained QAT ----------------
class _MarginEMA:
    """EMA of positive logit margin (detached) for SR-QAT scale penalty."""
    def __init__(self, momentum: float = 0.9):
        self.momentum = float(momentum)
        self.value = None  # torch.Tensor scalar on device

    @torch.no_grad()
    def update(self, margin: torch.Tensor) -> torch.Tensor:
        # Clamp to positive region as in doc (only correct margins contribute)
        m = torch.clamp(margin.detach(), min=0.0)
        mean_m = m.mean()
        if self.value is None:
            self.value = mean_m
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * mean_m
        return self.value


_SRQAT_MARGIN_EMA: _MarginEMA | None = None


def _compute_logit_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    margin m_i = z_{y} - max_{c != y} z_c
    """
    top2 = logits.topk(2, dim=1).values  # [B, 2]
    preds = logits.argmax(dim=1)
    z_y = logits[torch.arange(logits.size(0), device=logits.device), targets]
    z_comp = torch.where(preds == targets, top2[:, 1], top2[:, 0])
    return z_y - z_comp


def _get_module_weight_bits(module: nn.Module) -> int | None:
    """Best-effort read of current weight bit-width for a quantized module."""
    # Fixed-bit layers
    if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        fb = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
        if isinstance(fb, torch.Tensor):
            return int(fb.item())
        return int(fb)

    # Dynamic layers
    if hasattr(module, 'bits') and module.bits is not None:
        b = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
        if isinstance(b, torch.Tensor):
            return int(b.item())
        return int(b)

    return None


def _compute_srqat_scale_penalty(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    configs,
    *,
    epoch: int,
    is_faulted: bool,
) -> torch.Tensor:
    """
    SR-QAT penalty (from chat/doc.md):
        L_scale = lambda * sum_l s_l^2 / (m_bar^2 + eps)
    where m_bar is EMA of positive logit margin.
    """
    scale_cfg = getattr(configs, 'scale_penalty', None)
    if scale_cfg is None or not getattr(scale_cfg, 'enabled', False):
        return outputs.new_tensor(0.0)

    start_epoch = int(getattr(scale_cfg, 'start_epoch', 0))
    if epoch < start_epoch:
        return outputs.new_tensor(0.0)

    apply_on_faulted = bool(getattr(scale_cfg, 'apply_on_faulted', False))
    if is_faulted and not apply_on_faulted:
        return outputs.new_tensor(0.0)

    # Initialize EMA state once per process
    global _SRQAT_MARGIN_EMA
    if _SRQAT_MARGIN_EMA is None:
        _SRQAT_MARGIN_EMA = _MarginEMA(momentum=float(getattr(scale_cfg, 'margin_momentum', 0.9)))

    # IMPORTANT: do NOT update EMA with faulted forward unless explicitly requested
    margin = _compute_logit_margin(outputs, targets)
    if not is_faulted or apply_on_faulted:
        m_bar = _SRQAT_MARGIN_EMA.update(margin)
    else:
        # fallback: use current detached mean margin
        m_bar = torch.clamp(margin.detach(), min=0.0).mean()

    eps = float(getattr(scale_cfg, 'eps', 1e-6))
    denom = (m_bar ** 2) + eps

    # Sum of LSQ weight scales squared across layers
    scan_model = model.module if hasattr(model, 'module') else model
    sum_s2 = outputs.new_tensor(0.0)
    num_terms = 0

    for _name, module in scan_model.named_modules():
        if not isinstance(module, (QuanConv2d, QuanLinear)):
            continue
        if not hasattr(module, 'quan_w_fn') or module.quan_w_fn is None:
            continue
        if not isinstance(module.quan_w_fn, LsqQuan):
            continue

        wbits = _get_module_weight_bits(module)
        if wbits is None or wbits >= 32:
            continue

        s = module.quan_w_fn.get_scale(wbits, detach=False)
        sum_s2 = sum_s2 + (s ** 2).sum()
        num_terms += 1

    if num_terms == 0:
        return outputs.new_tensor(0.0)

    # Optional normalization (off by default to match doc)
    if bool(getattr(scale_cfg, 'normalize_by_num_layers', False)):
        sum_s2 = sum_s2 / float(num_terms)

    lam = float(getattr(scale_cfg, 'lambda_scale', 0.0))
    return (lam * sum_s2) / denom


def compute_overall_loss(
    outputs,
    teacher_outputs,
    targets,
    criterion,
    model,
    quantization_error_minimization=False,
    QE_loss_weight=.5,
    disable_smallest_regularization=True,
    configs=None,
    *,
    epoch: int = 0,
    is_faulted: bool = False,
):
    task_loss = loss_forward(outputs, teacher_outputs, targets, criterion)
    # SR-QAT: add penalty-based scale constraint (optional, controlled by configs.scale_penalty)
    if configs is not None:
        task_loss = task_loss + _compute_srqat_scale_penalty(
            outputs, targets, model, configs, epoch=epoch, is_faulted=is_faulted
        )

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
def update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, nr_random_sample, optimizer, optimizer_q, mode='training'):
    iters = len(meters) if mode == 'training' else 1
    for m in monitors:
        for i in range(iters):
            # if meters[i]['top1'].avg == 0.:
            #     continue
            p = meters[i]['name'] + ' '
            m.update(epoch, batch_idx + 1, steps_per_epoch, p + 'Training', {
                'Loss': meters[i]['loss'],
                'QE Loss': meters[i]['QE_loss'], 
                'Distribution Loss': meters[i]['dist_loss'], 
                'IDM Loss': meters[i]['IDM_loss'], 
                'Top1': meters[i]['top1'],
                'Top5': meters[i]['top5'],
                'LR': optimizer.param_groups[0]['lr'],
                'QLR': optimizer_q.param_groups[0]['lr'] if optimizer_q is not None else 0
            })
        
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
    """
    DISABLED: Distribution Loss disabled for fault tolerance study.
    Original code added KL divergence when teacher_outputs is not None:
        if teacher_outputs is not None:
            loss = 1/2 * loss + 1/2 * F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(teacher_outputs, dim=-1), reduction='batchmean')
    """
    loss = criterion(outputs, targets)

    # DISABLED: Distribution Loss (KL divergence) disabled for fault tolerance study
    # if teacher_outputs is not None:
    #     loss = 1/2 * loss + 1/2 * F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(teacher_outputs, dim=-1), reduction='batchmean')
    
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
    is_restorer_training = (optimizer is None and output_corrector is not None and corrector_optimizer is not None)
    if is_restorer_training:
        if device is None:
            device = next(model.parameters()).device
        logger_info(logger, f'Entered Stage 2 Restorer Training mode for epoch {epoch}.')
        model.eval()
        output_corrector.train()
        meters = {
            'restorer_loss': AverageMeter(),
            'clean_acc': AverageMeter(),
            'faulted_acc': AverageMeter(),
            'restored_acc': AverageMeter(),
            'improvement': AverageMeter(),
            'batch_time': AverageMeter()
        }
        end = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            corrector_optimizer.zero_grad()
            if fault_injector:
                # For restorer training: sample BER using Beta distribution (中间胖，两头瘦)
                if hasattr(fault_injector, 'use_random_flip_in_training') and fault_injector.use_random_flip_in_training:
                    import numpy as np
                    ber_min = 1e-2
                    ber_max = 1e-1
                    beta_alpha = 2.0  # Beta distribution shape parameter
                    beta_beta = 2.0  # Beta distribution shape parameter
                    # Beta(2, 2) gives a bell-shaped distribution (中间胖，两头瘦)
                    beta_sample = np.random.beta(beta_alpha, beta_beta)
                    effective_ber = ber_min + (ber_max - ber_min) * beta_sample
                    fault_injector.ber = effective_ber
                fault_injector.disable()
            with torch.no_grad():
                logits_clean = model(inputs)
            if fault_injector:
                fault_injector.enable()
                fault_injector.reset_forward_seed()
            collector = getattr(output_corrector, 'collector', None)
            if collector is not None:
                try:
                    # Re-register hooks if they were removed during validation
                    if not getattr(collector, 'handles', None) or len(collector.handles) == 0:
                        collector._register_hooks()
                    collector.clear()
                except Exception:
                    pass
            with torch.no_grad():
                logits_faulted = model(inputs)
            if collector is not None:
                _res = collector.build_layer_features(inputs.device)
                if isinstance(_res, tuple):
                    layer_features = _res[0]
                else:
                    layer_features = _res
                if not layer_features:
                    layer_features = []
            else:
                layer_features = []
            logits_restored, _gate = output_corrector(logits_faulted.detach(), layer_features)
            ce_loss = F.cross_entropy(logits_restored, targets)
            kl_loss = torch.tensor(0.0, device=inputs.device)
            if getattr(configs.sensitive_restorer, 'kl_div_weight', 0) > 0:
                T = getattr(configs.sensitive_restorer, 'temperature', 1.0)
                kl_loss = F.kl_div(
                    F.log_softmax(logits_restored / T, dim=1),
                    F.softmax(logits_clean.detach() / T, dim=1),
                    reduction='batchmean'
                )
            dir_loss = torch.tensor(0.0, device=inputs.device)
            if getattr(configs.sensitive_restorer, 'direction_weight', 0) > 0:
                pred_delta = logits_restored - logits_faulted.detach()
                target_delta = logits_clean.detach() - logits_faulted.detach()
                dir_loss = 1 - F.cosine_similarity(pred_delta, target_delta, dim=-1).mean()
            total_loss = ce_loss + getattr(configs.sensitive_restorer, 'kl_div_weight', 0) * kl_loss + getattr(configs.sensitive_restorer, 'direction_weight', 0) * dir_loss
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
            if (batch_idx + 1) % configs.log.print_freq == 0:
                logger_info(logger, f"Epoch: [{epoch}][{batch_idx+1}/{len(train_loader)}] | Time {meters['batch_time'].val:.3f} ({meters['batch_time'].avg:.3f}) | Loss {meters['restorer_loss'].avg:.4f} | Accs(C/F/R): {meters['clean_acc'].avg:.2f}/{meters['faulted_acc'].avg:.2f}/{meters['restored_acc'].avg:.2f} | Gain {meters['improvement'].avg:+.2f}%")
        return meters['restored_acc'].avg, 0, meters['restorer_loss'].avg
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
    if fault_injector is not None:
        fault_aware_training_config = getattr(configs, 'fault_aware_training', None)
        if fault_aware_training_config is not None:
            use_fault_aware_training = getattr(fault_aware_training_config, 'enabled', False)
            if use_fault_aware_training:
                trades_config = getattr(fault_aware_training_config, 'trades', {})
                use_kl = getattr(trades_config, 'use_kl', False)
                alpha = getattr(trades_config, 'alpha', 0.6)
                beta = getattr(trades_config, 'beta', 1.0)
                
                # 渐进式BER调度
                schedule_config = getattr(fault_aware_training_config, 'schedule', None)
                start_epoch = 0  # 初始化start_epoch
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
                    # 没有启用调度，使用固定BER
                    current_ber = getattr(fault_aware_training_config, 'ber', 1e-2)
                    # 确保current_ber是浮点数（YAML可能解析为字符串）
                    current_ber = float(current_ber)
                    fault_injector.ber = current_ber
                
                if use_fault_aware_training:
                    use_entropy = getattr(trades_config, 'use_entropy', False)
                    entropy_weight = getattr(trades_config, 'entropy_weight', 0.1)
                    entropy_mode = getattr(trades_config, 'entropy_mode', 'difference')
                    logger_info(logger, '=' * 80)
                    logger_info(logger, f'🔥 FAULT-AWARE TRAINING (FAT) - ACTIVE in train() function')
                    logger_info(logger, f'   Epoch {epoch}/{configs.epochs} (Progress: {epoch/configs.epochs*100:.1f}%), TRADES Loss: {"KL Div" if use_kl else "Simple"}')
                    logger_info(logger, f'   Current BER: {current_ber:.2e} (Progressive Schedule: {"Enabled" if schedule_config and getattr(schedule_config, "enabled", False) else "Disabled"})')
                    if not use_kl:
                        logger_info(logger, f'   Loss = {alpha} * loss_normal + {beta} * loss_faulted')
                    else:
                        logger_info(logger, f'   Loss = loss_normal + {beta} * KL(p_normal, p_faulted)')
                    if use_entropy:
                        logger_info(logger, f'   Entropy Regularization: Enabled (mode={entropy_mode}, weight={entropy_weight})')
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
        logger_info(logger, '⚠️  NOTE: Distribution Loss (KL divergence) is DISABLED in this version')
    
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

        # DISABLED: Distribution Loss disabled for fault tolerance study
        # external_teacher_outputs is not used in loss_forward anymore
        external_teacher_outputs = None
        # if teacher_model is not None and soft_criterion is not None:
        #     with torch.no_grad():
        #         external_teacher_outputs = teacher_model(inputs)

        QE_loss_weight = annealing_schedule(num_updates) # We use a scheduler for the weights of QE loss according to QAT Oscillations Overcoming [ICML'22]. 

        if sample_current_max:
            start_time = time.time()

            sample_max_cands(model, configs)

            if information_distortion_mitigation:
                target_features = []
                hooks = set_forward_hook_for_quantized_layers(model, target_features, is_max=True)

            max_outputs = model(inputs)

            loss, QE_loss, dist_loss = compute_overall_loss(
                max_outputs,
                external_teacher_outputs,
                targets,
                criterion,
                model,
                quantization_error_minimization=False,
                configs=configs,
                disable_smallest_regularization=True,
                epoch=epoch,
                is_faulted=False,
            )

            loss.backward()

            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

            teacher_outputs = max_outputs.clone().detach()
            
            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[0], loss, QE_loss, dist_loss, 0, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
                
        weight_conf_pool = []

        for iter_idx in range(nr_random_sample):
            start_time = time.time()

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
                outputs_normal = model(inputs)
                loss_normal, QE_loss_normal, dist_loss_normal = compute_overall_loss(
                    outputs_normal, teacher_outputs, targets, criterion, model, 
                    quantization_error_minimization=epoch>40, 
                    QE_loss_weight=QE_loss_weight, 
                    disable_smallest_regularization=True, 
                    configs=configs,
                    epoch=epoch,
                    is_faulted=False,
                )
                
                # 第二次forward: 故障注入
                fault_injector.enable()
                # Reset forward seed to ensure all layers in this forward use the same base_seed
                fault_injector.reset_forward_seed()
                if batch_idx == 0 and iter_idx == 0:
                    logger_info(logger, f'[FAT] Batch {batch_idx}, Iter {iter_idx}: Second forward (FAULTED, BER={current_ber:.2e})')
                outputs_faulted = model(inputs)
                loss_faulted, QE_loss_faulted, dist_loss_faulted = compute_overall_loss(
                    outputs_faulted, teacher_outputs, targets, criterion, model, 
                    quantization_error_minimization=epoch>40, 
                    QE_loss_weight=QE_loss_weight, 
                    disable_smallest_regularization=True, 
                    configs=configs,
                    epoch=epoch,
                    is_faulted=True,
                )
                
                # TRADES损失计算（支持信息熵正则化）
                use_entropy = getattr(trades_config, 'use_entropy', False)
                entropy_weight = getattr(trades_config, 'entropy_weight', 0.1)
                entropy_mode = getattr(trades_config, 'entropy_mode', 'difference')  # 'difference', 'constraint', 'balance'
                
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
                    
                    # 组合QE和dist loss
                    QE_loss = alpha * QE_loss_normal + beta * QE_loss_faulted
                    dist_loss = alpha * dist_loss_normal + beta * dist_loss_faulted
                
                # 使用normal输出计算准确率（用于显示）
                outputs = outputs_normal
                if batch_idx == 0 and iter_idx == 0:
                    entropy_info = ""
                    if use_entropy:
                        probs_n = F.softmax(outputs_normal, dim=1)
                        probs_f = F.softmax(outputs_faulted, dim=1)
                        entropy_n = compute_entropy(probs_n).mean().item()
                        entropy_f = compute_entropy(probs_f).mean().item()
                        entropy_info = f", entropy_normal={entropy_n:.4f}, entropy_faulted={entropy_f:.4f}"
                    logger_info(logger, f'[FAT] Batch {batch_idx}, Iter {iter_idx}: TRADES loss computed, normal_loss={loss_normal.item():.4f}, faulted_loss={loss_faulted.item():.4f}{entropy_info}')
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
                    configs=configs,
                    epoch=epoch,
                    is_faulted=False,
                )

            IDM_loss = 0
            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

                IDM_loss = sum([F.mse_loss(s, t).sum() if s is not None else 0 for s, t in zip(distorted_features, target_features)])
                loss += (IDM_loss * IDM_weight)
            
            loss.backward()
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[iter_idx+num_fixed_sample], loss, QE_loss, dist_loss, IDM_loss, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()
        if optimizer_q is not None:
            optimizer_q.step()

        num_updates += 1

        if model_ema is not None:
            model_ema.update(model)
        
        if (batch_idx + 1) % configs.log.print_freq == 0:
            update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, nr_random_sample, optimizer, optimizer_q, mode=mode)
            logger_info(logger, "="*115)

    show_training_info(meters, target_bits, nr_random_sample, mode=mode)
    
    return meters[0]['top1'].avg, meters[0]['top5'].avg, meters[0]['loss'].avg


def validate(data_loader, model, criterion, epoch, monitors, configs, nr_random_sample=3, alpha=1, train_loader=None, eval_predefined_arch=None, bops_limit=1e10, train_mode=False):
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

    def _eval(_loader, meter):
        for batch_idx, (inputs, targets) in enumerate(_loader):
            inputs = inputs.to(configs.device)
            targets = targets.to(configs.device)
            start_time = time.time()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))

            update_meter(meter, loss, None, None, None, acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
    
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
            
            _eval(data_loader, meters[idx])
            bops, size = model_profiling(model=model, return_layers=False)

            logger_info(logger, msg=f"Arch {idx}, BitOPs {round(bops, 2)} G, Size {round(size, 2)} MB, Top-1 Acc. {round(meters[idx]['top1'].avg, 2)}")
    
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

