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


def _compute_orthogonality_penalty(
    model: nn.Module,
    configs,
    *,
    device: torch.device,
    is_faulted: bool,
) -> torch.Tensor:
    """
    Orthogonality regularization to reduce inter-filter correlation (a proxy for limiting error propagation).

    For a weight matrix W (flattened conv/linear):
      - row-normalize: Wn = W / (||row||2 + eps)
      - penalize: ||G - I||_F^2 where G = Wn Wn^T (or Wn^T Wn, whichever is smaller)

    Config (configs.orthogonality_penalty):
      enabled: bool
      lambda_ortho: float
      layers: "conv" | "linear" | "both"   (default "conv")
      exclude_layers: [str, ...]           (module name substrings, optional)
      max_gram_dim: int                   (skip layers whose smaller Gram dim > this, default 2048)
      eps: float                          (default 1e-8)
      apply_on_faulted: bool              (default False)
    """
    ortho_cfg = getattr(configs, "orthogonality_penalty", None)
    if ortho_cfg is None or not getattr(ortho_cfg, "enabled", False):
        return torch.tensor(0.0, device=device)

    apply_on_faulted = bool(getattr(ortho_cfg, "apply_on_faulted", False))
    if is_faulted and not apply_on_faulted:
        return torch.tensor(0.0, device=device)

    lam = float(getattr(ortho_cfg, "lambda_ortho", 0.0))
    if lam <= 0:
        return torch.tensor(0.0, device=device)

    layers_mode = str(getattr(ortho_cfg, "layers", "conv")).lower()
    include_conv = layers_mode in ("conv", "both")
    include_linear = layers_mode in ("linear", "both")

    exclude_substrings = list(getattr(ortho_cfg, "exclude_layers", []) or [])
    max_gram_dim = int(getattr(ortho_cfg, "max_gram_dim", 2048))
    eps = float(getattr(ortho_cfg, "eps", 1e-8))

    scan_model = model.module if hasattr(model, "module") else model
    total = torch.tensor(0.0, device=device)
    used = 0

    for name, module in scan_model.named_modules():
        if exclude_substrings and any(s in name for s in exclude_substrings):
            continue

        # Only apply to quantized modules that actually have a weight parameter
        if isinstance(module, QuanConv2d) and include_conv:
            W = module.weight
            # [out, in, k, k] -> [out, in*k*k]
            Wm = W.view(W.size(0), -1)
        elif isinstance(module, QuanLinear) and include_linear:
            W = module.weight
            # [out, in] -> [out, in]
            Wm = W
        else:
            continue

        # Skip degenerate layers
        if Wm.dim() != 2 or Wm.size(0) < 2 or Wm.size(1) < 2:
            continue

        # Compute smaller Gram for efficiency: dim = min(out, in)
        out_dim, in_dim = int(Wm.size(0)), int(Wm.size(1))
        gram_dim = min(out_dim, in_dim)
        if gram_dim > max_gram_dim:
            continue

        # Row-normalize (scale-invariant; SR-QAT already cares about magnitude)
        Wf = Wm.float()
        row_norm = torch.norm(Wf, p=2, dim=1, keepdim=True).clamp_min(eps)
        Wn = Wf / row_norm

        if out_dim <= in_dim:
            G = Wn @ Wn.t()  # [out, out]
            I = torch.eye(out_dim, device=device, dtype=G.dtype)
            # Normalize by matrix size to keep loss scale comparable across layers
            loss = torch.mean((G - I) ** 2)
        else:
            # Use column Gram when out >> in to keep matrix small
            G = Wn.t() @ Wn  # [in, in]
            I = torch.eye(in_dim, device=device, dtype=G.dtype)
            loss = torch.mean((G - I) ** 2)

        total = total + loss.to(device=device)
        used += 1

    if used == 0:
        return torch.tensor(0.0, device=device)
    return lam * (total / float(used))


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
        # Orthogonality penalty (optional, controlled by configs.orthogonality_penalty)
        task_loss = task_loss + _compute_orthogonality_penalty(
            model, configs, device=outputs.device, is_faulted=is_faulted
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


class FeatureHook:
    """用于捕获指定层特征的简单钩子"""
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.handle.remove()


def project_bfat_gradients(clean_grads, bfat_grads, limit_norm=False, norm_ratio=0.5, projection_mode="direction"):
    """
    BFAT 梯度处理逻辑：
    1. 投影模式 (projection_mode):
       - "direction": 仅在夹角 > 90度时进行投影，剔除有害分量。
       - "orthogonal": 严格正交，无论夹角如何，始终剔除在基准梯度方向上的分量。
       - "cagrad": 冲突规避梯度下降，寻找两个任务的最佳妥协方向（共赢方向）。
    2. 模长限制 (limit_norm): 确保处理后的梯度模长不超过基准梯度的一定比例。
    """
    projected_bfat_grads = {}
    for p, g_b in bfat_grads.items():
        if p in clean_grads:
            g_c = clean_grads[p]
            
            # --- 1. 方向/空间处理 ---
            dot_product = torch.sum(g_c * g_b)
            norm_sq_c = torch.sum(g_c * g_c) + 1e-8
            norm_sq_b = torch.sum(g_b * g_b) + 1e-8
            
            if projection_mode == "orthogonal":
                # 严格正交模式
                projection = (dot_product / norm_sq_c) * g_c
                g_b_cleaned = g_b - projection
            elif projection_mode == "cagrad":
                # CAGrad 模式：寻找最佳妥协方向
                # 求解目标：max min(g_final^T g_c, g_final^T g_b)
                # 对于双任务，g_final = alpha * g_c + (1 - alpha) * g_b
                # alpha_opt = (g_b^T g_b - g_c^T g_b) / (g_c^T g_c + g_b^T g_b - 2 * g_c^T g_b)
                numerator = norm_sq_b - dot_product
                denominator = norm_sq_c + norm_sq_b - 2 * dot_product + 1e-8
                alpha = torch.clamp(numerator / denominator, 0.0, 1.0)
                
                g_target = alpha * g_c + (1.0 - alpha) * g_b
                # 为了适配后续的累加逻辑 (p.grad = g_c + g_b_ret)，
                # 我们返回一个修正后的 g_b_ret，使得 g_c + g_b_ret = g_target
                g_b_cleaned = g_target - g_c
            else:
                # 方向修正模式 (direction)
                if dot_product < 0:
                    projection = (dot_product / norm_sq_c) * g_c
                    g_b_cleaned = g_b - projection
                else:
                    g_b_cleaned = g_b
            
            # --- 2. 模长限制 ---
            if limit_norm:
                norm_c = torch.norm(g_c)
                norm_b_final = torch.norm(g_b_cleaned)
                target_norm = norm_c * norm_ratio
                
                if norm_b_final > target_norm:
                    scale = target_norm / (norm_b_final + 1e-8)
                    g_b_final = g_b_cleaned * scale
                else:
                    g_b_final = g_b_cleaned
                
                projected_bfat_grads[p] = g_b_final
            else:
                projected_bfat_grads[p] = g_b_cleaned
        else:
            projected_bfat_grads[p] = g_b
            
    return projected_bfat_grads

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
    
    # --- BFAT Setup ---
    bfat_cfg = getattr(configs, 'bfat', None)
    bfat_start_epoch = getattr(bfat_cfg, 'start_epoch', 0)
    use_bfat = bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False) and epoch >= bfat_start_epoch
    bfat_hook = None
    if use_bfat:
        bfat_loss_type = getattr(bfat_cfg, 'loss_type', 'feature_sim')
        
        # 仅在需要计算特征相似度时才寻找层并注册 hook
        if bfat_loss_type == 'feature_sim':
            target_layer_name = getattr(bfat_cfg, 'target_layer', 'avgpool')
            target_module = None
            for n, m in model.named_modules():
                if n == target_layer_name:
                    target_module = m
                    break
            if target_module is not None:
                bfat_hook = FeatureHook(target_module)
            else:
                logger_info(logger, f'⚠️ BFAT WARNING: Target layer {target_layer_name} not found for feature_sim!')
                use_bfat = False # 如果找不到层且需要 feature_sim，则无法运行

        # 配置与日志
        projection_base = getattr(bfat_cfg, 'projection_base', 'clean')
        bfat_ber = getattr(bfat_cfg, 'ber', 0.01)
        bfat_weight = getattr(bfat_cfg, 'loss_weight', 1.0)
        bfat_dual_bit = getattr(bfat_cfg, 'dual_bit', False)
        bfat_limit_norm = getattr(bfat_cfg, 'limit_norm', False)
        bfat_norm_ratio = getattr(bfat_cfg, 'norm_ratio', 0.5)
        bfat_freeze_bn = getattr(bfat_cfg, 'freeze_bn', False)
        bfat_proj_mode = getattr(bfat_cfg, 'projection_mode', 'direction')
        
        logger_info(logger, '=' * 80)
        logger_info(logger, f'🔥 BFAT (Bit-Flip Aware Training) - ACTIVE')
        if bfat_loss_type == 'feature_sim' and bfat_hook is not None:
            logger_info(logger, f'   Target Layer: {target_layer_name}')
        logger_info(logger, f'   Loss Type: {bfat_loss_type}')
        if bfat_loss_type == 'direct_ce':
            logger_info(logger, f'     → 直接使用 CE Loss * {bfat_weight}')
        else:
            logger_info(logger, f'     → Feature Similarity: (1 - cos_sim) * {bfat_weight}')
        
        if bfat_dual_bit:
            ber_msb = getattr(bfat_cfg, 'ber_msb', 0.01)
            ber_secondary = getattr(bfat_cfg, 'ber_secondary_msb', 0.01)
            logger_info(logger, f'   Mode: Dual Bit Flip (MSB + Secondary MSB)')
            logger_info(logger, f'   BER: MSB={ber_msb:.2e}, Secondary={ber_secondary:.2e}')
        else:
            logger_info(logger, f'   Mode: Single Bit Flip (MSB only)')
            logger_info(logger, f'   BER: {bfat_ber}')
        
        logger_info(logger, f'   Projection Base: {projection_base}')
        logger_info(logger, f'   Projection Mode: {bfat_proj_mode}')
        if bfat_proj_mode == 'orthogonal':
            logger_info(logger, f'     → 严格正交投影 (Null Space Training)')
        elif bfat_proj_mode == 'cagrad':
            logger_info(logger, f'     → CAGrad 模式 (Conflict-Averse 共赢搜索)')
        else:
            logger_info(logger, f'     → 仅方向修正 (Conflict-only Correction)')

        if bfat_limit_norm:
            logger_info(logger, f'   Magnitude Limiting: Enabled (Ratio: {bfat_norm_ratio})')
        else:
            logger_info(logger, f'   Magnitude Limiting: Disabled')
        
        if bfat_freeze_bn:
            logger_info(logger, f'   BN Stats: Frozen during BFAT (Eval Mode)')
        else:
            logger_info(logger, f'   BN Stats: Active during BFAT (Train Mode)')
        
        # 新增：是否恢复到最大位宽
        bfat_restore_max = getattr(bfat_cfg, 'restore_max_bits', True)
        logger_info(logger, f'   Restore Max Bits: {bfat_restore_max}')
        
        # 新增：是否按样本进行 BFAT
        bfat_per_sample = getattr(bfat_cfg, 'per_sample_injection', False)
        logger_info(logger, f'   Per-sample Injection: {bfat_per_sample}')
            
        if projection_base == 'combined':
            logger_info(logger, f'     → BFAT 梯度将与 (clean + nr_random) 综合梯度进行投影')
        else:
            logger_info(logger, f'     → BFAT 梯度将与 clean (max_bits) 梯度进行投影')
        logger_info(logger, '=' * 80)
    elif bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False) and epoch < bfat_start_epoch:
        logger_info(logger, f'⏳ BFAT is pending (enabled but start_epoch={bfat_start_epoch}, current_epoch={epoch})')

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx == 0:
            print(f'[DEBUG] Processing first batch, inputs shape: {inputs.shape}, targets shape: {targets.shape}')
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        # 用于存储梯度的字典（如果启用 BFAT）
        clean_grads = {}
        nr_grads = {}
        bfat_grads = {}
        clean_feature = None

        # 辅助函数：执行单次 BFAT 注入并累积梯度
        def _do_bfat_step(accum_dict):
            # 1. 根据配置决定是否恢复到最大位宽
            bfat_restore_max_inner = getattr(bfat_cfg, 'restore_max_bits', True)
            if bfat_restore_max_inner:
                sample_max_cands(model, configs)
            
            # 2. 冻结 BN (可选)
            bfat_freeze_bn_inner = getattr(bfat_cfg, 'freeze_bn', False)
            if bfat_freeze_bn_inner:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                        m.eval()

            # 3. 设置故障注入器参数
            old_only_msb = fault_injector.only_msb
            old_skip_msb = fault_injector.skip_msb
            old_bfat_idx = getattr(fault_injector, 'bfat_bit_index', None)
            old_bfat_dual = getattr(fault_injector, 'bfat_dual_bit', False)
            old_ber_msb = getattr(fault_injector, 'ber_msb', None)
            old_ber_secondary = getattr(fault_injector, 'ber_secondary_msb', None)
            old_ber = fault_injector.ber
            
            if getattr(bfat_cfg, 'dual_bit', False):
                fault_injector.bfat_dual_bit = True
                fault_injector.only_msb = False
                fault_injector.skip_msb = False
                fault_injector.bfat_bit_index = None
                fault_injector.ber_msb = getattr(bfat_cfg, 'ber_msb', 0.01)
                fault_injector.ber_secondary_msb = getattr(bfat_cfg, 'ber_secondary_msb', 0.01)
            else:
                fault_injector.bfat_dual_bit = False
                fault_injector.only_msb = True
                fault_injector.skip_msb = False
                fault_injector.bfat_bit_index = None
                fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)
            
            fault_injector.enable()
            fault_injector.reset_forward_seed()
            
            # 4. 前向传播与损失计算
            outputs_bfat = model(inputs)
            bfat_loss_type_inner = getattr(bfat_cfg, 'loss_type', 'feature_sim')
            
            if bfat_loss_type_inner == 'direct_ce':
                loss_bfat = criterion(outputs_bfat, targets) * getattr(bfat_cfg, 'loss_weight', 1.0)
            else:
                faulted_feature = bfat_hook.feature
                f_c = clean_feature.view(clean_feature.size(0), -1)
                f_f = faulted_feature.view(faulted_feature.size(0), -1)
                sim = F.cosine_similarity(f_c, f_f, dim=1).mean()
                loss_bfat = (1 - sim) * getattr(bfat_cfg, 'loss_weight', 1.0)
            
            # 5. 反向传播
            loss_bfat.backward()
            
            # 6. 累积梯度
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    accum_dict[p] = accum_dict.get(p, 0) + p.grad.clone()
            
            # 7. 恢复注入器状态
            fault_injector.disable()
            fault_injector.only_msb = old_only_msb
            fault_injector.skip_msb = old_skip_msb
            fault_injector.bfat_bit_index = old_bfat_idx
            fault_injector.bfat_dual_bit = old_bfat_dual
            fault_injector.ber_msb = old_ber_msb
            fault_injector.ber_secondary_msb = old_ber_secondary
            fault_injector.ber = old_ber
            
            if bfat_freeze_bn_inner:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                        m.train()

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
            
            # BFAT: 捕获 Clean 特征
            if use_bfat and bfat_hook is not None:
                clean_feature = bfat_hook.feature.clone().detach()

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

            # BFAT: 提取 Clean 梯度并清空，为后续 NR 准备
            if use_bfat:
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        clean_grads[p] = p.grad.clone()
                optimizer.zero_grad()
                if optimizer_q is not None:
                    optimizer_q.zero_grad()

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
                fault_injector.reset_forward_seed()
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
                
                # TRADES损失计算
                use_entropy = getattr(trades_config, 'use_entropy', False)
                entropy_weight = getattr(trades_config, 'entropy_weight', 0.1)
                entropy_mode = getattr(trades_config, 'entropy_mode', 'difference')
                
                if use_kl:
                    probs_normal = F.softmax(outputs_normal, dim=1)
                    log_probs_faulted = F.log_softmax(outputs_faulted, dim=1)
                    kl_div = F.kl_div(log_probs_faulted, probs_normal, reduction='batchmean')
                    loss = loss_normal + beta * kl_div
                    if use_entropy:
                        entropy_loss = compute_entropy_loss(probs_normal, F.softmax(outputs_faulted, dim=1), mode=entropy_mode)
                        loss = loss + entropy_weight * entropy_loss
                    QE_loss, dist_loss = QE_loss_normal, dist_loss_normal
                else:
                    loss = alpha * loss_normal + beta * loss_faulted
                    if use_entropy:
                        entropy_loss = compute_entropy_loss(F.softmax(outputs_normal, dim=1), F.softmax(outputs_faulted, dim=1), mode=entropy_mode)
                        loss = loss + entropy_weight * entropy_loss
                    QE_loss = alpha * QE_loss_normal + beta * QE_loss_faulted
                    dist_loss = alpha * dist_loss_normal + beta * dist_loss_faulted
                
                outputs = outputs_normal
            else:
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

            # --- BFAT 内部循环注入逻辑 (Per-Sample) ---
            if use_bfat and getattr(bfat_cfg, 'per_sample_injection', False):
                if fault_injector is None:
                    if batch_idx == 0 and iter_idx == 0:
                        logger_info(logger, '⚠️  WARNING: bfat.per_sample_injection is enabled but fault_injector is None. Skipping.')
                else:
                    # 记录并清除当前 NR 梯度
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            nr_grads[p] = nr_grads.get(p, 0) + p.grad.clone()
                    optimizer.zero_grad()
                    if optimizer_q is not None:
                        optimizer_q.zero_grad()
                    
                    # 执行 BFAT Step (注入并累积到 bfat_grads)
                    bfat_loss_type_check = getattr(bfat_cfg, 'loss_type', 'feature_sim')
                    if bfat_loss_type_check == 'direct_ce' or clean_feature is not None:
                        _do_bfat_step(bfat_grads)
                    
                    # 清除 BFAT 梯度，为下一个迭代做准备
                    optimizer.zero_grad()
                    if optimizer_q is not None:
                        optimizer_q.zero_grad()

        # [MODIFIED] BFAT 阶段：决定是在循环后执行一次，还是已经由 per_sample_injection 完成
        # 对于 direct_ce 模式，不需要 clean_feature；对于 feature_sim 模式，需要 clean_feature
        bfat_loss_type_check = getattr(bfat_cfg, 'loss_type', 'feature_sim') if bfat_cfg else 'feature_sim'
        bfat_can_run = use_bfat and fault_injector is not None and (
            bfat_loss_type_check == 'direct_ce' or clean_feature is not None
        )
        if bfat_can_run:
            # 记录 NR 累积的梯度并清空
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    nr_grads[p] = p.grad.clone()
            optimizer.zero_grad()
            if optimizer_q is not None:
                optimizer_q.zero_grad()

            # 配置故障注入器进行 BFAT 注入
            # 1. 根据配置决定是否恢复到最大位宽 (max_bits)
            # bfat_restore_max: 如果为 True (默认)，则在 BFAT 之前恢复到 max_bits
            # 如果为 False，则在当前 NR 采样后的位宽状态下进行 BFAT
            bfat_restore_max = getattr(bfat_cfg, 'restore_max_bits', True)
            if bfat_restore_max:
                sample_max_cands(model, configs)
                if batch_idx == 0:
                    logger_info(logger, '   [BFAT] Restored to max_bits for fault injection')
            else:
                if batch_idx == 0:
                    logger_info(logger, '   [BFAT] Running on current sampled bits (NO restoration to max_bits)')
            
            # --- 可选：将所有 BN 层设为 eval 模式，防止故障特征污染统计量 ---
            bfat_freeze_bn = getattr(bfat_cfg, 'freeze_bn', False)
            if bfat_freeze_bn:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                        m.eval()

            # 2. 临时备份并设置 BFAT 注入参数
            old_only_msb = fault_injector.only_msb
            old_skip_msb = fault_injector.skip_msb
            old_bfat_idx = getattr(fault_injector, 'bfat_bit_index', None)
            old_bfat_dual = getattr(fault_injector, 'bfat_dual_bit', False)
            old_ber_msb = getattr(fault_injector, 'ber_msb', None)
            old_ber_secondary = getattr(fault_injector, 'ber_secondary_msb', None)
            old_ber = fault_injector.ber
            
            # 设置 BFAT 专用参数
            bfat_dual_bit = getattr(bfat_cfg, 'dual_bit', False)
            if bfat_dual_bit:
                # 双位翻转模式：MSB + Secondary MSB
                fault_injector.bfat_dual_bit = True
                fault_injector.only_msb = False
                fault_injector.skip_msb = False
                fault_injector.bfat_bit_index = None
                fault_injector.ber_msb = getattr(bfat_cfg, 'ber_msb', 0.01)
                fault_injector.ber_secondary_msb = getattr(bfat_cfg, 'ber_secondary_msb', 0.01)
            else:
                # 默认：仅翻转 MSB
                fault_injector.bfat_dual_bit = False
                fault_injector.only_msb = True
                fault_injector.skip_msb = False
                fault_injector.bfat_bit_index = None
                fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)
            
            fault_injector.enable()
            # 确保 BFAT forward 使用的 seed 与 clean forward 保持某种联系或独立，
            # 这里重置 seed 确保注入发生
            fault_injector.reset_forward_seed()
            
            outputs_bfat = model(inputs)
            
            # 3. 计算 BFAT 损失
            # loss_type: "feature_sim" - 使用 feature 相似度损失 (1 - cos_sim)
            #            "direct_ce"   - 直接使用 CE loss
            bfat_loss_type = getattr(bfat_cfg, 'loss_type', 'feature_sim')
            
            if bfat_loss_type == 'direct_ce':
                # Direct CE 模式：直接计算故障注入后的 CE loss
                loss_bfat = criterion(outputs_bfat, targets) * getattr(bfat_cfg, 'loss_weight', 1.0)
            else:
                # Feature Similarity 模式（默认）：计算 feature 相似度损失
                faulted_feature = bfat_hook.feature
                f_c = clean_feature.view(clean_feature.size(0), -1)
                f_f = faulted_feature.view(faulted_feature.size(0), -1)
                sim = F.cosine_similarity(f_c, f_f, dim=1).mean()
                loss_bfat = (1 - sim) * getattr(bfat_cfg, 'loss_weight', 1.0)
            
            loss_bfat.backward()
            
            # 4. 记录 BFAT 梯度并恢复注入器状态
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    bfat_grads[p] = p.grad.clone()
            
            fault_injector.disable()
            fault_injector.only_msb = old_only_msb
            fault_injector.skip_msb = old_skip_msb
            fault_injector.bfat_bit_index = old_bfat_idx
            fault_injector.bfat_dual_bit = old_bfat_dual
            fault_injector.ber_msb = old_ber_msb
            fault_injector.ber_secondary_msb = old_ber_secondary
            fault_injector.ber = old_ber
            
            # --- 恢复：如果之前锁定了 BN，现在恢复为 train 模式 ---
            if getattr(bfat_cfg, 'freeze_bn', False):
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                        m.train()
            
            # --- 梯度投影与合并 ---
            # 根据配置选择投影基准梯度
            # projection_base: "clean" 仅使用 max_bits 的 clean 梯度
            #                  "combined" 使用 clean + nr_random 的综合梯度
            projection_base = getattr(bfat_cfg, 'projection_base', 'clean')
            limit_norm = getattr(bfat_cfg, 'limit_norm', False)
            norm_ratio = getattr(bfat_cfg, 'norm_ratio', 0.5)
            proj_mode = getattr(bfat_cfg, 'projection_mode', 'direction')
            
            if projection_base == 'combined':
                # 使用综合梯度作为投影基准：clean_grads + nr_grads
                base_grads = {}
                for p in model.parameters():
                    if p.requires_grad:
                        g_clean = clean_grads.get(p, None)
                        g_nr = nr_grads.get(p, None)
                        if g_clean is not None and g_nr is not None:
                            base_grads[p] = g_clean + g_nr
                        elif g_clean is not None:
                            base_grads[p] = g_clean
                        elif g_nr is not None:
                            base_grads[p] = g_nr
                projected_bfat = project_bfat_gradients(base_grads, bfat_grads, limit_norm=limit_norm, norm_ratio=norm_ratio, projection_mode=proj_mode)
            else:
                # 默认：仅使用 clean_grads 作为投影基准
                projected_bfat = project_bfat_gradients(clean_grads, bfat_grads, limit_norm=limit_norm, norm_ratio=norm_ratio, projection_mode=proj_mode)
            
            for p in model.parameters():
                if p.requires_grad:
                    # 合并三部分梯度：Clean (max) + NR Random + Projected BFAT
                    # 确保只有存在 Tensor 时才进行相加，否则保持 None
                    g_list = []
                    for g_dict in [clean_grads, nr_grads, projected_bfat]:
                        g = g_dict.get(p)
                        if g is not None:
                            g_list.append(g)
                    
                    if len(g_list) > 0:
                        g_final = g_list[0]
                        for i in range(1, len(g_list)):
                            g_final = g_final + g_list[i]
                        p.grad = g_final
                    else:
                        p.grad = None
        
        elif not use_bfat:
            # 如果不使用 BFAT，梯度已经由 max_sample 和 nr_random_sample 累积在 p.grad 中
            pass

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

