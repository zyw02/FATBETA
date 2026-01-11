import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from quan.func import QuanConv2d, QuanLinear, SwithableBatchNorm
from quan.quantizer.lsq import LsqQuan
from util import AverageMeter
from util.utils import accuracy, update_meter, set_global_seed
from util.dist import master_only, logger_info
from util.mpq import switch_bit_width

__all__ = ["train", "validate", "PerformanceScoreboard"]

logger = logging.getLogger()


# ---------------- SR-QAT: Penalty-based Scale-Constrained QAT ----------------
class _MarginEMA:
    """EMA of positive logit margin (detached) for SR-QAT scale penalty."""
    def __init__(self, momentum: float = 0.9):
        self.momentum = float(momentum)
        self.value = None

    @torch.no_grad()
    def update(self, margin: torch.Tensor) -> torch.Tensor:
        m = torch.clamp(margin.detach(), min=0.0)
        mean_m = m.mean()
        if self.value is None:
            self.value = mean_m
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * mean_m
        return self.value


_SRQAT_MARGIN_EMA: _MarginEMA | None = None


def _compute_logit_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """margin m_i = z_y - max_{c != y} z_c"""
    top2 = logits.topk(2, dim=1).values
    preds = logits.argmax(dim=1)
    z_y = logits[torch.arange(logits.size(0), device=logits.device), targets]
    z_comp = torch.where(preds == targets, top2[:, 1], top2[:, 0])
    return z_y - z_comp


def _get_module_weight_bits(module: nn.Module):
    """Best-effort read of current weight bit-width for a quantized module."""
    if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        fb = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
        if isinstance(fb, torch.Tensor):
            return int(fb.item())
        return int(fb)
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
    epoch: int,
) -> torch.Tensor:
    """
    SR-QAT penalty: L_scale = lambda * sum_l s_l^2 / (m_bar^2 + eps)
    """
    scale_cfg = getattr(configs, 'scale_penalty', None)
    if scale_cfg is None or not getattr(scale_cfg, 'enabled', False):
        return outputs.new_tensor(0.0)

    start_epoch = int(getattr(scale_cfg, 'start_epoch', 0))
    if epoch < start_epoch:
        return outputs.new_tensor(0.0)

    global _SRQAT_MARGIN_EMA
    if _SRQAT_MARGIN_EMA is None:
        _SRQAT_MARGIN_EMA = _MarginEMA(momentum=float(getattr(scale_cfg, 'margin_momentum', 0.9)))

    margin = _compute_logit_margin(outputs, targets)
    m_bar = _SRQAT_MARGIN_EMA.update(margin)

    eps = float(getattr(scale_cfg, 'eps', 1e-6))
    denom = (m_bar ** 2) + eps

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

    if bool(getattr(scale_cfg, 'normalize_by_num_layers', False)):
        sum_s2 = sum_s2 / float(num_terms)

    lam = float(getattr(scale_cfg, 'lambda_scale', 0.0))
    return (lam * sum_s2) / denom


def _max_target_bit(configs) -> int:
    target_bits = getattr(configs, "target_bits", [6, 5, 4, 3, 2])
    if isinstance(target_bits, (list, tuple)) and len(target_bits) > 0:
        return int(max(target_bits))
    return int(target_bits) if target_bits is not None else 6


def _force_max_bitwidth(model: nn.Module, configs) -> int:
    """Force all dynamic quantized layers to use max(target_bits) for both W/A."""
    max_bit = _max_target_bit(configs)
    try:
        switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
    except Exception:
        # Best-effort: training should still run even if BN switching differs
        pass
    return max_bit


def _get_meters(mode: str = "training"):
    return {
        "name": mode,
        "loss": AverageMeter(),
        # Keep key names consistent with util.utils.update_meter()
        "top1": AverageMeter(),
        "top5": AverageMeter(),
        "QE_loss": AverageMeter(),
        "dist_loss": AverageMeter(),
        "IDM_loss": AverageMeter(),
        "batch_time": AverageMeter(),
        # NEW: BFAT Simulation Losses
        "sim_res_loss": AverageMeter(), # Restricted BFAT Loss (avg of 3)
        "sim_bfat_loss": AverageMeter(), # Only MSB BFAT Loss
    }


class FeatureHook:
    """Simple hook to capture features from a target layer."""
    def __init__(self, module):
        self.handle = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.handle.remove()


def project_bfat_gradients(clean_grads, bfat_grads, limit_norm=False, norm_ratio=0.5, projection_mode="direction",
                           weight_relative_limit=False, weight_limit_ratio=0.01):
    """
    BFAT gradient processing logic:
    1. projection_mode:
       - "direction": Project only if angle > 90 deg, removing harmful component.
       - "orthogonal": Always project, forcing strict orthogonality to base gradient.
       - "cagrad": Conflict-Averse Gradient Descent.
    2. limit_norm: Ensure projected gradient magnitude doesn't exceed base gradient * ratio.
    3. weight_relative_limit: If True, limit_norm uses weight magnitude as base instead of gradient magnitude.
    """
    projected_bfat_grads = {}
    
    # 显式支持 none 模式，直接返回原始梯度
    if projection_mode == "none":
        for p, g_b in bfat_grads.items():
            projected_bfat_grads[p] = g_b.clone()
        return projected_bfat_grads

    for p, g_b in bfat_grads.items():
        if p in clean_grads:
            g_c = clean_grads[p]
            g_c_d = g_c.to(torch.float64)
            g_b_d = g_b.to(torch.float64)
            dot_product = torch.sum(g_c_d * g_b_d)
            norm_sq_c = torch.sum(g_c_d * g_c_d) + 1e-8
            projection = (dot_product / norm_sq_c) * g_c_d

            if projection_mode == "orthogonal":
                g_b_cleaned_d = g_b_d - projection
            elif projection_mode == "cagrad":
                norm_sq_b = torch.sum(g_b_d * g_b_d) + 1e-8
                numerator = norm_sq_b - dot_product
                denominator = norm_sq_c + norm_sq_b - 2 * dot_product + 1e-8
                alpha = torch.clamp(numerator / denominator, 0.0, 1.0)
                g_target_d = alpha * g_c_d + (1.0 - alpha) * g_b_d
                g_b_cleaned_d = g_target_d - g_c_d
            else:
                if dot_product < 0:
                    g_b_cleaned_d = g_b_d - projection
                else:
                    g_b_cleaned_d = g_b_d
            projected_bfat_grads[p] = g_b_cleaned_d.to(dtype=g_b.dtype)
        else:
            projected_bfat_grads[p] = g_b
            
    # --- 2. Magnitude Limiting ---
    if limit_norm:
        if weight_relative_limit:
            total_norm_base = torch.sqrt(sum(torch.sum(p.to(torch.float64)**2) for p in bfat_grads.keys()) + 1e-8)
            target_global_norm = total_norm_base * weight_limit_ratio
        else:
            total_norm_base = torch.sqrt(sum(torch.sum(g.to(torch.float64)**2) for g in clean_grads.values()) + 1e-8)
            target_global_norm = total_norm_base * norm_ratio
        total_norm_b = torch.sqrt(sum(torch.sum(g.to(torch.float64)**2) for g in projected_bfat_grads.values()) + 1e-8)
        
        if total_norm_b > target_global_norm:
            scale = target_global_norm / total_norm_b
            for p in projected_bfat_grads:
                projected_bfat_grads[p] = (projected_bfat_grads[p].to(torch.float64) * scale).to(projected_bfat_grads[p].dtype)

    return projected_bfat_grads


@master_only
def _update_monitors(monitors, meters, epoch, batch_idx, steps_per_epoch, optimizer, mode="training"):
    if monitors is None:
        return
    lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    stats = {
        "lr": lr,
        "loss": meters["loss"].avg,
        "acc1": meters["top1"].avg,
        "acc5": meters["top5"].avg,
    }
    # Add simulation losses if available
    if "sim_res_loss" in meters:
        stats["sim_res_loss"] = meters["sim_res_loss"].avg
    if "sim_bfat_loss" in meters:
        stats["sim_bfat_loss"] = meters["sim_bfat_loss"].avg
        
    for m in monitors:
        try:
            m.update(epoch, batch_idx, steps_per_epoch, mode=mode, **stats)
        except Exception:
            pass


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    monitors,
    configs,
    model_ema=None,
    nr_random_sample=0,  # ignored by design (NO multi-path)
    mode="training",
    soft_criterion=None,
    teacher_model=None,
    optimizer_q=None,
    annealing_schedule=None,
    freezing_annealing_schedule=None,
    IDM_weight=0.0,
    scaler=None,
    fault_injector=None,
    output_corrector=None,
    corrector_optimizer=None,
    device=None,
):
    """
    NUDE training (with optional SR-QAT):
    - No nr_random_sample / mixed-path sampling
    - Always train the max(target_bits) subnet
    - Loss = CrossEntropy + optional SR-QAT penalty
    """
    # Keep reproducibility consistent with the rest of the repo
    num_updates = epoch * len(train_loader)
    set_global_seed(num_updates + 1)

    model.train()
    if model_ema is not None:
        model_ema.ema.train()

    max_bit = _force_max_bitwidth(model, configs)
    logger_info(logger, f"[NUDE] Epoch {epoch}: force max bit-width W/A = {max_bit}")

    # --- [NUDE] BFAT Setup ---
    bfat_cfg = getattr(configs, 'bfat', None)
    bfat_start_epoch = getattr(bfat_cfg, 'start_epoch', 0)
    use_bfat = bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False) and epoch >= bfat_start_epoch
    bfat_hook = None
    if use_bfat:
        bfat_loss_type = getattr(bfat_cfg, 'loss_type', 'feature_sim')
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
                logger_info(logger, f'⚠️ [NUDE] BFAT WARNING: Target layer {target_layer_name} not found for feature_sim!')
                use_bfat = False

        # Config Logging
        logger_info(logger, '=' * 80)
        logger_info(logger, f'🔥 [NUDE] BFAT ACTIVE')
        logger_info(logger, f'   Loss Type: {bfat_loss_type}')
        logger_info(logger, f'   Proj Mode: {getattr(bfat_cfg, "projection_mode", "direction")}')
        logger_info(logger, f'   Injection: {"Dual Bit" if getattr(bfat_cfg, "dual_bit", False) else ("Only MSB" if getattr(bfat_cfg, "only_msb", False) else ("All Bits" if getattr(bfat_cfg, "all_bits", False) else "Standard"))}')
        logger_info(logger, '=' * 80)

    meters = _get_meters(mode=mode)
    end = time.time()
    steps_per_epoch = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        # Ensure bits are kept at max for dynamic layers
        _force_max_bitwidth(model, configs)

        # 1. Base Forward Pass (Clean or Restricted)
        # 优化点：如果 use_restricted_base 为 True，直接执行受限加噪的前向传播，减少一次冗余的 Clean Forward
        use_restricted_base = getattr(bfat_cfg, 'use_restricted_as_base', False) if use_bfat else False
        proj_mode = getattr(bfat_cfg, 'projection_mode', 'direction') if use_bfat else "direction"
        clean_grads = {}

        if use_restricted_base:
            # === 模式：使用受限故障注入作为基准 (优化性能，合并 Forward) ===
            # 备份原有 injector 状态
            old_state = {
                'only_msb': fault_injector.only_msb,
                'skip_msb': fault_injector.skip_msb,
                'skip_msbn': getattr(fault_injector, 'skip_msbn', False),
                'all_bits': getattr(fault_injector, 'all_bits', False),
                'ber': fault_injector.ber,
                'seed': fault_injector.seed,
                'seed_list': fault_injector.seed_list
            }

            # 配置受限注入：跳过高两位 (skip_msbn)，且 BER 单独配置
            fault_injector.only_msb = False
            fault_injector.skip_msb = False
            fault_injector.skip_msbn = True
            fault_injector.all_bits = False
            fault_injector.ber = getattr(bfat_cfg, 'ber_base_restricted', 0.01)
            fault_injector.seed_list = None
            # 使用一个独特的种子偏移，避免与后续攻击梯度种子重合
            fault_injector.seed = num_updates + batch_idx * 10 + 555 
            
            fault_injector.enable()
            fault_injector.reset_forward_seed()
            
            # 这里的 outputs 是受限注入的结果，它将作为后续计算的基准
            outputs = model(inputs)
            
            # 捕获特征 (这是受限状态下的特征，作为后续攻击对比的 base)
            clean_feature = None
            if use_bfat and bfat_hook is not None:
                clean_feature = bfat_hook.feature.clone().detach()

            ce_loss = criterion(outputs, targets)
            srqat_loss = _compute_srqat_scale_penalty(outputs, targets, model, configs, epoch)
            loss = ce_loss + srqat_loss # 这里的 loss 就是受限注入的 Loss
            
            # 如果需要梯度投影，计算基准梯度 (Clean Grads 现在实际上是 Restricted Grads)
            if proj_mode != "none":
                loss.backward()
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        clean_grads[p] = p.grad.clone()
                
                # 重置梯度，准备后续的攻击梯度计算
                optimizer.zero_grad()
                if optimizer_q is not None:
                    optimizer_q.zero_grad()
            
            # 还原 injector 状态
            fault_injector.disable()
            for k, v in old_state.items():
                setattr(fault_injector, k, v)
        else:
            # === 传统模式：执行标准 Clean Forward ===
            outputs = model(inputs)
            
            # Capture clean feature
            clean_feature = None
            if use_bfat and bfat_hook is not None:
                clean_feature = bfat_hook.feature.clone().detach()

            ce_loss = criterion(outputs, targets)
            srqat_loss = _compute_srqat_scale_penalty(outputs, targets, model, configs, epoch)
            loss = ce_loss + srqat_loss
            
            if use_bfat and proj_mode != "none":
                # 计算标准 Clean 梯度
                loss.backward()
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        clean_grads[p] = p.grad.clone()
                
                optimizer.zero_grad()
                if optimizer_q is not None:
                    optimizer_q.zero_grad()
            elif not use_bfat:
                # 不使用 BFAT 时，直接执行常规反传
                loss.backward()

        # 3. BFAT 攻击梯度捕获与投影
        if use_bfat and fault_injector is not None:
            accumulate_grad_mode = getattr(bfat_cfg, 'simulate_normal_accumulation', False)

            # Backup and set BFAT-specific injector state
            old_only_msb = fault_injector.only_msb
            old_skip_msb = fault_injector.skip_msb
            old_skip_msbn = getattr(fault_injector, 'skip_msbn', False)
            old_all_bits = getattr(fault_injector, 'all_bits', False)
            old_bfat_idx = getattr(fault_injector, 'bfat_bit_index', None)
            old_bfat_dual = getattr(fault_injector, 'bfat_dual_bit', False)
            old_ber_msb = getattr(fault_injector, 'ber_msb', None)
            old_ber_secondary = getattr(fault_injector, 'ber_secondary_msb', None)
            old_ber = fault_injector.ber
            # Backup seed list to ensure manual seed control works
            old_seed_list = fault_injector.seed_list
            # Temporarily disable seed_list to force usage of fault_injector.seed
            fault_injector.seed_list = None

            bfat_freeze_bn = getattr(bfat_cfg, 'freeze_bn', False)
            if bfat_freeze_bn:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                        m.eval()

            # --- 核心逻辑分支 ---
            if accumulate_grad_mode:
                # === 模式 A: 模拟 Normal 组的梯度累积 (Max + 3*Restricted + 1*OnlyMSB) ===
                
                # 1. 累积 3 次排除 MSB/SecMSB 的 BFAT 梯度 (作为 "Clean" 的补充)
                # 设置为 restricted 模式 (Skip MSBN: 忽略最高两位)
                fault_injector.bfat_dual_bit = False
                fault_injector.only_msb = False
                fault_injector.skip_msb = False
                fault_injector.skip_msbn = True # Skip highest 2 bits (MSB & SecMSB)
                fault_injector.all_bits = False
                fault_injector.bfat_bit_index = None
                # 使用 restricted 专用 BER，如果未定义则回退到全局 BER
                fault_injector.ber = getattr(bfat_cfg, 'ber_restricted', getattr(bfat_cfg, 'ber', 0.01))

                avg_restricted_loss = 0.0
                for i in range(2):
                    fault_injector.enable()
                    # Manually set a varying seed for each accumulation step
                    # Base seed: num_updates (epoch * len(loader)) + batch_idx * 10 + i
                    # This ensures uniqueness across batches and steps
                    current_iter_seed = num_updates + batch_idx * 10 + i
                    fault_injector.seed = current_iter_seed
                    fault_injector.reset_forward_seed()
                    
                    outputs_restricted = model(inputs)
                    
                    # 计算 restricted loss (直接使用 CE，视为一种数据增强)
                    loss_restricted = criterion(outputs_restricted, targets)
                    loss_restricted.backward()
                    
                    # 记录 Loss
                    avg_restricted_loss += loss_restricted.item()

                    # 累积梯度到 clean_grads
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            if p in clean_grads:
                                clean_grads[p] += p.grad.clone()
                            else:
                                clean_grads[p] = p.grad.clone()
                    
                    # 清空本次反传的梯度
                    optimizer.zero_grad()
                    if optimizer_q is not None:
                        optimizer_q.zero_grad()
                        
                avg_restricted_loss /= 2.0
                # 更新 sim_res_loss Meter
                if 'sim_res_loss' in meters:
                    meters['sim_res_loss'].update(avg_restricted_loss, inputs.size(0))

                # 2. 执行一次 Only MSB 的 BFAT (作为 "Robust" 攻击梯度)
                fault_injector.bfat_dual_bit = False
                fault_injector.only_msb = True # 关键：Only MSB，最强攻击
                fault_injector.skip_msb = False
                fault_injector.skip_msbn = False
                fault_injector.all_bits = False
                fault_injector.bfat_bit_index = None
                fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)

                fault_injector.enable()
                # Also vary seed for Only MSB step to be distinct from restricted steps
                current_msb_seed = num_updates + batch_idx * 10 + 99 # offset 99 to avoid collision
                fault_injector.seed = current_msb_seed
                fault_injector.reset_forward_seed()
                outputs_bfat = model(inputs)
                
                # Compute BFAT Loss (feature_sim or direct_ce)
                bfat_loss_type = getattr(bfat_cfg, 'loss_type', 'feature_sim')
                loss_bfat_val = 0.0
                if bfat_loss_type == 'direct_ce':
                    loss_bfat = criterion(outputs_bfat, targets) * getattr(bfat_cfg, 'loss_weight', 1.0)
                    loss_bfat_val = loss_bfat.item()
                else:
                    f_c = clean_feature.view(clean_feature.size(0), -1)
                    f_f = bfat_hook.feature.view(bfat_hook.feature.size(0), -1)
                    sim = F.cosine_similarity(f_c, f_f, dim=1).mean()
                    loss_bfat = (1 - sim) * getattr(bfat_cfg, 'loss_weight', 1.0)
                    loss_bfat_val = loss_bfat.item()
                
                # 更新 sim_bfat_loss Meter
                if 'sim_bfat_loss' in meters:
                    meters['sim_bfat_loss'].update(loss_bfat_val, inputs.size(0))
                
                loss_bfat.backward()
                
                # Capture Only MSB Gradients
                bfat_grads = {}
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        bfat_grads[p] = p.grad.clone()
                        
            else:
                # === 模式 B: 原有逻辑 (标准 Nude BFAT) ===
                
                # Apply BFAT settings from config
                fault_injector.bfat_dual_bit = getattr(bfat_cfg, 'dual_bit', False)
                fault_injector.only_msb = getattr(bfat_cfg, 'only_msb', False)
                fault_injector.skip_msb = getattr(bfat_cfg, 'skip_msb', False)
                fault_injector.skip_msbn = getattr(bfat_cfg, 'skip_msbn', False)
                fault_injector.all_bits = getattr(bfat_cfg, 'all_bits', False)
                fault_injector.bfat_bit_index = getattr(bfat_cfg, 'bit_index', None)
                
                fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)
                fault_injector.ber_msb = getattr(bfat_cfg, 'ber_msb', None)
                fault_injector.ber_secondary_msb = getattr(bfat_cfg, 'ber_secondary_msb', None)

                # [DDP] Add rank offset
                rank_offset = getattr(configs, 'rank', 0) * 100000
                fault_injector.seed = num_updates + batch_idx * 10 + 7 + rank_offset
                fault_injector.enable()
                fault_injector.reset_forward_seed()
                
                outputs_bfat = model(inputs)
                
                # Compute BFAT Loss
                bfat_loss_type = getattr(bfat_cfg, 'loss_type', 'feature_sim')
                if bfat_loss_type == 'direct_ce':
                    loss_bfat = criterion(outputs_bfat, targets) * getattr(bfat_cfg, 'loss_weight', 1.0)
                else:
                    # feature_sim
                    f_c = clean_feature.view(clean_feature.size(0), -1)
                    f_f = bfat_hook.feature.view(bfat_hook.feature.size(0), -1)
                    sim = F.cosine_similarity(f_c, f_f, dim=1).mean()
                    loss_bfat = (1 - sim) * getattr(bfat_cfg, 'loss_weight', 1.0)

                if proj_mode == "none":
                    # [叠加模式] 直接将两个 loss 相加后进行一次反向传播
                    (loss + loss_bfat).backward()
                else:
                    # [投影模式] 独立反传 BFAT loss 以便捕获其梯度
                    loss_bfat.backward()

                # Capture BFAT Gradients (仅在非叠加模式下需要)
                bfat_grads = {}
                if proj_mode != "none":
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            bfat_grads[p] = p.grad.clone()

            if fault_injector is not None:
                fault_injector.disable()
                fault_injector.only_msb = old_only_msb
                fault_injector.skip_msb = old_skip_msb
                fault_injector.skip_msbn = old_skip_msbn
                fault_injector.all_bits = old_all_bits
                fault_injector.bfat_bit_index = old_bfat_idx
                fault_injector.bfat_dual_bit = old_bfat_dual
                fault_injector.ber_msb = old_ber_msb
                fault_injector.ber_secondary_msb = old_ber_secondary
                fault_injector.ber = old_ber
                fault_injector.seed_list = old_seed_list

            if bfat_freeze_bn:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                        m.train()

            # 4. Projection and Merge
            # 如果是叠加模式，梯度已经在 p.grad 中了，不需要再处理
            if proj_mode != "none":
                limit_norm = getattr(bfat_cfg, 'limit_norm', False)
                norm_ratio = getattr(bfat_cfg, 'norm_ratio', 0.5)
                weight_rel_limit = getattr(bfat_cfg, 'weight_relative_limit', False)
                weight_limit_ratio = getattr(bfat_cfg, 'weight_limit_ratio', 0.01)
                
                projected_bfat = project_bfat_gradients(
                    clean_grads, bfat_grads, 
                    limit_norm=limit_norm, 
                    norm_ratio=norm_ratio, 
                    projection_mode=proj_mode,
                    weight_relative_limit=weight_rel_limit,
                    weight_limit_ratio=weight_limit_ratio
                )

                for p in model.parameters():
                    if p.requires_grad:
                        g_c = clean_grads.get(p, None)
                        g_b_proj = projected_bfat.get(p, None)
                        
                        g_final = None
                        if g_c is not None:
                            g_final = g_c
                        if g_b_proj is not None:
                            g_final = g_final + g_b_proj if g_final is not None else g_b_proj
                        
                        p.grad = g_final

        nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()
        if optimizer_q is not None:
            optimizer_q.step()

        if model_ema is not None:
            model_ema.update(model)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        update_meter(
            meters,
            loss,
            None,
            None,
            None,
            acc1,
            acc5,
            inputs.size(0),
            time.time() - end,
            configs.world_size,
        )
        end = time.time()

        if (batch_idx + 1) % configs.log.print_freq == 0:
            try:
                # 增强日志输出，包含模拟的 Loss 信息
                log_msg = (
                    f"[NUDE][TRAIN] Epoch {epoch}/{configs.epochs} "
                    f"Iter {batch_idx+1}/{steps_per_epoch} "
                    f"Loss {meters['loss'].avg:.4f} "
                )
                
                # 如果有模拟 Loss，追加显示
                if 'sim_res_loss' in meters and meters['sim_res_loss'].count > 0:
                    log_msg += f"ResLoss {meters['sim_res_loss'].avg:.4f} "
                if 'sim_bfat_loss' in meters and meters['sim_bfat_loss'].count > 0:
                    log_msg += f"BfatLoss {meters['sim_bfat_loss'].avg:.4f} "
                
                log_msg += (
                    f"Top1 {meters['top1'].avg:.2f} "
                    f"Top5 {meters['top5'].avg:.2f}"
                )
                
                logger_info(logger, log_msg)
            except Exception:
                pass
            _update_monitors(monitors, meters, epoch, batch_idx, steps_per_epoch, optimizer, mode=mode)

    if bfat_hook is not None:
        bfat_hook.remove()

    return meters["top1"].avg, meters["top5"].avg, meters["loss"].avg


def validate(
    data_loader,
    model,
    criterion,
    epoch,
    monitors,
    configs,
    nr_random_sample=0,  # ignored
    alpha=1,
    train_loader=None,
    eval_predefined_arch=None,
    bops_limit=1e10,
    train_mode=False,
):
    """
    NUDE validation: evaluate only the max(target_bits) subnet.
    """
    model.eval()
    max_bit = _force_max_bitwidth(model, configs)
    logger_info(logger, f"[NUDE] Validate: force max bit-width W/A = {max_bit}")

    meters = _get_meters(mode="validation")
    steps = len(data_loader)
    end = time.time()

    try:
        import torch.backends.cudnn as cudnn
        _prev_bench = cudnn.benchmark
        cudnn.benchmark = True
    except Exception:
        _prev_bench = None

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(
                meters,
                loss,
                None,
                None,
                None,
                acc1,
                acc5,
                inputs.size(0),
                time.time() - end,
                configs.world_size,
            )
            end = time.time()

    try:
        import torch.backends.cudnn as cudnn
        if _prev_bench is not None:
            cudnn.benchmark = _prev_bench
    except Exception:
        pass

    _update_monitors(monitors, meters, epoch, steps - 1, steps, optimizer=None, mode="validation")
    if getattr(configs, "distributed", False):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                s = torch.tensor([meters["top1"].sum, meters["top1"].count], device="cuda", dtype=torch.float64)
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                return float(s[0] / s[1])
        except Exception:
            pass
    return meters["top1"].avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores=3):
        self.num_best_scores = num_best_scores
        self.board = []

    def update(self, top1, top5, epoch):
        self.board.append((top1, top5, epoch))
        self.board = sorted(self.board, key=lambda x: x[0], reverse=True)[: self.num_best_scores]

    def is_best(self, top1):
        if len(self.board) == 0:
            return True
        return top1 >= max([x[0] for x in self.board])
