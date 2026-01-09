import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from quan.func import QuanConv2d, QuanLinear, SwithableBatchNorm
from quan.quantizer.lsq import LsqQuan, compute_thd
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


class PBSForwardWrapper:
    """
    A temporary wrapper for the quantizer's forward pass to inject PBS winner flips
    without physically modifying the weight.data parameters.
    Matches the FaultInjector's behavior of modifying the quantized tensor on-the-fly.
    """
    def __init__(self, original_forward, winners):
        self.original_forward = original_forward
        self.winners = winners # List of winners for this specific layer

    def __call__(self, x, bits, **kwargs):
        # 1. Get the original quantized weight tensor
        x_q = self.original_forward(x, bits, **kwargs)
        
        # 2. Apply PBS flips if bits matches (only on weights)
        is_activation = kwargs.get('is_activation', False)
        if not is_activation and bits is not None and bits < 32:
            # Create a delta tensor on the same device as x_q
            delta = torch.zeros_like(x_q)
            flat_delta = delta.view(-1)
            for w in self.winners:
                flat_delta[w['idx']] = w['delta_w']
            
            # 3. FaultInjector style: forward uses faulted, backward uses original
            # This ensures p.grad is calculated with respect to original p
            x_faulted = x_q + delta
            return x_faulted.detach() + (x_q - x_q.detach())
        
        return x_q


def _pbs_find_winner_bits(model, n_flips=1):
    """
    Optimized PBS search for training:
    Directly reuses existing gradients from the main backward pass to select winners
    via Taylor approximation (grad * delta_w). 
    Zero extra forward/backward passes during search.
    """
    model.eval()
    quantized_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            quantized_layers.append((name, module))

    candidate_pool = []
    for name, module in quantized_layers:
        w = module.weight
        # Re-use the gradient already computed in the main training loop
        if w.grad is None: continue
        grad = w.grad.data
        
        # Get quantization parameters
        wbits = _get_module_weight_bits(module)
        if wbits is None or wbits >= 32:
            continue
            
        thd_neg, thd_pos = compute_thd(module.quan_w_fn, wbits)
        s = module.quan_w_fn.get_scale(wbits, detach=True)
        
        # Compute quantized integer code
        # Note: round(w/s) is done on GPU, very fast.
        code = torch.round(w.data / s).clamp(thd_neg, thd_pos)
        code_unsigned = (code - thd_neg).to(torch.int64)
        
        for bit_i in range(wbits):
            bit_val = (code_unsigned >> bit_i) & 1
            delta_w = (1 - 2 * bit_val) * (2 ** bit_i) * s
            
            # Taylor approximation: Delta_L approx grad * delta_w
            sens_all = grad * delta_w
            
            # Find the most sensitive bit in this layer for this bit position
            v_max, i_max = torch.max(sens_all.view(-1), dim=0)
            
            if v_max > 0:
                candidate_pool.append({
                    'name': name, 'module': module, 'idx': i_max.item(), 
                    'bit_i': bit_i, 'delta_w': delta_w.view(-1)[i_max.item()].item(),
                    'sens': v_max.item()
                })
    
    # Sort and pick global winners based on Taylor sensitivity
    winners = sorted(candidate_pool, key=lambda x: x['sens'], reverse=True)[:n_flips]
    return winners


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
            
            # [Optimization] Use FP64 for projection calculations to ensure numerical stability
            original_dtype = g_b.dtype
            g_c_64 = g_c.to(torch.float64)
            g_b_64 = g_b.to(torch.float64)
            
            # --- 1. Directional / Space Processing ---
            dot_product = torch.sum(g_c_64 * g_b_64)
            norm_sq_c = torch.sum(g_c_64 * g_c_64) + 1e-12
            
            # 显式计算投影向量
            projection_64 = (dot_product / norm_sq_c) * g_c_64

            if projection_mode == "orthogonal":
                # Strict orthogonality
                g_b_cleaned_64 = g_b_64 - projection_64
            elif projection_mode == "cagrad":
                # CAGrad mode
                norm_sq_b = torch.sum(g_b_64 * g_b_64) + 1e-12
                numerator = norm_sq_b - dot_product
                denominator = norm_sq_c + norm_sq_b - 2 * dot_product + 1e-12
                alpha = torch.clamp(numerator / denominator, 0.0, 1.0)
                g_target_64 = alpha * g_c_64 + (1.0 - alpha) * g_b_64
                g_b_cleaned_64 = g_target_64 - g_c_64
            else:
                # "direction" mode
                if dot_product < 0:
                    g_b_cleaned_64 = g_b_64 - projection_64
                else:
                    g_b_cleaned_64 = g_b_64
            
            # Convert back to original dtype
            projected_bfat_grads[p] = g_b_cleaned_64.to(original_dtype)
        else:
            projected_bfat_grads[p] = g_b
            
    # --- 2. Magnitude Limiting ---
    if limit_norm:
        if weight_relative_limit:
            # Base is weight magnitude
            total_norm_base = torch.sqrt(sum(torch.sum(p.to(torch.float64)**2) for p in bfat_grads.keys()) + 1e-12)
            target_global_norm = total_norm_base * weight_limit_ratio
        else:
            # Base is clean gradient magnitude
            total_norm_base = torch.sqrt(sum(torch.sum(g.to(torch.float64)**2) for g in clean_grads.values()) + 1e-12)
            target_global_norm = total_norm_base * norm_ratio
            
        total_norm_b = torch.sqrt(sum(torch.sum(g.to(torch.float64)**2) for g in projected_bfat_grads.values()) + 1e-12)
        
        if total_norm_b > target_global_norm:
            scale = (target_global_norm / total_norm_b).to(torch.float32) # Scale is a scalar
            for p in projected_bfat_grads:
                projected_bfat_grads[p] = projected_bfat_grads[p] * scale

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

    # --- Hardware Consistency Setup ---
    # Disable TF32 to ensure consistent precision across Ampere/Hopper/Blackwell (5090) and Turing (2080Ti)
    # 5090 defaults to TF32 which has lower precision than FP32 used by 2080Ti
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Optional: Enable deterministic CuDNN if needed (may slow down training)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    logger_info(logger, f"[NUDE] Hardware Setup: TF32 Disabled for consistency. Device: {torch.cuda.get_device_name(0)}")
    # ----------------------------------

    model.train()
    if model_ema is not None:
        model_ema.ema.train()

    max_bit = _force_max_bitwidth(model, configs)
    logger_info(logger, f"[NUDE] Epoch {epoch}: force max bit-width W/A = {max_bit}")

    # --- [NUDE] BFAT Setup ---
    bfat_cfg = getattr(configs, 'bfat', None)
    bfat_start_epoch = getattr(bfat_cfg, 'start_epoch', 0)
    pbs_start_epoch = getattr(bfat_cfg, 'pbs_start_epoch', bfat_start_epoch)
    use_bfat = bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False) and epoch >= bfat_start_epoch
    use_pbs = bfat_cfg is not None and getattr(bfat_cfg, 'pbs_enabled', False) and epoch >= pbs_start_epoch
    bfat_freeze_bn = getattr(bfat_cfg, 'freeze_bn', False) if (use_bfat or use_pbs) else False
    bfat_hook = None
    if (use_bfat or use_pbs):
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
                logger_info(logger, f'⚠️ [NUDE] BFAT/PBS WARNING: Target layer {target_layer_name} not found for feature_sim!')
                use_bfat = False
                use_pbs = False

        # Config Logging
        logger_info(logger, '=' * 80)
        if use_bfat:
            logger_info(logger, f'🔥 [NUDE] BFAT ACTIVE')
        if use_pbs:
            logger_info(logger, f'🎯 [NUDE] PBS ACTIVE')
            logger_info(logger, f'   PBS Config: Flips={getattr(bfat_cfg, "pbs_flips", 1)}, Period={getattr(bfat_cfg, "pbs_period", 1)}, Start Epoch={pbs_start_epoch}')

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

        # --- [NEW] Optimized Base Pass Logic ---
        use_restricted_base = bfat_cfg is not None and getattr(bfat_cfg, 'use_restricted_as_base', False)
        proj_mode = getattr(bfat_cfg, 'projection_mode', 'direction') if (use_bfat or use_pbs) else "direction"
        
        if use_restricted_base and fault_injector is not None:
            # Step 1: Prepare Restricted State for "Base" pass (1F + 1B)
            fault_injector.enable()
            old_skip_msbn = getattr(fault_injector, 'skip_msbn', False)
            old_ber = fault_injector.ber
            fault_injector.skip_msbn = True
            fault_injector.ber = getattr(bfat_cfg, 'ber_base_restricted', 0.01)
            fault_injector.reset_forward_seed()
            
            # 1. "Restricted" forward pass (acting as base)
            outputs = model(inputs)
            
            # Cleanup injector for now
            fault_injector.disable()
            fault_injector.skip_msbn = old_skip_msbn
            fault_injector.ber = old_ber
        else:
            # 1. Clean forward pass (Standard)
            outputs = model(inputs)
        
        # Capture clean feature if needed (from whatever base we used)
        clean_feature = None
        if (use_bfat or use_pbs) and bfat_hook is not None:
            clean_feature = bfat_hook.feature.clone().detach()

        ce_loss = criterion(outputs, targets)
        srqat_loss = _compute_srqat_scale_penalty(outputs, targets, model, configs, epoch)
        loss = ce_loss + srqat_loss
        
        # 2. Base backward pass
        # 如果 proj_mode 为 "none"，则跳过此处的独立反传，后续与 BFAT/PBS loss 合并
        if not ((use_bfat or use_pbs) and proj_mode == "none"):
            loss.backward()

            # Update sim_res_loss meter if we used restricted base
            if use_restricted_base and 'sim_res_loss' in meters:
                meters['sim_res_loss'].update(loss.item(), inputs.size(0))

            # --- 捕获 base 梯度 ---
            clean_grads = {}
            projection_base_grads = None
            if proj_mode != "none" and (use_bfat or use_pbs):
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        clean_grads[p] = p.grad.clone()
                
                optimizer.zero_grad()
                if optimizer_q is not None:
                    optimizer_q.zero_grad()
            
            projection_base_grads = clean_grads # 默认投影基准

            # --- [NEW] PBS Adversarial Step ---
            pbs_grads = {}
            # 控制 PBS 频率：避免每个 batch 都做导致不收敛
            pbs_period = getattr(bfat_cfg, 'pbs_period', 1)
            if use_pbs and (batch_idx % pbs_period == 0):
                # 1. 使用 PBS 寻找当前 batch 下最敏感的比特 (极速模式)
                pbs_winners = _pbs_find_winner_bits(
                    model, 
                    n_flips=getattr(bfat_cfg, 'pbs_flips', 1)
                )
                
                if pbs_winners:
                    # 2. 准备 PBS 翻转 (像 fault_injector 一样动态包装)
                    winners_by_layer = {}
                    for w in pbs_winners:
                        if w['name'] not in winners_by_layer: winners_by_layer[w['name']] = []
                        winners_by_layer[w['name']].append(w)
                    
                    original_forwards = {}
                    try:
                        # 3. 临时包装 forward 方法
                        for name, module in model.named_modules():
                            if name in winners_by_layer:
                                original_forwards[name] = module.quan_w_fn.forward
                                module.quan_w_fn.forward = PBSForwardWrapper(
                                    original_forwards[name], winners_by_layer[name]
                                )
                        
                        # 4. 执行反向传播获取 pbs 对抗梯度 (1F + 1B)
                        _force_max_bitwidth(model, configs)
                        optimizer.zero_grad()
                        if optimizer_q is not None: optimizer_q.zero_grad()

                        outputs_pbs = model(inputs)
                        if bfat_loss_type == 'direct_ce':
                            loss_pbs = criterion(outputs_pbs, targets)
                        else:
                            f_c = clean_feature.view(clean_feature.size(0), -1)
                            f_f = bfat_hook.feature.view(bfat_hook.feature.size(0), -1)
                            sim_pbs = F.cosine_similarity(f_c, f_f, dim=1).mean()
                            loss_pbs = (1 - sim_pbs)
                        
                        loss_pbs.backward()

                        # 5. 捕获 PBS 梯度
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                pbs_grads[p] = p.grad.clone()
                                
                    finally:
                        # 6. 恢复原始 forward (无论是否报错)
                        for name, orig_fn in original_forwards.items():
                            for n, m in model.named_modules():
                                if n == name:
                                    m.quan_w_fn.forward = orig_fn
                                    break
                    
                    optimizer.zero_grad()
                    if optimizer_q is not None: optimizer_q.zero_grad()

            # --- [EXISTING] BFAT Logic ---
            bfat_grads = {}
            if use_bfat and fault_injector is not None:
                accumulate_grad_mode = getattr(bfat_cfg, 'simulate_normal_accumulation', False)
                
                # Backup injector state
                old_only_msb = fault_injector.only_msb
                old_skip_msb = fault_injector.skip_msb
                old_skip_msbn = getattr(fault_injector, 'skip_msbn', False)
                old_all_bits = getattr(fault_injector, 'all_bits', False)
                old_bfat_idx = getattr(fault_injector, 'bfat_bit_index', None)
                old_bfat_dual = getattr(fault_injector, 'bfat_dual_bit', False)
                old_ber_msb = getattr(fault_injector, 'ber_msb', None)
                old_ber_secondary = getattr(fault_injector, 'ber_secondary_msb', None)
                old_ber = fault_injector.ber
                old_seed_list = fault_injector.seed_list
                fault_injector.seed_list = None

                bfat_freeze_bn = getattr(bfat_cfg, 'freeze_bn', False)
                if bfat_freeze_bn:
                    for m in model.modules():
                        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)):
                            m.eval()

                if accumulate_grad_mode and not use_restricted_base:
                    # === Mode A: Refined Projection Base Mode ===
                    # [Optimization] Skip if use_restricted_as_base is True, 
                    # as restricted pass is already done.
                    fault_injector.stuck_at_0_random = False
                    fault_injector.bfat_dual_bit = False
                    fault_injector.only_msb = False
                    fault_injector.all_bits = False
                    fault_injector.skip_msb = False
                    fault_injector.skip_msbn = True
                    fault_injector.bfat_bit_index = None
                    fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)

                    skip_msbn_grads = {}
                    avg_skip_msbn_loss = 0.0
                    for i in range(1):
                        fault_injector.enable()
                        _force_max_bitwidth(model, configs)
                        current_iter_seed = num_updates + batch_idx * 10 + i
                        fault_injector.seed = current_iter_seed
                        fault_injector.reset_forward_seed()
                        outputs_skip = model(inputs)
                        loss_skip = criterion(outputs_skip, targets)
                        loss_skip.backward()
                        avg_skip_msbn_loss += loss_skip.item()
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                if p in skip_msbn_grads:
                                    skip_msbn_grads[p] += p.grad.clone()
                                else:
                                    skip_msbn_grads[p] = p.grad.clone()
                        optimizer.zero_grad()
                        if optimizer_q is not None: optimizer_q.zero_grad()
                            
                    avg_skip_msbn_loss /= 1.0
                    if 'sim_res_loss' in meters:
                        meters['sim_res_loss'].update(avg_skip_msbn_loss, inputs.size(0))
                        
                    sam_rho = getattr(bfat_cfg, 'sam_rho', 1.0)
                    refined_clean_grads = {}
                    for p in clean_grads.keys():
                        g_c = clean_grads[p]
                        g_s = skip_msbn_grads.get(p, None)
                        if g_s is not None:
                            refined_clean_grads[p] = g_c + (g_s - g_c) * sam_rho
                        else:
                            refined_clean_grads[p] = g_c
                            
                    projection_base_grads = refined_clean_grads
                    for p, g_s in skip_msbn_grads.items():
                         if p in clean_grads: clean_grads[p] += g_s 
                         else: clean_grads[p] = g_s
 
                    # Step 2: BFAT
                    fault_injector.bfat_dual_bit = getattr(bfat_cfg, 'dual_bit', False)
                    fault_injector.only_msb = getattr(bfat_cfg, 'only_msb', False)
                    fault_injector.all_bits = getattr(bfat_cfg, 'all_bits', False)
                    fault_injector.skip_msb = getattr(bfat_cfg, 'skip_msb', False)
                    fault_injector.skip_msbn = getattr(bfat_cfg, 'skip_msbn', False)
                    fault_injector.bfat_bit_index = getattr(bfat_cfg, 'bit_index', None)
                    fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)

                    fault_injector.enable()
                    _force_max_bitwidth(model, configs)
                    current_msb_seed = num_updates + batch_idx * 10 + 99
                    fault_injector.seed = current_msb_seed
                    fault_injector.reset_forward_seed()
                    outputs_bfat = model(inputs)
                    
                    if bfat_loss_type == 'direct_ce':
                        loss_bfat = criterion(outputs_bfat, targets)
                    else:
                        f_c = clean_feature.view(clean_feature.size(0), -1)
                        f_f = bfat_hook.feature.view(bfat_hook.feature.size(0), -1)
                        sim = F.cosine_similarity(f_c, f_f, dim=1).mean()
                        loss_bfat = (1 - sim)
                    
                    if 'sim_bfat_loss' in meters:
                        meters['sim_bfat_loss'].update(loss_bfat.item(), inputs.size(0))
                    
                    loss_bfat.backward()
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            bfat_grads[p] = p.grad.clone()
                            
                else:
                    # === Mode B: Standard Logic ===
                    projection_base_grads = clean_grads
                    fault_injector.bfat_dual_bit = getattr(bfat_cfg, 'dual_bit', False)
                    fault_injector.only_msb = getattr(bfat_cfg, 'only_msb', False)
                    fault_injector.skip_msb = getattr(bfat_cfg, 'skip_msb', False)
                    fault_injector.skip_msbn = getattr(bfat_cfg, 'skip_msbn', False)
                    fault_injector.all_bits = getattr(bfat_cfg, 'all_bits', False)
                    fault_injector.bfat_bit_index = getattr(bfat_cfg, 'bit_index', None)
                    
                    if fault_injector.bfat_dual_bit:
                        fault_injector.ber_msb = getattr(bfat_cfg, 'ber_msb', 0.01)
                        fault_injector.ber_secondary_msb = getattr(bfat_cfg, 'ber_secondary_msb', 0.01)
                    else:
                        fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)

                    fault_injector.enable()
                    _force_max_bitwidth(model, configs)
                    fault_injector.reset_forward_seed()
                    outputs_bfat = model(inputs)
                    
                    bfat_loss_type = getattr(bfat_cfg, 'loss_type', 'feature_sim')
                    if bfat_loss_type == 'direct_ce':
                        loss_bfat = criterion(outputs_bfat, targets)
                    else:
                        f_c = clean_feature.view(clean_feature.size(0), -1)
                        f_f = bfat_hook.feature.view(bfat_hook.feature.size(0), -1)
                        sim = F.cosine_similarity(f_c, f_f, dim=1).mean()
                        loss_bfat = (1 - sim)

                    if proj_mode == "none":
                        (loss + loss_bfat).backward()
                    else:
                        loss_bfat.backward()

                    if proj_mode != "none":
                        for p in model.parameters():
                            if p.requires_grad and p.grad is not None:
                                bfat_grads[p] = p.grad.clone()

            # Restore injector state
            if fault_injector is not None and use_bfat:
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

            _force_max_bitwidth(model, configs)
            if bfat_freeze_bn:
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, SwithableBatchNorm)): m.train()

            # 4. Final Unified Projection and Merge (Optimized for convergence)
            # Strategy: Combine adversarial gradients first, then project once to clean space.
            if proj_mode != "none" and (use_bfat or use_pbs):
                limit_norm = getattr(bfat_cfg, 'limit_norm', False)
                norm_ratio = getattr(bfat_cfg, 'norm_ratio', 0.5)
                weight_rel_limit = getattr(bfat_cfg, 'weight_relative_limit', False)
                weight_limit_ratio = getattr(bfat_cfg, 'weight_limit_ratio', 0.01)
                
                # --- Step 1: Combine adversarial sources by simple addition ---
                # We stop manual 0.5 scaling here to prevent "gradient jitter" between batches.
                # The total magnitude will be handled by unified projection and limit_norm.
                combined_adv_raw = {}
                all_adv_params = set(bfat_grads.keys()) | set(pbs_grads.keys())
                
                for p in all_adv_params:
                    g_b = bfat_grads.get(p, torch.zeros_like(p.data))
                    g_p = pbs_grads.get(p, torch.zeros_like(p.data))
                    combined_adv_raw[p] = g_b + g_p
                
                # --- Step 2: Unified Projection and Global Norm Limiting ---
                # Strategy: Treat total defense as a single budget. This ensures stability
                # even when PBS is only active occasionally.
                projected_adv_unified = project_bfat_gradients(
                    projection_base_grads, combined_adv_raw, 
                    limit_norm=limit_norm, 
                    norm_ratio=norm_ratio, 
                    projection_mode=proj_mode,
                    weight_relative_limit=weight_rel_limit,
                    weight_limit_ratio=weight_limit_ratio
                )

                # --- Step 3: Final Gradient Synthesis ---
                for p in model.parameters():
                    if p.requires_grad:
                        g_c = clean_grads.get(p, None)
                        g_a_proj = projected_adv_unified.get(p, None)
                        
                        g_final = g_c if g_c is not None else torch.zeros_like(p.data)
                        if g_a_proj is not None:
                            g_final = g_final + g_a_proj
                        
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

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            _force_max_bitwidth(model, configs)
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

    _update_monitors(monitors, meters, epoch, steps - 1, steps, optimizer=None, mode="validation")
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


