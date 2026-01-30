"""
Optimized Mixed-Precision QAT Training

Streamlined training without Gradient Surgery for maximum speed.
Key optimizations:
- Removed gradient projection (standard gradient accumulation)
- Vectorized SwitchableBatchNorm operations
- Cached layer references to avoid repeated module traversal
- Eliminated CPU-GPU synchronization in sampling
"""

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


def compute_overall_loss(outputs, teacher_outputs, targets, criterion, model, 
                         quantization_error_minimization=False, QE_loss_weight=.5, 
                         disable_smallest_regularization=True, configs=None):
    task_loss = loss_forward(outputs, teacher_outputs, targets, criterion)

    if quantization_error_minimization or disable_smallest_regularization:
        QE_loss, distribution_loss = auxiliary_quantized_loss(
            model, 
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
        logger.info('==> %s Top1: %.3f    Top5: %.3f    Loss: %.3f', 
                    meters[i]['name'], meters[i]['top1'].avg, meters[i]['top5'].avg, meters[i]['loss'].avg)


@master_only
def update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, 
                    nr_random_sample, optimizer, optimizer_q, mode='training'):
    iters = len(meters) if mode == 'training' else 1
    for m in monitors:
        for i in range(iters):
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


def evaluate_layer_sensitivity(model, loader, criterion, fault_injector, configs, epoch):
    """
    Evaluate sensitivity of each quantized layer to bit-flips.
    Returns: List of layer indices sorted by sensitivity (descending).
    """
    from util.mpq import get_cached_layers, sample_max_cands
    
    # 1. Setup diversity seed
    eval_seed = (epoch * 1000 + 42) % (2**31)
    fault_injector.seed = eval_seed
    fault_injector.ber = 4e-3 # Targeted BER for evaluation
    
    # 2. Get a single batch of data
    try:
        # Use a fresh iterator to avoid disturbing main loader state too much, 
        # though standard training resets loader every epoch anyway.
        data_iter = iter(loader)
        inputs, targets = next(data_iter)
    except StopIteration:
        return []
        
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    
    # Cache and set state
    was_training = model.training
    model.eval()
    
    # 3. Set bits to max for baseline
    sample_max_cands(model, configs)
    
    with torch.no_grad():
        # Baseline loss (no injection)
        baseline_outputs = model(inputs)
        baseline_loss = criterion(baseline_outputs, targets).item()
        
        cache = get_cached_layers(model, configs)
        quan_layers = cache['quan_layers']
        
        sensitivities = []
        
        logger_info(logger, f"🔍 [Sensitivity Analysis] Epoch {epoch} | Seed: {eval_seed} | BER: {fault_injector.ber}")
        
        for i, (module, name, layer_type) in enumerate(quan_layers):
            # Target ONLY this layer
            fault_injector.whitelist_layer = name
            fault_injector.enable()
            
            faulty_outputs = model(inputs)
            faulty_loss = criterion(faulty_outputs, targets).item()
            
            # Sensitivity is the loss increase caused by faults
            delta_loss = max(0.0, faulty_loss - baseline_loss)
            sensitivities.append((i, delta_loss))
            
            fault_injector.disable()
            fault_injector.whitelist_layer = None
            
    # Sort by sensitivity descending
    sensitivities.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in sensitivities]
    
    # Log top sensitive layers
    top_5_names = [quan_layers[idx][1] for idx in sorted_indices[:5]]
    logger_info(logger, f"🏆 Top 5 Sensitive: {', '.join(top_5_names)}")
    
    # Restore state
    if was_training:
        model.train()
        
    return sorted_indices


def loss_forward(outputs, teacher_outputs, targets, criterion):
    loss = criterion(outputs, targets)
    if teacher_outputs is not None:
        loss = 1/2 * loss + 1/2 * F.kl_div(
            F.log_softmax(outputs, dim=-1), 
            F.softmax(teacher_outputs, dim=-1), 
            reduction='batchmean'
        )
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


def project_bfat_gradients(clean_grads, bfat_grads, limit_norm=False, norm_ratio=0.5, projection_mode="direction",
                           weight_relative_limit=False, weight_limit_ratio=0.01):
    """
    Simpler BFAT projection logic for GS optimized training.
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
            elif projection_mode == "cagrad": # Support cagrad if needed, though mostly direction/orthogonal
                norm_sq_b = torch.sum(g_b_d * g_b_d) + 1e-8
                numerator = norm_sq_b - dot_product
                denominator = norm_sq_c + norm_sq_b - 2 * dot_product + 1e-8
                alpha = torch.clamp(numerator / denominator, 0.0, 1.0)
                g_target_d = alpha * g_c_d + (1.0 - alpha) * g_b_d
                g_b_cleaned_d = g_target_d - g_c_d
            else: # direction
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


def train(train_loader, model, criterion, optimizer, epoch, monitors, configs, 
          model_ema=None, nr_random_sample=2, mode='training', soft_criterion=None, 
          teacher_model=None, optimizer_q=None, annealing_schedule=None, 
          freezing_annealing_schedule=None, IDM_weight=0.01, scaler=None,
          fault_injector=None):
    """
    Optimized training loop for Mixed-Precision QAT with Sensitivity-Aware Sampling.
    """
    assert mode in ['finetuning', 'training']
    
    # 1. Handle Sensitivity-Aware Configuration
    warmup_epochs = getattr(configs, 'warmup_epochs', 5)
    sensitive_indices = None
    
    if mode == 'training' and epoch >= warmup_epochs and fault_injector is not None:
        # Perform Sensitivity Analysis at start of epoch
        sensitive_indices = evaluate_layer_sensitivity(
            model, train_loader, criterion, fault_injector, configs, epoch
        )
        
        # Calculate sensitive ratio using Cosine Annealing
        # Starts at 0.5 (conservative) and decays to 0.2 (robust)
        r_start, r_end = 0.5, 0.2
        total_epochs = configs.epochs
        if total_epochs > warmup_epochs:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            ratio = r_end + 0.5 * (r_start - r_end) * (1 + math.cos(math.pi * progress))
        else:
            ratio = r_start
            
        unwrapped_layers_count = len(sensitive_indices) if sensitive_indices else 0
        num_sensitive = max(1, int(unwrapped_layers_count * ratio))
        sensitive_indices = sensitive_indices[:num_sensitive]
        logger_info(logger, f"📊 Adaptive Policy [Epoch {epoch}]: ratio={ratio:.3f}, sensitive_layers={num_sensitive}/{unwrapped_layers_count}")

    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    sample_current_max = True
    sample_current_min = getattr(configs, 'sandwich_training', False)
    
    print("Bit-width candidates:", target_bits)
    
    meters, num_fixed_sample = get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min)

    total_sample = len(train_loader.sampler)
    batch_size = configs.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    information_distortion_mitigation = getattr(configs, 'information_distortion_mitigation', False)
    if information_distortion_mitigation:
        assert sample_current_max

    logger_info(logger, 'Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    gno_alpha = getattr(configs, 'gno_alpha', 0.0)
    if gno_alpha > 0:
        logger_info(logger, f"🛡️ GNO (Gradient-Noise Orthogonality) Enabled: alpha={gno_alpha}")
    else:
         logger_info(logger, f"🛡️ GNO Disabled")
    
    num_updates = epoch * len(train_loader)
    seed = num_updates
    set_global_seed(seed + 1)
    model.train()
    if model_ema:
        model_ema.ema.train()

    T = 2 if epoch <= int(configs.epochs * 0.72) else 15

    if configs.enable_dynamic_bit_training and epoch > 5 and (epoch + 1) % T == 0:
        freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
        freezing_metric = profile_layerwise_quantization_metric(model=model)
        freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio, 
                      progressive=False, logger=logger, org_cands=configs.target_bits)
        logger_info(logger=logger, msg=f'Current freezing ratio: {freezing_ratio}')

    if teacher_model is not None:
        teacher_model.eval()
        print("Training with KD...")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        external_teacher_outputs = None
        if teacher_model is not None and soft_criterion is not None:
            with torch.no_grad():
                external_teacher_outputs = teacher_model(inputs)

        QE_loss_weight = annealing_schedule(num_updates)

        # ============================================================
        # STEP 1: Max-bit subnet forward & backward
        # ============================================================
        teacher_outputs = None
        
        if sample_current_max:
            start_time = time.time()
            sample_max_cands(model, configs)

            if information_distortion_mitigation:
                target_features = []
                hooks = set_forward_hook_for_quantized_layers(model, target_features, is_max=True)

            max_outputs = model(inputs)
            loss, QE_loss, dist_loss = compute_overall_loss(
                max_outputs, external_teacher_outputs, targets, criterion, model, 
                quantization_error_minimization=False, configs=configs, 
                disable_smallest_regularization=True
            )
            loss.backward()  # Gradients accumulate

            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

            teacher_outputs = max_outputs.clone().detach()
            
            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[0], loss, QE_loss, dist_loss, 0, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        # ============================================================
        # STEP 2: Subnet sampling (standard gradient accumulation)
        # ============================================================
        for iter_idx in range(nr_random_sample):
            start_time = time.time()

            w_conf, a_conf, min_w_index = sample_one_mixed_policy(
                model, configs, sensitive_indices=sensitive_indices
            )
            
            if information_distortion_mitigation:
                distorted_features = []
                hooks = set_forward_hook_for_quantized_layers(model, distorted_features, is_max=False)

            outputs = model(inputs)
            loss, QE_loss, dist_loss = compute_overall_loss(
                outputs, teacher_outputs, targets, criterion, model, 
                quantization_error_minimization=epoch > 40, 
                QE_loss_weight=QE_loss_weight, 
                disable_smallest_regularization=True, 
                configs=configs
            )

            IDM_loss = 0
            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)
                IDM_loss = sum([F.mse_loss(s, t).sum() if s is not None else 0 
                               for s, t in zip(distorted_features, target_features)])
                loss += (IDM_loss * IDM_weight)
            
            loss.backward()  # Gradients accumulate directly (no projection)
            
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[iter_idx + num_fixed_sample], loss, QE_loss, dist_loss, IDM_loss, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)


        # ============================================================
        # [BFAT] Post-Accumulation Injection (GS Framework)
        # ============================================================
        bfat_cfg = getattr(configs, 'bfat', None)
        use_bfat = bfat_cfg is not None and getattr(bfat_cfg, 'enabled', False)
        if use_bfat and epoch < getattr(bfat_cfg, 'start_epoch', 0):
            use_bfat = False
            
        if use_bfat and fault_injector is not None:
             # 1. Stash accumulated "Clean" gradients (Max + All Random)
            clean_grads = {}
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    clean_grads[p] = p.grad.clone()
            
            # 2. Clear gradients for independent Faulty pass
            optimizer.zero_grad()
            if optimizer_q is not None:
                optimizer_q.zero_grad()
                
            # 3. Setup Fault Injector (All Bits, Single Pass)
            # Use current model state (last random subnet) ensuring we attack a valid subnet structure
            old_state = {
                'all_bits': getattr(fault_injector, 'all_bits', False),
                'only_msb': fault_injector.only_msb, # Ensure we don't accidentally stay in only_msb
                'ber': fault_injector.ber,
                'seed': fault_injector.seed
            }
            
            fault_injector.all_bits = getattr(bfat_cfg, 'all_bits', True) # Default to True for GS
            fault_injector.only_msb = False
            fault_injector.ber = getattr(bfat_cfg, 'ber', 0.01)
            # Unique seed for this injection step (Aligned with process_nude.py)
            rank_offset = getattr(configs, 'rank', 0) * 100000
            # Note: num_updates in process_gs increments per batch, but in process_nude it's epoch_start.
            # To match process_nude's formula: seed = epoch_start + batch_idx * 10 + 7
            # We reconstruct epoch_start or just use the formula explicitly.
            epoch_start_updates = epoch * len(train_loader)
            fault_injector.seed = epoch_start_updates + batch_idx * 10 + 7 + rank_offset 
            
            fault_injector.enable()
            fault_injector.reset_forward_seed()
            
            # 4. Faulty Forward & Backward
            # Re-run forward on the LAST sampled subnet configuration (which is still active in model)
            # If information_distortion_mitigation was on, hooks are already removed, so safe to run.
            # Note: We use the same inputs/targets
            outputs_bfat = model(inputs)
            
            # Loss: Standard CE Loss (as requested)
            loss_bfat = criterion(outputs_bfat, targets) * getattr(bfat_cfg, 'loss_weight', 1.0)
            loss_bfat.backward()
            
            # 5. Capture Faulty Gradients
            bfat_grads = {}
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    bfat_grads[p] = p.grad.clone()
            
            # 6. Restore Injector State
            fault_injector.disable()
            fault_injector.all_bits = old_state['all_bits']
            fault_injector.only_msb = old_state['only_msb']
            fault_injector.ber = old_state['ber']
            fault_injector.seed = old_state['seed']
            
            # 7. Projection & Update
            proj_mode = getattr(bfat_cfg, 'projection_mode', 'direction')
            projected_bfat = project_bfat_gradients(
                clean_grads, bfat_grads, 
                limit_norm=getattr(bfat_cfg, 'limit_norm', False),
                norm_ratio=getattr(bfat_cfg, 'norm_ratio', 0.5),
                projection_mode=proj_mode,
                weight_relative_limit=getattr(bfat_cfg, 'weight_relative_limit', False),
                weight_limit_ratio=getattr(bfat_cfg, 'weight_limit_ratio', 0.01)
            )
            
            # 8. Final Accumulation: p.grad = Clean (stashed) + Robust (projected)
            # Note: If mode is 'none', projected_bfat is just bfat_grads, so we add raw faulty gradient
            for p in model.parameters():
                if p.requires_grad:
                    # Restore clean first because we zeroed it
                    g_clean = clean_grads.get(p, None)
                    g_rob = projected_bfat.get(p, None)
                    
                    # Accumulate: grad = clean + robust
                    # (Robust is already "processed faulty" - usually orthogonal component)
                    g_final = None
                    if g_clean is not None:
                         g_final = g_clean
                    
                    if g_rob is not None:
                        if g_final is None:
                            g_final = g_rob
                        else:
                            g_final += g_rob
                            
                    if g_final is not None:
                        p.grad = g_final


        # ============================================================
        # STEP 3: Single optimizer step (all gradients accumulated)
        # ============================================================
        # OPTIMIZATION: Vectorized gradient clipping using foreach
        params_to_clip = [p.grad for p in model.parameters() if p.grad is not None]
        if params_to_clip:
            torch._foreach_clamp_min_(params_to_clip, -1.0)
            torch._foreach_clamp_max_(params_to_clip, 1.0)

        # [GNO] Gradient-Noise Orthogonality Projector
        gno_alpha = getattr(configs, 'gno_alpha', 0.0)
        if gno_alpha > 0:
            for m in model.modules():
                # Check directly for attributes to avoid imports/instance checks
                if hasattr(m, 'q_error') and m.q_error is not None and hasattr(m, 'weight') and m.weight.grad is not None:
                    g = m.weight.grad
                    e = m.q_error
                    
                    # Project gradient: g_new = g - alpha * proj_e(g)
                    # proj_e(g) = (g . e) / (e . e) * e
                    e_norm = torch.sum(e * e)
                    if e_norm > 1e-9:
                        dot = torch.sum(g * e)
                        proj = (dot / e_norm) * e
                        g.data.sub_(gno_alpha * proj)
                    
                    # Clear to save memory
                    m.q_error = None

        optimizer.step()
        if optimizer_q is not None:
            optimizer_q.step()

        num_updates += 1

        if model_ema is not None:
            model_ema.update(model)
        
        if (batch_idx + 1) % configs.log.print_freq == 0:
            update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, 
                           nr_random_sample, optimizer, optimizer_q, mode=mode)
            logger_info(logger, "=" * 115)

    show_training_info(meters, target_bits, nr_random_sample, mode=mode)
    
    return meters[0]['top1'].avg, meters[0]['top5'].avg, meters[0]['loss'].avg


def validate(data_loader, model, criterion, epoch, monitors, configs, 
             nr_random_sample=3, alpha=1, train_loader=None, eval_predefined_arch=None, 
             bops_limit=1e10, train_mode=False):
    """Validation function"""
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
            update_meter(meter, loss, None, None, None, acc1, acc5, 
                        inputs.size(0), time.time() - start_time, configs.world_size)
    
    if train_mode:
        logger_info(logger, msg='Using training mode...')
        model.train()

    if eval_predefined_arch is None:
        from policy import MIN_POLICY
        eval_predefined_arch = [MIN_POLICY]
    
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
                calibrate_batchnorm_state(
                    model, loader=train_loader, reset=True, 
                    distributed_training=True, 
                    num_batch=7000 // torch.distributed.get_world_size() // configs.dataloader.batch_size
                )
            
            _eval(data_loader, meters[idx])
            bops, size = model_profiling(model=model, return_layers=False)

            logger_info(logger, msg=f"Arch {idx}, BitOPs {round(bops, 2)} G, Size {round(size, 2)} MB, Top-1 Acc. {round(meters[idx]['top1'].avg, 2)}")
    
    return [meters[idx]['top1'].avg for idx in range(len(eval_predefined_arch))]


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board, key=operator.itemgetter('top1', 'top5', 'epoch'), 
                           reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
