"""
Gradient Surgery for Mixed-Precision QAT Training

This module implements gradient projection methods to resolve gradient conflicts
between different bit-width subnets during mixed-precision quantization training.

Key Innovation: Instead of projecting fault-injection gradients (like BFAT),
we project subnet gradients against max-bit gradient to reduce gradient conflicts.
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


def project_subnet_gradients(base_grads, subnet_grads, projection_mode="direction", limit_norm=False, norm_ratio=0.5):
    """
    Project subnet gradients to reduce conflicts with base (max-bit) gradients.
    
    Args:
        base_grads: dict {param: gradient} from max-bit subnet (the "teacher" direction)
        subnet_grads: dict {param: gradient} from lower-bit subnet
        projection_mode: 
            - "direction": Only project if angle > 90 deg (remove conflicting component)
            - "orthogonal": Always project, forcing strict orthogonality
            - "none": No projection, return original gradients
        limit_norm: Whether to limit the magnitude of projected gradients
        norm_ratio: If limit_norm, projected grad magnitude <= base grad magnitude * ratio
    
    Returns:
        dict {param: projected_gradient}
    """
    projected_grads = {}
    
    if projection_mode == "none":
        for p, g_sub in subnet_grads.items():
            projected_grads[p] = g_sub.clone()
        return projected_grads
    
    for p, g_sub in subnet_grads.items():
        if p in base_grads:
            g_base = base_grads[p]
            # Use float64 for numerical stability
            g_base_d = g_base.to(torch.float64)
            g_sub_d = g_sub.to(torch.float64)
            
            # Compute projection of g_sub onto g_base
            dot_product = torch.sum(g_base_d * g_sub_d)
            norm_sq_base = torch.sum(g_base_d * g_base_d) + 1e-8
            projection = (dot_product / norm_sq_base) * g_base_d
            
            if projection_mode == "orthogonal":
                # Always remove the component along g_base direction
                g_projected = g_sub_d - projection
            else:  # "direction" mode
                # Only project if gradient conflict exists (dot product < 0)
                if dot_product < 0:
                    g_projected = g_sub_d - projection  # Remove conflicting component
                else:
                    g_projected = g_sub_d  # Keep original if aligned
            
            projected_grads[p] = g_projected.to(dtype=g_sub.dtype)
        else:
            projected_grads[p] = g_sub
    
    # Magnitude limiting (optional)
    if limit_norm:
        total_norm_base = torch.sqrt(sum(torch.sum(g.to(torch.float64)**2) for g in base_grads.values()) + 1e-8)
        target_norm = total_norm_base * norm_ratio
        total_norm_proj = torch.sqrt(sum(torch.sum(g.to(torch.float64)**2) for g in projected_grads.values()) + 1e-8)
        
        if total_norm_proj > target_norm:
            scale = target_norm / total_norm_proj
            for p in projected_grads:
                projected_grads[p] = (projected_grads[p].to(torch.float64) * scale).to(projected_grads[p].dtype)
    
    return projected_grads


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


def train(train_loader, model, criterion, optimizer, epoch, monitors, configs, 
          model_ema=None, nr_random_sample=2, mode='training', soft_criterion=None, 
          teacher_model=None, optimizer_q=None, annealing_schedule=None, 
          freezing_annealing_schedule=None, IDM_weight=0.01, scaler=None):
    """
    Training with Gradient Surgery for Mixed-Precision QAT.
    
    Key difference from standard training:
    - Max-bit subnet gradient is captured first
    - Subnet gradients are projected against max-bit gradient to remove conflicts
    - Final gradient is accumulated from max-bit + projected subnets
    """
    assert mode in ['finetuning', 'training']

    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    sample_current_max = True
    sample_current_min = getattr(configs, 'sandwich_training', False)
    
    # Gradient surgery config
    gs_config = getattr(configs, 'gradient_surgery', None)
    gs_enabled = gs_config is not None and getattr(gs_config, 'enabled', False)
    gs_mode = getattr(gs_config, 'projection_mode', 'direction') if gs_config else 'direction'
    gs_limit_norm = getattr(gs_config, 'limit_norm', False) if gs_config else False
    gs_norm_ratio = getattr(gs_config, 'norm_ratio', 0.5) if gs_config else 0.5
    
    if gs_enabled:
        logger_info(logger, f'[GS-MPQ] Gradient Surgery ENABLED | Mode: {gs_mode} | Limit Norm: {gs_limit_norm}')
    
    print("Bit-width candidates:", target_bits)
    
    meters, num_fixed_sample = get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min)

    total_sample = len(train_loader.sampler)
    batch_size = configs.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    information_distortion_mitigation = getattr(configs, 'information_distortion_mitigation', False)
    if information_distortion_mitigation:
        assert sample_current_max

    logger_info(logger, 'Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    
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
        # STEP 1: Max-bit subnet forward & backward → capture g_max
        # ============================================================
        max_grads = {}
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
            loss.backward()

            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

            # Capture max-bit gradients for projection
            if gs_enabled:
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        max_grads[p] = p.grad.clone()
                optimizer.zero_grad()
                if optimizer_q is not None:
                    optimizer_q.zero_grad()

            teacher_outputs = max_outputs.clone().detach()
            
            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[0], loss, QE_loss, dist_loss, 0, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        # ============================================================
        # STEP 2: Subnet sampling with gradient surgery
        # ============================================================
        accumulated_subnet_grads = {}  # Accumulated projected gradients from subnets

        for iter_idx in range(nr_random_sample):
            start_time = time.time()

            w_conf, a_conf, min_w_index = sample_one_mixed_policy(model, configs)
            
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
            
            loss.backward()
            
            # Gradient Surgery: project subnet gradients against max-bit gradients
            if gs_enabled and max_grads:
                subnet_grads = {}
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        subnet_grads[p] = p.grad.clone()
                
                # Project subnet gradients
                projected_grads = project_subnet_gradients(
                    max_grads, subnet_grads, 
                    projection_mode=gs_mode,
                    limit_norm=gs_limit_norm,
                    norm_ratio=gs_norm_ratio
                )
                
                # Accumulate projected gradients
                for p, g_proj in projected_grads.items():
                    if p in accumulated_subnet_grads:
                        accumulated_subnet_grads[p] = accumulated_subnet_grads[p] + g_proj
                    else:
                        accumulated_subnet_grads[p] = g_proj
                
                optimizer.zero_grad()
                if optimizer_q is not None:
                    optimizer_q.zero_grad()
            
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[iter_idx + num_fixed_sample], loss, QE_loss, dist_loss, IDM_loss, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        # ============================================================
        # STEP 3: Apply final gradients (max + projected subnets)
        # ============================================================
        if gs_enabled:
            for p in model.parameters():
                if p.requires_grad:
                    g_max = max_grads.get(p, None)
                    g_sub = accumulated_subnet_grads.get(p, None)
                    
                    if g_max is not None or g_sub is not None:
                        p.grad = (g_max if g_max is not None else 0) + \
                                 (g_sub if g_sub is not None else 0)

        nn.utils.clip_grad_value_(model.parameters(), 1.0)

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
    """Validation function (same as process1.py)"""
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
