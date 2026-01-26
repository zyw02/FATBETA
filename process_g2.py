import logging
import time
import torch
import torch.nn as nn
import random
from quan.func import QuanConv2d, QuanLinear, SwithableBatchNorm
from util import AverageMeter
from util.utils import accuracy, update_meter, set_global_seed
from util.qat import auxiliary_quantized_loss
from util.mpq import switch_bit_width, sample_max_cands, unwrap_model
from util.dist import logger_info, master_only

__all__ = ["train", "validate", "PerformanceScoreboard"]

logger = logging.getLogger()

@master_only
def update_monitors(monitors, meters, configs, epoch, batch_idx, steps_per_epoch, optimizer, optimizer_q, mode='training'):
    iters = len(meters)
    for m in monitors:
        for i in range(iters):
            p = meters[i]['name'] + ' '
            m.update(epoch, batch_idx + 1, steps_per_epoch, p + 'Training', {
                'Loss': meters[i]['loss'],
                'QE Loss': meters[i]['QE_loss'], 
                'Distribution Loss': meters[i]['dist_loss'], 
                'Top1': meters[i]['top1'],
                'Top5': meters[i]['top5'],
                'LR': optimizer.param_groups[0]['lr'],
                'QLR': optimizer_q.param_groups[0]['lr'] if optimizer_q is not None else 0
            })

@master_only
def show_training_info(meters):
    for i in range(len(meters)):
        logger.info('==> %s Top1: %.3f    Top5: %.3f    Loss: %.3f   QE_Loss: %.3f   Dist_Loss: %.3f', 
                    meters[i]['name'], meters[i]['top1'].avg, meters[i]['top5'].avg, 
                    meters[i]['loss'].avg, meters[i]['QE_loss'].avg, meters[i]['dist_loss'].avg)


def _get_meters(mode: str = "training"):
    meters = []
    # Helper to create full meter dict
    def create_full_meter(name):
        return {
            "name": name,
            "loss": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
            "QE_loss": AverageMeter(),
            "dist_loss": AverageMeter(),
            "IDM_loss": AverageMeter(),
            "batch_time": AverageMeter(),
        }

    # Meter for Max Bit
    meters.append(create_full_meter("Max"))
    
    # Meter for other candidates (averaged)
    meters.append(create_full_meter("Subnets"))
    return meters

def compute_overall_loss(model, task_loss, configs, epoch, QE_loss_weight=0.1):
    """
    Sync with process_normal.py logic: compute QE_loss and distribution_loss
    """
    quantization_error_minimization = (epoch > 40)
    
    # Compute auxiliary loss
    QE_loss, distribution_loss = auxiliary_quantized_loss(
        model, 
        quantization_error_minimization=quantization_error_minimization, 
        fairness_regularization=True
    )
    
    # Weight processing
    QE_loss = QE_loss * QE_loss_weight
    
    # Distribution loss weight logic
    adaptive_region_weight_decay = getattr(configs, 'adaptive_region_weight_decay', configs.weight_decay)
    dist_weight = (adaptive_region_weight_decay - configs.weight_decay)
    distribution_loss = distribution_loss * dist_weight
    
    return task_loss + QE_loss + distribution_loss, QE_loss, distribution_loss

def train(train_loader, model, criterion, optimizer, epoch, monitors, configs, 
          similarity_results=None, nr_random_sample=2, optimizer_q=None, 
          annealing_schedule=None):
    
    model.train()
    num_updates = epoch * len(train_loader)
    steps_per_epoch = len(train_loader)
    set_global_seed(num_updates + 1)
    
    meters = _get_meters()
    
    target_bits = sorted(configs.target_bits, reverse=True)
    quan_scheduler = configs.quan
    max_bit = target_bits[0]
    
    # --- Stratified Gradient Sampling Setup ---
    # 1. Define Buckets
    candidate_pool_static = [max_bit] # Used for Warmup
    buckets = []
    subnets = target_bits[1:]
    if len(subnets) > 0:
        mid_idx = len(subnets) // 2
        bucket_high = subnets[:mid_idx] 
        bucket_low = subnets[mid_idx:] if mid_idx < len(subnets) else []
        if bucket_high: buckets.append(("High", bucket_high))
        if bucket_low: buckets.append(("Low", bucket_low))
    
    # 2. Check Warmup
    warmup_epochs = getattr(configs, 'warmup_epochs', 0)
    in_warmup = (epoch < warmup_epochs)
    if in_warmup:
        logger_info(logger, f"G-PIPELINE: Epoch {epoch} (Warmup) - Static Pool: {candidate_pool_static}")

    # 3. Pre-compute Sampling Probabilities (Outside Loop)
    bucket_samplers = [] # List of (name, list, probs)
    if not in_warmup and similarity_results:
         # Compute Avg Similarity
         avg_similarity = {}
         for b in subnets:
             sim_list = similarity_results.get(b, [])
             if isinstance(sim_list, list):
                 valid_sims = [s for s in sim_list if s is not None]
                 avg_sim = sum(valid_sims) / len(valid_sims) if valid_sims else -1.0
             else:
                 avg_sim = float(sim_list)
             avg_similarity[b] = avg_sim

         # Compute Softmax Probs
         import numpy as np
         T = 1.0 # Temperature coefficient (Higher = More Random)
         
         for b_name, b_list in buckets:
             sims = np.array([avg_similarity.get(b, -1.0) for b in b_list])
             
             # Apply Softmax directly
             exp_sims = np.exp(sims / T)
             probs = exp_sims / np.sum(exp_sims)
             
             bucket_samplers.append((b_name, b_list, probs))
             
             # Log once per epoch setup
             sim_str = ", ".join([f"{b}:{avg_similarity.get(b,0):.3f}" for b in b_list])
             prob_str = ", ".join([f"{p:.2f}" for p in probs])
             logger_info(logger, f"  [{b_name} Bucket Config] Sims:[{sim_str}] -> Probs:[{prob_str}]")

    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        
        # --- Generate Dynamic Candidate Pool (Inside Loop) ---
        candidate_pool = [max_bit]
        
        if in_warmup:
            pass # Only max_bit
        elif bucket_samplers:
            # Sample based on pre-computed probs
            import numpy as np
            for b_name, b_list, probs in bucket_samplers:
                cand = np.random.choice(b_list, p=probs)
                candidate_pool.append(int(cand))
        elif not in_warmup:
             # Fallback Uniform if no similarity results
             for _, b_list in buckets:
                 if b_list:
                     candidate_pool.append(random.choice(b_list))
        
        # Log sampled pool occasionally
        if (batch_idx + 1) % configs.log.print_freq == 0:
             logger_info(logger, f"    Batch {batch_idx}: Sampled Architecture {candidate_pool}")
        
        # Get current QE loss annealing weight
        qe_w = annealing_schedule(num_updates) if annealing_schedule else 0.1
        
        optimizer.zero_grad()
        if optimizer_q: optimizer_q.zero_grad()
        
        # Iterate over global pool
        for pool_idx, bit in enumerate(candidate_pool):
            is_max = (bit == max_bit)
            
            # Switch architecture
            if is_max:
                sample_max_cands(model, configs)
            else:
                switch_bit_width(model, quan_scheduler=quan_scheduler, wbit=bit, abits=bit)
                
            # Forward
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)
            
            # Auxiliary Loss
            total_loss, qe_loss, dist_loss = compute_overall_loss(model, ce_loss, configs, epoch, QE_loss_weight=qe_w)
            
            # Backward (Accumulate gradients)
            total_loss.backward()
            
            # Update Meters
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            
            meter_idx = 0 if is_max else 1
            # If we only have Max bit in pool, meter_idx 1 is unused, which is fine.
            
            update_meter(meters[meter_idx], total_loss, qe_loss, dist_loss, 0, acc1, acc5, inputs.size(0), time.time()-end, configs.world_size)
        
        # Clip Comp
        nn.utils.clip_grad_value_(model.parameters(), 1.0)
        
        # Optimizer Step
        optimizer.step()
        if optimizer_q: optimizer_q.step()
        
        num_updates += 1
        end = time.time()
        
        if (batch_idx + 1) % configs.log.print_freq == 0:
            update_monitors(monitors, meters, configs, epoch, batch_idx, steps_per_epoch, optimizer, optimizer_q)
            logger_info(logger, "="*115)

    show_training_info(meters)
    return meters[0]['top1'].avg, meters[0]['top5'].avg, meters[0]['loss'].avg

def validate(data_loader, model, criterion, epoch, monitors, configs):
    from process_normal import validate as normal_validate
    return validate_min(data_loader, model, criterion, epoch, monitors, configs)

def validate_min(data_loader, model, criterion, epoch, monitors, configs):
    # Force switch to min bit width for validation as robustness reference
    from util.mpq import sample_min_cands
    model.eval()
    sample_min_cands(model, configs)
    
    val_meters = {
        "loss": AverageMeter(),
        "top1": AverageMeter(),
        "top5": AverageMeter(),
        "QE_loss": AverageMeter(),
        "dist_loss": AverageMeter(),
        "IDM_loss": AverageMeter(),
        "batch_time": AverageMeter(),
    }
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(val_meters, loss, 0, 0, 0, acc1, acc5, inputs.size(0), 0, configs.world_size)
            
    logger_info(logger, f"[G-VAL] Epoch {epoch} Min-Bit Top1: {val_meters['top1'].avg:.2f}")
    return val_meters['top1'].avg

class PerformanceScoreboard:
    def __init__(self, num_best_scores=3):
        self.board = []
        self.num_best_scores = num_best_scores
    def update(self, top1, top5, epoch):
        self.board.append((top1, epoch))
        self.board = sorted(self.board, key=lambda x: x[0], reverse=True)[:self.num_best_scores]
    def is_best(self, top1):
        return len(self.board) > 0 and top1 >= self.board[0][0]

