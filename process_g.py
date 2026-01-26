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
    
    # --- Determine Global Candidate Pool ---
    # Always include max_bit
    candidate_pool = [max_bit]
    
    # Check warmup
    warmup_epochs = getattr(configs, 'warmup_epochs', 0)
    in_warmup = (epoch < warmup_epochs)
    
    if in_warmup:
        logger_info(logger, f"G-PIPELINE: Epoch {epoch} (Warmup) - Only training Max Bit: {max_bit}")
    else:
        # Filter other bits based on global similarity
        if similarity_results:
             # similarity_results is {bit: [sim_layer1, sim_layer2, ...]}
             # We need to compute the average similarity for each bit-width
             avg_similarity = {}
             for b in target_bits[1:]:
                 sim_list = similarity_results.get(b, [])
                 if isinstance(sim_list, list):
                     valid_sims = [s for s in sim_list if s is not None]
                     if valid_sims:
                         avg_sim = sum(valid_sims) / len(valid_sims)
                     else:
                         avg_sim = -1.0
                 else:
                     # dynamic fallback if it is already a float
                     avg_sim = float(sim_list)
                     
                 avg_similarity[b] = avg_sim

             for b in target_bits[1:]:
                 sim = avg_similarity.get(b, -1.0)
                 if sim > 0:
                     candidate_pool.append(b)
             
             # Fallback: if only max_bit is in pool and we have other options, pick the best one
             if len(candidate_pool) == 1 and len(target_bits) > 1:
                 best_b = None
                 best_sim = -float('inf')
                 for b in target_bits[1:]:
                     sim = avg_similarity.get(b, -1.0)
                     if sim > best_sim:
                         best_sim = sim
                         best_b = b
                 
                 if best_b is not None:
                     candidate_pool.append(best_b)
                     logger_info(logger, f"  [Fallback] Added best alternative bit {best_b} (sim={best_sim:.4f})")
        else:
             # If no similarity results yet (e.g. first epoch if not pre-calculated, or failed), 
             # fallback to all bits or just max bit? 
             # Usually align analysis runs before train if configured.
             # Let's fallback to all target bits to be safe/exploratory if no info.
             candidate_pool = list(target_bits)
        
        logger_info(logger, f"G-PIPELINE: Epoch {epoch} Global Candidate Pool: {candidate_pool}")
        if similarity_results:
             # logger_info(logger, f"  Similarities: {similarity_results}")
             pass


    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        
        # Get current QE loss annealing weight
        qe_w = annealing_schedule(num_updates) if annealing_schedule else 0.1
        
        optimizer.zero_grad()
        if optimizer_q: optimizer_q.zero_grad()
        
        # Iterate over global pool
        # First bit is always Max Bit (by construction of candidate_pool)
        
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

