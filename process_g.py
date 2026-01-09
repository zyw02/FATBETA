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


def _get_meters(mode: str = "training", nr_random_sample=2):
    meters = []
    # 辅助函数：生成包含所有必要键的字典
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
    
    # Meters for Random Samples
    for i in range(nr_random_sample):
        meters.append(create_full_meter(f"Align-Sample-{i}"))
    return meters

def sample_alignment_aware_policy(model, configs, similarity_results):
    """
    核心逻辑：根据梯度相似度结果指导采样。
    对于每一层：
    1. 找出相似度 > 0 的 bit 位。
    2. 如果存在，则从这些 bit 中随机选一个。
    3. 如果不存在，则选相似度最高的一个。
    """
    target_bits = sorted(configs.target_bits, reverse=True)
    max_bit = target_bits[0]
    quan_scheduler = configs.quan
    
    # 获取所有动态量化层名称（为了匹配相似度列表的索引）
    dynamic_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)) and name not in quan_scheduler.excepts:
            dynamic_layer_names.append(name)
            
    # 执行采样
    layer_id = 0
    weights, acts = [], []
    
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)) and name not in quan_scheduler.excepts:
            # 获取该层在不同位宽下的相似度
            pool = [] # 候选位宽池
            
            # 始终允许最大位宽（相似度默认为 1）
            pool.append(max_bit)
            
            if similarity_results:
                # 检查其他位宽
                for b in target_bits[1:]:
                    sim = similarity_results[b][layer_id]
                    if sim is not None and sim > 0:
                        pool.append(b)
                
                # 如果池子里只有 max_bit (即相似度全 <= 0)，则补充相似度最高的一个
                if len(pool) == 1 and len(target_bits) > 1:
                    best_b = target_bits[1]
                    best_sim = -2.0
                    for b in target_bits[1:]:
                        sim = similarity_results[b][layer_id]
                        if sim is not None and sim > best_sim:
                            best_sim = sim
                            best_b = b
                    pool.append(best_b)
            else:
                # 初始轮次没有相似度结果，回退到全量随机
                pool = target_bits
            
            # 从候选池中随机采样
            sampled_bit = random.choice(pool)
            module.bits = (sampled_bit, sampled_bit)
            weights.append(sampled_bit)
            acts.append(sampled_bit)
            
            # 同步 BN
            layer_id += 1
        elif isinstance(module, SwithableBatchNorm):
            # 简单的 BN 切换逻辑，这里假设采样是对称的 (W=A)
            if hasattr(model, 'last_sampled_bits'):
                module.switch_bn(model.last_sampled_bits)

    return weights, acts

def compute_overall_loss(model, task_loss, configs, epoch, QE_loss_weight=0.1):
    """
    同步 process_normal.py 的逻辑：计算 QE_loss 和 distribution_loss
    """
    # normal 里面是: epoch > 40 时启用 QE 最小化
    quantization_error_minimization = (epoch > 40)
    
    # 计算辅助损失
    QE_loss, distribution_loss = auxiliary_quantized_loss(
        model, 
        quantization_error_minimization=quantization_error_minimization, 
        fairness_regularization=True
    )
    
    # 权重处理 (同步 normal)
    QE_loss = QE_loss * QE_loss_weight
    
    # Distribution loss 权重逻辑 (同步 normal)
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
    
    meters = _get_meters(nr_random_sample=nr_random_sample)
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        
        # 获取当前的 QE loss 退火权重
        qe_w = annealing_schedule(num_updates) if annealing_schedule else 0.1
        
        # 1. Max Bit Pass
        optimizer.zero_grad()
        if optimizer_q: optimizer_q.zero_grad()
        
        sample_max_cands(model, configs)
        outputs_max = model(inputs)
        ce_loss_max = criterion(outputs_max, targets)
        
        # 计算辅助损失
        total_loss_max, qe_max, dist_max = compute_overall_loss(model, ce_loss_max, configs, epoch, QE_loss_weight=qe_w)
        
        total_loss_max.backward()
        
        acc1, acc5 = accuracy(outputs_max.data, targets.data, topk=(1, 5))
        update_meter(meters[0], total_loss_max, qe_max, dist_max, 0, acc1, acc5, inputs.size(0), time.time()-end, configs.world_size)
        
        # 2. Alignment-Aware Random Passes
        for i in range(nr_random_sample):
            start_sample = time.time()
            sample_alignment_aware_policy(unwrap_model(model), configs, similarity_results)
            
            outputs_rand = model(inputs)
            ce_loss_rand = criterion(outputs_rand, targets)
            
            # 同样计算辅助损失
            total_loss_rand, qe_rand, dist_rand = compute_overall_loss(model, ce_loss_rand, configs, epoch, QE_loss_weight=qe_w)
            
            total_loss_rand.backward()
            
            acc1, acc5 = accuracy(outputs_rand.data, targets.data, topk=(1, 5))
            update_meter(meters[i+1], total_loss_rand, qe_rand, dist_rand, 0, acc1, acc5, inputs.size(0), time.time()-start_sample, configs.world_size)
            
        # 3. Step
        nn.utils.clip_grad_value_(model.parameters(), 1.0)
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
    # 强制切换到最小位宽进行验证，作为鲁棒性参考
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
            # 同样修复这里的验证集 meter 更新
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
