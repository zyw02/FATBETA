import logging
import time
import torch
import operator
from util.utils import accuracy, update_meter
from util import AverageMeter
from util.dist import master_only, logger_info

logger = logging.getLogger()

def train(train_loader, model, criterion, optimizer, epoch, monitors, configs):
    model.train()
    meters = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'QE_loss': AverageMeter(),
        'dist_loss': AverageMeter(),
        'IDM_loss': AverageMeter(),
        'batch_time': AverageMeter()
    }
    
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        update_meter(meters, loss, None, None, None, acc1, acc5, inputs.size(0), time.time() - end, configs.world_size)
        end = time.time()
        
        if (batch_idx + 1) % configs.log.print_freq == 0:
            logger_info(logger, f'Epoch: [{epoch}][{batch_idx+1}/{len(train_loader)}] '
                               f'Loss {meters["loss"].val:.4f} ({meters["loss"].avg:.4f}) '
                               f'Top1 {meters["top1"].val:.2f} ({meters["top1"].avg:.2f}) '
                               f'LR {optimizer.param_groups[0]["lr"]:.6f}')
            
    return meters['top1'].avg, meters['top5'].avg, meters['loss'].avg

def validate(val_loader, model, criterion, epoch, monitors, configs):
    model.eval()
    meters = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'QE_loss': AverageMeter(),
        'dist_loss': AverageMeter(),
        'IDM_loss': AverageMeter(),
        'batch_time': AverageMeter()
    }
    
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters, loss, None, None, None, acc1, acc5, inputs.size(0), time.time() - end, configs.world_size)
            end = time.time()
            
    logger_info(logger, f'Test:  Loss {meters["loss"].avg:.4f} Top1 {meters["top1"].avg:.2f} Top5 {meters["top5"].avg:.2f}')
    return meters['top1'].avg, meters['top5'].avg, meters['loss'].avg

class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board, key=operator.itemgetter('top1', 'top5', 'epoch'), reverse=True)[0:curr_len]

    def is_best(self, epoch):
        if not self.board: return False
        return self.board[0]['epoch'] == epoch

