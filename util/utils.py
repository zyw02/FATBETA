from copy import deepcopy
import torch
import torch.distributed as dist
from quan.func import QuanConv2d,QuanLinear,SwithableBatchNorm
from timm.utils import reduce_tensor, distribute_bn
from torch.distributed import get_world_size
import numpy as np
import random
import torch.nn as nn
from timm.scheduler import create_scheduler
import os
import shutil
import logging

logger = logging.getLogger()

def create_optimizer_and_lr_scheduler(model, configs):
    all_parameters = model.parameters()
    weight_parameters = []
    bn_parameters = []
    quant_parameters = []

    for pname, p in model.named_parameters():
        if p.ndimension() == 4 and 'bias' not in pname:
            # print('weight_param:', pname)
            weight_parameters.append(p)
        if 'quan_a_fn.s' in pname or 'quan_w_fn.s' in pname or 'quan3.a' in pname or 'scale' in pname or 'start' in pname:
            # print('alpha_param:', pname)
            quant_parameters.append(p)

    weight_parameters_id = list(map(id, weight_parameters))
    alpha_parameters_id = list(map(id, quant_parameters))
    other_parameters1 = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1))

    quantizer_optim = 'adam'
    if quantizer_optim == 'adam':
        optimizer_q = torch.optim.Adam(
            [
                {'params' : quant_parameters, 'lr': getattr(configs, 'q_lr', 1e-5)}
            ],
            lr = getattr(configs, 'q_lr', 1e-5)
        )

        optimizer = torch.optim.SGD(
            [
                {'params' : weight_parameters, 'weight_decay': configs.weight_decay, 'lr': configs.lr},
                {'params' : other_parameters, 'lr': configs.lr},
            ],
            nesterov=True,
            momentum=configs.momentum
        )

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs.epochs, eta_min=0)
        lr_scheduler, _ = create_scheduler(configs, optimizer)
        lr_scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=configs.epochs, eta_min=0)
    else:
        optimizer = torch.optim.Adam(
            [   {'params' : weight_parameters, 'weight_decay': configs.weight_decay, 'lr': configs.lr},
                {'params' : quant_parameters, 'lr': configs.lr*.1},
                {'params' : other_parameters, 'lr': configs.lr},
                
            ],
                betas=(0.9, 0.999)
        )
        
        optimizer_q, lr_scheduler_q = None, None
    
    return optimizer, optimizer_q, lr_scheduler, lr_scheduler_q

def preprocess_model(model, configs):
    dropout_p = getattr(configs, 'dropout', .0)

    if dropout_p > 0.:
        return model
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            
                print('droupout -> ', module.p)
        
        if 'mobilenet' in configs.arch:
            model.classifier = model.classifier[1:]
    
    return model

def model_profiling(model: torch.nn.Module, first_last_layer_act_bits=8, first_last_layer_weight_bits=8, return_layers=False):
    bitops = 0.
    model_size = 0.
    quantized_layers = []
    bn = []
    next_bn = False

    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d):
            if hasattr(module, 'bits') and (module.bits is not None and len(module.bits) > 1):
                # If next_bn is still True from previous Conv, it means that Conv had no BN
                # Add None to bn list for that Conv layer
                if next_bn and return_layers:
                    bn.append(None)
                
                # module: torch.nn.Conv2d
                assert isinstance(module.bits, (list, tuple))
                next_bn = True
                quantized_layers.append(module)

                wbits, abits = module.bits
                # Convert to Python int if tensor or list
                if isinstance(wbits, torch.Tensor):
                    wbits = int(wbits.item())
                elif isinstance(wbits, (list, tuple)):
                    wbits = int(wbits[0])
                else:
                    wbits = int(wbits)
                    
                if isinstance(abits, torch.Tensor):
                    abits = int(abits.item())
                elif isinstance(abits, (list, tuple)):
                    abits = int(abits[0])
                else:
                    abits = int(abits)
                    
                bitops += (wbits*abits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels*module.output_size)//module.groups
                model_size += (wbits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels)//module.groups
            
            elif module.bits is None:
                bitops += first_last_layer_act_bits*first_last_layer_weight_bits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels*module.output_size
                
                model_size += first_last_layer_weight_bits*module.kernel_size[-1]*module.kernel_size[-2]*module.in_channels*module.out_channels
        
        if isinstance(module, SwithableBatchNorm) and next_bn:
            bn.append(module)
            next_bn = False
        
        if isinstance(module, QuanLinear):
            # If next_bn is still True, it means previous Conv had no BN, add None for it
            if next_bn and return_layers:
                bn.append(None)
                next_bn = False
            
            # Check if this Linear layer has dynamic bits (not fixed_bits)
            if hasattr(module, 'bits') and module.bits is not None and len(module.bits) > 1:
                # Linear layer with dynamic bits - add to quantized_layers
                quantized_layers.append(module)
                # Linear layers don't have BN, so add None to bn list
                if return_layers:
                    bn.append(None)
                
                wbits, abits = module.bits
                # Convert to Python int if tensor or list
                if isinstance(wbits, torch.Tensor):
                    wbits = int(wbits.item())
                elif isinstance(wbits, (list, tuple)):
                    wbits = int(wbits[0])
                else:
                    wbits = int(wbits)
                    
                if isinstance(abits, torch.Tensor):
                    abits = int(abits.item())
                elif isinstance(abits, (list, tuple)):
                    abits = int(abits[0])
                else:
                    abits = int(abits)
                    
                bitops += wbits*abits*module.in_features*module.out_features
                model_size += wbits*module.in_features*module.out_features
            else:
                # Fixed bits Linear layer (first/last layer)
                bitops += first_last_layer_act_bits*first_last_layer_weight_bits*module.in_features*module.out_features
                model_size += first_last_layer_weight_bits*module.in_features*module.out_features
    
    # Handle case where last Conv layer had no BN (next_bn still True at end)
    if next_bn and return_layers:
        bn.append(None)
    
    bitops /= 1e9
    model_size /= (8*1024*1024)
    
    if return_layers:
        # Debug: print layer information if mismatch
        if len(quantized_layers) != len(bn):
            print(f"[DEBUG] Layer count mismatch: quantized_layers={len(quantized_layers)}, bn={len(bn)}")
            for i, layer in enumerate(quantized_layers):
                layer_type = "QuanConv2d" if isinstance(layer, QuanConv2d) else "QuanLinear"
                print(f"  Layer {i}: {layer_type}, bits={layer.bits}, bn={bn[i] if i < len(bn) else 'MISSING'}")
        assert len(quantized_layers) == len(bn), f"quantized_layers count ({len(quantized_layers)}) != bn count ({len(bn)})"
        return bitops, model_size, quantized_layers, bn
    else:
        return bitops, model_size

def reset_batchnorm_stats(m):
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        m.momentum = None

@torch.no_grad()
def calibrate_batchnorm_state(model, loader, num_batch=30, reset=False, distributed_training=True, epoch=0):

    if epoch >= 0 and hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    model.eval()

    if reset:
        for _, module in model.named_modules():
            reset_batchnorm_stats(module)

    for batch_idx, (inputs, _) in enumerate(loader):
            if batch_idx > num_batch:
                break
            
            # print(batch_idx)
            
            inputs = inputs.cuda()
            model(inputs)
        
    if distributed_training: # all reduce for each GPU
        if dist.is_initialized():
            distribute_bn(model, world_size=get_world_size(), reduce=True)
    
    model.eval()

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_meter(meter, loss, QE_loss, dist_loss, IDM_loss, acc1, acc5, size, batch_time, world_size):
    # 确保loss是tensor，如果是标量则转换为tensor
    if not isinstance(loss, torch.Tensor):
        if loss is None or loss == 0:
            loss = torch.tensor(0.0)
        else:
            loss = torch.tensor(float(loss))
    
    # 获取device（如果loss是tensor）
    device = loss.device if isinstance(loss, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保loss是可reshape的
    if isinstance(loss, torch.Tensor) and loss.dim() == 0:
        loss_reshaped = loss.reshape(1)
    elif isinstance(loss, torch.Tensor):
        loss_reshaped = loss.data.reshape(1) if hasattr(loss, 'data') else loss.reshape(1)
    else:
        loss_reshaped = torch.tensor([float(loss)], device=device)
    
    data = torch.cat([loss_reshaped, acc1.reshape(1), acc5.reshape(1), 
                QE_loss.data.reshape(1) if QE_loss is not None and QE_loss != 0 and isinstance(QE_loss, torch.Tensor) else torch.zeros(1, device=device), 
                dist_loss.data.reshape(1) if dist_loss is not None and dist_loss != 0 and isinstance(dist_loss, torch.Tensor) else torch.zeros(1, device=device), 
                IDM_loss.data.reshape(1) if IDM_loss is not None and IDM_loss != 0 and isinstance(IDM_loss, torch.Tensor) else torch.zeros(1, device=device), 
                 ])
    # Only reduce if distributed training is initialized
    if dist.is_initialized() and world_size > 1:
        reduced_data = reduce_tensor(data, world_size)
    else:
        reduced_data = data
    reduced_loss, reduced_top1, reduced_top5, reduced_QE_loss, reduced_dist_loss, reduced_IDM_loss = reduced_data
    
    meter['dist_loss'].update(reduced_dist_loss.item(), size)
    meter['IDM_loss'].update(reduced_IDM_loss.item(), size)
    meter['QE_loss'].update(reduced_QE_loss.item(), size)
    meter['loss'].update(reduced_loss.item(), size)
    meter['top1'].update(reduced_top1.item(), size)
    meter['top5'].update(reduced_top5.item(), size)
    meter['batch_time'].update(batch_time)


def copy_code(logger, src=None, dst="./code/", exclude_dirs=None, exclude_files=None):
    """
    Copy code files from source directory to destination directory.
    Similar to SAQ's copy_code function, saves experiment code for reproducibility.
    
    Args:
        logger: Logger instance
        src: Source directory (default: current working directory)
        dst: Destination directory (default: "./code/")
        exclude_dirs: List of directory names to exclude (default: ["output", "log", "training", "eval", "search", "data", "__pycache__", ".git"])
        exclude_files: List of file patterns to exclude (default: ["__pycache__", ".pyc", ".pyo"])
    """
    if src is None:
        src = os.path.abspath(".")
    
    if exclude_dirs is None:
        exclude_dirs = ["output", "log", "training", "eval", "search", "data", "__pycache__", ".git", "code"]
    
    if exclude_files is None:
        exclude_files = ["__pycache__", ".pyc", ".pyo", ".pth", ".pth.tar", ".ckpt", ".pt"]
    
    # Only copy on main process (rank 0)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            for f in os.listdir(src):
                # Skip excluded directories and files
                if f in exclude_dirs:
                    continue
                if any(pattern in f for pattern in exclude_files):
                    continue
                
                src_file = os.path.join(src, f)
                
                # Copy Python files
                if f.endswith(".py"):
                    if not os.path.isdir(dst):
                        os.makedirs(dst, exist_ok=True)
                    dst_file = os.path.join(dst, f)
                    try:
                        shutil.copy2(src=src_file, dst=dst_file)
                        logger.info(f"Copied code file: {f}")
                    except Exception as e:
                        logger.warning(f"Failed to copy file {src_file} to {dst_file}: {e}")
                
                # Recursively copy directories (but skip excluded ones)
                elif os.path.isdir(src_file):
                    # Skip if it's an excluded directory
                    if f in exclude_dirs:
                        continue
                    deeper_dst = os.path.join(dst, f)
                    copy_code(logger, src=src_file, dst=deeper_dst, exclude_dirs=exclude_dirs, exclude_files=exclude_files)
            
            logger.info(f"Code backup completed. Code saved to: {dst}")
        except Exception as e:
            logger.error(f"Error during code backup: {e}")