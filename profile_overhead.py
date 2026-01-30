"""
Profile EMA and Logging overhead in the real training loop context.
"""
import time
import torch
import torch.nn as nn
import sys
sys.argv = ['', 'configs/training/r20.yaml']

from pathlib import Path
from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util import get_config, preprocess_model
from util.model_ema import ModelEma
from util.utils import accuracy, update_meter

# Setup
configs = get_config(default_file=Path('.') / 'template.yaml')
model = create_model('resnet20', dataset='cifar10', pre_trained=False)
model = preprocess_model(model, configs)
model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
model.cuda()

ema = ModelEma(model, decay=0.999)

# Profiling
N = 100
x = torch.randn(128, 10).cuda() # dummy output
y = torch.randint(0, 10, (128,)).cuda()

# 1. EMA Update
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    ema.update(model)
torch.cuda.synchronize()
ema_time = (time.time() - start) / N * 1000

# 2. Accuracy + Update Meter
from util.monitor import AverageMeter
meter = {
    "name": "test",
    "loss": AverageMeter(),
    "top1": AverageMeter(),
    "top5": AverageMeter(),
    "QE_loss": AverageMeter(),
    "dist_loss": AverageMeter(),
    "IDM_loss": AverageMeter(),
    "batch_time": AverageMeter(),
}

torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    acc1, acc5 = accuracy(x, y, topk=(1, 5))
    update_meter(meter, torch.tensor(1.0).cuda(), 0, 0, 0, acc1, acc5, 128, 0.05, 1)
torch.cuda.synchronize()
logging_time = (time.time() - start) / N * 1000

print(f"EMA Update: {ema_time:.3f} ms")
print(f"Accuracy + Meter: {logging_time:.3f} ms")

# Check if there are any other syncs in ModelEma.update
# Already optimized manually before, but check if there's still a crawl.
print("\nAnalyzing EMA update internals...")
msd = model.state_dict()
start = time.time()
for _ in range(N):
    _ = model.state_dict()
torch.cuda.synchronize()
sd_time = (time.time() - start) / N * 1000
print(f"model.state_dict() duration: {sd_time:.3f} ms")
