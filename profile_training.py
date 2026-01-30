"""
Profiling script to identify training bottlenecks.
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
from util.mpq import sample_max_cands, sample_one_mixed_policy, get_cached_layers
from util.qat import auxiliary_quantized_loss

# Setup
configs = get_config(default_file=Path('.') / 'template.yaml')
model = create_model('resnet20', dataset='cifar10', pre_trained=False)
model = preprocess_model(model, configs)
model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
model.cuda()
model.train()

# Warmup
x = torch.randn(128, 3, 32, 32).cuda()
y = torch.randint(0, 10, (128,)).cuda()
criterion = nn.CrossEntropyLoss().cuda()

for _ in range(5):
    sample_max_cands(model, configs)
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
torch.cuda.synchronize()

# Profiling
N = 100
timings = {}

# 1. Sample max cands
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    sample_max_cands(model, configs)
torch.cuda.synchronize()
timings['sample_max_cands'] = (time.time() - start) / N * 1000

# 2. Sample one mixed policy
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    sample_one_mixed_policy(model, configs)
torch.cuda.synchronize()
timings['sample_one_mixed_policy'] = (time.time() - start) / N * 1000

# 3. Forward pass
sample_max_cands(model, configs)
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    out = model(x)
torch.cuda.synchronize()
timings['forward'] = (time.time() - start) / N * 1000

# 4. Criterion (CE loss)
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    loss = criterion(out, y)
torch.cuda.synchronize()
timings['criterion'] = (time.time() - start) / N * 1000

# 5. Auxiliary quantized loss
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    qe, dist = auxiliary_quantized_loss(model, fairness_regularization=True)
torch.cuda.synchronize()
timings['auxiliary_quantized_loss'] = (time.time() - start) / N * 1000

# 6. Backward pass
out = model(x)
loss = criterion(out, y)
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    model.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
torch.cuda.synchronize()
timings['forward+backward'] = (time.time() - start) / N * 1000

# 7. Optimizer step
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    optimizer.step()
torch.cuda.synchronize()
timings['optimizer.step'] = (time.time() - start) / N * 1000

# 8. Clip grad
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    nn.utils.clip_grad_value_(model.parameters(), 1.0)
torch.cuda.synchronize()
timings['clip_grad_value'] = (time.time() - start) / N * 1000

print("\n" + "="*60)
print("PROFILING RESULTS (ms per call)")
print("="*60)
for name, t in sorted(timings.items(), key=lambda x: -x[1]):
    print(f"{name:35s}: {t:8.3f} ms")

print("\n" + "="*60)
print("ESTIMATED EPOCH TIME BREAKDOWN")
print("="*60)
batches = 391
# Per batch: sample_max + forward + backward + aux_loss + sample_mixed + forward + backward + aux_loss + clip + optimizer
per_batch = (
    timings['sample_max_cands'] + 
    timings['forward+backward'] + 
    timings['auxiliary_quantized_loss'] +  # for max
    timings['sample_one_mixed_policy'] + 
    timings['forward+backward'] +
    timings['auxiliary_quantized_loss'] +  # for mixed
    timings['clip_grad_value'] +
    timings['optimizer.step']
)
print(f"Estimated per batch: {per_batch:.2f} ms")
print(f"Estimated epoch time: {per_batch * batches / 1000:.1f} s")
print(f"Pure forward+backward only (2x): {timings['forward+backward'] * 2 * batches / 1000:.1f} s")
