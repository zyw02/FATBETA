#!/usr/bin/env python
"""调试故障注入：检查权重是否真的被修改"""
import torch
import sys
sys.path.insert(0, '.')
from util.fault_injector import FaultInjector
from model.model import create_model
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.config import get_config
from util.checkpoint import load_checkpoint
from util.mpq import switch_bit_width

# 加载配置
import sys
original_argv = sys.argv[:]
sys.argv = [sys.argv[0], 'configs/training/train_alexnet_cifar10_sensitive_stage1.yaml']
try:
    configs = get_config(default_file='configs/training/train_alexnet_cifar10_sensitive_stage1.yaml')
finally:
    sys.argv = original_argv

model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
model = model.cuda()
model.eval()

load_checkpoint(model, 'training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar', strict=False)
target_bits = configs.target_bits
max_bit = max(target_bits)
switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

dummy_input = torch.randn(1, 3, 32, 32).cuda()
with torch.no_grad():
    _ = model(dummy_input)

# 保存原始权重
original_weights = {}
for name, module in model.named_modules():
    if hasattr(module, 'weight') and hasattr(module, 'quan_w_fn'):
        if name not in ['features.0', 'classifier.6']:  # 跳过 first/last
            original_weights[name] = module.weight.clone()

print("Testing fault injection with BER=1e-1...")
injector = FaultInjector(
    model=model,
    mode='ber',
    ber=1e-1,
    device='cuda:0',
    enable_in_training=False,
    enable_in_inference=True,
    skip_first_last=True,
    seed=42,
    seed_list=None,
)

injector.enable()

# 进行一次 forward pass
with torch.no_grad():
    output = model(dummy_input)

# 检查权重是否被修改
print("\nChecking if weights were modified during forward pass...")
weight_modified = False
for name, module in model.named_modules():
    if hasattr(module, 'weight') and hasattr(module, 'quan_w_fn'):
        if name not in ['features.0', 'classifier.6']:  # 跳过 first/last
            if name in original_weights:
                diff = (module.weight - original_weights[name]).abs().max().item()
                if diff > 1e-6:
                    print(f"  {name}: weight modified! Max diff = {diff:.6f}")
                    weight_modified = True
                else:
                    print(f"  {name}: weight NOT modified (diff = {diff:.10f})")

if not weight_modified:
    print("\n⚠️  WARNING: Weights were NOT modified! Fault injection may not be working!")
    print("   This could explain why accuracy is so high - no faults are being injected!")
else:
    print("\n✓ Weights were modified, fault injection is working")

injector.disable()



