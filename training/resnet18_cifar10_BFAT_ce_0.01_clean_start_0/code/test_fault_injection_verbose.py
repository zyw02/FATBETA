#!/usr/bin/env python
"""详细调试故障注入：检查是否真的进入了故障注入逻辑"""
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

print("Creating FaultInjector with BER=1e-1...")
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

print(f"FaultInjector settings:")
print(f"  enable_in_training = {injector.enable_in_training}")
print(f"  enable_in_inference = {injector.enable_in_inference}")
print(f"  ber = {injector.ber}")
print(f"  model.training = {model.training}")

injector.enable()

# 检查一个层的量化器是否被包装
print("\nChecking if quantizer forward is wrapped...")
for name, module in model.named_modules():
    if hasattr(module, 'quan_w_fn') and name not in ['features.0', 'classifier.6']:
        orig_fn = getattr(module.quan_w_fn, '_original_forward', None)
        if orig_fn is None:
            print(f"  {name}: quan_w_fn.forward is NOT wrapped (no _original_forward found)")
        else:
            print(f"  {name}: quan_w_fn.forward IS wrapped")
            # 检查 wrapped function
            wrapped_fn = module.quan_w_fn.forward
            print(f"    Wrapped function: {wrapped_fn}")
        break  # 只检查第一个

# 进行一次 forward pass，并检查量化后的权重
print("\nPerforming forward pass...")
with torch.no_grad():
    # 手动检查一个层的量化过程
    test_module = None
    for name, module in model.named_modules():
        if hasattr(module, 'quan_w_fn') and name not in ['features.0', 'classifier.6']:
            test_module = module
            test_name = name
            break
    
    if test_module:
        print(f"Testing layer: {test_name}")
        # 获取原始权重
        weight = test_module.weight
        bits = test_module.bits[0] if test_module.bits else 8
        print(f"  Weight shape: {weight.shape}, bits: {bits}")
        
        # 手动调用量化器
        x_q1 = test_module.quan_w_fn(weight, bits, is_activation=False)
        x_q2 = test_module.quan_w_fn(weight, bits, is_activation=False)
        
        diff = (x_q1 - x_q2).abs().max().item()
        print(f"  Quantized weight difference between two calls: {diff:.10f}")
        
        if diff < 1e-6:
            print(f"  ⚠️  WARNING: Quantized weights are identical! Fault injection may not be working!")
        else:
            print(f"  ✓ Quantized weights differ, fault injection is working")
    
    # 完整的 forward pass
    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")

injector.disable()

