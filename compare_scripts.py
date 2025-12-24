#!/usr/bin/env python
"""对比两个脚本的执行流程，找出差异"""
import torch
import sys
sys.path.insert(0, '.')
from model.model import create_model
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.config import get_config
from util.checkpoint import load_checkpoint
from util.mpq import switch_bit_width
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector
from util.gradient_statistics_restorer import create_gradient_statistics_restorer

# 加载配置
import sys
original_argv = sys.argv[:]
sys.argv = [sys.argv[0], 'configs/training/train_alexnet_cifar10_sensitive_stage1.yaml']
try:
    configs = get_config(default_file='configs/training/train_alexnet_cifar10_sensitive_stage1.yaml')
finally:
    sys.argv = original_argv

device = torch.device('cuda:0')

print("="*80)
print("模拟 eval_with_fault_injection.py 的流程")
print("="*80)

# 创建模型
model1 = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
model1 = replace_module_by_names(model1, find_modules_to_quantize(model1, configs))
model1 = model1.to(device)
model1.eval()

# 加载 checkpoint
load_checkpoint(model1, 'training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar', strict=False)

# 切换 bit width
target_bits = configs.target_bits
max_bit = max(target_bits)
switch_bit_width(model1, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

# Dummy forward pass
dummy_input = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    _ = model1(dummy_input)

# 检查量化器的 scale
print("\n量化器 scale 状态 (eval_with_fault_injection.py 流程):")
for name, module in model1.named_modules():
    if hasattr(module, 'quan_w_fn') and hasattr(module.quan_w_fn, 's'):
        s = module.quan_w_fn.s
        init_state = getattr(module.quan_w_fn, 'init_state', None)
        if init_state is not None:
            init_status = init_state[0].item() if len(init_state) > 0 else 0
            print(f"  {name}: scale={s[0].item():.6f}, init_state={init_status}")

print("\n" + "="*80)
print("模拟 eval_gradient_statistics_restorer.py 的流程")
print("="*80)

# 创建模型
model2 = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
model2 = replace_module_by_names(model2, find_modules_to_quantize(model2, configs))
model2 = model2.to(device)
model2.eval()

# 加载 checkpoint
load_checkpoint(model2, 'training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar', strict=False)

# 切换 bit width
switch_bit_width(model2, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

# Dummy forward pass
with torch.no_grad():
    _ = model2(dummy_input)

# 检查量化器的 scale（统计收集前）
print("\n量化器 scale 状态 (统计收集前):")
for name, module in model2.named_modules():
    if hasattr(module, 'quan_w_fn') and hasattr(module.quan_w_fn, 's'):
        s = module.quan_w_fn.s
        init_state = getattr(module.quan_w_fn, 'init_state', None)
        if init_state is not None:
            init_status = init_state[0].item() if len(init_state) > 0 else 0
            print(f"  {name}: scale={s[0].item():.6f}, init_state={init_status}")

# 创建 data loader
train_loader, val_loader, test_loader, _, _ = init_dataloader(configs.dataloader, configs.arch)

# 统计收集（关键步骤！）
print("\n进行统计收集（50 batches）...")
restorer = create_gradient_statistics_restorer(
    model=model2,
    data_loader=test_loader,
    k=4.0,
    kernel_size=3,
    num_statistics_batches=50,
    layer_names=None,
)

# 检查量化器的 scale（统计收集后）
print("\n量化器 scale 状态 (统计收集后):")
for name, module in model2.named_modules():
    if hasattr(module, 'quan_w_fn') and hasattr(module.quan_w_fn, 's'):
        s = module.quan_w_fn.s
        init_state = getattr(module.quan_w_fn, 'init_state', None)
        if init_state is not None:
            init_status = init_state[0].item() if len(init_state) > 0 else 0
            print(f"  {name}: scale={s[0].item():.6f}, init_state={init_status}")

# 比较两个模型的量化器 scale
print("\n" + "="*80)
print("比较两个模型的量化器 scale 差异:")
print("="*80)
for name, module1 in model1.named_modules():
    if hasattr(module1, 'quan_w_fn') and hasattr(module1.quan_w_fn, 's'):
        module2 = dict(model2.named_modules())[name]
        if hasattr(module2, 'quan_w_fn') and hasattr(module2.quan_w_fn, 's'):
            s1 = module1.quan_w_fn.s[0].item()
            s2 = module2.quan_w_fn.s[0].item()
            diff = abs(s1 - s2)
            if diff > 1e-6:
                print(f"  {name}: scale1={s1:.6f}, scale2={s2:.6f}, diff={diff:.6f} ⚠️ DIFFERENT!")
            else:
                print(f"  {name}: scale1={s1:.6f}, scale2={s2:.6f}, diff={diff:.10f}")



