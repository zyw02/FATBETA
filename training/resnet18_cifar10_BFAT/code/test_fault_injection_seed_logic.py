#!/usr/bin/env python
"""测试故障注入的 seed 逻辑，看看两个脚本是否使用相同的逻辑"""
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

print("="*80)
print("测试 1: eval_with_fault_injection.py 的配置")
print("="*80)
print("配置: seed=42, seed_list=None, use_position_based_mask=False (default)")

injector1 = FaultInjector(
    model=model,
    mode='ber',
    ber=1e-1,
    device='cuda:0',
    enable_in_training=False,
    enable_in_inference=True,
    skip_first_last=True,
    seed=42,
    seed_list=None,  # 关键：不使用 seed_list
    use_position_based_mask=False,  # 默认值
)

injector1.enable()

# 进行多次 forward pass，检查输出是否相同
outputs1 = []
with torch.no_grad():
    for i in range(3):
        output = model(dummy_input)
        outputs1.append(output.clone())

injector1.disable()

# 检查输出差异
diffs1 = []
for i in range(len(outputs1)):
    for j in range(i+1, len(outputs1)):
        diff = (outputs1[i] - outputs1[j]).abs().mean().item()
        diffs1.append(diff)

print(f"  输出差异: {diffs1}")
print(f"  平均差异: {sum(diffs1)/len(diffs1):.10f}")
if all(d < 1e-6 for d in diffs1):
    print("  ⚠️  所有 forward pass 的输出完全相同！")
else:
    print("  ✓ Forward pass 的输出有差异")

print("\n" + "="*80)
print("测试 2: eval_gradient_statistics_restorer.py 的配置")
print("="*80)
print("配置: seed=42, seed_list=None, use_position_based_mask=False (default)")

injector2 = FaultInjector(
    model=model,
    mode='ber',
    ber=1e-1,
    device='cuda:0',
    enable_in_training=False,
    enable_in_inference=True,
    skip_first_last=True,
    seed=42,
    seed_list=None,  # 关键：不使用 seed_list
    use_position_based_mask=False,  # 默认值
)

injector2.enable()

# 进行多次 forward pass，检查输出是否相同
outputs2 = []
with torch.no_grad():
    for i in range(3):
        output = model(dummy_input)
        outputs2.append(output.clone())

injector2.disable()

# 检查输出差异
diffs2 = []
for i in range(len(outputs2)):
    for j in range(i+1, len(outputs2)):
        diff = (outputs2[i] - outputs2[j]).abs().mean().item()
        diffs2.append(diff)

print(f"  输出差异: {diffs2}")
print(f"  平均差异: {sum(diffs2)/len(diffs2):.10f}")
if all(d < 1e-6 for d in diffs2):
    print("  ⚠️  所有 forward pass 的输出完全相同！")
else:
    print("  ✓ Forward pass 的输出有差异")

# 比较两个 injector 的输出
print("\n" + "="*80)
print("比较两个 injector 的输出:")
print("="*80)
diff_between = (outputs1[0] - outputs2[0]).abs().mean().item()
print(f"  Injector1 和 Injector2 的第一次输出差异: {diff_between:.10f}")
if diff_between < 1e-6:
    print("  ✓ 两个 injector 产生相同的输出（符合预期）")
else:
    print("  ⚠️  两个 injector 产生不同的输出！")



