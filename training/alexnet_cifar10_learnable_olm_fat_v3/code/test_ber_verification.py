#!/usr/bin/env python
"""快速验证 BER 是否真的生效"""
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

# 加载 checkpoint
print("Loading checkpoint...")
load_checkpoint(model, 'training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar', strict=False)

# 切换 bit width
target_bits = configs.target_bits
max_bit = max(target_bits)
print(f"Switching to {max_bit}-bit...")
switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)

# 初始化 output_size
dummy_input = torch.randn(1, 3, 32, 32).cuda()
with torch.no_grad():
    _ = model(dummy_input)
print("Model initialized.")

# 测试不同 BER 下的故障注入
print("\n" + "="*60)
print("Testing fault injection with different BER values...")
print("="*60)

for ber in [1e-3, 1e-2, 1e-1]:
    print(f"\nBER = {ber:.1e}")
    
    # 创建新的 FaultInjector
    injector = FaultInjector(
        model=model,
        mode='ber',
        ber=ber,
        device='cuda:0',
        enable_in_training=False,
        enable_in_inference=True,
        skip_first_last=True,
        seed=42,
        seed_list=None,
    )
    
    injector.enable()
    
    # 进行多次 forward pass，检查输出是否变化
    outputs = []
    with torch.no_grad():
        for i in range(5):
            output = model(dummy_input)
            outputs.append(output.clone())
    
    injector.disable()
    
    # 计算输出之间的差异
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff = (outputs[i] - outputs[j]).abs().mean().item()
            diffs.append(diff)
    
    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0.0
    
    print(f"  Average output difference: {avg_diff:.6f}")
    print(f"  Max output difference: {max_diff:.6f}")
    
    # 检查 BER 是否生效
    if ber == 1e-1:
        if avg_diff < 0.01:
            print(f"  ⚠️  WARNING: BER={ber:.1e} but output difference is very small!")
            print(f"     Fault injection may NOT be working correctly!")
        else:
            print(f"  ✓ BER={ber:.1e} produces significant output variation")
    elif ber == 1e-3:
        if avg_diff > 10.0:
            print(f"  ⚠️  WARNING: BER={ber:.1e} but output difference is very large!")
        else:
            print(f"  ✓ BER={ber:.1e} produces expected small output variation")

print("\n" + "="*60)
print("Test complete!")
print("="*60)

