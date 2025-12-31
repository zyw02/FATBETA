#!/usr/bin/env python
"""详细调试故障注入：检查是否真的在修改量化后的权重"""
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
print("测试故障注入是否真的修改了量化后的权重")
print("="*80)

# 选择一个层进行详细检查
test_layer = None
test_name = None
for name, module in model.named_modules():
    if hasattr(module, 'quan_w_fn') and name not in ['features.0', 'classifier.6']:
        test_layer = module
        test_name = name
        break

if test_layer is None:
    print("ERROR: No test layer found!")
    sys.exit(1)

print(f"测试层: {test_name}")
print(f"权重形状: {test_layer.weight.shape}")

# 手动量化权重（不使用故障注入）
weight = test_layer.weight
bits = test_layer.bits[0] if test_layer.bits else 6
scale = test_layer.quan_w_fn.get_scale(bits, detach=True)

print(f"\n不使用故障注入:")
x_q_clean = test_layer.quan_w_fn(weight, bits, is_activation=False)
print(f"  量化后的权重范围: [{x_q_clean.min().item():.6f}, {x_q_clean.max().item():.6f}]")
print(f"  量化后的权重均值: {x_q_clean.mean().item():.6f}")

# 创建故障注入器
injector = FaultInjector(
    model=model,
    mode='ber',
    ber=1e-1,  # 高 BER，应该产生明显的故障
    device='cuda:0',
    enable_in_training=False,
    enable_in_inference=True,
    skip_first_last=True,
    seed=42,
    seed_list=None,
)

injector.enable()

# 手动调用量化器（应该会触发故障注入）
print(f"\n使用故障注入 (BER=1e-1):")
x_q_faulted = test_layer.quan_w_fn(weight, bits, is_activation=False)
print(f"  量化后的权重范围: [{x_q_faulted.min().item():.6f}, {x_q_faulted.max().item():.6f}]")
print(f"  量化后的权重均值: {x_q_faulted.mean().item():.6f}")

# 比较差异
diff = (x_q_clean - x_q_faulted).abs()
print(f"\n差异统计:")
print(f"  最大差异: {diff.max().item():.6f}")
print(f"  平均差异: {diff.mean().item():.6f}")
print(f"  非零差异的数量: {(diff > 1e-6).sum().item()} / {diff.numel()}")

if diff.max().item() < 1e-6:
    print("\n⚠️  WARNING: 量化后的权重没有变化！故障注入可能没有生效！")
else:
    print("\n✓ 量化后的权重有变化，故障注入正在工作")

# 再次调用，检查是否产生相同的故障
print(f"\n第二次调用（应该产生相同的故障，因为使用固定 seed）:")
x_q_faulted2 = test_layer.quan_w_fn(weight, bits, is_activation=False)
diff2 = (x_q_faulted - x_q_faulted2).abs()
print(f"  两次故障注入的差异: 最大={diff2.max().item():.10f}, 平均={diff2.mean().item():.10f}")

if diff2.max().item() < 1e-6:
    print("  ✓ 两次故障注入产生相同的故障（符合预期，因为使用固定 seed）")
else:
    print("  ⚠️  两次故障注入产生不同的故障！")

injector.disable()



