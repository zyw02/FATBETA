#!/usr/bin/env python
"""测试 JSON 配置加载是否正确"""
import torch
import sys
sys.path.insert(0, '.')
from model.model import create_model
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.config import get_config
from util.checkpoint import load_checkpoint
from util.fault_injector import setup_model_with_bit_width_config

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

# 初始化 output_size
dummy_input = torch.randn(1, 3, 32, 32).cuda()
with torch.no_grad():
    _ = model(dummy_input)

print("="*80)
print("测试 JSON 配置加载")
print("="*80)

# 检查加载前的位宽
print("\n加载 JSON 配置前的位宽:")
for name, module in model.named_modules():
    if hasattr(module, 'bits') and module.bits is not None:
        print(f"  {name}: bits={module.bits}")
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        print(f"  {name}: fixed_bits={module.fixed_bits}")

# 加载 JSON 配置
json_path = "search/alexnet_cifar10_FAT_a92b25search_bit_width_config.json"
print(f"\n加载 JSON 配置: {json_path}")
try:
    weight_bits, act_bits = setup_model_with_bit_width_config(
        model,
        json_path,
        config_index=0,
        verbose=True
    )
    print(f"✓ 成功加载配置: {len(weight_bits)} 层")
    print(f"  Weight bits: {weight_bits}")
    print(f"  Act bits: {act_bits}")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查加载后的位宽
print("\n加载 JSON 配置后的位宽:")
for name, module in model.named_modules():
    if hasattr(module, 'bits') and module.bits is not None:
        print(f"  {name}: bits={module.bits}")
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        print(f"  {name}: fixed_bits={module.fixed_bits}")

# 验证位宽是否匹配 JSON 配置
print("\n验证位宽是否匹配 JSON 配置:")
config_idx = 0
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
            print(f"  {name}: fixed_bits={module.fixed_bits} (跳过，不在 JSON 配置中)")
        elif hasattr(module, 'bits') and module.bits is not None:
            expected_w = weight_bits[config_idx] if config_idx < len(weight_bits) else None
            expected_a = act_bits[config_idx] if config_idx < len(act_bits) else None
            actual_w, actual_a = module.bits
            match = (expected_w == actual_w and expected_a == actual_a)
            status = "✓" if match else "✗"
            print(f"  {name}: bits={module.bits}, expected=({expected_w}, {expected_a}) {status}")
            if not match:
                print(f"    WARNING: 位宽不匹配！")
            config_idx += 1

print("\n" + "="*80)
print("测试完成")
print("="*80)



