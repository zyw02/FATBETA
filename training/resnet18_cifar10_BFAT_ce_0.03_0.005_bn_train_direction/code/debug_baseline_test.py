#!/usr/bin/env python3
"""
Debug baseline模型容错能力测试
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector

def evaluate_model_quick(model, dataloader, device, max_batches=5):
    """快速评估模型准确率（只测试前几个batch）"""
    import torch

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy

def main():
    print("Debug: Baseline ResNet18容错能力测试")

    config_path = "configs/eval/eval_resnet18_cifar10_fault_tolerance_test.yaml"
    ckpt_path = "/root/autodl-tmp/retraining-free-quantization/training/resnet18_cifar10_baseline/resnet18_cifar10_baseline_checkpoint.pth.tar"

    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config_path]
    try:
        config = get_config(default_file=config_path)
    finally:
        sys.argv = original_argv

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)

    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)

    load_checkpoint(model, ckpt_path, model_device=device)

    # 设置dynamic层bits
    max_target_bit = max(config.target_bits) if config.target_bits else 6
    from quan.func import QuanConv2d, QuanLinear
    dynamic_layers_set = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if not hasattr(module, 'fixed_bits') or module.fixed_bits is None:
                module.bits = (max_target_bit, max_target_bit)
                dynamic_layers_set += 1

    print(f"设置了 {dynamic_layers_set} 个dynamic层为{max_target_bit}-bit")

    # 加载数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)

    # 测试baseline
    print("测试baseline...")
    baseline_acc = evaluate_model_quick(model, test_loader, device, max_batches=5)
    print(f"Baseline准确率: {baseline_acc:.2f}%")

    # 测试单个BER值
    ber = 0.01
    print(f"\n测试BER = {ber}...")

    injector = FaultInjector(
        model=model,
        mode='ber',
        ber=ber,
        device=device,
        enable_in_inference=True,
        enable_statistics=True
    )

    print("启用FaultInjector...")
    injector.enable()

    print("评估准确率...")
    fault_acc = evaluate_model_quick(model, test_loader, device, max_batches=5)

    print(f"故障注入后准确率: {fault_acc:.2f}%")
    print(f"相对下降: {baseline_acc - fault_acc:.2f}%")

    injector.disable()
    print("测试完成！")

if __name__ == "__main__":
    main()
