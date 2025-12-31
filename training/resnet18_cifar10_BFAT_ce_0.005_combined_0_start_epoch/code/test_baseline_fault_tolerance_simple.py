#!/usr/bin/env python3
"""
简单评估baseline ResNet18模型的容错能力
对所有层所有bit启用故障注入
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

def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    import torch

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy

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
    print("评估Baseline ResNet18模型容错能力")
    print("对所有层所有bit启用故障注入")
    print("=" * 50)

    # 使用我们之前创建的eval配置
    config_path = "configs/eval/eval_resnet18_cifar10_fault_tolerance_test.yaml"
    ckpt_path = "/root/autodl-tmp/retraining-free-quantization/training/resnet18_cifar10_baseline/resnet18_cifar10_baseline_checkpoint.pth.tar"

    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config_path]
    try:
        config = get_config(default_file=config_path)
    finally:
        sys.argv = original_argv

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)

    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)

    # 加载checkpoint
    load_checkpoint(model, ckpt_path, model_device=device)

    # 设置dynamic层的bits为target_bits的最大值，确保所有层都能被故障注入
    print("设置dynamic层的bits为target_bits的最大值...")
    max_target_bit = max(config.target_bits) if config.target_bits else 6

    from quan.func import QuanConv2d, QuanLinear
    dynamic_layers_set = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if not hasattr(module, 'fixed_bits') or module.fixed_bits is None:
                module.bits = (max_target_bit, max_target_bit)
                dynamic_layers_set += 1

    print(f"已设置 {dynamic_layers_set} 个dynamic层的bits为{max_target_bit}-bit")

    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)

    # 首先测试baseline（无故障）
    print("\n测试Baseline（无故障）...")
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"Baseline准确率: {baseline_acc:.2f}%")

    # 测试不同BER值
    ber_values = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]  # 测试多个BER值

    print("\n开始容错能力测试...")
    print("BER值\tTop-1准确率\t相对下降")
    print("-" * 40)

    for ber in ber_values:
        # 创建FaultInjector - 对所有bit进行注入（包括MSB）
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=ber,
            device=device,
            enable_in_inference=True,
            enable_statistics=True
        )

        injector.enable()

        # 评估准确率（快速测试，只用前5个batch）
        accuracy = evaluate_model_quick(model, test_loader, device, max_batches=5)
        relative_drop = baseline_acc - accuracy

        print(f"{ber:.1e}\t{accuracy:5.2f}%\t\t{relative_drop:5.2f}%")

        injector.disable()

    print("\n评估完成！")

if __name__ == "__main__":
    main()
