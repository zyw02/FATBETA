#!/usr/bin/env python3
"""
简单测试ResNet18模型的SEU容错能力（跳过MSB注入）
"""

import torch
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

def main():
    # 使用我们配置的YAML
    config_path = "configs/eval/eval_resnet18_cifar10_fault_tolerance_test.yaml"
    ckpt_path = "training/resnet18_cifar10_nude_srqat_ls0/resnet18_cifar10_nude_srqat_ls0_checkpoint.pth.tar"

    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], config_path]
    try:
        config = get_config(default_file=config_path)
    finally:
        sys.argv = original_argv

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
    print(f"target_bits: {config.target_bits}, max_target_bit: {max_target_bit}")

    from quan.func import QuanConv2d, QuanLinear
    dynamic_layers_set = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            # 跳过fixed_bits层（它们已经有fixed_bits设置）
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                continue
            # 直接设置bits属性为最大值
            module.bits = (max_target_bit, max_target_bit)
            dynamic_layers_set += 1

    print(f"已设置 {dynamic_layers_set} 个dynamic层的bits为{max_target_bit}-bit")
    print()

    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)

    print("开始测试...")

    # 测试不同BER值
    ber_values = [1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1]

    for ber in ber_values:
        print(f"\n测试 BER = {ber}")

        # 创建FaultInjector，启用skip_msb
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=ber,
            device=device,
            enable_in_inference=True,
            skip_msb=True,  # 跳过MSB注入
            # only_msb=True,  # 只反转MSB
            enable_statistics=True
        )

        injector.enable()

        # 评估准确率
        accuracy = evaluate_model(model, test_loader, device)

        print(f"BER {ber}: Top-1 Accuracy = {accuracy:.2f}%")

        # 获取统计信息
        stats = injector.get_flip_statistics()
        if stats:
            total_flipped = sum(layer_stats['flipped_bits'] for layer_stats in stats.values())
            total_bits = sum(layer_stats['total_bits'] for layer_stats in stats.values())
            flip_ratio = 100.0 * total_flipped / total_bits if total_bits > 0 else 0
            print(f"  实际翻转比例: {flip_ratio:.3f}% (期望BER: {ber*100:.3f}%)")

        injector.disable()

    print("\n测试完成！")

if __name__ == "__main__":
    main()
