#!/usr/bin/env python3
"""
评估baseline ResNet18模型的容错能力
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
    # 配置路径
    config_path = "configs/training/train_resnet18_cifar10_single_gpu.yaml"
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

    print("开始评估baseline ResNet18模型的容错能力...")
    print("=" * 60)

    # 测试不同BER值 - 对所有层所有bit进行故障注入
    ber_values = [1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1]

    results = []

    for ber in ber_values:
        print(f"\n测试 BER = {ber} (对所有层所有bit注入)")

        # 创建FaultInjector - 默认设置，对所有bit进行注入（包括MSB）
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=ber,
            device=device,
            enable_in_inference=True,
            enable_statistics=True
            # 不设置skip_msb或only_msb，默认对所有bit进行注入
        )

        injector.enable()

        # 评估准确率
        accuracy = evaluate_model(model, test_loader, device)

        # 获取统计信息
        stats = injector.get_flip_statistics()
        if stats:
            total_flipped = sum(layer_stats['flipped_bits'] for layer_stats in stats.values())
            total_bits = sum(layer_stats['total_bits'] for layer_stats in stats.values())
            flip_ratio = 100.0 * total_flipped / total_bits if total_bits > 0 else 0
            print(f"Top-1 Accuracy = {accuracy:.2f}%")
            print(f"实际翻转比例: {flip_ratio:.3f}% (期望BER: {ber*100:.3f}%)")
        else:
            print(f"Top-1 Accuracy = {accuracy:.2f}%")
            print("没有统计信息")

        results.append({
            'ber': ber,
            'accuracy': accuracy,
            'flip_ratio': flip_ratio if 'flip_ratio' in locals() else None
        })

        injector.disable()

    print("\n" + "=" * 60)
    print("容错能力评估结果总结")
    print("=" * 60)
    print(f"模型: ResNet18 Baseline")
    print(f"评估模式: 所有层所有bit故障注入")
    print()

    for result in results:
        ber = result['ber']
        acc = result['accuracy']
        flip_ratio = result.get('flip_ratio', 'N/A')
        print(f"BER {ber:6.1e}: 准确率 {acc:5.2f}%, 翻转比例 {flip_ratio}")

    print("\n评估完成！")

if __name__ == "__main__":
    main()
