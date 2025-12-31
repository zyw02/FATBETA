#!/usr/bin/env python
"""
快速测试 Gradient Statistics Restorer

使用方法:
1. 修改下面的 CHECKPOINT_PATH 为你的stage1 checkpoint路径
2. 运行: python quick_test_gradient_restorer.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).parent
sys.path.insert(0, str(_project_root))

import torch
from model.model import create_model
from util.data_loader import init_dataloader
from util.config import get_config
from util.fault_injector import FaultInjector
from util.gradient_statistics_restorer import create_gradient_statistics_restorer
from quan.utils import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.utils import set_global_seed

# ========== 配置区域 ==========
# 请修改为你的实际路径
CONFIG_FILE = "configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
# 默认使用stage1的checkpoint，如果不存在请修改路径
CHECKPOINT_PATH = "training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 测试参数
K = 4.0
KERNEL_SIZE = 3
NUM_STATISTICS_BATCHES = 20  # 快速测试用较少batch
BER_VALUES = [1e-4, 1e-3, 1e-2, 2e-2]  # 快速测试用较少BER值
# =============================


def main():
    print("="*60)
    print("Gradient Statistics Restorer 快速测试")
    print("="*60)
    
    # 检查checkpoint路径
    if not CHECKPOINT_PATH:
        print("\n错误: 请先设置 CHECKPOINT_PATH 变量")
        print("例如: CHECKPOINT_PATH = 'training/alexnet_cifar10_sensitive_stage1/best.pth.tar'")
        return
    
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"\n错误: Checkpoint文件不存在: {CHECKPOINT_PATH}")
        print("请检查路径是否正确")
        return
    
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        print(f"\n错误: 配置文件不存在: {CONFIG_FILE}")
        return
    
    set_global_seed(seed=42)
    device = torch.device(DEVICE)
    
    print(f"\n配置:")
    print(f"  配置文件: {CONFIG_FILE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  设备: {DEVICE}")
    print(f"  k={K}, kernel_size={KERNEL_SIZE}")
    print(f"  统计batch数: {NUM_STATISTICS_BATCHES}")
    print(f"  测试BER值: {BER_VALUES}")
    print()
    
    # 加载配置
    print("1. 加载配置...")
    configs = get_config(default_file=CONFIG_FILE)
    
    # 创建模型
    print("2. 创建模型...")
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    model.eval()
    
    # 加载checkpoint
    print(f"3. 加载checkpoint: {CHECKPOINT_PATH}...")
    load_checkpoint(model, str(checkpoint_path))
    print("   ✓ 模型加载成功")
    
    # 创建数据加载器
    print("4. 创建数据加载器...")
    train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, configs.arch)
    print(f"   ✓ 测试集大小: {len(test_loader.dataset)}")
    
    # 创建fault injector
    print("5. 创建fault injector...")
    fault_injector = FaultInjector(
        model=model,
        mode="ber",
        ber=1e-3,
        device=device,
        enable_in_training=False,
        enable_in_inference=True,
        use_random_flip_in_training=False,
    )
    print("   ✓ Fault injector创建成功")
    
    # 创建gradient statistics restorer
    print(f"6. 创建gradient statistics restorer...")
    print(f"   收集统计信息（{NUM_STATISTICS_BATCHES} batches）...")
    restorer = create_gradient_statistics_restorer(
        model=model,
        data_loader=test_loader,
        k=K,
        kernel_size=KERNEL_SIZE,
        num_statistics_batches=NUM_STATISTICS_BATCHES,
        layer_names=None,  # 应用到所有层
    )
    print(f"   ✓ Restorer创建成功")
    print(f"   统计的层数: {len(restorer.thresholds)}")
    
    # 测试不同BER值
    print("\n" + "="*60)
    print("开始测试不同BER值")
    print("="*60)
    
    results = {}
    baseline_results = {}
    
    for ber in BER_VALUES:
        print(f"\n测试 BER = {ber:.2e}")
        print("-" * 60)
        
        # 设置BER
        fault_injector.set_ber(ber)
        
        # 测试with restorer
        fault_injector.enable()
        restorer.enable()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if (batch_idx + 1) % 20 == 0:
                    acc = 100. * correct / total
                    print(f"  Batch {batch_idx+1}/{len(test_loader)}, Acc: {acc:.2f}%")
        
        restorer_acc = 100. * correct / total
        results[ber] = restorer_acc
        restorer.disable()
        
        # 测试baseline (no restorer)
        fault_injector.enable()
        restorer.disable()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        baseline_acc = 100. * correct / total
        baseline_results[ber] = baseline_acc
        fault_injector.disable()
        
        improvement = restorer_acc - baseline_acc
        print(f"  Baseline: {baseline_acc:.2f}%")
        print(f"  With Restorer: {restorer_acc:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
    
    # 打印总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"{'BER':<15} {'Baseline (%)':<15} {'Restorer (%)':<15} {'Improvement':<15}")
    print("-" * 60)
    for ber in BER_VALUES:
        baseline_acc = baseline_results[ber]
        restorer_acc = results[ber]
        improvement = restorer_acc - baseline_acc
        print(f"{ber:<15.2e} {baseline_acc:<15.2f} {restorer_acc:<15.2f} {improvement:+.2f}%")
    print("="*60)
    print("\n测试完成！")


if __name__ == '__main__':
    main()

