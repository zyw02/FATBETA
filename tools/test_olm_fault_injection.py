#!/usr/bin/env python3
"""
测试OLM编码的故障注入效果

对比：
1. Baseline（无故障）
2. 标准二进制编码 + 故障注入
3. 格雷码编码 + 故障注入
4. OLM编码 + 故障注入
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.olm_encoder import create_olm_encoder


def evaluate_model(model, dataloader, device, max_samples=None):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if max_samples is not None and total >= max_samples:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test OLM encoding fault injection')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit error rate')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for evaluation')
    parser.add_argument('--layer', type=str, default='features.0', help='Layer to test')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 加载配置
    original_argv = sys.argv
    sys.argv = [sys.argv[0], '--config', args.config]
    config = get_config()
    sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    
    # 加载checkpoint
    load_checkpoint(model, args.ckpt, device=device)
    
    # 准备数据
    _, test_loader = init_dataloader(config, distributed=False)
    
    print("="*80)
    print("OLM编码故障注入测试")
    print("="*80)
    print(f"测试层: {args.layer}")
    print(f"BER: {args.ber}")
    print(f"样本数: {args.num_samples}")
    print()
    
    # Test 1: Baseline（无故障）
    print("Test 1: Baseline (无故障注入)")
    accuracy_baseline = evaluate_model(model, test_loader, device, max_samples=args.num_samples)
    print(f"准确率: {accuracy_baseline:.2f}%")
    print()
    
    # Test 2: 标准二进制编码
    print("Test 2: 标准二进制编码 + 故障注入")
    injector_binary = FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True
    )
    injector_binary.enable()
    accuracy_binary = evaluate_model(model, test_loader, device, max_samples=args.num_samples)
    injector_binary.disable()
    print(f"准确率: {accuracy_binary:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_binary:.2f}%")
    print()
    
    # Test 3: 格雷码编码
    print("Test 3: 格雷码编码 + 故障注入")
    injector_gray = FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        gray_code_layers=[args.layer],
        enable_statistics=True
    )
    injector_gray.enable()
    accuracy_gray = evaluate_model(model, test_loader, device, max_samples=args.num_samples)
    injector_gray.disable()
    print(f"准确率: {accuracy_gray:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_gray:.2f}%")
    print(f"相对二进制改进: {accuracy_gray - accuracy_binary:.2f}%")
    print()
    
    # Test 4: OLM编码
    print("Test 4: OLM编码 + 故障注入")
    print("  正在生成OLM映射...")
    try:
        value_to_code, code_to_value, lrobust = create_olm_encoder(
            model, args.layer, method='greedy', num_samples=1000
        )
        print(f"  LRobust值: {lrobust:.4f}")
        print(f"  映射表大小: {len(value_to_code)}")
        
        injector_olm = FaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            olm_layers={args.layer: value_to_code},
            enable_statistics=True
        )
        injector_olm.enable()
        accuracy_olm = evaluate_model(model, test_loader, device, max_samples=args.num_samples)
        injector_olm.disable()
        print(f"准确率: {accuracy_olm:.2f}%")
        print(f"相对Baseline下降: {accuracy_baseline - accuracy_olm:.2f}%")
        print(f"相对二进制改进: {accuracy_olm - accuracy_binary:.2f}%")
        print(f"相对格雷码改进: {accuracy_olm - accuracy_gray:.2f}%")
    except Exception as e:
        print(f"  OLM编码失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*80)
    print("测试完成")
    print("="*80)


if __name__ == '__main__':
    main()

