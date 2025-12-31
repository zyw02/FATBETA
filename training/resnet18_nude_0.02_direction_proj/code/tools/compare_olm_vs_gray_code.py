#!/usr/bin/env python3
"""
比较当前OLM编码方案和标准格雷码在BER=1e-1下的准确率
"""

import argparse
import json
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


def evaluate_model(model, dataloader, device):
    """评估模型准确率（在整个验证集上）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='比较OLM编码和格雷码在BER=1e-1下的准确率')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--olm_json', type=str, required=True, help='OLM encoding JSON file path')
    parser.add_argument('--layer', type=str, required=True, help='Layer name to test')
    parser.add_argument('--ber', type=float, default=0.1, help='Bit-error-rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    # 创建模型
    print("创建模型...")
    dataset = config.dataloader.dataset if hasattr(config.dataloader, 'dataset') else config.dataset if hasattr(config, 'dataset') else 'cifar10'
    model = create_model(config.arch, dataset=dataset, pre_trained=getattr(config, 'pre_trained', False))
    model = model.to(device)
    
    # 应用量化
    print("应用量化...")
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        print("加载bit-width配置...")
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    
    # 加载checkpoint
    print("加载checkpoint...")
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 获取验证集大小
    total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * test_loader.batch_size
    
    print("="*80)
    print("OLM编码 vs 格雷码 准确率比较")
    print("="*80)
    print(f"测试层: {args.layer}")
    print(f"BER: {args.ber}")
    print(f"验证集大小: {total_samples} 样本")
    print()
    
    # Test 1: Baseline（无故障）
    print("Test 1: Baseline (无故障注入)")
    model.eval()  # 确保模型处于评估模式
    accuracy_baseline = evaluate_model(model, test_loader, device)
    print(f"  准确率: {accuracy_baseline:.2f}%")
    if accuracy_baseline < 85:
        print(f"  ⚠️  警告：Baseline准确率偏低，可能模型加载或评估有问题")
    print()
    
    # Test 2: 标准格雷码编码
    print("Test 2: 标准格雷码编码 + 故障注入")
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
    accuracy_gray = evaluate_model(model, test_loader, device)
    injector_gray.disable()
    print(f"  准确率: {accuracy_gray:.2f}%")
    print(f"  相对Baseline下降: {accuracy_baseline - accuracy_gray:.2f}%")
    print()
    
    # Test 3: OLM编码
    print("Test 3: OLM编码 + 故障注入")
    try:
        # 加载OLM映射
        with open(args.olm_json, 'r') as f:
            olm_data = json.load(f)
        
        # 支持两种格式：
        # 1. 新格式：{"layer_mappings": {"layer_name": {...}}}
        # 2. 旧格式：{"layer_name": "...", "value_to_code": {...}, ...}
        if 'layer_mappings' in olm_data:
            # 新格式
            layer_mappings = olm_data.get('layer_mappings', {})
            if args.layer not in layer_mappings:
                print(f"  ❌ 错误：层 {args.layer} 不在OLM映射文件中")
                print(f"  可用层: {list(layer_mappings.keys())}")
                return
            mapping = layer_mappings[args.layer]
        elif 'value_to_code' in olm_data:
            # 旧格式（单层直接映射）
            mapping = olm_data
            # 检查层名是否匹配
            if 'layer_name' in olm_data and olm_data['layer_name'] != args.layer:
                print(f"  ⚠️  警告：JSON文件中的层名 ({olm_data.get('layer_name', 'N/A')}) 与指定层名 ({args.layer}) 不匹配")
                print(f"  → 继续使用JSON文件中的映射")
        else:
            print(f"  ❌ 错误：无法识别OLM映射文件格式")
            print(f"  文件应包含 'layer_mappings' 或 'value_to_code' 字段")
            return
        value_to_code = {
            int(k): int(v) 
            for k, v in mapping['value_to_code'].items()
        }
        code_to_value = {
            int(k): int(v)
            for k, v in mapping['code_to_value'].items()
        }
        
        print(f"  OLM映射信息:")
        print(f"    位宽: {mapping.get('bit_width', 'N/A')}")
        print(f"    方法: {mapping.get('method', 'N/A')}")
        print(f"    LRobust: {mapping.get('lrobust', 'N/A')}")
        print(f"    使用敏感度: {mapping.get('use_sensitivity', False)}")
        print(f"    映射表大小: {len(value_to_code)}")
        
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
        
        # 设置code_to_value
        injector_olm.olm_code_to_value = {args.layer: code_to_value}
        
        injector_olm.enable()
        accuracy_olm = evaluate_model(model, test_loader, device)
        injector_olm.disable()
        
        print(f"  准确率: {accuracy_olm:.2f}%")
        print(f"  相对Baseline下降: {accuracy_baseline - accuracy_olm:.2f}%")
        print(f"  相对格雷码改进: {accuracy_olm - accuracy_gray:.2f}%")
        print(f"  相对格雷码改进百分比: {((accuracy_olm - accuracy_gray) / max(accuracy_gray, 1e-6) * 100):.2f}%")
        
    except Exception as e:
        print(f"  ❌ OLM编码失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*80)
    print("比较总结")
    print("="*80)
    print(f"Baseline准确率: {accuracy_baseline:.2f}%")
    print(f"格雷码准确率: {accuracy_gray:.2f}% (下降 {accuracy_baseline - accuracy_gray:.2f}%)")
    if 'accuracy_olm' in locals():
        print(f"OLM编码准确率: {accuracy_olm:.2f}% (下降 {accuracy_baseline - accuracy_olm:.2f}%)")
        print(f"OLM相对格雷码改进: {accuracy_olm - accuracy_gray:.2f}% ({((accuracy_olm - accuracy_gray) / max(accuracy_gray, 1e-6) * 100):.2f}%)")
    print("="*80)


if __name__ == '__main__':
    main()

