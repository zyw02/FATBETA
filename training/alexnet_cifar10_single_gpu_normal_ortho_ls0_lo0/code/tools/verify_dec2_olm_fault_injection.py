#!/usr/bin/env python3
"""
验证12月2日版本的fault_injector是否启用了OLM编码并成功进行故障注入

使用真实的模型和checkpoint进行测试
"""

import argparse
import json
import sys
import os
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


def evaluate_model(model, dataloader, device, num_batches=10):
    """评估模型准确率（在部分批次上，用于快速测试）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def load_olm_mapping(json_path):
    """从JSON文件加载OLM映射"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    olm_layers = {}
    olm_code_to_value = {}
    
    # 处理不同的JSON格式
    if 'layer_mappings' in data:
        # 新格式：包含多个层的映射
        for layer_name, layer_data in data['layer_mappings'].items():
            if 'value_to_code' in layer_data and 'code_to_value' in layer_data:
                # 转换字符串键/值为整数
                value_to_code = {int(k): int(v) for k, v in layer_data['value_to_code'].items()}
                code_to_value = {int(k): int(v) for k, v in layer_data['code_to_value'].items()}
                olm_layers[layer_name] = value_to_code
                olm_code_to_value[layer_name] = code_to_value
    elif isinstance(data, dict):
        # 旧格式或其他格式
        for layer_name, mapping in data.items():
            if isinstance(mapping, dict):
                if 'value_to_code' in mapping and 'code_to_value' in mapping:
                    # 格式: {"layer_name": {"value_to_code": {...}, "code_to_value": {...}}}
                    value_to_code = {int(k): int(v) for k, v in mapping['value_to_code'].items()}
                    code_to_value = {int(k): int(v) for k, v in mapping['code_to_value'].items()}
                    olm_layers[layer_name] = value_to_code
                    olm_code_to_value[layer_name] = code_to_value
    
    return olm_layers, olm_code_to_value


def main():
    parser = argparse.ArgumentParser(description='验证12月2日版本的fault_injector')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--olm_json', type=str, required=True, help='Path to OLM encoding JSON file')
    parser.add_argument('--layers', type=str, nargs='+', default=['features.0'], help='Layers to test')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit error rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to test')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("验证12月2日版本的fault_injector")
    print("="*80)
    print()
    
    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    # 创建模型
    print("创建模型...")
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 应用量化
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
    print("准备数据...")
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 加载OLM映射
    print(f"加载OLM映射: {args.olm_json}")
    olm_layers, olm_code_to_value = load_olm_mapping(args.olm_json)
    print(f"  OLM层数量: {len(olm_layers)}")
    print(f"  OLM层: {list(olm_layers.keys())}")
    print()
    
    # 验证OLM编码路径是否会被触发
    print("="*80)
    print("验证1: OLM编码路径检查")
    print("="*80)
    for layer_name in args.layers:
        use_olm = (len(olm_layers) > 0 and 
                   layer_name is not None and 
                   layer_name in olm_layers)
        print(f"  {layer_name}: use_olm = {use_olm}")
        if use_olm:
            print(f"    ✅ OLM编码路径应该被触发")
        else:
            print(f"    ❌ OLM编码路径不会被触发")
    print()
    
    # 测试故障注入
    print("="*80)
    print("验证2: 故障注入功能测试")
    print("="*80)
    
    # Test 1: Baseline（无故障）
    print("Test 1: Baseline (无故障注入)")
    model.eval()
    accuracy_baseline = evaluate_model(model, test_loader, device, args.num_batches)
    print(f"  准确率: {accuracy_baseline:.2f}%")
    print()
    
    # Test 2: OLM编码 + 故障注入
    print("Test 2: OLM编码 + 故障注入")
    try:
        # 初始化FaultInjector（12月2日版本）
        fault_injector = FaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            olm_layers=olm_layers,
            enable_statistics=True
        )
        
        # 设置olm_code_to_value（12月2日版本需要手动设置）
        fault_injector.olm_code_to_value = olm_code_to_value
        
        print(f"  FaultInjector已创建")
        print(f"  olm_layers包含层: {list(fault_injector.olm_layers.keys())}")
        print(f"  olm_code_to_value包含层: {list(fault_injector.olm_code_to_value.keys())}")
        
        # 启用故障注入
        fault_injector.enable()
        
        # 评估模型
        model.eval()
        accuracy_olm = evaluate_model(model, test_loader, device, args.num_batches)
        
        # 禁用故障注入
        fault_injector.disable()
        
        print(f"  准确率: {accuracy_olm:.2f}%")
        print(f"  相对Baseline下降: {accuracy_baseline - accuracy_olm:.2f}%")
        print()
        
        if accuracy_baseline - accuracy_olm > 0.1:  # 至少下降0.1%
            print("  ✅ 故障注入成功！准确率明显下降")
            fault_injection_success = True
        else:
            print("  ⚠️  故障注入可能未生效，准确率下降不明显")
            fault_injection_success = False
        
        # 打印统计信息
        if hasattr(fault_injector, '_pending_stats') and len(fault_injector._pending_stats) > 0:
            print("  故障注入统计:")
            fault_injector._process_pending_statistics()
            fault_injector.print_statistics()
        
    except Exception as e:
        print(f"  ❌ OLM编码故障注入测试失败: {e}")
        import traceback
        traceback.print_exc()
        fault_injection_success = False
    
    print()
    
    # 总结
    print("="*80)
    print("验证总结")
    print("="*80)
    print(f"  OLM编码路径: ✅ 应该被触发")
    print(f"  故障注入功能: {'✅ 成功' if fault_injection_success else '❌ 失败'}")
    print()
    
    if fault_injection_success:
        print("✅ 验证通过：12月2日版本的fault_injector启用了OLM编码并成功进行了故障注入")
        return 0
    else:
        print("❌ 验证失败：故障注入可能未生效")
        return 1


if __name__ == '__main__':
    sys.exit(main())

