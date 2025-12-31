#!/usr/bin/env python3
"""
测试脚本：检查OLM编码时layer_name和layer_name_for_stats的值
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


def evaluate_model_with_debug(model, dataloader, device, olm_layers_dict, num_batches=1):
    """评估模型准确率，并输出调试信息"""
    model.eval()
    correct = 0
    total = 0
    
    # 创建FaultInjector
    injector = FaultInjector(
        model=model,
        mode='ber',
        ber=0.1,
        device=device,
        enable_in_inference=True,
        seed=42,
        olm_layers=olm_layers_dict,
        enable_statistics=True
    )
    
    print("="*80)
    print("FaultInjector配置检查:")
    print("="*80)
    print(f"olm_layers.keys() = {list(injector.olm_layers.keys())}")
    print(f"olm_code_to_value.keys() = {list(injector.olm_code_to_value.keys())}")
    print()
    
    injector.enable()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            print(f"\n处理batch {batch_idx+1}/{num_batches}...")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    injector.disable()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test OLM layer_name check')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--olm_json', type=str, required=True, help='OLM encoding JSON file')
    parser.add_argument('--layers', type=str, nargs='+', default=None, help='Layers to test')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to process')
    
    args = parser.parse_args()
    
    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载OLM映射
    print("="*80)
    print("加载OLM映射...")
    print("="*80)
    with open(args.olm_json, 'r') as f:
        olm_data = json.load(f)
    
    if 'layer_mappings' in olm_data:
        olm_layers_dict = {}
        for layer_name, layer_data in olm_data['layer_mappings'].items():
            olm_layers_dict[layer_name] = layer_data['value_to_code']
        print(f"从JSON加载了 {len(olm_layers_dict)} 个层的OLM映射:")
        for layer_name in olm_layers_dict.keys():
            print(f"  - {layer_name}")
    else:
        # 旧格式
        olm_layers_dict = {args.layers[0]: olm_data['value_to_code']}
        print(f"从JSON加载了1个层的OLM映射: {args.layers[0]}")
    
    # 确定要测试的层
    if args.layers:
        test_layers = args.layers
    else:
        test_layers = list(olm_layers_dict.keys())
    
    # 只使用测试层的映射
    test_olm_mappings = {layer: olm_layers_dict[layer] for layer in test_layers if layer in olm_layers_dict}
    print(f"\n将测试以下层的OLM编码: {list(test_olm_mappings.keys())}")
    print()
    
    # 创建模型
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    else:
        from util.mpq import switch_bit_width
        target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
        max_bit = max(target_bits) if target_bits else 6
        print(f"未提供bit_width_config，使用target_bits的最大值: {max_bit}-bit")
        switch_bit_width(model, quan_scheduler=config.quan, wbit=max_bit, abits=max_bit)
    
    # 加载checkpoint
    load_checkpoint(model, args.ckpt, model_device=device, lean=True)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 运行测试
    print("="*80)
    print("开始测试（只处理前几个batch以查看调试信息）...")
    print("="*80)
    accuracy = evaluate_model_with_debug(
        model, test_loader, device, test_olm_mappings, num_batches=args.num_batches
    )
    
    print()
    print("="*80)
    print(f"测试完成，准确率: {accuracy:.2f}%")
    print("="*80)
    print("\n请查看上面的[OLM DEBUG]输出，检查:")
    print("  1. layer_name是否为None")
    print("  2. layer_name_for_stats的值")
    print("  3. 它们是否在olm_layers/olm_code_to_value中")


if __name__ == '__main__':
    main()


