#!/usr/bin/env python3
"""
测试从checkpoint加载的Search OLM编码的故障注入效果

对比：
1. Baseline（无故障）
2. 标准二进制编码 + 故障注入
3. 格雷码编码 + 故障注入
4. Search OLM编码 + 故障注入（从checkpoint加载）
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


def load_search_olm_from_checkpoint(checkpoint_path):
    """
    从checkpoint中加载Search OLM映射
    
    Returns:
        search_olm_mappings: Dict[str, Dict[int, int]] - {layer_name: {value: code}}
        search_olm_code_to_value: Dict[str, Dict[int, int]] - {layer_name: {code: value}}
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    search_olm_mappings = checkpoint.get('search_olm_mappings', {})
    search_olm_code_to_value = checkpoint.get('search_olm_code_to_value', {})
    
    # 确保键是字符串类型（JSON加载后可能是字符串）
    if search_olm_mappings:
        search_olm_mappings = {
            str(layer_name): {
                int(k): int(v) for k, v in value_to_code.items()
            }
            for layer_name, value_to_code in search_olm_mappings.items()
        }
    
    if search_olm_code_to_value:
        search_olm_code_to_value = {
            str(layer_name): {
                int(k): int(v) for k, v in code_to_value.items()
            }
            for layer_name, code_to_value in search_olm_code_to_value.items()
        }
    
    return search_olm_mappings, search_olm_code_to_value


def main():
    parser = argparse.ArgumentParser(description='Test Search OLM encoding from checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit error rate')
    parser.add_argument('--layers', type=str, nargs='+', default=None, 
                        help='Layers to test (default: all layers with OLM mappings)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    
    # 从checkpoint加载Search OLM映射
    print("="*80)
    print("正在从checkpoint加载Search OLM映射...")
    print("="*80)
    search_olm_mappings, search_olm_code_to_value = load_search_olm_from_checkpoint(args.ckpt)
    
    if not search_olm_mappings:
        print("⚠️  警告: checkpoint中没有找到Search OLM映射！")
        print("   请确保checkpoint是从使用Search OLM训练的模型中保存的。")
        return
    
    print(f"✅ 成功加载 {len(search_olm_mappings)} 个层的OLM映射:")
    for layer_name in search_olm_mappings.keys():
        mapping_size = len(search_olm_mappings[layer_name])
        print(f"   - {layer_name}: {mapping_size} 个映射")
    print()
    
    # 确定要测试的层
    if args.layers:
        test_layers = args.layers
        # 检查所有请求的层是否都有映射
        missing_layers = [layer for layer in test_layers if layer not in search_olm_mappings]
        if missing_layers:
            print(f"⚠️  警告: 以下层没有OLM映射: {', '.join(missing_layers)}")
            test_layers = [layer for layer in test_layers if layer in search_olm_mappings]
    else:
        # 测试所有有映射的层
        test_layers = list(search_olm_mappings.keys())
        print(f"将测试所有 {len(test_layers)} 个有OLM映射的层")
    
    if not test_layers:
        print("❌ 没有可测试的层！")
        return
    
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
        # 如果没有提供bit_width_config，使用target_bits的最大值（对于动态位宽训练模型）
        from util.mpq import switch_bit_width
        target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
        max_bit = max(target_bits) if target_bits else 6
        print(f"未提供bit_width_config，使用target_bits的最大值: {max_bit}-bit")
        print(f"注意: fixed_bits层（features.0, classifier.6）将保持8-bit")
        switch_bit_width(model, quan_scheduler=config.quan, wbit=max_bit, abits=max_bit)
    
    # 加载checkpoint（只加载模型权重，不加载OLM映射，因为我们手动处理）
    load_checkpoint(model, args.ckpt, model_device=device, lean=True)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 获取验证集大小
    total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * test_loader.batch_size
    
    print("="*80)
    print("Search OLM编码故障注入测试")
    print("="*80)
    print(f"测试层: {', '.join(test_layers)}")
    print(f"BER: {args.ber}")
    print(f"验证集大小: {total_samples} 样本")
    print()
    
    # Test 1: Baseline（无故障）
    print("Test 1: Baseline (无故障注入)")
    accuracy_baseline = evaluate_model(model, test_loader, device)
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
    accuracy_binary = evaluate_model(model, test_loader, device)
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
        gray_code_layers=test_layers,
        enable_statistics=True
    )
    injector_gray.enable()
    accuracy_gray = evaluate_model(model, test_loader, device)
    injector_gray.disable()
    print(f"准确率: {accuracy_gray:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_gray:.2f}%")
    print(f"相对二进制改进: {accuracy_gray - accuracy_binary:.2f}%")
    print()
    
    # Test 4: Search OLM编码
    print("Test 4: Search OLM编码 + 故障注入")
    try:
        # 只使用测试层的映射
        test_olm_mappings = {layer: search_olm_mappings[layer] for layer in test_layers}
        
        injector_olm = FaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            olm_layers=test_olm_mappings,
            enable_statistics=True
        )
        injector_olm.enable()
        accuracy_olm = evaluate_model(model, test_loader, device)
        injector_olm.disable()
        
        print(f"准确率: {accuracy_olm:.2f}%")
        print(f"相对Baseline下降: {accuracy_baseline - accuracy_olm:.2f}%")
        print(f"相对二进制改进: {accuracy_olm - accuracy_binary:.2f}%")
        print(f"相对格雷码改进: {accuracy_olm - accuracy_gray:.2f}%")
        print()
        
        # 打印故障统计信息
        print("故障注入统计信息:")
        injector_olm.print_flip_statistics()
        print()
        
    except Exception as e:
        print(f"  ❌ Search OLM编码测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 汇总结果
    print("="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"Baseline准确率:        {accuracy_baseline:.2f}%")
    print(f"二进制编码准确率:      {accuracy_binary:.2f}%  (下降 {accuracy_baseline - accuracy_binary:.2f}%)")
    print(f"格雷码编码准确率:      {accuracy_gray:.2f}%  (下降 {accuracy_baseline - accuracy_gray:.2f}%)")
    if 'accuracy_olm' in locals():
        print(f"Search OLM编码准确率:   {accuracy_olm:.2f}%  (下降 {accuracy_baseline - accuracy_olm:.2f}%)")
        print()
        print("改进幅度:")
        print(f"  Search OLM vs 二进制:  {accuracy_olm - accuracy_binary:+.2f}%")
        print(f"  Search OLM vs 格雷码:  {accuracy_olm - accuracy_gray:+.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()

