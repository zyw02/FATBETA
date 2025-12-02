#!/usr/bin/env python3
"""
训练OLM编码器

针对指定模型的指定层，收集量化权重分布，优化编码映射，并保存结果。
可选择自动运行故障注入测试，对比baseline、二进制、格雷码、OLM的效果。

使用方法：
    # 基本用法（单个层）
    python tools/train_olm_encoder.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --layer features.0 \
        --output olm_encoding_features_0.json \
        --method both
    
    # 多个层（用逗号分隔）
    python tools/train_olm_encoder.py \
        --config ... \
        --ckpt ... \
        --bit_width_config ... \
        --layer "features.0,classifier.1" \
        --output olm_encoding_multi_layer.json \
        --method both \
        --test_fault_injection \
        --test_ber 0.1
    
    注意：classifier.0是Dropout层，不是Linear层。应该使用classifier.1（第一个Linear层）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.data_loader import init_dataloader
from util.olm_encoder import (
    collect_quantized_value_distribution,
    optimize_olm_mapping,
    create_olm_encoder
)


def main():
    parser = argparse.ArgumentParser(description='Train OLM encoder for a specific layer')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--layer', type=str, required=True, 
                       help='Layer name(s) to train OLM for. Multiple layers can be specified with comma, e.g., "features.0,classifier.1"')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--method', type=str, default='both', choices=['greedy', 'simulated_annealing', 'both'],
                       help='Optimization method: greedy, simulated_annealing, or both (try both and choose better)')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples for distribution collection (-1 for all)')
    parser.add_argument('--max_iterations', type=int, default=3000,
                       help='Max iterations for simulated annealing (default: 3000, recommended: 3000-5000 for 8-bit)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_fault_injection', action='store_true',
                       help='After training, automatically test fault injection with baseline, binary, gray code, and OLM')
    parser.add_argument('--test_ber', type=float, default=0.1,
                       help='BER for fault injection test (default: 0.1, i.e., 1e-1)')
    
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
    
    # 解析层名称（支持多个层，用逗号分隔）
    layer_names = [name.strip() for name in args.layer.split(',')]
    
    print("="*80)
    print("OLM编码器训练")
    print("="*80)
    print(f"配置: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"目标层: {', '.join(layer_names)} ({len(layer_names)} 个层)")
    print(f"优化方法: {args.method}")
    print(f"采样数量: {args.num_samples if args.num_samples > 0 else '全部'}")
    print()
    
    # 创建模型
    print("步骤1: 创建模型...")
    dataset = getattr(config.dataloader, 'dataset', 'cifar10')
    pre_trained = getattr(config, 'pre_trained', False)
    model = create_model(config.arch, dataset=dataset, pre_trained=pre_trained)
    model = model.to(device)
    
    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载bit-width配置
    if args.bit_width_config:
        print(f"步骤2: 加载bit-width配置: {args.bit_width_config}")
        # 检查文件是否存在
        if not os.path.exists(args.bit_width_config):
            raise FileNotFoundError(
                f"Bit-width config file not found: {args.bit_width_config}\n"
                f"Please provide a valid path to the bit-width configuration JSON file."
            )
        setup_model_with_bit_width_config(
            model, 
            json_path=args.bit_width_config, 
            config_index=0,
            verbose=True
        )
    
    # 加载checkpoint
    print(f"步骤3: 加载checkpoint: {args.ckpt}")
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 确保模型处于eval模式
    model.eval()
    
    # 验证所有层是否存在
    modules = dict(model.named_modules())
    for layer_name in layer_names:
        if layer_name not in modules:
            raise ValueError(f"Layer {layer_name} not found in model")
        target_module = modules[layer_name]
        if not isinstance(target_module, (torch.nn.Conv2d, torch.nn.Linear)):
            raise ValueError(f"Layer {layer_name} is not a Conv2d or Linear layer")
    
    # 为每个层训练OLM编码
    all_olm_mappings = {}  # {layer_name: {'value_to_code': ..., 'code_to_value': ..., ...}}
    
    for layer_idx, layer_name in enumerate(layer_names):
        print("="*80)
        print(f"处理层 {layer_idx+1}/{len(layer_names)}: {layer_name}")
        print("="*80)
        
        target_module = modules[layer_name]
        
        # 检查是否有量化配置（支持 bits 和 fixed_bits）
        wbits = None
        if hasattr(target_module, 'bits') and target_module.bits is not None:
            wbits = target_module.bits[0] if isinstance(target_module.bits, (list, tuple)) else target_module.bits
        elif hasattr(target_module, 'fixed_bits') and target_module.fixed_bits is not None:
            wbits = target_module.fixed_bits[0] if isinstance(target_module.fixed_bits, (list, tuple)) else target_module.fixed_bits
        
        if wbits is None:
            raise ValueError(f"Layer {layer_name} has no bit-width configuration (neither bits nor fixed_bits)")
        
        if isinstance(wbits, torch.Tensor):
            wbits = int(wbits.item())
        else:
            wbits = int(wbits)
        
        print(f"  层位宽: {wbits} bits")
        print()
        
        # 收集量化值分布
        print(f"步骤4.{layer_idx+1}: 收集量化值分布 ({layer_name})...")
        try:
            distribution = collect_quantized_value_distribution(
                model, layer_name, num_samples=args.num_samples
            )
            print(f"  收集到 {len(distribution)} 个不同的量化值")
            print(f"  总权重数量: {sum(distribution.values())}")
            
            # 显示最常见的量化值
            sorted_dist = sorted(distribution.items(), key=lambda x: -x[1])
            print(f"  最常见的10个量化值:")
            for val, freq in sorted_dist[:10]:
                print(f"    值 {val:4d}: 出现 {freq:8d} 次 ({100*freq/sum(distribution.values()):.2f}%)")
            print()
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 优化编码映射
        print(f"步骤5.{layer_idx+1}: 优化编码映射 ({layer_name}, 方法: {args.method})...")
        try:
            if args.method == 'both':
                # 尝试两种方法，选择更好的
                print("  尝试贪婪搜索...")
                import time
                start_time = time.time()
                value_to_code_greedy, code_to_value_greedy, lrobust_greedy = optimize_olm_mapping(
                    distribution, wbits, method='greedy'
                )
                time_greedy = time.time() - start_time
                print(f"    完成! LRobust: {lrobust_greedy:.4f}, 耗时: {time_greedy:.2f}秒")
                
                print("  尝试模拟退火...")
                start_time = time.time()
                value_to_code_sa, code_to_value_sa, lrobust_sa = optimize_olm_mapping(
                    distribution, wbits, method='simulated_annealing',
                    max_iterations=args.max_iterations
                )
                time_sa = time.time() - start_time
                print(f"    完成! LRobust: {lrobust_sa:.4f}, 耗时: {time_sa:.2f}秒")
                
                # 选择更好的（LRobust更小的）
                if lrobust_sa < lrobust_greedy:
                    print(f"  → 选择模拟退火的结果 (LRobust更小: {lrobust_sa:.4f} < {lrobust_greedy:.4f})")
                    value_to_code, code_to_value, lrobust = value_to_code_sa, code_to_value_sa, lrobust_sa
                    selected_method = 'simulated_annealing'
                else:
                    print(f"  → 选择贪婪搜索的结果 (LRobust更小: {lrobust_greedy:.4f} < {lrobust_sa:.4f})")
                    value_to_code, code_to_value, lrobust = value_to_code_greedy, code_to_value_greedy, lrobust_greedy
                    selected_method = 'greedy'
                print()
            elif args.method == 'greedy':
                value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                    distribution, wbits, method='greedy'
                )
                selected_method = 'greedy'
            else:
                value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                    distribution, wbits, method='simulated_annealing',
                    max_iterations=args.max_iterations
                )
                selected_method = 'simulated_annealing'
            
            print(f"  优化完成!")
            print(f"  使用方法: {selected_method}")
            print(f"  LRobust值: {lrobust:.4f}")
            print(f"  映射表大小: {len(value_to_code)}")
            print()
            
            # 显示一些映射示例
            print("  映射示例 (前10个):")
            sorted_mappings = sorted(value_to_code.items(), key=lambda x: -distribution.get(x[0], 0))
            for val, code in sorted_mappings[:10]:
                freq = distribution.get(val, 0)
                print(f"    量化值 {val:4d} (频率: {freq:8d}) → 编码 {code:4d} (二进制: {code:0{wbits}b})")
            print()
            
            # 保存该层的结果
            all_olm_mappings[layer_name] = {
                'bit_width': wbits,
                'method': selected_method,
                'lrobust': lrobust,
                'distribution': distribution,
                'value_to_code': value_to_code,
                'code_to_value': code_to_value,
                'num_values': len(value_to_code),
                'total_weights': sum(distribution.values())
            }
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存所有层的结果
    print("="*80)
    print(f"步骤6: 保存结果到 {args.output}...")
    output_data = {
        'layers': layer_names,
        'num_layers': len(layer_names),
        'layer_mappings': all_olm_mappings
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  保存完成!")
    print()
    
    # 总结
    print("="*80)
    print("训练完成!")
    print("="*80)
    print(f"输出文件: {args.output}")
    print(f"处理的层数: {len(all_olm_mappings)}")
    for layer_name, layer_data in all_olm_mappings.items():
        print(f"  {layer_name}: LRobust={layer_data['lrobust']:.4f}, 映射表大小={layer_data['num_values']}")
    print()
    print("使用方法:")
    print(f"  from util.fault_injector import FaultInjector")
    print(f"  import json")
    print(f"  ")
    print(f"  with open('{args.output}', 'r') as f:")
    print(f"      olm_data = json.load(f)")
    print(f"  ")
    print(f"  # 构建olm_layers字典")
    print(f"  olm_layers = {{}}")
    print(f"  for layer_name in olm_data['layers']:")
    print(f"      olm_layers[layer_name] = olm_data['layer_mappings'][layer_name]['value_to_code']")
    print(f"  ")
    print(f"  injector = FaultInjector(")
    print(f"      model=model,")
    print(f"      mode='ber',")
    print(f"      ber=1e-1,")
    print(f"      olm_layers=olm_layers")
    print(f"  )")
    print()
    
    # 自动故障注入测试
    if args.test_fault_injection:
        print("="*80)
        print("步骤7: 自动故障注入测试")
        print("="*80)
        print(f"测试层: {', '.join(layer_names)}")
        print(f"BER: {args.test_ber}")
        print(f"使用整个测试集")
        print()
        
        # 准备数据
        _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
        
        def evaluate_model(model, dataloader, device):
            """评估模型准确率（使用整个测试集）"""
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
        
        # Test 1: Baseline（无故障）
        print("Test 1: Baseline (无故障注入)")
        accuracy_baseline = evaluate_model(model, test_loader, device)
        print(f"  准确率: {accuracy_baseline:.2f}%")
        print()
        
        # Test 2: 标准二进制编码
        print("Test 2: 标准二进制编码 + 故障注入")
        injector_binary = FaultInjector(
            model=model,
            mode='ber',
            ber=args.test_ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            enable_statistics=True
        )
        injector_binary.enable()
        accuracy_binary = evaluate_model(model, test_loader, device)
        injector_binary.disable()
        print(f"  准确率: {accuracy_binary:.2f}%")
        print(f"  相对Baseline下降: {accuracy_baseline - accuracy_binary:.2f}%")
        print()
        
        # Test 3: 格雷码编码
        print("Test 3: 格雷码编码 + 故障注入")
        injector_gray = FaultInjector(
            model=model,
            mode='ber',
            ber=args.test_ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            gray_code_layers=layer_names,
            enable_statistics=True
        )
        injector_gray.enable()
        accuracy_gray = evaluate_model(model, test_loader, device)
        injector_gray.disable()
        print(f"  准确率: {accuracy_gray:.2f}%")
        print(f"  相对Baseline下降: {accuracy_baseline - accuracy_gray:.2f}%")
        print(f"  相对二进制改进: {accuracy_gray - accuracy_binary:.2f}%")
        print()
        
        # Test 4: OLM编码
        print("Test 4: OLM编码 + 故障注入")
        try:
            # 构建olm_layers字典
            olm_layers_dict = {}
            for layer_name in layer_names:
                if layer_name in all_olm_mappings:
                    olm_layers_dict[layer_name] = all_olm_mappings[layer_name]['value_to_code']
                    print(f"  {layer_name}: OLM映射表大小={len(olm_layers_dict[layer_name])}")
            
            injector_olm = FaultInjector(
                model=model,
                mode='ber',
                ber=args.test_ber,
                device=device,
                enable_in_inference=True,
                seed=args.seed,
                olm_layers=olm_layers_dict,
                enable_statistics=True
            )
            print(f"  FaultInjector已创建，olm_layers包含层: {list(injector_olm.olm_layers.keys())}")
            injector_olm.enable()
            accuracy_olm = evaluate_model(model, test_loader, device)
            injector_olm.disable()
            print(f"  准确率: {accuracy_olm:.2f}%")
            print(f"  相对Baseline下降: {accuracy_baseline - accuracy_olm:.2f}%")
            print(f"  相对二进制改进: {accuracy_olm - accuracy_binary:.2f}%")
            print(f"  相对格雷码改进: {accuracy_olm - accuracy_gray:.2f}%")
        except Exception as e:
            print(f"  OLM编码测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("="*80)
        print("故障注入测试完成!")
        print("="*80)
        print(f"Baseline:        {accuracy_baseline:.2f}%")
        print(f"二进制编码:      {accuracy_binary:.2f}% (下降 {accuracy_baseline - accuracy_binary:.2f}%)")
        print(f"格雷码编码:      {accuracy_gray:.2f}% (改进 {accuracy_gray - accuracy_binary:.2f}%)")
        if 'accuracy_olm' in locals():
            print(f"OLM编码:         {accuracy_olm:.2f}% (改进 {accuracy_olm - accuracy_binary:.2f}%)")
            if accuracy_olm > accuracy_gray:
                print(f"  → OLM编码优于格雷码 {accuracy_olm - accuracy_gray:.2f}%")
            elif accuracy_gray > accuracy_olm:
                print(f"  → 格雷码优于OLM编码 {accuracy_gray - accuracy_olm:.2f}%")
            else:
                print(f"  → OLM编码与格雷码效果相同")
        print()


if __name__ == '__main__':
    main()

