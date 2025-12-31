#!/usr/bin/env python3
"""
比较Learnable OLM与传统OLM在容错能力上的优劣

对比：
1. Baseline（无故障）
2. 传统OLM编码 + 故障注入
3. Learnable OLM编码 + 故障注入

使用方法：
    python tools/test_learnable_olm_vs_traditional_olm.py \
        --config configs/training/train_alexnet_cifar10_learnable_olm_fat.yaml \
        --ckpt training/alexnet_cifar10_learnable_olm_fat/alexnet_cifar10_learnable_olm_fat_checkpoint.pth.tar \
        --ber 1e-2 \
        --layers features.0,features.3,features.6,classifier.0,classifier.3,classifier.6 \
        --log_file debug.log
    
    或者使用命令行重定向（更简单）：
    python tools/test_learnable_olm_vs_traditional_olm.py ... 2> debug.log
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from quan.func import QuanConv2d, QuanLinear
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.learnable_olm import LearnableOLMManager
from util.olm_encoder import create_olm_encoder
from util.qat import get_quantized_layers


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


def load_learnable_olm_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    layer_names: List[str],
    device: torch.device,
    config
) -> Optional[LearnableOLMManager]:
    """
    从checkpoint加载Learnable OLM编码器
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 量化模型
        layer_names: 层名称列表
        device: 设备
        config: 配置对象
    
    Returns:
        LearnableOLMManager实例，如果加载失败则返回None
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取bit-width配置
        # 注意：get_quantized_layers返回的是(layers, bns)，其中layers是模块列表，不包含名称
        # 我们需要通过named_modules来获取层名称
        bit_widths = {}
        q_layers, _ = get_quantized_layers(model)
        q_layers_set = set(q_layers)  # 转换为set以便快速查找
        
        # 构建层名称映射
        layer_name_map = {}
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                layer_name_map[module] = name
        
        for layer_name in layer_names:
            # 通过named_modules查找对应的模块
            module = None
            for name, mod in model.named_modules():
                if name == layer_name and isinstance(mod, (QuanConv2d, QuanLinear)):
                    module = mod
                    break
            
            if module is not None:
                # 获取位宽
                wbits = None
                if hasattr(module, 'bits') and module.bits is not None:
                    wbits = module.bits
                    if isinstance(wbits, (list, tuple)):
                        wbits = wbits[0] if len(wbits) > 0 else 8
                    if isinstance(wbits, torch.Tensor):
                        wbits = wbits.item() if wbits.numel() == 1 else int(wbits[0].item())
                    bit_widths[layer_name] = int(wbits)
                elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    wbits = module.fixed_bits
                    if isinstance(wbits, (list, tuple)):
                        wbits = wbits[0] if len(wbits) > 0 else 8
                    if isinstance(wbits, torch.Tensor):
                        wbits = wbits.item() if wbits.numel() == 1 else int(wbits[0].item())
                    bit_widths[layer_name] = int(wbits)
                else:
                    bit_widths[layer_name] = 8  # 默认8bit
            else:
                # 如果找不到，使用默认值
                bit_widths[layer_name] = 8
        
        # 创建LearnableOLMManager
        learnable_olm_config = getattr(config, 'learnable_olm', None)
        if learnable_olm_config is None:
            print("  ⚠️  配置文件中没有learnable_olm配置，使用默认值")
            init_method = 'identity'
            temperature = 1.0
            use_straight_through = True
        else:
            init_method = getattr(learnable_olm_config, 'init_method', 'identity')
            temperature = getattr(learnable_olm_config, 'temperature', 1.0)
            use_straight_through = getattr(learnable_olm_config, 'use_straight_through', True)
        
        olm_manager = LearnableOLMManager(
            model=model,
            layer_names=layer_names,
            bit_widths=bit_widths,
            device=device,
            init_method=init_method,
            temperature=temperature,
            use_straight_through=use_straight_through,
        )
        
        # 尝试从checkpoint中加载编码器参数
        # 方法1: 从checkpoint的learnable_olm_state字段加载（这是正确的保存位置）
        checkpoint_olm_state = checkpoint.get('learnable_olm_state', None)
        
        # 方法2: 从extras中查找（向后兼容）
        extras = checkpoint.get('extras', {})
        learnable_olm_state = extras.get('learnable_olm_state', None)
        
        # 方法3: 从state_dict中查找（如果编码器被注册为模型的一部分）
        state_dict = checkpoint.get('state_dict', {})
        encoder_state_dict = {}
        
        # 查找所有learnable OLM编码器的参数
        for key, value in state_dict.items():
            # 查找包含encoding_logits的键
            if 'encoding_logits' in key:
                # 尝试提取层名称
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part in layer_names:
                        layer_name = part
                        # 提取参数名（encoding_logits）
                        param_name = parts[-1] if parts[-1] == 'encoding_logits' else 'encoding_logits'
                        if layer_name not in encoder_state_dict:
                            encoder_state_dict[layer_name] = {}
                        encoder_state_dict[layer_name][param_name] = value
                        break
        
        # 优先从checkpoint['learnable_olm_state']加载（这是正确的保存位置）
        if checkpoint_olm_state is not None:
            print("  📦 从checkpoint['learnable_olm_state']加载...")
            for layer_name, layer_state in checkpoint_olm_state.items():
                if layer_name in olm_manager.encoders:
                    encoder = olm_manager.encoders[layer_name]
                    try:
                        encoder.load_state_dict(layer_state, strict=False)
                        print(f"  ✅ 已加载 {layer_name} 的编码器参数")
                    except Exception as e:
                        print(f"  ⚠️  加载 {layer_name} 的编码器参数失败: {e}")
        # 向后兼容：从extras中加载
        elif learnable_olm_state is not None:
            print("  📦 从extras中加载learnable OLM状态（向后兼容）...")
            for layer_name, layer_state in learnable_olm_state.items():
                if layer_name in olm_manager.encoders:
                    encoder = olm_manager.encoders[layer_name]
                    try:
                        encoder.load_state_dict(layer_state, strict=False)
                        print(f"  ✅ 已加载 {layer_name} 的编码器参数")
                    except Exception as e:
                        print(f"  ⚠️  加载 {layer_name} 的编码器参数失败: {e}")
        # 从state_dict中加载（如果编码器被注册为模型的一部分）
        elif encoder_state_dict:
            print("  📦 从state_dict中加载learnable OLM编码器参数...")
            for layer_name, layer_state in encoder_state_dict.items():
                if layer_name in olm_manager.encoders:
                    encoder = olm_manager.encoders[layer_name]
                    try:
                        encoder.load_state_dict(layer_state, strict=False)
                        print(f"  ✅ 已加载 {layer_name} 的编码器参数")
                    except Exception as e:
                        print(f"  ⚠️  加载 {layer_name} 的编码器参数失败: {e}")
        else:
            print("  ⚠️  在checkpoint中未找到learnable OLM编码器参数")
            print("  💡 提示：如果模型使用了learnable OLM训练，编码器参数应该保存在checkpoint['learnable_olm_state']中")
            print("  💡 如果checkpoint中没有保存，将使用默认初始化（可能影响测试结果）")
            print("  💡 建议：检查训练代码是否正确保存了learnable OLM的状态")
        
        # 设置编码器为eval模式（推理时使用hard mapping）
        for encoder in olm_manager.encoders.values():
            encoder.eval()
        
        return olm_manager
        
    except Exception as e:
        print(f"  ❌ 加载Learnable OLM失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_traditional_olm_mappings(
    model: torch.nn.Module,
    layer_names: List[str],
    olm_json: Optional[str] = None,
    num_samples: int = 1000,
    method: str = 'greedy',
    max_iterations: int = 3000
) -> Dict[str, Dict[int, int]]:
    """
    获取传统OLM映射
    
    Args:
        model: 量化模型
        layer_names: 层名称列表
        olm_json: OLM映射JSON文件路径（如果提供，从文件加载）
        num_samples: 如果生成映射，使用的采样数量
    
    Returns:
        {layer_name: {value: code}} 映射字典
    """
    mappings = {}
    
    if olm_json and Path(olm_json).exists():
        # 从JSON文件加载
        print(f"  正在从JSON文件加载传统OLM映射: {olm_json}")
        with open(olm_json, 'r') as f:
            olm_data = json.load(f)
        
        # 检查JSON格式
        if 'layer_mappings' in olm_data:
            # 新格式：包含多个层的映射
            for layer_name in layer_names:
                if layer_name in olm_data['layer_mappings']:
                    layer_data = olm_data['layer_mappings'][layer_name]
                    value_to_code = {int(k): int(v) for k, v in layer_data['value_to_code'].items()}
                    mappings[layer_name] = value_to_code
                    print(f"  ✅ 已加载 {layer_name} 的映射（{len(value_to_code)} 个值）")
                else:
                    print(f"  ⚠️  {layer_name} 不在JSON文件中")
        else:
            # 旧格式：直接包含value_to_code（只支持单层）
            if len(layer_names) == 1:
                value_to_code = {int(k): int(v) for k, v in olm_data['value_to_code'].items()}
                mappings[layer_names[0]] = value_to_code
                print(f"  ✅ 已加载映射（{len(value_to_code)} 个值）")
            else:
                print(f"  ⚠️  JSON文件格式不支持多层，只加载第一层")
                value_to_code = {int(k): int(v) for k, v in olm_data['value_to_code'].items()}
                mappings[layer_names[0]] = value_to_code
    else:
        # 生成OLM映射
        print(f"  正在生成传统OLM映射（采样数量: {num_samples}, 方法: {method}）...")
        print(f"  目标层: {', '.join(layer_names)} ({len(layer_names)} 个层)")
        print()
        for layer_idx, layer_name in enumerate(layer_names):
            try:
                print(f"    [{layer_idx+1}/{len(layer_names)}] 生成 {layer_name} 的映射...")
                
                # 需要先收集分布和获取位宽（用于both方法或simulated_annealing）
                from util.olm_encoder import collect_quantized_value_distribution, optimize_olm_mapping
                distribution = collect_quantized_value_distribution(model, layer_name, num_samples)
                
                # 获取位宽
                module = dict(model.named_modules())[layer_name]
                wbits = None
                if hasattr(module, 'bits') and module.bits is not None:
                    wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
                elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
                if wbits is None:
                    raise ValueError(f"Layer {layer_name} has no bits or fixed_bits attribute")
                if isinstance(wbits, torch.Tensor):
                    wbits = int(wbits.item()) if wbits.numel() == 1 else int(wbits[0].item())
                else:
                    wbits = int(wbits)
                
                # 根据方法选择优化策略
                if method == 'both':
                    # 尝试两种方法，选择更好的
                    import time
                    print("      尝试贪婪搜索...")
                    start_time = time.time()
                    value_to_code_greedy, code_to_value_greedy, lrobust_greedy = optimize_olm_mapping(
                        distribution, wbits, method='greedy'
                    )
                    time_greedy = time.time() - start_time
                    print(f"        完成! LRobust: {lrobust_greedy:.4f}, 耗时: {time_greedy:.2f}秒")
                    
                    print("      尝试模拟退火...")
                    start_time = time.time()
                    value_to_code_sa, code_to_value_sa, lrobust_sa = optimize_olm_mapping(
                        distribution, wbits, method='simulated_annealing', max_iterations=max_iterations
                    )
                    time_sa = time.time() - start_time
                    print(f"        完成! LRobust: {lrobust_sa:.4f}, 耗时: {time_sa:.2f}秒")
                    
                    # 选择更好的（LRobust更小的）
                    if lrobust_sa < lrobust_greedy:
                        print(f"      → 选择模拟退火的结果 (LRobust更小: {lrobust_sa:.4f} < {lrobust_greedy:.4f})")
                        value_to_code, code_to_value, lrobust = value_to_code_sa, code_to_value_sa, lrobust_sa
                        selected_method = 'simulated_annealing'
                    else:
                        print(f"      → 选择贪婪搜索的结果 (LRobust更小: {lrobust_greedy:.4f} < {lrobust_sa:.4f})")
                        value_to_code, code_to_value, lrobust = value_to_code_greedy, code_to_value_greedy, lrobust_greedy
                        selected_method = 'greedy'
                elif method == 'simulated_annealing':
                    # 使用模拟退火
                    value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                        distribution, wbits, method='simulated_annealing', max_iterations=max_iterations
                    )
                    selected_method = 'simulated_annealing'
                else:
                    # 使用贪心方法
                    value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                        distribution, wbits, method='greedy'
                    )
                    selected_method = 'greedy'
                
                mappings[layer_name] = value_to_code
                print(f"    ✅ {layer_name}: LRobust={lrobust:.4f}, 映射大小={len(value_to_code)}, 方法={selected_method}")
                print()
            except Exception as e:
                print(f"    ❌ {layer_name} 生成失败: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        # 总结
        print(f"  生成完成: 成功 {len(mappings)}/{len(layer_names)} 个层")
        if len(mappings) < len(layer_names):
            failed_layers = [name for name in layer_names if name not in mappings]
            print(f"  ⚠️  失败的层: {', '.join(failed_layers)}")
        print()
    
    return mappings


def main():
    parser = argparse.ArgumentParser(description='Compare Learnable OLM vs Traditional OLM')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path (with learnable OLM)')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--ber', type=float, default=1e-2, help='Bit error rate')
    parser.add_argument('--layers', type=str, default='features.0', 
                       help='Comma-separated layer names to test (e.g., "features.0,features.3")')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--traditional_olm_json', type=str, default=None, 
                       help='Path to traditional OLM encoding JSON file (if provided, will load from file instead of generating)')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of samples for generating traditional OLM (if not loading from JSON)')
    parser.add_argument('--olm_method', type=str, default='both', 
                       choices=['greedy', 'simulated_annealing', 'both'],
                       help='Optimization method for traditional OLM: greedy (fast), simulated_annealing (better but slower), or both (try both and choose better)')
    parser.add_argument('--olm_max_iterations', type=int, default=3000,
                       help='Max iterations for simulated annealing (default: 3000)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file for debug output (if not provided, will use stderr)')
    
    args = parser.parse_args()
    
    # 设置日志文件（如果提供）
    log_file = None
    original_stderr = sys.stderr
    try:
        if args.log_file:
            log_file_path = Path(args.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_file_path, 'w', encoding='utf-8')
            sys.stderr = log_file
            print(f"调试信息将输出到: {log_file_path}", file=original_stderr)
            print(f"日志文件: {log_file_path}", file=original_stderr)
            print()
        
        # 加载配置
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], args.config]
        try:
            config = get_config(default_file=args.config)
        finally:
            sys.argv = original_argv
        
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(args.seed)
        
        # 解析层名称（如果使用默认值，稍后从checkpoint自动检测）
        use_auto_detect = (args.layers == 'features.0')
        layer_names = [name.strip() for name in args.layers.split(',')] if not use_auto_detect else None
        
        # 创建模型
        print("="*80)
        print("Learnable OLM vs Traditional OLM 容错能力对比测试")
        print("="*80)
        print(f"配置: {args.config}")
        print(f"Checkpoint: {args.ckpt}")
        if layer_names:
            print(f"测试层: {', '.join(layer_names)} ({len(layer_names)} 个层)")
        else:
            print(f"测试层: 将从checkpoint自动检测")
        print(f"BER: {args.ber}")
        print()
        
        print("步骤1: 创建模型...")
        model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
        model = model.to(device)
        
        # 应用量化
        print("步骤2: 应用量化...")
        modules_to_replace = find_modules_to_quantize(model, config)
        replace_module_by_names(model, modules_to_replace)
        
        # 加载checkpoint（必须在bit-width设置之前）
        print(f"步骤3: 加载checkpoint: {args.ckpt}")
        load_checkpoint(model, args.ckpt, model_device=device)
    
        # 如果使用默认值，从checkpoint自动检测层
        if use_auto_detect:
            try:
                checkpoint = torch.load(args.ckpt, map_location='cpu')
                if 'learnable_olm_state' in checkpoint and checkpoint['learnable_olm_state']:
                    auto_detected_layers = list(checkpoint['learnable_olm_state'].keys())
                    if auto_detected_layers:
                        print(f"✅ 检测到checkpoint中包含 {len(auto_detected_layers)} 个层的Learnable OLM编码")
                        print(f"   自动检测到的层: {', '.join(auto_detected_layers)}")
                        print(f"   使用自动检测的层（如需指定其他层，请使用 --layers 参数）")
                        print()
                        layer_names = auto_detected_layers
                    else:
                        print(f"⚠️  checkpoint中没有Learnable OLM编码，使用默认值: features.0")
                        layer_names = ['features.0']
                else:
                    print(f"⚠️  checkpoint中没有learnable_olm_state，使用默认值: features.0")
                    layer_names = ['features.0']
            except Exception as e:
                print(f"⚠️  无法从checkpoint自动检测层，使用默认值: features.0")
                print(f"   错误: {e}")
                print()
                layer_names = ['features.0']
    
        # 更新显示信息
        print(f"最终测试层: {', '.join(layer_names)} ({len(layer_names)} 个层)")
        print()
    
        # 加载bit-width配置（必须在checkpoint之后，因为需要设置dynamic bit-width）
        print("步骤4: 设置bit-width配置...")
        if args.bit_width_config:
            print(f"  从文件加载bit-width配置: {args.bit_width_config}")
            setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
        else:
            # 如果没有提供bit-width配置文件，从训练配置中读取target_bits并设置
            target_bits = getattr(config, 'target_bits', [8])
            enable_dynamic = getattr(config, 'enable_dynamic_bit_training', False)
            
            # switch_bit_width 需要单个值，不能是列表
            if isinstance(target_bits, list):
                if len(target_bits) > 0:
                    bit_width_value = max(target_bits)
                else:
                    bit_width_value = 8
            else:
                bit_width_value = int(target_bits)
            
            from util.mpq import switch_bit_width
            if enable_dynamic:
                print(f"  从配置读取: enable_dynamic_bit_training=True, target_bits={target_bits}")
                print(f"  设置所有dynamic layers为 {bit_width_value}-bit (使用target_bits的最大值)")
            else:
                print(f"  使用配置: enable_dynamic_bit_training={enable_dynamic}, target_bits={target_bits}")
                print(f"  设置所有层为 {bit_width_value}-bit")
            
            switch_bit_width(model, quan_scheduler=config.quan, wbit=bit_width_value, abits=bit_width_value)
    
        model.eval()
    
        # 准备数据
        print("步骤4: 准备数据...")
        _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
        total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * test_loader.batch_size
        print(f"验证集大小: {total_samples} 样本")
        print()
    
        # Test 1: Baseline（无故障）
        print("="*80)
        print("Test 1: Baseline (无故障注入)")
        print("="*80)
        accuracy_baseline = evaluate_model(model, test_loader, device)
        print(f"准确率: {accuracy_baseline:.2f}%")
        print()
    
        # Test 0: 验证故障注入是否正常工作（不使用OLM）
        print("="*80)
        print("Test 0: 验证故障注入（不使用OLM编码）")
        print("="*80)
        print(f"  BER: {args.ber} ({args.ber * 100:.2f}%)")
        print(f"  测试层: {', '.join(layer_names)}")
        print()
        injector_test = FaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            seed=args.seed,
            enable_statistics=True
        )
        injector_test.enable()
        print(f"  ✅ FaultInjector已启用，包装的层数: {len(injector_test._wrapped)}")
        accuracy_test = evaluate_model(model, test_loader, device)
        if injector_test.enable_statistics and injector_test._pending_stats:
            injector_test._process_pending_statistics()
        injector_test.disable()
        print(f"  准确率: {accuracy_test:.2f}%")
        print(f"  相对Baseline下降: {accuracy_baseline - accuracy_test:.2f}%")
        stats_test = injector_test.get_flip_statistics()
        if stats_test:
            total_flipped = sum(s['flipped_bits'] for s in stats_test.values())
            total_bits = sum(s['total_bits'] for s in stats_test.values())
            if total_flipped > 0:
                actual_ber = total_flipped / total_bits if total_bits > 0 else 0
                print(f"  实际翻转比例: {actual_ber:.6f} ({actual_ber*100:.4f}%)")
                if actual_ber < args.ber * 0.1:
                    print(f"  ⚠️  警告: 实际翻转比例远低于配置BER")
            else:
                print(f"  ⚠️  警告: 没有bit被翻转！故障注入可能没有工作")
                print(f"  💡 可能的原因:")
                print(f"     1. BER值太小 ({args.ber:.2e})，导致翻转概率极低")
                print(f"     2. 故障注入没有正确应用到权重上")
                print(f"     3. 模型权重可能没有被量化（检查模型配置）")
        else:
            print(f"  ⚠️  警告: 没有统计信息，故障注入可能没有工作")
        print()
    
        # Test 2: 传统OLM编码
        print("="*80)
        print("Test 2: 传统OLM编码 + 故障注入")
        print("="*80)
        print("  ⚠️  注意: 传统OLM只对 features.0 层进行编解码，但故障注入会对整个模型生效")
        print("="*80)
        try:
            # 只对 features.0 层生成传统OLM映射
            traditional_olm_layers = ['features.0']
            print(f"  传统OLM目标层: {', '.join(traditional_olm_layers)} (仅用于编解码)")
            print(f"  故障注入目标层: 整个模型的所有量化层")
            print()
            traditional_olm_mappings = get_traditional_olm_mappings(
                model, traditional_olm_layers, args.traditional_olm_json, args.num_samples,
                method=args.olm_method, max_iterations=args.olm_max_iterations
            )
            
            if traditional_olm_mappings:
                # 验证所有请求的层都在映射中
                missing_layers = [name for name in traditional_olm_layers if name not in traditional_olm_mappings]
                if missing_layers:
                    print(f"  ⚠️  警告: 以下层没有映射: {', '.join(missing_layers)}")
                    print(f"  ⚠️  这些层将不会使用OLM编码进行故障注入")
                
                # 验证映射格式并显示信息（参照train_olm_encoder.py的做法）
                print(f"  BER: {args.ber} ({args.ber * 100:.2f}%)")
                print(f"  使用OLM编码的层: {list(traditional_olm_mappings.keys())} (仅 features.0 使用OLM编解码)")
                print(f"  其他层将使用普通二进制编码进行故障注入")
                
                # 验证映射格式：确保是 {layer_name: {value: code}} 格式
                for layer_name, mapping in traditional_olm_mappings.items():
                    if not isinstance(mapping, dict):
                        raise ValueError(f"Layer {layer_name} 的映射格式错误: 期望dict，得到 {type(mapping)}")
                    if len(mapping) == 0:
                        print(f"  ⚠️  警告: Layer {layer_name} 的映射为空")
                    # 验证映射中的键值对都是整数
                    for val, code in list(mapping.items())[:5]:  # 只检查前5个作为示例
                        if not isinstance(val, int) or not isinstance(code, int):
                            raise ValueError(f"Layer {layer_name} 的映射包含非整数键值对: {val} -> {code}")
                    print(f"    {layer_name}: OLM映射表大小={len(mapping)}")
                
                print()
                injector_traditional = FaultInjector(
                    model=model,
                    mode='ber',
                    ber=args.ber,
                    device=device,
                    enable_in_inference=True,
                    seed=args.seed,
                    olm_layers=traditional_olm_mappings,
                    enable_statistics=True
                )
                print(f"  FaultInjector已创建，olm_layers包含层: {list(injector_traditional.olm_layers.keys())}")
                if len(injector_traditional.olm_layers) == 0:
                    print("  ⚠️  警告: olm_layers为空，故障注入可能无法正常工作")
                injector_traditional.enable()
                
                # 验证故障注入器是否正确包装了层
                print(f"  ✅ FaultInjector已启用")
                print(f"  📊 包装的层数: {len(injector_traditional._wrapped)}")
                if len(injector_traditional._wrapped) == 0:
                    print("  ⚠️  警告: 没有包装任何层，故障注入可能无法工作！")
                print(f"  📊 启用统计功能: {injector_traditional.enable_statistics}")
                print(f"  📊 初始pending统计信息数量: {len(injector_traditional._pending_stats)}")
                print(f"  📊 初始_flip_stats数量: {len(injector_traditional._flip_stats)}")
                
                accuracy_traditional = evaluate_model(model, test_loader, device)
                
                # 检查评估后的pending统计信息
                print(f"  📊 评估后pending统计信息数量: {len(injector_traditional._pending_stats)}")
                if injector_traditional._pending_stats:
                    print(f"  📊 Pending统计信息的stats_key: {[item[3] for item in injector_traditional._pending_stats[:5]]}")  # 显示前5个
                # 在disable之前，先处理pending的统计信息
                if injector_traditional.enable_statistics:
                    if injector_traditional._pending_stats:
                        print(f"  📊 处理 {len(injector_traditional._pending_stats)} 个pending的统计信息...")
                        injector_traditional._process_pending_statistics()
                    else:
                        print(f"  ⚠️  警告: 没有pending的统计信息")
                injector_traditional.disable()
                print(f"准确率: {accuracy_traditional:.2f}%")
                print(f"相对Baseline下降: {accuracy_baseline - accuracy_traditional:.2f}%")
                print()
                print("="*80)
                print("故障注入统计信息")
                print("="*80)
                # 确保在打印统计信息前处理pending的统计信息
                if injector_traditional.enable_statistics:
                    if injector_traditional._pending_stats:
                        print(f"  📊 处理 {len(injector_traditional._pending_stats)} 个pending的统计信息...")
                        injector_traditional._process_pending_statistics()
                    # 检查_flip_stats是否有数据
                    if injector_traditional._flip_stats:
                        print(f"  ✅ _flip_stats包含 {len(injector_traditional._flip_stats)} 个层的统计信息: {list(injector_traditional._flip_stats.keys())}")
                    else:
                        print(f"  ⚠️  警告: _flip_stats为空，统计信息可能没有被正确收集")
                injector_traditional.print_flip_statistics(verbose=True)
                
                # 验证是否有实际的故障注入
                stats = injector_traditional.get_flip_statistics()
                if stats:
                    total_flipped = sum(s['flipped_bits'] for s in stats.values())
                    total_bits = sum(s['total_bits'] for s in stats.values())
                    if total_flipped == 0:
                        print()
                        print("  ⚠️  警告: 统计信息显示没有bit被翻转！")
                        print(f"  💡 可能的原因:")
                        print(f"     1. BER值太小 ({args.ber:.2e})，导致翻转概率极低")
                        print(f"     2. 故障注入没有正确应用到权重上")
                        print(f"     3. 只对OLM层进行了故障注入，但这些层可能没有被调用")
                    else:
                        actual_ber = total_flipped / total_bits if total_bits > 0 else 0
                        print()
                        print(f"  ✅ 实际翻转比例: {actual_ber:.6f} ({actual_ber*100:.4f}%)")
                        print(f"  ✅ 配置BER: {args.ber:.6f} ({args.ber*100:.4f}%)")
                        if actual_ber < args.ber * 0.1:
                            print(f"  ⚠️  警告: 实际翻转比例远低于配置BER，可能存在问题")
                else:
                    print("  ⚠️  警告: 没有统计信息，故障注入可能没有工作")
                print()
            else:
                print("  ❌ 无法获取传统OLM映射")
                accuracy_traditional = None
        except Exception as e:
            print(f"  ❌ 传统OLM测试失败: {e}")
            import traceback
            traceback.print_exc()
            accuracy_traditional = None
        print()
    
        # Test 3: Learnable OLM编码
        print("="*80)
        print("Test 3: Learnable OLM编码 + 故障注入")
        print("="*80)
        try:
            print("  正在加载Learnable OLM编码器...")
            learnable_olm_manager = load_learnable_olm_from_checkpoint(
                args.ckpt, model, layer_names, device, config
            )
            
            if learnable_olm_manager is not None:
                # 获取hard映射（用于FaultInjector）
                learnable_olm_mappings = learnable_olm_manager.get_hard_mappings()
                print(f"  ✅ 已获取 {len(learnable_olm_mappings)} 个层的映射")
                
                # 验证learnable_olm_manager的层名称
                if not learnable_olm_manager.layer_names:
                    raise ValueError("learnable_olm_manager.layer_names为空")
                
                # 显示映射信息
                for layer_name, mapping in learnable_olm_mappings.items():
                    if not isinstance(mapping, dict):
                        raise ValueError(f"Layer {layer_name} 的映射格式错误: 期望dict，得到 {type(mapping)}")
                    print(f"    {layer_name}: {len(mapping)} 个值")
                
                print(f"  BER: {args.ber} ({args.ber * 100:.2f}%)")
                print(f"  使用Learnable OLM编码的层: {learnable_olm_manager.layer_names}")
                print()
                injector_learnable = FaultInjector(
                    model=model,
                    mode='ber',
                    ber=args.ber,
                    device=device,
                    enable_in_inference=True,
                    seed=args.seed,
                    learnable_olm_manager=learnable_olm_manager,
                    enable_statistics=True
                )
                print(f"  FaultInjector已创建，learnable_olm_manager包含层: {list(injector_learnable.learnable_olm_manager.layer_names) if injector_learnable.learnable_olm_manager else 'None'}")
                if injector_learnable.learnable_olm_manager is None:
                    print("  ⚠️  警告: learnable_olm_manager为None，故障注入可能无法正常工作")
                elif len(injector_learnable.learnable_olm_manager.layer_names) == 0:
                    print("  ⚠️  警告: learnable_olm_manager.layer_names为空，故障注入可能无法正常工作")
                injector_learnable.enable()
                
                # 验证故障注入器是否正确包装了层
                print(f"  ✅ FaultInjector已启用")
                print(f"  📊 包装的层数: {len(injector_learnable._wrapped)}")
                if len(injector_learnable._wrapped) == 0:
                    print("  ⚠️  警告: 没有包装任何层，故障注入可能无法工作！")
                print(f"  📊 启用统计功能: {injector_learnable.enable_statistics}")
                print(f"  📊 初始pending统计信息数量: {len(injector_learnable._pending_stats)}")
                print(f"  📊 初始_flip_stats数量: {len(injector_learnable._flip_stats)}")
                
                accuracy_learnable = evaluate_model(model, test_loader, device)
                
                # 检查评估后的pending统计信息
                print(f"  📊 评估后pending统计信息数量: {len(injector_learnable._pending_stats)}")
                if injector_learnable._pending_stats:
                    print(f"  📊 Pending统计信息的stats_key: {[item[3] for item in injector_learnable._pending_stats[:5]]}")  # 显示前5个
                # 在disable之前，先处理pending的统计信息
                if injector_learnable.enable_statistics:
                    if injector_learnable._pending_stats:
                        print(f"  📊 处理 {len(injector_learnable._pending_stats)} 个pending的统计信息...")
                        injector_learnable._process_pending_statistics()
                    else:
                        print(f"  ⚠️  警告: 没有pending的统计信息")
                injector_learnable.disable()
                print(f"准确率: {accuracy_learnable:.2f}%")
                print(f"相对Baseline下降: {accuracy_baseline - accuracy_learnable:.2f}%")
                print()
                print("="*80)
                print("故障注入统计信息")
                print("="*80)
                # 确保在打印统计信息前处理pending的统计信息
                if injector_learnable.enable_statistics:
                    if injector_learnable._pending_stats:
                        print(f"  📊 处理 {len(injector_learnable._pending_stats)} 个pending的统计信息...")
                        injector_learnable._process_pending_statistics()
                    # 检查_flip_stats是否有数据
                    if injector_learnable._flip_stats:
                        print(f"  ✅ _flip_stats包含 {len(injector_learnable._flip_stats)} 个层的统计信息: {list(injector_learnable._flip_stats.keys())}")
                    else:
                        print(f"  ⚠️  警告: _flip_stats为空，统计信息可能没有被正确收集")
                injector_learnable.print_flip_statistics(verbose=True)
                
                # 验证是否有实际的故障注入
                stats = injector_learnable.get_flip_statistics()
                if stats:
                    total_flipped = sum(s['flipped_bits'] for s in stats.values())
                    total_bits = sum(s['total_bits'] for s in stats.values())
                    if total_flipped == 0:
                        print()
                        print("  ⚠️  警告: 统计信息显示没有bit被翻转！")
                        print(f"  💡 可能的原因:")
                        print(f"     1. BER值太小 ({args.ber:.2e})，导致翻转概率极低")
                        print(f"     2. 故障注入没有正确应用到权重上")
                        print(f"     3. 只对Learnable OLM层进行了故障注入，但这些层可能没有被调用")
                    else:
                        actual_ber = total_flipped / total_bits if total_bits > 0 else 0
                        print()
                        print(f"  ✅ 实际翻转比例: {actual_ber:.6f} ({actual_ber*100:.4f}%)")
                        print(f"  ✅ 配置BER: {args.ber:.6f} ({args.ber*100:.4f}%)")
                        if actual_ber < args.ber * 0.1:
                            print(f"  ⚠️  警告: 实际翻转比例远低于配置BER，可能存在问题")
                else:
                    print("  ⚠️  警告: 没有统计信息，故障注入可能没有工作")
                print()
                if accuracy_traditional is not None:
                    improvement = accuracy_learnable - accuracy_traditional
                    baseline_drop = accuracy_baseline - accuracy_traditional
                    if abs(baseline_drop) > 0.01:  # 避免除以0
                        relative_improvement = improvement / abs(baseline_drop) * 100
                        print(f"相对传统OLM改进: {improvement:+.2f}% ({relative_improvement:.1f}% 相对改进率)")
                    else:
                        print(f"相对传统OLM改进: {improvement:+.2f}% (传统OLM准确率与Baseline相同，无法计算相对改进率)")
            else:
                print("  ❌ 无法加载Learnable OLM编码器")
                accuracy_learnable = None
        except Exception as e:
            print(f"  ❌ Learnable OLM测试失败: {e}")
            import traceback
            traceback.print_exc()
            accuracy_learnable = None
        print()
    
        # 总结
        print("="*80)
        print("测试总结")
        print("="*80)
        print(f"Baseline准确率: {accuracy_baseline:.2f}%")
        if accuracy_traditional is not None:
            print(f"传统OLM准确率: {accuracy_traditional:.2f}% (下降 {accuracy_baseline - accuracy_traditional:.2f}%)")
        if accuracy_learnable is not None:
            print(f"Learnable OLM准确率: {accuracy_learnable:.2f}% (下降 {accuracy_baseline - accuracy_learnable:.2f}%)")
        if accuracy_traditional is not None and accuracy_learnable is not None:
            improvement = accuracy_learnable - accuracy_traditional
            if improvement > 0:
                print(f"✅ Learnable OLM优于传统OLM: +{improvement:.2f}%")
            elif improvement < 0:
                print(f"⚠️  传统OLM优于Learnable OLM: {improvement:.2f}%")
            else:
                print(f"➡️  Learnable OLM与传统OLM性能相同")
        print("="*80)
    
    finally:
        # 恢复stderr并关闭日志文件（确保在异常情况下也能关闭）
        if log_file:
            sys.stderr = original_stderr
            log_file.close()
            print(f"日志文件已保存: {args.log_file}", file=original_stderr)


if __name__ == '__main__':
    main()

