#!/usr/bin/env python3
"""
测试ResNet18模型的SEU容错能力（Baseline，无编码保护）

直接对baseline模型进行故障注入，评估模型在SEU环境下的准确率下降。
"""

import argparse
import sys
from pathlib import Path

import torch
import random
import numpy as np

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
    parser = argparse.ArgumentParser(description='Test ResNet18 SEU fault tolerance (baseline, no encoding)')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file (optional, will auto-generate if not provided)')
    parser.add_argument('--ber', type=float, nargs='+', default=[1e-1], help='Bit error rate(s)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exclude_layers', type=str, nargs='+', default=None, 
                       help='Layer names to exclude from fault injection (e.g., conv1 fc)')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline evaluation (no fault)')
    parser.add_argument('--skip_fault', action='store_true', help='Skip fault injection evaluation')
    parser.add_argument('--skip_msb', action='store_true', help='Skip MSB (highest bit) injection')
    parser.add_argument('--only_msb', action='store_true', help='Only inject faults on MSB (highest bit)')
    parser.add_argument('--dynamic_bits', type=int, default=None, help='Force bit-width for dynamic layers (e.g., 4 for w4a4)')
    
    args = parser.parse_args()
    
    # 加载配置
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        config = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Best-effort determinism for evaluation reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # 创建模型
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 应用量化
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 加载checkpoint（先加载，然后再设置位宽）
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 准备数据（需要先准备数据以便进行前向传播初始化output_size）
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 进行一次前向传播以初始化output_size（避免model_profiling报错）
    print("初始化模型output_size...")
    model.eval()
    with torch.no_grad():
        # 获取一个batch的数据
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        # 前向传播
        _ = model(inputs)
    print("✅ 模型output_size初始化完成")
    print()
    
    # 加载bit-width配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    else:
        # 如果没有提供bit_width_config，根据配置文件自动设置
        # fixed_bits (excepts) 设置为 8
        # dynamic layers 设置为 target_bits 的最大值（6）
        print("="*80)
        print("自动设置位宽配置")
        print("="*80)
        print("根据配置文件设置：")
        print("  - fixed_bits (excepts): 8-bit")
        print("  - dynamic layers: 6-bit (target_bits最大值)")
        print()
        
        # 获取target_bits的最大值
        if args.dynamic_bits is not None:
            max_target_bit = args.dynamic_bits
            print(f"  强制设置 dynamic layers 位宽为: {max_target_bit}-bit")
        else:
            target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
            max_target_bit = max(target_bits) if target_bits else 6
            print(f"  根据 target_bits 自动选择最大位宽: {max_target_bit}-bit")
        print()
        
        # 直接设置所有dynamic layers的位宽
        from quan.func import QuanConv2d, QuanLinear
        from util.qat import set_bit_width
        
        print(f"  正在将所有 dynamic layers 强制设置为 {max_target_bit}-bit...")
        dynamic_layers_names = []
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                # 跳过fixed_bits层（它们已经有fixed_bits设置）
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    continue
                # 跳过excepts中的层（这些是fixed_bits层）
                if hasattr(config.quan, 'excepts') and name in config.quan.excepts:
                    continue
                dynamic_layers_names.append(name)
        
        # 构造位宽列表进行统一设置
        w_bits_list = [max_target_bit] * len(dynamic_layers_names)
        a_bits_list = [max_target_bit] * len(dynamic_layers_names)
        
        # 使用官方工具设置位宽
        set_bit_width(model, w_bits_list, a_bits_list)
        
        # 核心修复：强制遍历所有量化层，通过闭包或直接赋值锁定推理时的 bits
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                # 跳过 fixed_bits 层（通常是 conv1 和 fc）
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    continue
                
                # 显式覆盖 module 的 bits 属性，这是 FaultInjector 的第一判断依据
                module.bits = (max_target_bit, max_target_bit)
                
                # 如果是 LSQ 量化器，同步更新其内部缓存
                if hasattr(module, 'current_bit_cands_w'):
                    module.current_bit_cands_w = [torch.tensor(max_target_bit).to(device)]
                if hasattr(module, 'current_bit_cands_a'):
                    module.current_bit_cands_a = [torch.tensor(max_target_bit).to(device)]
                
                # 为 FaultInjector 强制开启“量化状态”
                # 在某些版本中，quan_w_fn 可能会读取自己的属性
                if hasattr(module, 'quan_w_fn') and module.quan_w_fn is not None:
                    if hasattr(module.quan_w_fn, 'bits'):
                        module.quan_w_fn.bits = max_target_bit

        # 再次确认：打印前两层的状态
        print(f"  验证层状态:")
        for name, module in list(model.named_modules())[:30]:
            if isinstance(module, QuanConv2d):
                print(f"    - {name}: bits={getattr(module, 'bits', 'N/A')}, fixed_bits={getattr(module, 'fixed_bits', 'N/A')}")
                break # 只看第一个

        print(f"  已强制锁定 {len(dynamic_layers_names)} 个 dynamic layers 的量化状态")
        
        # 也调用switch_bit_width来更新BN层（如果需要）
        from util.mpq import switch_bit_width
        switch_bit_width(model, quan_scheduler=config.quan, wbit=max_target_bit, abits=max_target_bit)
        print("✅ 位宽配置完成")
        print()
        
        # 在设置位宽后，再次进行前向传播以更新output_size（如果位宽改变了）
        print("重新初始化模型output_size（位宽已更新）...")
        model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(test_loader))
            inputs = inputs.to(device)
            _ = model(inputs)
        print("✅ output_size已更新")
        print()
    
    # 获取验证集大小
    total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * test_loader.batch_size
    
    print("="*80)
    print("ResNet18 SEU容错能力测试（Baseline，无编码保护）")
    print("="*80)
    print(f"模型: {config.arch}")
    print(f"数据集: {config.dataloader.dataset}")
    print(f"BER: {args.ber}")
    print(f"验证集大小: {total_samples} 样本")
    print(f"随机种子: {args.seed}")
    if args.exclude_layers:
        print(f"排除层: {args.exclude_layers}")
    print()
    
    # 打印位宽信息
    print("位宽配置:")
    from quan.func import QuanConv2d, QuanLinear
    from util.qat import get_quantized_layers
    
    q_layers, _ = get_quantized_layers(model)
    fixed_bits_layers = []
    dynamic_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                fixed_bits_layers.append((name, module.fixed_bits))
            elif module in q_layers:
                # 获取当前位宽：优先从我们刚刚设置的 bits 属性里读
                if hasattr(module, 'bits') and module.bits is not None:
                    wbits = module.bits[0]
                    abits = module.bits[1]
                elif hasattr(module, 'current_bit_cands_w'):
                    wbits = int(module.current_bit_cands_w[0].item()) if len(module.current_bit_cands_w) > 0 else 'unknown'
                    abits = int(module.current_bit_cands_a[0].item()) if len(module.current_bit_cands_a) > 0 else 'unknown'
                else:
                    wbits = 'unknown'
                    abits = 'unknown'
                dynamic_layers.append((name, wbits, abits))
    
    if fixed_bits_layers:
        print(f"  Fixed-bits层 (8-bit): {len(fixed_bits_layers)} 层")
        for name, bits in fixed_bits_layers[:5]:  # 只显示前5个
            print(f"    - {name}: {bits}-bit")
        if len(fixed_bits_layers) > 5:
            print(f"    ... 还有 {len(fixed_bits_layers) - 5} 层")
    
    if dynamic_layers:
        print(f"  Dynamic-bits层 ({max_target_bit if 'max_target_bit' in locals() else 'N/A'}-bit): {len(dynamic_layers)} 层")
        for name, wbits, abits in dynamic_layers[:5]:  # 只显示前5个
            print(f"    - {name}: W{wbits}-bit, A{abits}-bit")
        if len(dynamic_layers) > 5:
            print(f"    ... 还有 {len(dynamic_layers) - 5} 层")
    print()
    
    accuracy_baseline = None
    if not args.skip_baseline:
        # Test 1: Baseline（无故障）
        print("Test 1: Baseline (无故障注入)")
        accuracy_baseline = evaluate_model(model, test_loader, device)
        print(f"准确率: {accuracy_baseline:.2f}%")
        print()
    
    # 开始循环测试
    bers = args.ber
    for current_ber in bers:
        print("\n" + "="*40)
        print(f"正在测试 BER = {current_ber}")
        print("="*40)
        
        accuracy_binary = None
        if not args.skip_fault:
            # Test 2: 标准二进制编码 + 故障注入（无编码保护）
            exclude_layers = args.exclude_layers if args.exclude_layers else []
            if exclude_layers:
                print(f"Test 2: 标准二进制编码 + 故障注入 (无编码保护, 排除层: {exclude_layers})")
            else:
                print("Test 2: 标准二进制编码 + 故障注入 (无编码保护)")
            injector_binary = FaultInjector(
                model=model,
                mode='ber',
                ber=current_ber,
                device=device,
                enable_in_inference=True,
                seed=args.seed,
                enable_statistics=True,
                exclude_layers=exclude_layers,
                skip_msb=args.skip_msb,
                only_msb=args.only_msb
            )
            injector_binary.enable()
        
            # 打印被包装的层信息（仅在第一个BER时打印）
            if current_ber == bers[0]:
                print()
                print("故障注入器已启用，被包装的层:")
                from quan.func import QuanConv2d, QuanLinear
                wrapped_layers = []
                excluded_layers_list = []
                for name, module in model.named_modules():
                    if isinstance(module, (QuanConv2d, QuanLinear)):
                        if hasattr(module, 'quan_w_fn') and module.quan_w_fn is not None:
                            key = id(module.quan_w_fn)
                            if key in injector_binary._wrapped:
                                layer_type = "fixed_bits" if (hasattr(module, 'fixed_bits') and module.fixed_bits is not None) else "dynamic"
                                bits_info = f"fixed_bits={module.fixed_bits}" if (hasattr(module, 'fixed_bits') and module.fixed_bits is not None) else f"bits={module.bits}"
                                wrapped_layers.append((name, layer_type, bits_info))
                            elif exclude_layers and name in exclude_layers:
                                excluded_layers_list.append(name)
                print(f"  共 {len(wrapped_layers)} 层被包装:")
                for name, layer_type, bits_info in wrapped_layers[:10]:  # 只显示前10个
                    print(f"    - {name} ({layer_type}, {bits_info})")
                if len(wrapped_layers) > 10:
                    print(f"    ... 还有 {len(wrapped_layers) - 10} 层")
                if excluded_layers_list:
                    print(f"  排除的层 ({len(excluded_layers_list)} 层):")
                    for name in excluded_layers_list:
                        print(f"    - {name} (已排除，不进行故障注入)")
                print()

            accuracy_binary = evaluate_model(model, test_loader, device)
            
            # 打印故障注入统计信息
            if injector_binary.enable_statistics:
                stats = injector_binary.get_flip_statistics()
                if stats:
                    print()
                    print(f"故障注入统计信息 (BER={current_ber}):")
                    total_flipped_bits = 0
                    total_bits = 0
                    total_affected_params = 0
                    total_params = 0
                    
                    for layer_name, layer_stats in stats.items():
                        total_flipped_bits += layer_stats['flipped_bits']
                        total_bits += layer_stats['total_bits']
                        total_affected_params += layer_stats['affected_params']
                        total_params += layer_stats['total_params']
                    
                    print(f"  总翻转bit数: {total_flipped_bits}/{total_bits} ({100.0 * total_flipped_bits / total_bits if total_bits > 0 else 0:.4f}%)")
                    print(f"  总受影响参数: {total_affected_params}/{total_params} ({100.0 * total_affected_params / total_params if total_params > 0 else 0:.2f}%)")
                    print(f"  期望BER: {current_ber} ({current_ber * 100:.4f}%)")
                else:
                    print("  ⚠️  警告：没有统计信息！可能故障注入没有生效")

            injector_binary.disable()
            print(f"BER {current_ber} 故障注入后准确率: {accuracy_binary:.2f}%")
            if accuracy_baseline is not None:
                print(f"相对Baseline下降: {accuracy_baseline - accuracy_binary:.2f}%")
                print(f"准确率保持率: {accuracy_binary / accuracy_baseline * 100:.2f}%")
    
    print("\n" + "="*80)
    print("所有测试完成")
    print("="*80)
    if accuracy_baseline is not None:
        print(f"Baseline准确率: {accuracy_baseline:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()

