#!/usr/bin/env python3
"""
测试ResNet18模型的SEU容错能力（Baseline，无编码保护）

直接对baseline模型进行故障注入，评估模型在SEU环境下的准确率下降。
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


def main():
    parser = argparse.ArgumentParser(description='Test ResNet18 SEU fault tolerance (baseline, no encoding)')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file (optional, will auto-generate if not provided)')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit error rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exclude_layers', type=str, nargs='+', default=None, 
                       help='Layer names to exclude from fault injection (e.g., conv1 fc)')
    
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
        target_bits = getattr(config, 'target_bits', [6, 5, 4, 3, 2])
        max_target_bit = max(target_bits) if target_bits else 6
        print(f"  target_bits: {target_bits}")
        print(f"  max_target_bit: {max_target_bit}")
        print()
        
        # 直接设置所有dynamic layers的bits属性
        # switch_bit_width只会在hasattr(module, 'bits')为True时设置，但dynamic layers可能没有bits属性
        # 所以我们需要直接设置所有层的bits属性
        from quan.func import QuanConv2d, QuanLinear
        
        print(f"  直接设置所有dynamic layers的bits属性为{max_target_bit}-bit...")
        dynamic_layers_set = 0
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                # 跳过fixed_bits层（它们已经有fixed_bits设置）
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    continue
                # 跳过excepts中的层（这些是fixed_bits层）
                if hasattr(config.quan, 'excepts') and name in config.quan.excepts:
                    continue
                # 直接设置bits属性
                module.bits = (max_target_bit, max_target_bit)
                dynamic_layers_set += 1
        print(f"  已设置 {dynamic_layers_set} 个dynamic layers的bits属性")
        
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
                # 获取当前位宽
                if hasattr(module, 'current_bit_cands_w'):
                    wbits = int(module.current_bit_cands_w[0].item()) if len(module.current_bit_cands_w) > 0 else 'unknown'
                else:
                    wbits = 'unknown'
                if hasattr(module, 'current_bit_cands_a'):
                    abits = int(module.current_bit_cands_a[0].item()) if len(module.current_bit_cands_a) > 0 else 'unknown'
                else:
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
    
    # Test 1: Baseline（无故障）
    print("Test 1: Baseline (无故障注入)")
    accuracy_baseline = evaluate_model(model, test_loader, device)
    print(f"准确率: {accuracy_baseline:.2f}%")
    print()
    
    # Test 2: 标准二进制编码 + 故障注入（无编码保护）
    exclude_layers = args.exclude_layers if args.exclude_layers else []
    if exclude_layers:
        print(f"Test 2: 标准二进制编码 + 故障注入 (无编码保护, 排除层: {exclude_layers})")
    else:
        print("Test 2: 标准二进制编码 + 故障注入 (无编码保护)")
    injector_binary = FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        exclude_layers=exclude_layers
    )
    injector_binary.enable()
    
    # 打印被包装的层信息（用于调试）
    print()
    print("故障注入器已启用，被包装的层:")
    from quan.func import QuanConv2d, QuanLinear
    wrapped_layers = []
    excluded_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'quan_w_fn') and module.quan_w_fn is not None:
                key = id(module.quan_w_fn)
                if key in injector_binary._wrapped:
                    layer_type = "fixed_bits" if (hasattr(module, 'fixed_bits') and module.fixed_bits is not None) else "dynamic"
                    bits_info = f"fixed_bits={module.fixed_bits}" if (hasattr(module, 'fixed_bits') and module.fixed_bits is not None) else f"bits={module.bits}"
                    wrapped_layers.append((name, layer_type, bits_info))
                elif exclude_layers and name in exclude_layers:
                    excluded_layers.append(name)
    print(f"  共 {len(wrapped_layers)} 层被包装:")
    for name, layer_type, bits_info in wrapped_layers[:10]:  # 只显示前10个
        print(f"    - {name} ({layer_type}, {bits_info})")
    if len(wrapped_layers) > 10:
        print(f"    ... 还有 {len(wrapped_layers) - 10} 层")
    if excluded_layers:
        print(f"  排除的层 ({len(excluded_layers)} 层):")
        for name in excluded_layers:
            print(f"    - {name} (已排除，不进行故障注入)")
    print()
    accuracy_binary = evaluate_model(model, test_loader, device)
    injector_binary.disable()
    print(f"准确率: {accuracy_binary:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_binary:.2f}%")
    print(f"准确率保持率: {accuracy_binary / accuracy_baseline * 100:.2f}%")
    
    # 打印故障注入统计信息
    if injector_binary.enable_statistics:
        stats = injector_binary.get_flip_statistics()
        if stats:
            print()
            print("故障注入统计信息:")
            total_flipped_bits = 0
            total_bits = 0
            total_affected_params = 0
            total_params = 0
            
            for layer_name, layer_stats in stats.items():
                print(f"  {layer_name}:")
                print(f"    翻转bit数: {layer_stats['flipped_bits']}/{layer_stats['total_bits']} ({layer_stats['flip_ratio']:.2f}%)")
                print(f"    受影响参数: {layer_stats['affected_params']}/{layer_stats['total_params']} ({layer_stats['affected_ratio']:.2f}%)")
                total_flipped_bits += layer_stats['flipped_bits']
                total_bits += layer_stats['total_bits']
                total_affected_params += layer_stats['affected_params']
                total_params += layer_stats['total_params']
            
            print()
            print("总体统计:")
            print(f"  总翻转bit数: {total_flipped_bits}/{total_bits} ({100.0 * total_flipped_bits / total_bits if total_bits > 0 else 0:.2f}%)")
            print(f"  总受影响参数: {total_affected_params}/{total_params} ({100.0 * total_affected_params / total_params if total_params > 0 else 0:.2f}%)")
            print(f"  期望BER: {args.ber} ({args.ber * 100:.2f}%)")
            print()
            print("分析:")
            print(f"  - 统计层数: {len(stats)} 层")
            print(f"  - BER={args.ber} 意味着每个bit有 {args.ber*100:.1f}% 的概率被翻转")
            print(f"  - 对于 {max_target_bit if 'max_target_bit' in locals() else 'N/A'}-bit 量化，每个参数有 {args.ber * (max_target_bit if 'max_target_bit' in locals() else 8) * 100:.1f}% 的概率至少有一个bit被翻转")
            print(f"  - 准确率大幅下降是正常的，因为：")
            print(f"    1. BER=1e-1 (10%) 是非常高的故障率（实际SEU环境通常 < 1e-6）")
            print(f"    2. 所有层都被注入故障，累积误差很大")
            print(f"    3. 没有使用任何编码保护（格雷码/OLM）")
            print()
            print(f"  - ResNet18 vs AlexNet 容错能力差异分析：")
            print(f"    1. **架构差异**：")
            print(f"       - ResNet18 有残差连接，故障可能通过残差路径传播和累积")
            print(f"       - ResNet18 层数更深（18层 vs AlexNet 约8层），累积误差更大")
            print(f"       - ResNet18 参数更多，受故障影响的参数总数更大")
            print(f"    2. **位宽配置差异**：")
            print(f"       - AlexNet 测试使用了搜索得到的混合位宽配置")
            print(f"       - 混合位宽可能包含更低位宽（4-bit/5-bit）的层")
            print(f"       - 低位宽层在BER=1e-1时受影响相对较小")
            print(f"       - ResNet18 当前测试所有dynamic layers都是6-bit")
            print(f"    3. **故障传播路径**：")
            print(f"       - ResNet18 的残差连接使故障可以从任意层传播到后续所有层")
            print(f"       - AlexNet 的串行结构使故障传播路径相对简单")
            print(f"    4. **建议**：")
            print(f"       - 使用搜索得到的bit_width_config可能提升容错能力")
            print(f"       - 考虑对ResNet18进行FAT（Fault-Aware Training）训练")
            print(f"       - 考虑使用编码保护（格雷码或OLM）")
        else:
            print("  ⚠️  警告：没有统计信息！可能故障注入没有生效")
    
    print()
    print("="*80)
    print("测试完成")
    print("="*80)
    print(f"Baseline准确率: {accuracy_baseline:.2f}%")
    print(f"故障注入后准确率: {accuracy_binary:.2f}%")
    print(f"准确率下降: {accuracy_baseline - accuracy_binary:.2f}%")
    print(f"准确率保持率: {accuracy_binary / accuracy_baseline * 100:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()

