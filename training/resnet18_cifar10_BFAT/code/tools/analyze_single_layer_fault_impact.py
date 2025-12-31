"""
分析只对单个层（features.0）注入故障时的性能损失
使用 w6a6 配置，测试不同 BER 值
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from util.checkpoint import load_checkpoint
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.config import get_config
from util.data_loader import init_dataloader
from util.utils import preprocess_model
from quan import find_modules_to_quantize, replace_module_by_names


def compute_accuracy(model, data_loader, device, max_batches=None):
    """计算模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, total


def analyze_single_layer_fault(model, layer_name, data_loader, bers, device, num_samples=1000):
    """分析单个层故障注入的性能损失"""
    print(f"\n{'='*70}")
    print(f"Analyzing fault injection impact on layer: {layer_name}")
    print(f"{'='*70}")
    
    # 1. Clean accuracy (baseline)
    print("\n1. Computing clean accuracy (baseline)...")
    clean_acc, _ = compute_accuracy(model, data_loader, device, max_batches=None)
    print(f"   Clean Accuracy: {clean_acc:.2f}%")
    
    results = {
        'layer': layer_name,
        'clean_acc': clean_acc,
        'ber_results': {}
    }
    
    # 2. Fault injection at different BERs
    print("\n2. Testing fault injection at different BERs...")
    print(f"   Target layer: {layer_name}")
    print(f"   BER values: {bers}")
    
    for ber in bers:
        print(f"\n   Testing BER={ber:.1e}...")
        
        # Create fault injector with whitelist (only inject on target layer)
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=ber,
            device=device,
            enable_in_inference=True,
            seed=42,
            skip_first_last=False,
            whitelist_layer=layer_name,  # 只对指定层注入故障
        )
        
        injector.enable()
        
        # Compute accuracy with fault injection
        fault_acc, num_samples_used = compute_accuracy(model, data_loader, device, max_batches=None)
        
        injector.disable()
        
        # Calculate accuracy drop
        acc_drop = clean_acc - fault_acc
        acc_drop_pct = (acc_drop / clean_acc) * 100 if clean_acc > 0 else 0
        
        results['ber_results'][ber] = {
            'accuracy': fault_acc,
            'drop': acc_drop,
            'drop_pct': acc_drop_pct,
            'num_samples': num_samples_used
        }
        
        print(f"     Fault Accuracy: {fault_acc:.2f}%")
        print(f"     Accuracy Drop: {acc_drop:.2f}% ({acc_drop_pct:.2f}% relative)")
    
    return results


def print_summary(results):
    """打印结果摘要"""
    print(f"\n{'='*70}")
    print("Summary: Single Layer Fault Injection Impact")
    print(f"{'='*70}")
    print(f"Layer: {results['layer']}")
    print(f"Clean Accuracy: {results['clean_acc']:.2f}%")
    print(f"\n{'BER':<12} {'Fault Acc':<15} {'Drop (abs)':<15} {'Drop (rel)':<15}")
    print('-' * 70)
    
    for ber in sorted(results['ber_results'].keys()):
        r = results['ber_results'][ber]
        print(f"{ber:<12.1e} {r['accuracy']:<15.2f} {r['drop']:<15.2f} {r['drop_pct']:<15.2f}%")
    
    print(f"\n{'='*70}")
    print("Key Findings:")
    print(f"{'='*70}")
    
    # Find most severe impact
    max_drop_ber = max(results['ber_results'].keys(), 
                      key=lambda b: results['ber_results'][b]['drop'])
    max_drop = results['ber_results'][max_drop_ber]
    
    print(f"  - Most severe impact at BER={max_drop_ber:.1e}:")
    print(f"    Accuracy drop: {max_drop['drop']:.2f}% ({max_drop['drop_pct']:.2f}% relative)")
    print(f"    Remaining accuracy: {max_drop['accuracy']:.2f}%")
    
    # Calculate sensitivity (drop per BER unit)
    bers_sorted = sorted(results['ber_results'].keys())
    if len(bers_sorted) >= 2:
        ber_range = bers_sorted[-1] - bers_sorted[0]
        drop_range = (results['ber_results'][bers_sorted[-1]]['drop'] - 
                     results['ber_results'][bers_sorted[0]]['drop'])
        sensitivity = drop_range / ber_range if ber_range > 0 else 0
        print(f"\n  - Sensitivity: {sensitivity:.2f}% accuracy drop per BER unit")
        print(f"    (From BER={bers_sorted[0]:.1e} to {bers_sorted[-1]:.1e})")


def main():
    parser = argparse.ArgumentParser(description='Analyze single layer fault injection impact')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--layer', type=str, default='features.0', help='Target layer name')
    parser.add_argument('--bers', type=str, default='1e-3,1e-2,5e-2', help='Comma-separated BER values')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--bit_width_config', type=str, default=None, help='Bit width config JSON')
    parser.add_argument('--config_index', type=int, default=0, help='Config index')
    parser.add_argument('--force_w6a6', action='store_true', help='Force w6a6 for all dynamic layers')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Parse BER values
    bers = [float(x.strip()) for x in args.bers.split(',')]
    
    # Load model
    print("Loading model...")
    import sys as sys_module
    original_argv = sys_module.argv.copy()
    sys_module.argv = ['analyze_single_layer_fault_impact.py', args.config]
    try:
        configs = get_config(args.config)
    finally:
        sys_module.argv = original_argv
    
    if not hasattr(configs, 'local_rank'):
        configs.local_rank = 0
    if not hasattr(configs, 'world_size'):
        configs.world_size = 1
    if not hasattr(configs, 'rank'):
        configs.rank = 0
    
    model = create_model(configs.arch, dataset=configs.dataloader.dataset)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    model.eval()
    
    load_checkpoint(model, args.ckpt, model_device=str(device), strict=False)
    
    # Set bit-width configuration
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, args.config_index)
    
    # Force w6a6 for dynamic layers if requested
    if args.force_w6a6:
        print("\nForcing w6a6 for all dynamic layers...")
        from util.mpq import switch_bit_width, switch_bit_width_bn
        switch_bit_width(model, configs.quan, wbit=6, abits=6)
        switch_bit_width_bn(model, 6, 6)
        print("  ✓ All dynamic layers set to w6a6")
        print("  Note: Fixed_bits layers (features.0, classifier.6) remain 8-bit")
    
    # Get data loader
    _, _, test_loader, _, _ = init_dataloader(configs.dataloader, configs.arch)
    
    # Analyze
    results = analyze_single_layer_fault(
        model, 
        args.layer, 
        test_loader, 
        bers, 
        device
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = Path(f"logs/single_layer_fault_{args.layer.replace('.', '_')}_w6a6.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    # Convert to JSON-serializable format
    results_json = {
        'layer': results['layer'],
        'clean_acc': results['clean_acc'],
        'ber_results': {str(k): v for k, v in results['ber_results'].items()}
    }
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

