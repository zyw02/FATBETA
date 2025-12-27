#!/usr/bin/env python3
"""
分析Corrector的logits修正效果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
from munch import Munch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from util.checkpoint import load_checkpoint
from util.data_loader import init_dataloader
from util.output_corrector import create_output_corrector
from util.qat import set_forward_hook_for_conv_linear_layers, remove_hook_for_quantized_layers
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.config import get_config

def analyze_corrector_logits(
    model,
    corrector,
    data_loader,
    device,
    num_samples=100,
    ber=3e-2,
    seed=42
):
    """
    分析Corrector的logits修正效果
    
    Args:
        model: 模型
        corrector: Corrector
        data_loader: 数据加载器
        device: 设备
        num_samples: 分析的样本数量
        ber: 故障注入率
        seed: 随机种子
    """
    model.eval()
    corrector.eval()
    
    samples_collected = 0
    all_results = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            if samples_collected >= num_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_size = inputs.size(0)
            remaining = num_samples - samples_collected
            actual_batch_size = min(batch_size, remaining)
            
            inputs = inputs[:actual_batch_size]
            targets = targets[:actual_batch_size]
            
            # 1. 正常forward
            normal_activations = []
            normal_hooks = set_forward_hook_for_conv_linear_layers(model, normal_activations)
            logits_normal = model(inputs)
            remove_hook_for_quantized_layers(normal_hooks)
            
            # 2. 故障forward
            # 创建故障注入器
            fault_injector = FaultInjector(
                model=model,
                mode="ber",
                ber=ber,
                device=device,
                enable_in_training=False,
                enable_in_inference=True,
                seed=seed,
                use_position_based_mask=False,
                seed_list=None,
                skip_first_last=True
            )
            fault_injector.enable()
            
            faulted_activations = []
            faulted_hooks = set_forward_hook_for_conv_linear_layers(model, faulted_activations)
            logits_faulted = model(inputs)
            remove_hook_for_quantized_layers(faulted_hooks)
            
            fault_injector.disable()
            
            # 3. Corrector修正
            logits_corrected = corrector(
                logits_faulted,
                activations=faulted_activations,
                targets=None  # 推理时不用targets
            )
            
            # 4. 分析每个样本
            for i in range(actual_batch_size):
                logits_n = logits_normal[i].cpu().numpy()
                logits_f = logits_faulted[i].cpu().numpy()
                logits_c = logits_corrected[i].cpu().numpy()
                target = targets[i].item()
                
                # 预测
                pred_n = logits_n.argmax()
                pred_f = logits_f.argmax()
                pred_c = logits_c.argmax()
                
                # 概率
                probs_n = F.softmax(torch.from_numpy(logits_n), dim=0).numpy()
                probs_f = F.softmax(torch.from_numpy(logits_f), dim=0).numpy()
                probs_c = F.softmax(torch.from_numpy(logits_c), dim=0).numpy()
                
                # 修正量
                correction = logits_c - logits_f
                
                # 分析
                result = {
                    'target': target,
                    'pred_normal': pred_n,
                    'pred_faulted': pred_f,
                    'pred_corrected': pred_c,
                    'correct_normal': (pred_n == target),
                    'correct_faulted': (pred_f == target),
                    'correct_corrected': (pred_c == target),
                    'logits_normal': logits_n,
                    'logits_faulted': logits_f,
                    'logits_corrected': logits_c,
                    'correction': correction,
                    'probs_normal': probs_n,
                    'probs_faulted': probs_f,
                    'probs_corrected': probs_c,
                    'logit_target_normal': logits_n[target],
                    'logit_target_faulted': logits_f[target],
                    'logit_target_corrected': logits_c[target],
                    'logit_max_normal': logits_n.max(),
                    'logit_max_faulted': logits_f.max(),
                    'logit_max_corrected': logits_c.max(),
                    'correction_target': correction[target],
                    'correction_max': correction.max(),
                    'correction_min': correction.min(),
                    'correction_norm': np.linalg.norm(correction),
                }
                
                all_results.append(result)
            
            samples_collected += actual_batch_size
    
    return all_results

def print_analysis(results):
    """打印分析结果"""
    print("=" * 80)
    print("📊 Corrector Logits修正分析")
    print("=" * 80)
    
    # 统计
    total = len(results)
    correct_normal = sum(r['correct_normal'] for r in results)
    correct_faulted = sum(r['correct_faulted'] for r in results)
    correct_corrected = sum(r['correct_corrected'] for r in results)
    
    # 需要修正的样本（faulted错误，但normal正确）
    need_correction = [r for r in results if r['correct_normal'] and not r['correct_faulted']]
    num_need_correction = len(need_correction)
    
    # 成功修正的样本
    successful_corrections = [r for r in need_correction if r['correct_corrected']]
    num_successful = len(successful_corrections)
    
    # 错误修正的样本（faulted正确，但corrected错误）
    wrong_corrections = [r for r in results if r['correct_faulted'] and not r['correct_corrected']]
    num_wrong_corrections = len(wrong_corrections)
    
    print(f"\n【总体统计】")
    print(f"  总样本数: {total}")
    print(f"  Normal准确率: {correct_normal}/{total} = {100*correct_normal/total:.2f}%")
    print(f"  Faulted准确率: {correct_faulted}/{total} = {100*correct_faulted/total:.2f}%")
    print(f"  Corrected准确率: {correct_corrected}/{total} = {100*correct_corrected/total:.2f}%")
    print(f"  Corrector Gain: {100*(correct_corrected-correct_faulted)/total:.2f}%")
    
    print(f"\n【需要修正的样本】")
    print(f"  数量: {num_need_correction}/{total} ({100*num_need_correction/total:.2f}%)")
    print(f"  成功修正: {num_successful}/{num_need_correction} ({100*num_successful/num_need_correction if num_need_correction > 0 else 0:.2f}%)")
    print(f"  错误修正: {num_wrong_corrections}/{total} ({100*num_wrong_corrections/total:.2f}%)")
    
    if num_need_correction > 0:
        print(f"\n【需要修正但未成功的样本分析（前10个）】")
        failed_corrections = [r for r in need_correction if not r['correct_corrected']]
        for i, r in enumerate(failed_corrections[:10]):
            print(f"\n  样本 {i+1}:")
            print(f"    Target: {r['target']}, Pred_Normal: {r['pred_normal']}, Pred_Faulted: {r['pred_faulted']}, Pred_Corrected: {r['pred_corrected']}")
            print(f"    Logit_Target: N={r['logit_target_normal']:.3f}, F={r['logit_target_faulted']:.3f}, C={r['logit_target_corrected']:.3f}")
            print(f"    Logit_Max: N={r['logit_max_normal']:.3f}, F={r['logit_max_faulted']:.3f}, C={r['logit_max_corrected']:.3f}")
            print(f"    Correction_Target: {r['correction_target']:.3f}, Correction_Max: {r['correction_max']:.3f}")
            print(f"    Correction_Norm: {r['correction_norm']:.3f}")
            
            # 分析修正方向
            if r['correction_target'] > 0:
                print(f"    ✅ 修正方向正确（增加了target的logit）")
            else:
                print(f"    ❌ 修正方向错误（减少了target的logit）")
            
            # 分析修正量是否足够
            gap = r['logit_max_faulted'] - r['logit_target_faulted']
            correction_needed = gap + 0.1  # 需要至少超过max logit 0.1
            if r['correction_target'] >= correction_needed:
                print(f"    ✅ 修正量足够（需要{correction_needed:.3f}，实际{r['correction_target']:.3f}）")
            else:
                print(f"    ❌ 修正量不足（需要{correction_needed:.3f}，实际{r['correction_target']:.3f}，差{gap:.3f}）")
    
    if num_successful > 0:
        print(f"\n【成功修正的样本分析（前5个）】")
        for i, r in enumerate(successful_corrections[:5]):
            print(f"\n  样本 {i+1}:")
            print(f"    Target: {r['target']}, Pred_Faulted: {r['pred_faulted']} → Pred_Corrected: {r['pred_corrected']}")
            print(f"    Logit_Target: F={r['logit_target_faulted']:.3f} → C={r['logit_target_corrected']:.3f} (Δ={r['correction_target']:.3f})")
            print(f"    Logit_Max: F={r['logit_max_faulted']:.3f} → C={r['logit_max_corrected']:.3f}")
    
    # 修正量统计
    corrections_target = [r['correction_target'] for r in results]
    corrections_norm = [r['correction_norm'] for r in results]
    
    print(f"\n【修正量统计】")
    print(f"  Correction_Target: mean={np.mean(corrections_target):.3f}, std={np.std(corrections_target):.3f}")
    print(f"  Correction_Target: min={np.min(corrections_target):.3f}, max={np.max(corrections_target):.3f}")
    print(f"  Correction_Norm: mean={np.mean(corrections_norm):.3f}, std={np.std(corrections_norm):.3f}")
    
    # 修正方向分析
    positive_corrections = sum(1 for c in corrections_target if c > 0)
    negative_corrections = sum(1 for c in corrections_target if c < 0)
    zero_corrections = sum(1 for c in corrections_target if c == 0)
    
    print(f"\n【修正方向分析】")
    print(f"  增加target logit: {positive_corrections}/{total} ({100*positive_corrections/total:.2f}%)")
    print(f"  减少target logit: {negative_corrections}/{total} ({100*negative_corrections/total:.2f}%)")
    print(f"  不修正target: {zero_corrections}/{total} ({100*zero_corrections/total:.2f}%)")
    
    # 需要修正的样本的修正方向
    if num_need_correction > 0:
        need_correction_target = [r['correction_target'] for r in need_correction]
        positive_need = sum(1 for c in need_correction_target if c > 0)
        negative_need = sum(1 for c in need_correction_target if c < 0)
        
        print(f"\n【需要修正样本的修正方向】")
        print(f"  增加target logit: {positive_need}/{num_need_correction} ({100*positive_need/num_need_correction:.2f}%)")
        print(f"  减少target logit: {negative_need}/{num_need_correction} ({100*negative_need/num_need_correction:.2f}%)")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze Corrector logits correction')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--bit_width_config', type=str, required=True, help='Bit width config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--ber', type=float, default=3e-2, help='BER value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--config_index', type=int, default=0, help='Bit width config index')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    configs = Munch(config_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    from util.utils import preprocess_model
    from quan import find_modules_to_quantize, replace_module_by_names
    dataset = getattr(configs.dataloader, 'dataset', 'cifar10')
    model = create_model(configs.arch, dataset=dataset, pre_trained=getattr(configs, 'pre_trained', False))
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.to(device)
    
    # 加载bit width config
    from util.bit_width_config import load_bit_width_config
    bit_width_config = load_bit_width_config(args.bit_width_config, args.config_index)
    model = setup_model_with_bit_width_config(model, bit_width_config, device)
    
    # 创建数据加载器
    from util.data_loader import init_dataloader
    train_loader, test_loader = init_dataloader(configs)
    
    # 加载checkpoint
    checkpoint_path = args.checkpoint
    load_checkpoint(model, checkpoint_path, device, strict=False)
    
    # 加载corrector
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    output_corrector = None
    if 'output_corrector' in checkpoint:
        # 检测corrector类型
        corrector_state = checkpoint['output_corrector']
        if 'prototypes' in corrector_state:
            # V7 corrector
            num_classes = configs.dataloader.num_classes
            num_layers = 8
            output_corrector = create_output_corrector(
                num_classes=num_classes,
                num_layers=num_layers,
                device=device,
                num_prototypes=10,
                use_dynamic_topk=True,
                topk_k=5
            )
        else:
            # V6 or earlier
            num_classes = configs.dataloader.num_classes
            num_layers = 8
            output_corrector = create_output_corrector(
                num_classes=num_classes,
                num_layers=num_layers,
                device=device
            )
        
        output_corrector.load_state_dict(corrector_state)
        output_corrector.eval()
        print(f"✓ Loaded corrector with {output_corrector.get_total_parameters()} parameters")
    
    if output_corrector is None:
        print("❌ No corrector found in checkpoint!")
        return
    
    # 校准corrector
    print("Calibrating corrector...")
    output_corrector.calibrate_from_samples(
        model=model,
        data_loader=test_loader,
        num_samples=500,
        device=device
    )
    
    # 分析
    print(f"\nAnalyzing {args.num_samples} samples with BER={args.ber}...")
    results = analyze_corrector_logits(
        model=model,
        corrector=output_corrector,
        data_loader=test_loader,
        device=device,
        num_samples=args.num_samples,
        ber=args.ber,
        seed=args.seed
    )
    
    # 打印分析结果
    print_analysis(results)

if __name__ == '__main__':
    main()

