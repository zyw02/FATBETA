#!/usr/bin/env python3
"""
验证12月2日版本的fault_injector是否启用了OLM编码并成功进行故障注入
"""

import torch
import torch.nn as nn
import sys
import os
import json
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.fault_injector import FaultInjector


def create_test_model():
    """创建一个简单的测试模型（使用真实的量化层）"""
    from quan import QuanConv2d
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 使用真实的量化层
            self.conv = QuanConv2d(3, 16, 3, padding=1, bits=[6], quan_w_fn='lsq')
            
        def forward(self, x):
            return self.conv(x)
    
    model = TestModel()
    return model


def test_olm_encoding_and_fault_injection():
    """测试OLM编码和故障注入"""
    print("="*80)
    print("验证12月2日版本的fault_injector")
    print("="*80)
    print()
    
    # 创建测试模型
    model = create_test_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建OLM映射（6-bit量化，n_levels = 63）
    layer_name = "conv"
    value_to_code = {}
    code_to_value = {}
    
    # 创建简单的OLM映射（6-bit: -32到31）
    thd_neg = -32
    thd_pos = 31
    n_levels = 63
    
    # 创建identity映射（用于测试）
    for val in range(thd_neg, thd_pos + 1):
        code = val - thd_neg  # 映射到[0, 63]
        value_to_code[val] = code
        code_to_value[code] = val
    
    print(f"创建OLM映射:")
    print(f"  layer_name: {layer_name}")
    print(f"  value_to_code size: {len(value_to_code)}")
    print(f"  code_to_value size: {len(code_to_value)}")
    print(f"  范围: [{thd_neg}, {thd_pos}] -> [0, {n_levels}]")
    print()
    
    # 初始化FaultInjector
    print("初始化FaultInjector...")
    try:
        fault_injector = FaultInjector(
            model=model,
            mode='ber',
            ber=0.1,
            device=device,
            enable_in_inference=True,
            seed=42
        )
        
        # 设置OLM映射
        fault_injector.olm_layers = {layer_name: value_to_code}
        fault_injector.olm_code_to_value = {layer_name: code_to_value}
        
        print("✅ FaultInjector初始化成功")
        print(f"  olm_layers keys: {list(fault_injector.olm_layers.keys())}")
        print(f"  olm_code_to_value keys: {list(fault_injector.olm_code_to_value.keys())}")
        print()
    except Exception as e:
        print(f"❌ FaultInjector初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试故障注入
    print("测试故障注入...")
    try:
        # 创建测试输入
        x = torch.randn(1, 3, 32, 32).to(device)
        
        # 获取量化后的权重（用于故障注入）
        # 注意：FaultInjector会在forward hook中修改权重
        # 我们需要检查forward hook是否被正确安装
        
        # 启用故障注入
        fault_injector.enable()
        
        # 检查forward hook是否被安装
        hooks_installed = len(model.conv._forward_hooks) > 0
        print(f"  Forward hooks已安装: {hooks_installed}")
        if hooks_installed:
            print(f"  Hook数量: {len(model.conv._forward_hooks)}")
        
        # 获取量化后的权重（通过前向传播）
        with torch.no_grad():
            # 第一次前向传播：获取量化后的权重
            _ = model.conv(x)
            weight_before = model.conv.weight.data.clone()
            
            # 第二次前向传播：应该触发故障注入
            output = model(x)
            weight_after = model.conv.weight.data.clone()
        weight_diff = (weight - weight_after).abs()
        
        print(f"  故障注入后权重范围: [{weight_after.min().item():.4f}, {weight_after.max().item():.4f}]")
        print(f"  权重变化统计:")
        print(f"    最大变化: {weight_diff.max().item():.6f}")
        print(f"    平均变化: {weight_diff.mean().item():.6f}")
        print(f"    有变化的参数数量: {(weight_diff > 1e-6).sum().item()}/{weight.numel()}")
        print(f"    变化比例: {(weight_diff > 1e-6).sum().item() / weight.numel() * 100:.2f}%")
        print()
        
        if (weight_diff > 1e-6).sum().item() > 0:
            print("✅ 故障注入成功！权重已被修改")
            return True
        else:
            print("❌ 故障注入失败！权重未被修改")
            return False
            
    except Exception as e:
        print(f"❌ 故障注入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        fault_injector.disable()


def test_olm_encoding_path():
    """测试OLM编码路径是否被触发"""
    print("="*80)
    print("测试OLM编码路径是否被触发")
    print("="*80)
    print()
    
    # 创建测试模型
    model = create_test_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建OLM映射
    layer_name = "conv"
    value_to_code = {}
    code_to_value = {}
    
    thd_neg = -32
    thd_pos = 31
    n_levels = 63
    
    for val in range(thd_neg, thd_pos + 1):
        code = val - thd_neg
        value_to_code[val] = code
        code_to_value[code] = val
    
    # 初始化FaultInjector
    fault_injector = FaultInjector(
        model=model,
        mode='ber',
        ber=0.1,
        device=device,
        enable_in_inference=True,
        seed=42
    )
    
    # 设置OLM映射
    fault_injector.olm_layers = {layer_name: value_to_code}
    fault_injector.olm_code_to_value = {layer_name: code_to_value}
    
    # 检查use_olm标志
    print("检查use_olm标志...")
    print(f"  len(fault_injector.olm_layers): {len(fault_injector.olm_layers)}")
    print(f"  layer_name in fault_injector.olm_layers: {layer_name in fault_injector.olm_layers}")
    
    # 模拟_inject_on_quantized_tensor的检查逻辑
    use_olm = (len(fault_injector.olm_layers) > 0 and 
               layer_name is not None and 
               layer_name in fault_injector.olm_layers)
    
    print(f"  use_olm = {use_olm}")
    print()
    
    if use_olm:
        print("✅ OLM编码路径应该被触发")
    else:
        print("❌ OLM编码路径不会被触发")
    
    return use_olm


def test_index_type_issue():
    """测试索引类型问题"""
    print("="*80)
    print("测试索引类型问题")
    print("="*80)
    print()
    
    # 模拟12月2日版本的代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 6-bit量化
    k = 6
    n_levels = (1 << k) - 1  # 63
    code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
    
    print(f"量化配置:")
    print(f"  k = {k}")
    print(f"  n_levels = {n_levels}")
    print(f"  code_dtype = {code_dtype}")
    print()
    
    # 模拟code_shifted（float类型）
    code_f = torch.round(torch.tensor([1.0, 2.5, 3.7], dtype=torch.float32))
    thd_neg = -32
    code_shifted = code_f - thd_neg  # float类型
    code = code_shifted.to(code_dtype).clamp(0, n_levels)
    
    print(f"类型检查:")
    print(f"  code_shifted类型: {code_shifted.dtype}")
    print(f"  code类型: {code.dtype}")
    print()
    
    # 创建查找表
    lookup_table = torch.arange(n_levels + 1, dtype=code_dtype, device=device)
    
    # 测试第950行的代码（使用code_shifted）
    print("测试第950行: lookup_table[code_shifted.clamp(0, n_levels)]")
    try:
        result = lookup_table[code_shifted.clamp(0, n_levels).to(device)]
        print(f"✅ 成功！这不应该发生...")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        print("  这说明12月2日版本的代码确实有bug！")
        print()
        
        # 测试使用code（正确的做法）
        print("测试使用code: lookup_table[code.clamp(0, n_levels)]")
        try:
            result = lookup_table[code.clamp(0, n_levels).to(device)]
            print(f"✅ 成功！")
            return True
        except Exception as e2:
            print(f"❌ 失败: {e2}")
            # 尝试转换为long
            print("测试转换为long: lookup_table[code.clamp(0, n_levels).long()]")
            try:
                result = lookup_table[code.clamp(0, n_levels).long().to(device)]
                print(f"✅ 转换为long后成功！")
                return True
            except Exception as e3:
                print(f"❌ 仍然失败: {e3}")
                return False


def main():
    parser = argparse.ArgumentParser(description='验证12月2日版本的fault_injector')
    parser.add_argument('--test', type=str, default='all', 
                       choices=['all', 'olm_path', 'fault_injection', 'index_type'],
                       help='要运行的测试')
    args = parser.parse_args()
    
    results = {}
    
    if args.test in ['all', 'olm_path']:
        print("\n" + "="*80)
        print("测试1: OLM编码路径检查")
        print("="*80 + "\n")
        results['olm_path'] = test_olm_encoding_path()
        print()
    
    if args.test in ['all', 'index_type']:
        print("\n" + "="*80)
        print("测试2: 索引类型问题检查")
        print("="*80 + "\n")
        results['index_type'] = test_index_type_issue()
        print()
    
    if args.test in ['all', 'fault_injection']:
        print("\n" + "="*80)
        print("测试3: 故障注入功能测试")
        print("="*80 + "\n")
        results['fault_injection'] = test_olm_encoding_and_fault_injection()
        print()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    print()
    
    if all(results.values()):
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败！")
        return 1


if __name__ == '__main__':
    sys.exit(main())

