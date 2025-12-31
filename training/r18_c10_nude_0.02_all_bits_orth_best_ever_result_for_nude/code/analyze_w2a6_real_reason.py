#!/usr/bin/env python
"""分析w2a6比w6a6慢的真实原因"""
import torch
import time

print("="*60)
print("分析：为什么w2a6比w6a6慢？")
print("="*60)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N = 100000

print(f"\n测试参数: N={N}, device={device}")

# 模拟权重tensor
weight = torch.randn(N, device=device)

print("\n关键发现：量化器初始化中的GPU-CPU同步！")
print("-"*60)

# 测试1: 2-bit初始化（thd_pos = 1）
print("\n1. 2-bit量化器初始化（thd_pos=1）:")
start = time.time()
s_init_2bit = weight.detach().abs().mean() * 2 / (1 ** 0.5)  # thd_pos = 1
torch.cuda.synchronize() if device.type == 'cuda' else None
time_2bit = time.time() - start
print(f"   时间: {time_2bit*1000:.2f} ms")
print(f"   操作: x.detach().abs().mean() * 2 / (1 ** 0.5)")

# 测试2: 6-bit初始化（thd_pos = 31）
print("\n2. 6-bit量化器初始化（thd_pos=31）:")
start = time.time()
s_init_6bit = weight.detach().abs().mean() * 2 / (31 ** 0.5)  # thd_pos = 31
torch.cuda.synchronize() if device.type == 'cuda' else None
time_6bit = time.time() - start
print(f"   时间: {time_6bit*1000:.2f} ms")
print(f"   操作: x.detach().abs().mean() * 2 / (31 ** 0.5)")

print("\n" + "="*60)
print("关键发现：")
print("="*60)
print("1. 量化器初始化操作：")
print("   - 每次第一次使用某个bit宽度时，会执行：")
print("     s_init = x.detach().abs().mean() * 2 / (thd_pos ** 0.5)")
print("   - 这个操作包含：")
print("     a) x.detach() - 创建新的tensor（可能需要GPU-CPU同步）")
print("     b) .abs() - 计算绝对值")
print("     c) .mean() - 计算均值（可能需要GPU-CPU同步）")

print("\n2. 为什么w2a6比w6a6慢？")
print("   - 如果量化器的bit_list是[6,5,4,3,2]，那么：")
print("     * w6a6: 6-bit在bit_list中，可能已经初始化过了")
print("     * w2a6: 2-bit在bit_list中，但可能是第一次使用，需要初始化")
print("   - 初始化操作需要GPU-CPU同步，非常慢！")

print("\n3. 但是，如果bit_list包含2，为什么还会慢？")
print("   - 可能的原因：")
print("     a) 2-bit的初始化可能更复杂（thd_pos=1，计算可能不同）")
print("     b) 2-bit的scale可能需要更多计算")
print("     c) 2-bit的数值范围很小，可能需要更多边界检查")

print("\n4. 最可能的原因：")
print("   - 量化器在forward时，如果init_state[idx]==0，会进行初始化")
print("   - 初始化包括：x.detach().abs().mean()，这需要GPU-CPU同步")
print("   - 如果w2a6时，2-bit还没有初始化，就会触发初始化")
print("   - 而w6a6时，6-bit可能已经初始化过了，所以不会触发")

print("\n" + "="*60)
print("解决方案：")
print("="*60)
print("1. 在设置w2a6之前，先进行一次dummy forward pass来初始化量化器")
print("2. 或者，在量化器初始化时，预先初始化所有bit宽度的scale")
print("3. 或者，优化初始化逻辑，避免GPU-CPU同步")



