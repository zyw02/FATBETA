#!/usr/bin/env python3
"""计算不同位宽权重在不同 BER 下的翻转概率"""

import numpy as np

def calculate_flip_probability(ber, bits):
    """计算至少一个 bit 翻转的概率"""
    return 1 - (1 - ber) ** bits

bits_list = [2, 4, 6, 8, 16, 32]
bers = [1e-3, 1e-2, 5e-2, 1e-1]

print('Weight Flip Probability for Different Bit-widths:')
print('=' * 90)
header = f"{'BER':<10} "
for bits in bits_list:
    header += f"{bits}-bit".ljust(15)
print(header)
print('-' * 90)

for ber in bers:
    row = f"{ber:<10.1e} "
    for bits in bits_list:
        prob = calculate_flip_probability(ber, bits)
        row += f"{prob:.4f} ({100*prob:>6.2f}%)".ljust(15)
    print(row)

print()
print('=' * 90)
print('Key Comparison: BER=1e-2 vs BER=1e-1')
print('=' * 90)
print(f"{'Bit-width':<12} {'BER=1e-2':<15} {'BER=1e-1':<15} {'Ratio':<15} {'Linear Approx':<20}")
print('-' * 90)

for bits in bits_list:
    prob_1e2 = calculate_flip_probability(1e-2, bits)
    prob_1e1 = calculate_flip_probability(1e-1, bits)
    ratio = prob_1e1 / prob_1e2
    linear_1e2 = bits * 1e-2
    linear_1e1 = bits * 1e-1
    print(f"{bits:<12} {prob_1e2:.4f} ({100*prob_1e2:>5.2f}%) {prob_1e1:.4f} ({100*prob_1e1:>5.2f}%) {ratio:<15.2f}x {linear_1e1/linear_1e2:<20.2f}x")

print()
print('Special Case: 32-bit weights')
print('-' * 90)
prob_32_1e2 = calculate_flip_probability(1e-2, 32)
prob_32_1e1 = calculate_flip_probability(1e-1, 32)
print(f'BER=1e-2: P(flip) = {prob_32_1e2:.6f} ({100*prob_32_1e2:.2f}%)')
print(f'BER=1e-1: P(flip) = {prob_32_1e1:.6f} ({100*prob_32_1e1:.2f}%)')
print(f'Ratio: {prob_32_1e1 / prob_32_1e2:.2f}x')
print()
print('Linear approximation (32 × BER):')
print(f'BER=1e-2: 32 × 0.01 = {32 * 1e-2:.2f} = {100*32*1e-2:.0f}% (严重高估!)')
print(f'BER=1e-1: 32 × 0.1 = {32 * 1e-1:.2f} = {100*32*1e-1:.0f}% (严重高估!)')
print()
print('Key Insight:')
print(f'  - For 32-bit: BER increases 10x → Weight flip probability increases {prob_32_1e1 / prob_32_1e2:.2f}x')
print(f'  - This is MUCH closer to 10x than for 8-bit (which was 7.37x)')
print(f'  - Why? More bits → More opportunities for at least one flip')
print()
print('Detailed breakdown for 32-bit, BER=1e-1:')
print(f'  - P(0 bits flip) = {(1-0.1)**32:.6f} ({100*(1-0.1)**32:.2f}%)')
print(f'  - P(≥1 bit flip) = {prob_32_1e1:.6f} ({100*prob_32_1e1:.2f}%)')
print(f'  - Expected number of flipped bits per weight (if flipped): ≈ {32 * 0.1 / prob_32_1e1:.2f} bits')


