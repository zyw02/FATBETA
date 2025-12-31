#!/usr/bin/env python3
"""
测试混合保护OLM编码的故障注入效果

方案：
- bit7, bit0, bit1: 做冗余（3倍冗余保护bit7）
- bit6-2: 提取出来当做无符号数，做OLM训练（5位，32个值）

解码逻辑：
1. 提取bit7, bit0, bit1，进行多数投票得到bit7_corrected
2. 提取bit2-6的编码，通过OLM映射解码得到bit2-6的值
3. 组合bit7_corrected和bit2-6的值，得到最终解码值

使用方法：
    python tools/test_hybrid_olm_fault_injection.py \
        --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
        --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
        --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
        --olm_json olm_encoding_hybrid_v2_features_0.json \
        --layer features.0 \
        --ber 1e-1
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config


def extract_bit7(code_shifted: torch.Tensor, k: int) -> torch.Tensor:
    """从编码中提取bit7的值（0或1）"""
    bit7_idx = k - 1
    return (code_shifted >> bit7_idx) & 1


def extract_bits_2_to_6(code_shifted: torch.Tensor, k: int) -> torch.Tensor:
    """从编码中提取bit2-6的值（5位，范围0-31），当做无符号数"""
    bits_2_to_6 = torch.zeros_like(code_shifted)
    for i in range(2, 7):  # bit2到bit6
        bit_val = (code_shifted >> i) & 1
        bits_2_to_6 |= (bit_val << (i - 2))
    return bits_2_to_6


def binary_to_gray(binary: torch.Tensor, k: int = 8) -> torch.Tensor:
    """将二进制编码转换为格雷码"""
    # G = B ^ (B >> 1)
    return binary ^ (binary >> 1)


def gray_to_binary(gray: torch.Tensor, k: int = 8) -> torch.Tensor:
    """将格雷码转换为二进制编码"""
    # B = G ^ (G >> 1) ^ (G >> 2) ^ ... ^ (G >> (k-1))
    binary = gray.clone()
    for i in range(1, k):
        binary ^= (gray >> i)
    return binary


def majority_vote(bit7: torch.Tensor, bit0: torch.Tensor, bit1: torch.Tensor) -> torch.Tensor:
    """对bit7, bit0, bit1进行多数投票，得到bit7_corrected"""
    # 计算三个bit的和，如果>=2，则bit7_corrected=1，否则=0
    sum_bits = bit7 + bit0 + bit1
    return (sum_bits >= 2).long()


def hybrid_encode(
    value_shifted: torch.Tensor,
    k: int,
    value_to_code: dict,
    thd_neg: int
) -> torch.Tensor:
    """
    混合编码：量化值 → 混合OLM编码
    
    Args:
        value_shifted: 量化值（已shift到[0, 2^k-1]范围）
        k: 位宽（8）
        value_to_code: 量化值到编码的映射 {value: code}
        thd_neg: 量化阈值下界
    
    Returns:
        编码后的值（已shift到[0, 2^k-1]范围）
    """
    device = value_shifted.device
    dtype = value_shifted.dtype
    
    # 转换回原始范围以查找映射
    value = value_shifted.to(torch.long) + thd_neg
    
    # 创建查找表
    min_value = min(value_to_code.keys()) if value_to_code else thd_neg
    max_value = max(value_to_code.keys()) if value_to_code else -thd_neg + (1 << k) - 1
    value_range = max_value - min_value + 1
    
    # 创建编码查找表（使用偏移量）
    encode_lookup = torch.zeros(value_range, dtype=torch.long, device=device)
    for val, code in value_to_code.items():
        idx = val - min_value
        if 0 <= idx < value_range:
            encode_lookup[idx] = int(code) - thd_neg  # 转换到shifted范围
    
    # 对于未映射的值，使用identity映射
    value_idx = (value - min_value).clamp(0, value_range - 1)
    encoded_shifted = encode_lookup[value_idx]
    
    # 对于超出范围的值，使用identity映射
    mask = (value < min_value) | (value > max_value)
    if mask.any():
        encoded_shifted[mask] = value_shifted[mask].long()
    
    return encoded_shifted.to(dtype)


def decode_redundancy_only(
    code: torch.Tensor,  # 补码形式，范围[-128, 127]，可能已注入故障
    k: int,
    thd_neg: int,
    bit2_6_encoding: str = 'binary'  # 'binary' or 'gray'
) -> torch.Tensor:
    """
    只使用bit7冗余解码，bit2-6保持二进制或格雷码
    
    Args:
        code: 编码值（补码形式，范围[-128, 127]，可能已注入故障）
        k: 位宽（8）
        thd_neg: 量化阈值下界（未使用，保留接口兼容性）
        bit2_6_encoding: bit2-6的编码方式（'binary'或'gray'）
    
    Returns:
        解码后的量化值（补码形式，范围[-128, 127]）
    """
    device = code.device
    dtype = code.dtype
    
    # 转换为int8（自动处理补码表示）
    code_int8 = code.to(torch.int8)
    
    # 1. 从补码提取bit7, bit0, bit1
    bit7 = ((code_int8 >> (k - 1)) & 1).to(torch.int8)
    bit0 = ((code_int8 >> 0) & 1).to(torch.int8)
    bit1 = ((code_int8 >> 1) & 1).to(torch.int8)
    
    # 2. 多数投票得到bit7_corrected
    bit7_corrected = majority_vote(bit7.to(torch.long), bit0.to(torch.long), bit1.to(torch.long)).to(torch.int8)
    
    # 3. 从补码提取bit2-6的编码
    bits_2_to_6_code = torch.zeros_like(code_int8, dtype=torch.int8)
    for i in range(2, 7):  # bit2到bit6
        bit_val = (code_int8 >> i) & 1
        bits_2_to_6_code |= (bit_val << (i - 2))
    
    # 4. 根据编码方式解码bit2-6
    if bit2_6_encoding == 'gray':
        # 格雷码解码：格雷码转二进制
        # bit2-6是5位，所以k=5
        bits_2_to_6_value = gray_to_binary(bits_2_to_6_code.to(torch.long), k=5).to(torch.int8)
    else:  # 'binary'
        # 二进制解码：直接使用
        bits_2_to_6_value = bits_2_to_6_code
    
    # 5. 组合bit7_corrected和bit2-6_value，得到最终解码值（补码形式，使用int8）
    # 注意：bit0和bit1是bit7的冗余副本，用于保护bit7
    # 编码时，bit0和bit1被设置为bit7的值（覆盖原始值）
    # 解码时，bit0和bit1应该等于bit7_corrected（这是设计的代价，但bit0和bit1的重要性远低于bit7）
    decoded = torch.zeros_like(code_int8, dtype=torch.int8)
    decoded |= (bit7_corrected << (k - 1))  # bit7
    decoded |= (bit7_corrected << 0)  # bit0 = bit7_corrected (冗余副本)
    decoded |= (bit7_corrected << 1)  # bit1 = bit7_corrected (冗余副本)
    # bit2-6的值需要正确放置到bit2-6位置
    for i in range(5):  # bit2-6共5位
        bit_val = (bits_2_to_6_value >> i) & 1
        bit_idx = i + 2  # bit2-6
        # 使用 (bit_val != 0) 来避免tensor布尔值判断错误，并转换为int8
        decoded |= ((bit_val != 0).int().to(torch.int8) << bit_idx)
    
    # 转换为long（用于返回，保持与输入dtype一致）
    decoded_signed = decoded.to(torch.long)
    
    return decoded_signed.to(dtype)


def encode_redundancy_only(
    value_original: torch.Tensor,  # 补码形式，范围[-128, 127]
    k: int,
    thd_neg: int,
    bit2_6_encoding: str = 'binary'  # 'binary' or 'gray'
) -> torch.Tensor:
    """
    只使用bit7冗余编码，bit2-6保持二进制或格雷码
    
    Args:
        value_original: 量化值（补码形式，范围[-128, 127]）
        k: 位宽（8）
        thd_neg: 量化阈值下界（未使用，保留接口兼容性）
        bit2_6_encoding: bit2-6的编码方式（'binary'或'gray'）
    
    Returns:
        编码后的值（补码形式，范围[-128, 127]）
    """
    device = value_original.device
    dtype = value_original.dtype
    
    # 直接使用原始值（补码形式），转换为int8进行补码操作
    value_int8 = value_original.to(torch.int8)  # [-128, 127]
    
    # 1. 从原始值提取bit7（使用8位有符号整数的bit7）
    # 对于有符号整数，bit7是符号位
    bit7 = ((value_int8 >> (k - 1)) & 1).to(torch.int8)
    
    # 2. 从原始值提取bit2-6的值（5位，范围0-31）
    # 使用int8时，补码表示是自动的，不需要手动处理符号扩展
    bits_2_to_6_value = torch.zeros_like(value_int8, dtype=torch.int8)
    for i in range(2, 7):  # bit2到bit6
        bit_val = (value_int8 >> i) & 1
        bits_2_to_6_value |= (bit_val << (i - 2))
    
    # 3. 根据编码方式编码bit2-6
    if bit2_6_encoding == 'gray':
        # 格雷码编码：二进制转格雷码
        # bit2-6是5位，所以k=5
        bits_2_to_6_code = binary_to_gray(bits_2_to_6_value.to(torch.long), k=5).to(torch.int8)
    else:  # 'binary'
        # 二进制编码：直接使用
        bits_2_to_6_code = bits_2_to_6_value
    
    # 4. 构建完整编码（使用int8，补码表示自动处理）
    encoded_original = torch.zeros_like(value_int8, dtype=torch.int8)
    encoded_original |= (bit7 << (k - 1))  # bit7
    encoded_original |= (bit7 << 0)  # bit0 = bit7 (冗余)
    encoded_original |= (bit7 << 1)  # bit1 = bit7 (冗余)
    # bit2-6的编码
    for i in range(5):  # bit2-6共5位
        bit_val = (bits_2_to_6_code >> i) & 1
        bit_idx = i + 2  # bit2-6
        # 使用 (bit_val != 0) 来避免tensor布尔值判断错误，并转换为int8
        encoded_original |= ((bit_val != 0).int().to(torch.int8) << bit_idx)
    
    # 5. 返回补码形式的编码值（shifted转换在函数外部进行）
    return encoded_original.to(dtype)


def encode_backup_bit7_bit6(
    value_original: torch.Tensor,  # 补码形式，范围[-128, 127]
    k: int,
    thd_neg: int,
) -> torch.Tensor:
    """
    新方案编码（仅features.0等受保护层使用）：
    - bit1 备份 bit7
    - bit0 备份 bit6
    其余bit保持不变。
    
    Args:
        value_original: 量化值（补码形式，范围[-128, 127]）
        k: 位宽（8）
        thd_neg: 量化阈值下界（未使用，保留接口兼容性）
    
    Returns:
        编码后的值（补码形式，范围[-128, 127]）
    """
    dtype = value_original.dtype
    v = value_original.to(torch.int8)
    # 提取 bit7 / bit6
    bit7 = ((v >> (k - 1)) & 1).to(torch.int8)
    bit6 = ((v >> (k - 2)) & 1).to(torch.int8)
    # 覆盖 bit1 / bit0
    encoded = v.clone()
    # 清除 bit0/bit1
    encoded = (encoded & torch.tensor(~0b11, dtype=torch.int8, device=encoded.device))
    # 设置 bit0=bit6, bit1=bit7
    encoded |= (bit6 << 0)
    encoded |= (bit7 << 1)
    return encoded.to(dtype)


def decode_backup_bit7_bit6(
    code: torch.Tensor,  # 补码形式，范围[-128, 127]，可能已注入故障
    k: int,
    thd_neg: int,
) -> torch.Tensor:
    """
    新方案解码/纠错（仅features.0等受保护层使用）：
    - 若 bit7 != bit1：认为符号位出错，直接将该权重置零（code=0）
    - 否则再判断 bit6 与 bit0：
        - 若不一致且 bit7==1：强制 bit6=1
        - 若不一致且 bit7==0：强制 bit6=0
    其余bit保持当前值不变。
    
    Args:
        code: 编码值（补码形式，范围[-128, 127]，可能已注入故障）
        k: 位宽（8）
        thd_neg: 量化阈值下界（未使用，保留接口兼容性）
    
    Returns:
        解码后的量化值（补码形式，范围[-128, 127]）
    """
    dtype = code.dtype
    v8 = code.to(torch.int8)
    # 用 int16 做位运算更安全（避免int8下按位非/掩码溢出问题）
    v = v8.to(torch.int16)
    bit7 = ((v >> (k - 1)) & 1)
    bit6 = ((v >> (k - 2)) & 1)
    bit1 = ((v >> 1) & 1)
    bit0 = ((v >> 0) & 1)
    
    # 1) 符号位检测：bit7 与 bit1 不一致 -> 置零
    sign_mismatch = (bit7 ^ bit1).to(torch.bool)
    
    corrected = v.clone()
    if sign_mismatch.any():
        corrected[sign_mismatch] = 0
    
    # 2) bit6 检测/纠正：仅对符号一致的样本处理
    ok_mask = ~sign_mismatch
    if ok_mask.any():
        bit7_ok = bit7[ok_mask]
        bit6_ok = bit6[ok_mask]
        bit0_ok = bit0[ok_mask]
        mismatch6 = (bit6_ok ^ bit0_ok).to(torch.bool)
        if mismatch6.any():
            # 强制 bit6 = bit7（按用户描述：负数置1，正数置0）
            force_val = bit7_ok[mismatch6]  # 0/1
            idx = ok_mask.nonzero(as_tuple=True)[0][mismatch6]
            bit6_mask = (1 << (k - 2))  # bit6
            # 先清bit6，再按force_val置位
            corrected[idx] = corrected[idx] & (~bit6_mask)
            corrected[idx] = corrected[idx] | (force_val.to(corrected.dtype) << (k - 2))
    
    return corrected.to(torch.int8).to(torch.long).to(dtype)

def hybrid_decode(
    code_shifted: torch.Tensor,
    k: int,
    bits_2_to_6_code_to_value: dict,
    thd_neg: int
) -> torch.Tensor:
    """
    混合解码：bit7冗余 + bit2-6 OLM
    
    Args:
        code_shifted: 编码值（已shift到[0, 2^k-1]范围，可能已注入故障）
        k: 位宽（8）
        bits_2_to_6_code_to_value: bit2-6的OLM解码映射 {code: value}
        thd_neg: 量化阈值下界
    
    Returns:
        解码后的量化值（原始范围，未shift）
    """
    device = code_shifted.device
    dtype = code_shifted.dtype
    
    # 1. 提取bit7, bit0, bit1
    bit7 = extract_bit7(code_shifted, k)
    bit0 = (code_shifted >> 0) & 1
    bit1 = (code_shifted >> 1) & 1
    
    # 2. 多数投票得到bit7_corrected
    bit7_corrected = majority_vote(bit7, bit0, bit1)
    
    # 3. 提取bit2-6的编码
    bits_2_to_6_code = extract_bits_2_to_6(code_shifted, k)
    
    # 4. 通过OLM映射解码bit2-6
    # 创建查找表
    max_olm_code = max(bits_2_to_6_code_to_value.keys()) if bits_2_to_6_code_to_value else 31
    olm_lookup = torch.zeros(max_olm_code + 1, dtype=torch.long, device=device)
    for code, value in bits_2_to_6_code_to_value.items():
        if 0 <= code <= max_olm_code:
            olm_lookup[code] = int(value)
    
    # 对于未映射的编码，使用identity映射
    bits_2_to_6_value = olm_lookup[bits_2_to_6_code.clamp(0, max_olm_code).long()]
    
    # 5. 组合bit7_corrected和bit2-6_value，得到最终解码值
    # 重建完整编码：bit7_corrected (bit7) + bits_2_to_6_value (bit2-6)
    # bit0-1在解码时不需要，因为它们是冗余副本
    decoded_code_shifted = torch.zeros_like(code_shifted, dtype=torch.long)
    decoded_code_shifted |= (bit7_corrected << (k - 1))  # bit7
    # bit2-6的值需要正确放置到bit2-6位置
    for i in range(5):  # bit2-6共5位
        bit_val = (bits_2_to_6_value >> i) & 1
        bit_idx = i + 2  # bit2-6
        # 使用 (bit_val != 0) 来避免tensor布尔值判断错误
        decoded_code_shifted |= ((bit_val != 0).long() << bit_idx)
    
    # 转换回原始量化值范围
    decoded_value = decoded_code_shifted.to(dtype) + thd_neg
    
    return decoded_value


class HybridOLMFaultInjector(FaultInjector):
    """支持混合OLM解码的故障注入器"""
    
    def __init__(self, *args, hybrid_olm_mappings=None, protection_scheme='hybrid_olm', debug_log_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid_olm_mappings = hybrid_olm_mappings or {}
        self.protection_scheme = protection_scheme  # 'hybrid_olm', 'hybrid_olm_bit2_6_only', 'redundancy_binary', 'redundancy_gray', 'redundancy_binary_bit2_6_only', 'redundancy_gray_bit2_6_only', 'backup_b7b6_detect_zero', 'full_gray'
        self.debug_log_file = debug_log_file  # 存储调试日志文件句柄
        # 存储bit2-6的OLM映射和value_to_code映射
        self.bits_2_to_6_code_to_value = {}
        # 存储bit2-6的反向映射：value -> code（用于编码，保证与hybrid_decode严格一致）
        self.bits_2_to_6_value_to_code = {}
        for layer_name, mapping_info in self.hybrid_olm_mappings.items():
            if 'bits_2_to_6_code_to_value' in mapping_info:
                # 转换字符串key为整数
                self.bits_2_to_6_code_to_value[layer_name] = {
                    int(k): int(v) for k, v in mapping_info['bits_2_to_6_code_to_value'].items()
                }
                # 构建反向映射：value -> code
                inv = {}
                for code, value in self.bits_2_to_6_code_to_value[layer_name].items():
                    # 若出现冲突，保留第一个（理论上不应冲突：应为一一映射）
                    if int(value) not in inv:
                        inv[int(value)] = int(code)
                self.bits_2_to_6_value_to_code[layer_name] = inv
            # 确保value_to_code也在mapping_info中（用于编码）
            if 'value_to_code' not in mapping_info:
                # 如果没有，从code_to_value反向构建
                code_to_value = mapping_info.get('code_to_value', {})
                mapping_info['value_to_code'] = {v: k for k, v in code_to_value.items()}
    
    def _inject_on_quantized_tensor(
        self, x_q: torch.Tensor, k: int, scale: torch.Tensor,
        layer_name=None, forward_seed=None, layer_name_for_stats=None
    ) -> torch.Tensor:
        """重写故障注入逻辑：先编码，再注入故障，最后解码（完全按照FaultInjector的标准流程）"""
        import sys
        debug_file = self.debug_log_file if self.debug_log_file else sys.stderr
        if layer_name:
            print(f"[DEBUG] _inject_on_quantized_tensor: layer_name={layer_name}, layer_name_for_stats={layer_name_for_stats}, k={k}, protection_scheme={self.protection_scheme}", file=debug_file, flush=True)
        device = x_q.device if self.device is None else self.device
        thd_neg = -(1 << (k - 1))
        thd_pos = (1 << (k - 1)) - 1
        n_levels = (1 << k) - 1
        
        # Handle scale as tensor or scalar
        if isinstance(scale, torch.Tensor):
            s = scale.to(device)
            if s.dim() > 0 and s.numel() > 1:
                while s.dim() < x_q.dim():
                    s = s.unsqueeze(-1)
        else:
            s = torch.tensor(float(scale), device=device, dtype=x_q.dtype)
        
        # Step 1: 从量化后的浮点数反推整数码（与FaultInjector一致）
        code_f = torch.round(x_q.to(device) / s)
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        # 直接转换为int8（补码形式），用于冗余编码
        code_int8 = code_f.to(torch.int8)  # 补码形式，范围[-128, 127]
        n_levels = (1 << k) - 1
        
        # Step 2: 根据保护方案进行编码
        # 注意：编码/解码只应用到指定层（hybrid_olm_mappings中的层），其他层使用标准二进制
        check_name = layer_name if layer_name is not None else layer_name_for_stats
        is_protected_layer = check_name and check_name in self.hybrid_olm_mappings
        
        if self.protection_scheme in ['hybrid_olm', 'hybrid_olm_bit2_6_only']:
            # 混合OLM编码（bit7冗余 + bit2-6 OLM）
            if is_protected_layer:
                # 重要：严格按方案组装码字，避免value_to_code整体替换导致bit7/bit0/bit1关系被破坏
                # - bit7 来自原始码（符号位）
                # - bit0/bit1 = bit7（冗余）
                # - bit2-6 使用 OLM 映射（5位无符号码）进行编码
                code_dtype = torch.int16 if n_levels <= 32767 else torch.int32

                # 原始码（shifted域 [0, n_levels]）
                code_shifted = (code_int8.to(torch.long) - thd_neg).clamp(0, n_levels)
                bit7 = extract_bit7(code_shifted, k).to(torch.long)  # 0/1

                # 原始bit2-6的“值”（5位无符号 0..31）
                bits_2_to_6_value = extract_bits_2_to_6(code_shifted, k).to(torch.long)

                # value -> code（5位OLM码字 0..31）；若未映射则identity
                value_to_code_5 = self.bits_2_to_6_value_to_code.get(check_name, {})
                # 构建lookup（长度32）
                lookup_5 = torch.arange(32, dtype=torch.long, device=device)
                for v, c in value_to_code_5.items():
                    if 0 <= int(v) < 32 and 0 <= int(c) < 32:
                        lookup_5[int(v)] = int(c)
                bits_2_to_6_code = lookup_5[bits_2_to_6_value.clamp(0, 31)]

                # 组装最终shifted域编码：bit7 + 冗余bit0/bit1 + bit2-6(OLM码字)
                encoded_shifted = torch.zeros_like(code_shifted, dtype=torch.long)
                encoded_shifted |= (bit7 << (k - 1))  # bit7
                encoded_shifted |= (bit7 << 0)  # bit0
                encoded_shifted |= (bit7 << 1)  # bit1
                encoded_shifted |= (bits_2_to_6_code << 2)  # bit2-6

                code = encoded_shifted.to(code_dtype).clamp(0, n_levels)
            else:
                # 其他层：标准二进制，转换为合适的整数类型
                code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
                code = code_int8.to(code_dtype)
        elif self.protection_scheme == 'redundancy_binary':
            # 只使用bit7冗余，bit2-6保持二进制
            # 只对指定层进行编码，其他层保持标准二进制
            # 确保code_dtype在所有分支中都被定义
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            if is_protected_layer:
                # 直接使用code_int8（补码形式）进行编码，返回补码形式
                encoded_code = encode_redundancy_only(code_int8, k, thd_neg, bit2_6_encoding='binary')
                code = encoded_code.to(torch.int8)
            else:
                # 其他层：标准二进制，转换为合适的整数类型
                code = code_int8.to(code_dtype)
            # else: 保持原code（标准二进制）
        elif self.protection_scheme == 'redundancy_gray':
            # 只使用bit7冗余，bit2-6使用格雷码
            # 只对指定层进行编码，其他层保持标准二进制
            # 确保code_dtype在所有分支中都被定义
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            if is_protected_layer:
                # 直接使用code_int8（补码形式）进行编码，返回补码形式
                encoded_code = encode_redundancy_only(code_int8, k, thd_neg, bit2_6_encoding='gray')
                code = encoded_code.to(torch.int8)
            else:
                # 其他层：标准二进制，转换为合适的整数类型
                code = code_int8.to(code_dtype)
        elif self.protection_scheme == 'backup_b7b6_detect_zero':
            # 新方案：bit1备份bit7，bit0备份bit6；解码时检测bit7!=bit1则置零，否则根据bit7强制修正bit6
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            if is_protected_layer:
                encoded_code = encode_backup_bit7_bit6(code_int8, k, thd_neg)
                code = encoded_code.to(torch.int8)
            else:
                code = code_int8.to(code_dtype)
        elif self.protection_scheme == 'redundancy_binary_bit2_6_only':
            # 只使用bit7冗余，bit2-6保持二进制（编码/解码与redundancy_binary相同，但故障注入时只对bit2-6注入）
            # 只对指定层进行编码，其他层保持标准二进制
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            if is_protected_layer:
                # 直接使用code_int8（补码形式）进行编码，返回补码形式
                encoded_code = encode_redundancy_only(code_int8, k, thd_neg, bit2_6_encoding='binary')
                code = encoded_code.to(torch.int8)
            else:
                # 其他层：标准二进制，转换为合适的整数类型
                code = code_int8.to(code_dtype)
        elif self.protection_scheme == 'redundancy_gray_bit2_6_only':
            # 只使用bit7冗余，bit2-6使用格雷码（编码/解码与redundancy_gray相同，但故障注入时只对bit2-6注入）
            # 只对指定层进行编码，其他层保持标准二进制
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            if is_protected_layer:
                # 直接使用code_int8（补码形式）进行编码，返回补码形式
                encoded_code = encode_redundancy_only(code_int8, k, thd_neg, bit2_6_encoding='gray')
                code = encoded_code.to(torch.int8)
            else:
                # 其他层：标准二进制，转换为合适的整数类型
                code = code_int8.to(code_dtype)
        elif self.protection_scheme == 'full_gray':
            # 全部使用格雷码（包括bit7）
            # 对所有层都应用格雷码编码（与标准FaultInjector完全一致）
            # Shift to non-negative range [0, 2^k-1] for bit operations (与标准FaultInjector完全一致)
            code_shifted = code_f - thd_neg  # Now in [0, n_levels]
            # Use compact integer dtype for efficiency (与标准FaultInjector完全一致)
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            code = code_shifted.to(code_dtype).clamp(0, n_levels)
            if code.device != device:
                code = code.to(device)
            # 向量化：G = B ^ (B >> 1)（与标准FaultInjector一致）
            code = code ^ (code >> 1)
        else:
            # 标准二进制（不编码），转换为合适的整数类型
            code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
            code = code_int8.to(code_dtype)
        
        # Flatten for bit operations (与FaultInjector一致)
        # 注意：bit-flip 必须发生在“shifted无符号域”[0, 2^k-1] 上
        # - full_gray：code 已经是 shifted 域的格雷码，直接使用
        # - hybrid_olm：受保护层经过 OLM 编码后，code 已经是 shifted 域（不是补码），直接使用
        # - 其他情况：code 是补码形式，需要先转回 shifted 域再做 bit-flip
        if self.protection_scheme == 'full_gray' or (self.protection_scheme in ['hybrid_olm', 'hybrid_olm_bit2_6_only'] and is_protected_layer):
            flat = code.view(-1)
        else:
            code_shifted_after_encode = code.to(torch.long) - thd_neg  # 补码 -> shifted
            flat = code_shifted_after_encode.view(-1)
        N = flat.numel()
        
        # Generate flip mask [N, k]
        mask_seed = forward_seed if forward_seed is not None else self.seed
        flip_mask = self._generate_flip_mask(N, k, device, layer_name=layer_name, mask_seed=mask_seed)
        
        # 对于受保护层且使用 *_bit2_6_only 方案时，只对bit2-6做故障注入（不对bit0, bit1, bit7注入）
        if is_protected_layer and self.protection_scheme in ['redundancy_binary_bit2_6_only', 'redundancy_gray_bit2_6_only', 'hybrid_olm_bit2_6_only']:
            # 将bit0, bit1, bit7的mask设置为False（不翻转）
            flip_mask[:, 0] = False  # bit0
            flip_mask[:, 1] = False  # bit1
            flip_mask[:, k - 1] = False  # bit7
            import sys
            debug_file = self.debug_log_file if self.debug_log_file else sys.stderr
            print(f"[DEBUG] HybridOLMFaultInjector: 受保护层 {check_name} 使用{self.protection_scheme}方案，只对bit2-6做故障注入（bit0, bit1, bit7不注入）", file=debug_file, flush=True)
        
        # 记录统计信息（与FaultInjector一致）
        if self.enable_statistics:
            total_bits = N * k
            total_params = N
            # Use layer_name_for_stats if provided, otherwise fall back to layer_name or "unknown"
            stats_key = layer_name_for_stats if layer_name_for_stats is not None else (layer_name if layer_name is not None else "unknown")
            # 计算受影响的参数数量（至少有一个bit被翻转的参数）
            affected_params_sum = (flip_mask.sum(dim=1) > 0).sum()
            flip_mask_sum = flip_mask.sum()
            self._pending_stats.append((flip_mask_sum, total_bits, total_params, stats_key, affected_params_sum))
            # 调试信息：打印所有层的统计信息（特别是受保护层）
            import sys
            debug_file = self.debug_log_file if self.debug_log_file else sys.stderr
            if stats_key and stats_key in self.hybrid_olm_mappings:
                print(f"[DEBUG] HybridOLMFaultInjector: 记录统计信息 for 受保护层 {stats_key}, N={N}, k={k}, total_bits={total_bits}, total_params={total_params}", file=debug_file, flush=True)
            elif stats_key:
                print(f"[DEBUG] HybridOLMFaultInjector: 记录统计信息 for 普通层 {stats_key}, N={N}, k={k}, total_bits={total_bits}, total_params={total_params}", file=debug_file, flush=True)
            else:
                print(f"[DEBUG] HybridOLMFaultInjector: 记录统计信息 for 未知层 (layer_name={layer_name}, layer_name_for_stats={layer_name_for_stats}), N={N}, k={k}, total_bits={total_bits}, total_params={total_params}", file=debug_file, flush=True)
        
        # Step 3: 在编码空间（二进制或格雷码）中进行位翻转（与FaultInjector完全一致）
        bit_positions = torch.arange(k, device=device, dtype=torch.int64)
        bit_weights = (1 << bit_positions).to(torch.int64)
        
        flat_int64 = flat.to(torch.int64)
        if flat_int64.device != device:
            flat_int64 = flat_int64.to(device)
        bits = ((flat_int64.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)
        flipped_bits = bits ^ flip_mask
        
        flat_faulted = (flipped_bits.to(torch.int64) * bit_weights).sum(-1)
        # 先clamp，再转换类型，避免类型转换时的溢出问题
        # 添加异常处理以便定位问题
        try:
            flat_faulted = flat_faulted.clamp(0, n_levels)
            # 确保在正确的设备上，然后再转换类型
            if flat_faulted.device != device:
                flat_faulted = flat_faulted.to(device)
            flat_faulted = flat_faulted.to(code_dtype)
        except Exception as e:
            import sys
            debug_file = self.debug_log_file if self.debug_log_file else sys.stderr
            print(f"[ERROR] 在test_hybrid_olm_fault_injection.py:477处发生异常: {type(e).__name__}: {e}", file=debug_file, flush=True)
            print(f"[ERROR] flat_faulted.shape={flat_faulted.shape}, flat_faulted.dtype={flat_faulted.dtype}, flat_faulted.device={flat_faulted.device}", file=debug_file, flush=True)
            print(f"[ERROR] n_levels={n_levels}, code_dtype={code_dtype}, device={device}", file=debug_file, flush=True)
            if flat_faulted.numel() > 0:
                print(f"[ERROR] flat_faulted.min()={flat_faulted.min().item()}, flat_faulted.max()={flat_faulted.max().item()}", file=debug_file, flush=True)
            import traceback
            traceback.print_exc(file=debug_file)
            raise
        
        # 详细调试信息：记录受保护层的翻转详情（只记录前10个被翻转的样本）
        if is_protected_layer and self.enable_statistics:
            import sys
            debug_file = self.debug_log_file if self.debug_log_file else sys.stderr
            # 找出哪些位置发生了翻转
            flip_indices = (flip_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            if len(flip_indices) > 0:
                # 只记录前10个被翻转的样本
                num_samples = min(10, len(flip_indices))
                sample_indices = flip_indices[:num_samples].cpu().numpy()
                print(f"\n[DEBUG] ========== {check_name} 故障注入详情 (前{num_samples}个被翻转的样本) ==========", file=debug_file, flush=True)
                print(f"[DEBUG] 保护方案: {self.protection_scheme}", file=debug_file, flush=True)
                # 计算原始量化值（shifted范围）- 根据保护方案不同，计算方式不同
                if self.protection_scheme == 'full_gray':
                    # full_gray: code_shifted已经定义
                    orig_code_shifted_tensor = code_shifted.view(-1)
                else:
                    # 其他方案：需要从code_int8计算
                    orig_code_shifted_tensor = (code_int8.to(torch.long) - thd_neg).view(-1)
                for idx in sample_indices:
                    idx = int(idx)
                    # 原始量化值（shifted范围）
                    orig_code_shifted = orig_code_shifted_tensor[idx].item()
                    # 编码后的值（shifted范围）- 注意：这里的flat是编码后的值
                    encoded_code_shifted = flat[idx].item()
                    # 故障注入后的值（shifted范围）
                    faulted_code_shifted = flat_faulted[idx].item()
                    # 哪些bit被翻转了
                    flipped_bit_positions = flip_mask[idx].nonzero(as_tuple=True)[0].cpu().numpy()
                    flipped_bits_str = ','.join([f'bit{i}' for i in flipped_bit_positions])
                    # 原始量化值（原始范围）
                    orig_code_original = orig_code_shifted + thd_neg
                    # 编码后的值（原始范围）
                    encoded_code_original = encoded_code_shifted + thd_neg
                    # 故障注入后的值（原始范围，在编码空间）
                    faulted_code_original_encoded = faulted_code_shifted + thd_neg
                    # 打印二进制表示以便调试
                    orig_binary = format(int(orig_code_shifted), '08b')
                    encoded_binary = format(int(encoded_code_shifted), '08b')
                    faulted_binary = format(int(faulted_code_shifted), '08b')
                    print(f"  [样本 {idx}]", file=debug_file, flush=True)
                    print(f"    原始量化值: {orig_code_original} (shifted: {orig_code_shifted}, 二进制: {orig_binary})", file=debug_file, flush=True)
                    print(f"    编码后值: {encoded_code_original} (shifted: {encoded_code_shifted}, 二进制: {encoded_binary})", file=debug_file, flush=True)
                    print(f"    被翻转的bit: [{flipped_bits_str}]", file=debug_file, flush=True)
                    print(f"    故障注入后值(编码空间): {faulted_code_original_encoded} (shifted: {faulted_code_shifted}, 二进制: {faulted_binary})", file=debug_file, flush=True)
        
        # Step 4: 根据保护方案进行解码
        # 注意：解码只应用到指定层（hybrid_olm_mappings中的层），其他层使用标准二进制解码
        if self.protection_scheme in ['hybrid_olm', 'hybrid_olm_bit2_6_only']:
            # 混合OLM解码（bit7冗余 + bit2-6 OLM）
            if is_protected_layer and check_name in self.bits_2_to_6_code_to_value:
                decoded_value = hybrid_decode(
                    flat_faulted.to(torch.long),
                    k,
                    self.bits_2_to_6_code_to_value[check_name],
                    thd_neg
                )
                code_faulted = decoded_value.to(x_q.dtype)  # hybrid_decode返回原始范围的值
            else:
                # 其他层：标准二进制解码
                code_faulted_shifted = flat_faulted.clamp(0, n_levels)
                code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        elif self.protection_scheme == 'redundancy_binary':
            # 只使用bit7冗余解码，bit2-6保持二进制
            if is_protected_layer:
                # flat_faulted是shifted范围，需要先转换回补码形式
                flat_faulted_original = (flat_faulted.to(torch.long) + thd_neg)
                # decode_redundancy_only期望补码形式作为输入，返回补码形式
                decoded_value = decode_redundancy_only(flat_faulted_original, k, thd_neg, bit2_6_encoding='binary')
                code_faulted = decoded_value.to(x_q.dtype)  # 已经是补码形式，直接使用
            else:
                # 其他层：标准二进制解码
                code_faulted_shifted = flat_faulted.clamp(0, n_levels)
                code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        elif self.protection_scheme == 'redundancy_gray':
            # 只使用bit7冗余解码，bit2-6使用格雷码
            if is_protected_layer:
                # flat_faulted是shifted范围，需要先转换回补码形式
                flat_faulted_original = (flat_faulted.to(torch.long) + thd_neg)
                # decode_redundancy_only期望补码形式作为输入，返回补码形式
                decoded_value = decode_redundancy_only(flat_faulted_original, k, thd_neg, bit2_6_encoding='gray')
                code_faulted = decoded_value.to(x_q.dtype)  # 已经是补码形式，直接使用
            else:
                # 其他层：标准二进制解码
                code_faulted_shifted = flat_faulted.clamp(0, n_levels)
                code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        elif self.protection_scheme == 'backup_b7b6_detect_zero':
            if is_protected_layer:
                # flat_faulted是shifted范围，需要先转换回补码形式
                flat_faulted_original = (flat_faulted.to(torch.long) + thd_neg)
                decoded_value = decode_backup_bit7_bit6(flat_faulted_original, k, thd_neg)
                code_faulted = decoded_value.to(x_q.dtype)  # 补码形式
            else:
                code_faulted_shifted = flat_faulted.clamp(0, n_levels)
                code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        elif self.protection_scheme == 'redundancy_binary_bit2_6_only':
            # 只使用bit7冗余解码，bit2-6保持二进制（解码与redundancy_binary相同）
            if is_protected_layer:
                # flat_faulted是shifted范围，需要先转换回补码形式
                flat_faulted_original = (flat_faulted.to(torch.long) + thd_neg)
                # decode_redundancy_only期望补码形式作为输入，返回补码形式
                decoded_value = decode_redundancy_only(flat_faulted_original, k, thd_neg, bit2_6_encoding='binary')
                code_faulted = decoded_value.to(x_q.dtype)  # 已经是补码形式，直接使用
            else:
                # 其他层：标准二进制解码
                code_faulted_shifted = flat_faulted.clamp(0, n_levels)
                code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        elif self.protection_scheme == 'redundancy_gray_bit2_6_only':
            # 只使用bit7冗余解码，bit2-6使用格雷码（解码与redundancy_gray相同）
            if is_protected_layer:
                # flat_faulted是shifted范围，需要先转换回补码形式
                flat_faulted_original = (flat_faulted.to(torch.long) + thd_neg)
                # decode_redundancy_only期望补码形式作为输入，返回补码形式
                decoded_value = decode_redundancy_only(flat_faulted_original, k, thd_neg, bit2_6_encoding='gray')
                code_faulted = decoded_value.to(x_q.dtype)  # 已经是补码形式，直接使用
            else:
                # 其他层：标准二进制解码
                code_faulted_shifted = flat_faulted.clamp(0, n_levels)
                code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        elif self.protection_scheme == 'full_gray':
            # 全部使用格雷码解码（与标准FaultInjector完全一致）
            # 对所有层都应用格雷码解码
            # flat_faulted 是shifted范围（[0, n_levels]）的格雷码
            if flat_faulted.device != device:
                flat_faulted = flat_faulted.to(device)
            # 向量化转换：B = G ^ (G >> 1) ^ (G >> 2) ^ ... ^ (G >> (k-1))
            # 与标准FaultInjector完全一致
            gray_orig = flat_faulted  # 保存原始格雷码值
            binary = gray_orig
            if k >= 2:
                binary = binary ^ (gray_orig >> 1)
            if k >= 3:
                binary = binary ^ (gray_orig >> 2)
            if k >= 4:
                binary = binary ^ (gray_orig >> 3)
            if k >= 5:
                binary = binary ^ (gray_orig >> 4)
            if k >= 6:
                binary = binary ^ (gray_orig >> 5)
            if k >= 7:
                binary = binary ^ (gray_orig >> 6)
            if k >= 8:
                binary = binary ^ (gray_orig >> 7)
            if k > 8:
                for i in range(8, min(k, 16)):
                    binary = binary ^ (gray_orig >> i)
            flat_faulted = binary
            # Reshape back to original shape
            code_faulted = flat_faulted.view_as(code)
            # Shift back to original range [thd_neg, thd_pos]
            code_faulted_shifted = code_faulted.to(x_q.dtype) + thd_neg
            code_faulted = code_faulted_shifted
        else:
            # 标准二进制解码
            code_faulted_shifted = flat_faulted.clamp(0, n_levels)
            code_faulted = code_faulted_shifted.to(x_q.dtype) + thd_neg
        
        # 继续记录解码后的值（只对受保护层）
        if is_protected_layer and self.enable_statistics:
            import sys
            debug_file = self.debug_log_file if self.debug_log_file else sys.stderr
            flip_indices = (flip_mask.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            if len(flip_indices) > 0:
                num_samples = min(10, len(flip_indices))
                sample_indices = flip_indices[:num_samples].cpu().numpy()
                # 计算原始量化值（shifted范围）- 根据保护方案不同，计算方式不同
                if self.protection_scheme == 'full_gray':
                    # full_gray: code_shifted已经定义
                    orig_code_shifted_tensor = code_shifted.view(-1)
                else:
                    # 其他方案：需要从code_int8计算
                    orig_code_shifted_tensor = (code_int8.to(torch.long) - thd_neg).view(-1)
                for idx in sample_indices:
                    idx = int(idx)
                    # 解码后的值（原始范围）
                    decoded_code_original = code_faulted.view(-1)[idx].item()
                    # 原始量化值（原始范围）
                    orig_code_original = orig_code_shifted_tensor[idx].item() + thd_neg
                    print(f"    解码后值: {decoded_code_original} (原始值: {orig_code_original}, 变化: {decoded_code_original - orig_code_original})", file=debug_file, flush=True)
                print(f"[DEBUG] ========== {check_name} 故障注入详情结束 ==========\n", file=debug_file, flush=True)
        
        # Step 5: 反量化
        x_faulted = code_faulted * s
        
        # 数值安全
        if torch.is_floating_point(x_faulted):
            max_range = torch.abs(s) * (thd_pos + 1)
            x_faulted = torch.clamp(x_faulted, -max_range, max_range)
        
        # Reshape back to original shape
        x_faulted = x_faulted.view_as(x_q)
        
        return x_faulted


def load_hybrid_olm_mapping(json_path: str) -> dict:
    """加载混合OLM映射JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hybrid_mappings = {}
    
    if 'layer_mappings' in data:
        for layer_name, layer_data in data['layer_mappings'].items():
            if layer_data.get('protection_scheme') == 'hybrid_v2':
                k = layer_data.get('bit_width', 8)
                
                # 优先使用JSON中直接存储的bits_2_to_6_code_to_value
                if 'bits_2_to_6_code_to_value' in layer_data:
                    bits_2_to_6_code_to_value = {
                        int(k): int(v) for k, v in layer_data['bits_2_to_6_code_to_value'].items()
                    }
                else:
                    # 如果没有直接存储，从完整映射中提取
                    code_to_value = {int(k): int(v) for k, v in layer_data.get('code_to_value', {}).items()}
                    bits_2_to_6_code_to_value = {}
                    
                    for code, value in code_to_value.items():
                        # 提取bit2-6的编码和值
                        code_shifted = code - (-(1 << (k - 1)))  # 转换到[0, 2^k-1]
                        bits_2_to_6_code = extract_bits_2_to_6(torch.tensor([code_shifted]), k).item()
                        
                        value_shifted = value - (-(1 << (k - 1)))
                        bits_2_to_6_value = extract_bits_2_to_6(torch.tensor([value_shifted]), k).item()
                        
                        bits_2_to_6_code_to_value[bits_2_to_6_code] = bits_2_to_6_value
                
                value_to_code = {int(k): int(v) for k, v in layer_data.get('value_to_code', {}).items()}
                code_to_value = {int(k): int(v) for k, v in layer_data.get('code_to_value', {}).items()}
                
                hybrid_mappings[layer_name] = {
                    'bit_width': k,
                    'bits_2_to_6_code_to_value': bits_2_to_6_code_to_value,
                    'value_to_code': value_to_code,
                    'code_to_value': code_to_value
                }
    
    return hybrid_mappings


def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test hybrid OLM encoding fault injection')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--bit_width_config', type=str, help='Bit-width config JSON file')
    parser.add_argument('--olm_json', type=str, required=True, help='Path to hybrid OLM encoding JSON file')
    parser.add_argument('--ber', type=float, default=1e-1, help='Bit error rate')
    parser.add_argument('--layer', type=str, default='features.0', help='Layer to test')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug_log', type=str, default='debug.log', help='Debug log file path')
    
    args = parser.parse_args()
    
    # 打开调试日志文件
    debug_log_file = open(args.debug_log, 'w', encoding='utf-8')
    print(f"[DEBUG] 调试日志文件已打开: {args.debug_log}", file=debug_log_file, flush=True)
    print(f"[DEBUG] 开始测试...", file=debug_log_file, flush=True)
    
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
    
    # 加载bit-width配置
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=True)
    
    # 加载checkpoint
    load_checkpoint(model, args.ckpt, model_device=device)
    
    # 准备数据
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 加载混合OLM映射
    print("="*80)
    print("混合保护OLM编码故障注入测试")
    print("="*80)
    print(f"测试层: {args.layer}")
    print(f"BER: {args.ber}")
    print(f"OLM映射文件: {args.olm_json}")
    print()
    
    hybrid_mappings = load_hybrid_olm_mapping(args.olm_json)
    if args.layer not in hybrid_mappings:
        print(f"❌ 错误：层 {args.layer} 不在OLM映射文件中")
        print(f"可用层: {list(hybrid_mappings.keys())}")
        return
    
    print(f"✅ 成功加载混合OLM映射")
    layer_mapping = hybrid_mappings[args.layer]
    print(f"   位宽: {layer_mapping['bit_width']}")
    print(f"   Bit2-6 OLM映射数量: {len(layer_mapping['bits_2_to_6_code_to_value'])}")
    print()
    
    # Test 1: Baseline（无故障）
    print("Test 1: Baseline (无故障注入)")
    accuracy_baseline = evaluate_model(model, test_loader, device)
    print(f"准确率: {accuracy_baseline:.2f}%")
    print()
    
    # Test 1.5: 只对features.0做bit7冗余编码/解码，不注入故障
    # 跳过此测试
    # print("Test 1.5: 只对features.0做bit7冗余编码/解码 (无故障注入)")
    # injector_redundancy_clean = HybridOLMFaultInjector(
    #     model=model,
    #     mode='ber',
    #     ber=0.0,  # 不注入故障
    #     device=device,
    #     enable_in_inference=True,
    #     seed=args.seed,
    #     enable_statistics=False,
    #     hybrid_olm_mappings={args.layer: layer_mapping},
    #     protection_scheme='redundancy_binary',
    #     debug_log_file=debug_log_file
    # )
    # injector_redundancy_clean.enable()
    # accuracy_redundancy_clean = evaluate_model(model, test_loader, device)
    # injector_redundancy_clean.disable()
    # print(f"准确率: {accuracy_redundancy_clean:.2f}%")
    # print(f"相对Baseline变化: {accuracy_redundancy_clean - accuracy_baseline:.2f}%")
    # print()
    accuracy_redundancy_clean = None  # 占位符，避免后续引用错误
    
    # Test 2: 标准二进制编码 + 故障注入
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
    
    # Test 3: 只使用bit7冗余，bit2-6保持二进制
    print("Test 3: 只使用bit7冗余 (bit7/bit0/bit1冗余)，bit2-6保持二进制 + 故障注入")
    # 注意：编码/解码只应用到指定层（features.0），其他层使用标准二进制
    # 但故障注入应用到所有层
    injector_redundancy_binary = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},  # 只对指定层进行编码/解码
        protection_scheme='redundancy_binary',
        debug_log_file=debug_log_file
    )
    injector_redundancy_binary.enable()
    accuracy_redundancy_binary = evaluate_model(model, test_loader, device)
    injector_redundancy_binary.disable()
    print(f"准确率: {accuracy_redundancy_binary:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_redundancy_binary:.2f}%")
    print(f"相对二进制改进: {accuracy_redundancy_binary - accuracy_binary:.2f}%")
    # 打印故障注入统计信息
    # 在调用get_flip_statistics之前，先检查pending stats
    print(f"[DEBUG] 调用get_flip_statistics前，_pending_stats长度: {len(injector_redundancy_binary._pending_stats)}")
    if injector_redundancy_binary._pending_stats:
        print(f"[DEBUG] _pending_stats中的stats_key: {[item[3] for item in injector_redundancy_binary._pending_stats]}")
    print(f"[DEBUG] _flip_stats中的键: {list(injector_redundancy_binary._flip_stats.keys())}")
    stats = injector_redundancy_binary.get_flip_statistics()
    print(f"故障注入统计信息:")
    print(f"[DEBUG] stats.keys() = {list(stats.keys()) if stats else 'None'}")
    print(f"[DEBUG] args.layer = {args.layer}")
    if stats:
        for layer_name, layer_stats in stats.items():
            print(f"  {layer_name}:")
            print(f"    翻转bit数: {layer_stats['flipped_bits']}/{layer_stats['total_bits']} ({layer_stats['flip_ratio']:.2f}%)")
            print(f"    受影响参数: {layer_stats['affected_params']}/{layer_stats['total_params']} ({layer_stats['affected_ratio']:.2f}%)")
            if layer_name == args.layer:
                print(f"    ⭐ 这是受保护的层 (bit7冗余编码/解码)")
    else:
        print(f"  ⚠️  警告：没有统计信息！可能故障注入没有生效")
    print()
    
    # Test 4: 只使用bit7冗余，bit2-6使用格雷码
    # 跳过此测试
    # print("Test 4: 只使用bit7冗余 (bit7/bit0/bit1冗余)，bit2-6使用格雷码 + 故障注入")
    # # 注意：编码/解码只应用到指定层（features.0），其他层使用标准二进制
    # # 但故障注入应用到所有层
    # injector_redundancy_gray = HybridOLMFaultInjector(
    #     model=model,
    #     mode='ber',
    #     ber=args.ber,
    #     device=device,
    #     enable_in_inference=True,
    #     seed=args.seed,
    #     enable_statistics=True,
    #     hybrid_olm_mappings={args.layer: layer_mapping},  # 只对指定层进行编码/解码
    #     protection_scheme='redundancy_gray',
    #     debug_log_file=debug_log_file
    # )
    # injector_redundancy_gray.enable()
    # accuracy_redundancy_gray = evaluate_model(model, test_loader, device)
    # injector_redundancy_gray.disable()
    # print(f"准确率: {accuracy_redundancy_gray:.2f}%")
    # print(f"相对Baseline下降: {accuracy_baseline - accuracy_redundancy_gray:.2f}%")
    # print(f"相对二进制改进: {accuracy_redundancy_gray - accuracy_binary:.2f}%")
    # # 打印故障注入统计信息
    # stats = injector_redundancy_gray.get_flip_statistics()
    # print(f"故障注入统计信息:")
    # print(f"[DEBUG] stats.keys() = {list(stats.keys()) if stats else 'None'}")
    # print(f"[DEBUG] args.layer = {args.layer}")
    # if stats:
    #     for layer_name, layer_stats in stats.items():
    #         print(f"  {layer_name}:")
    #         print(f"    翻转bit数: {layer_stats['flipped_bits']}/{layer_stats['total_bits']} ({layer_stats['flip_ratio']:.2f}%)")
    #         print(f"    受影响参数: {layer_stats['affected_params']}/{layer_stats['total_params']} ({layer_stats['affected_ratio']:.2f}%)")
    #         if layer_name == args.layer:
    #             print(f"    ⭐ 这是受保护的层 (bit7冗余编码/解码)")
    # if not stats:
    #     print(f"  ⚠️  警告：没有统计信息！可能故障注入没有生效")
    # print()
    accuracy_redundancy_gray = None  # 占位符，避免后续引用错误
    
    # Test 5: 全部使用格雷码
    print("Test 5: 全部使用格雷码 (bit0-7) + 故障注入")
    # 注意：格雷码编码/解码应用到所有层
    # 故障注入也应用到所有层
    # 注意：hybrid_olm_mappings参数仍然需要提供（用于兼容性），但full_gray会对所有层应用格雷码
    injector_full_gray = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},  # 用于兼容性，但full_gray会对所有层应用格雷码
        protection_scheme='full_gray',
        debug_log_file=debug_log_file
    )
    injector_full_gray.enable()
    accuracy_full_gray = evaluate_model(model, test_loader, device)
    injector_full_gray.disable()
    print(f"准确率: {accuracy_full_gray:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_full_gray:.2f}%")
    print(f"相对二进制改进: {accuracy_full_gray - accuracy_binary:.2f}%")
    print()
    
    # Test 5.5: 只使用bit7冗余，bit2-6保持二进制，但故障注入时只对bit2-6注入（不对bit0, bit1, bit7注入）
    print("Test 5.5: bit7冗余 (bit2-6二进制)，故障注入时只对bit2-6注入 + 故障注入")
    # 注意：编码/解码只应用到指定层（features.0），其他层使用标准二进制
    # 对于features.0层：故障注入只对bit2-6注入（不对bit0, bit1, bit7注入）
    # 对于其他层：正常按位宽做故障注入
    injector_redundancy_binary_bit2_6_only = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},  # 只对指定层进行编码/解码
        protection_scheme='redundancy_binary_bit2_6_only',
        debug_log_file=debug_log_file
    )
    injector_redundancy_binary_bit2_6_only.enable()
    accuracy_redundancy_binary_bit2_6_only = evaluate_model(model, test_loader, device)
    injector_redundancy_binary_bit2_6_only.disable()
    print(f"准确率: {accuracy_redundancy_binary_bit2_6_only:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_redundancy_binary_bit2_6_only:.2f}%")
    print(f"相对二进制改进: {accuracy_redundancy_binary_bit2_6_only - accuracy_binary:.2f}%")
    print(f"相对bit7冗余(全bit注入)改进: {accuracy_redundancy_binary_bit2_6_only - accuracy_redundancy_binary:.2f}%")
    # 打印故障注入统计信息
    stats = injector_redundancy_binary_bit2_6_only.get_flip_statistics()
    print(f"故障注入统计信息:")
    if stats:
        for layer_name, layer_stats in stats.items():
            print(f"  {layer_name}:")
            print(f"    翻转bit数: {layer_stats['flipped_bits']}/{layer_stats['total_bits']} ({layer_stats['flip_ratio']:.2f}%)")
            print(f"    受影响参数: {layer_stats['affected_params']}/{layer_stats['total_params']} ({layer_stats['affected_ratio']:.2f}%)")
            if layer_name == args.layer:
                print(f"    ⭐ 这是受保护的层 (bit7冗余编码/解码，故障注入时只对bit2-6注入)")
    else:
        print(f"  ⚠️  警告：没有统计信息！可能故障注入没有生效")
    print()
    
    # Test 5.6: 只使用bit7冗余，bit2-6使用格雷码，但故障注入时只对bit2-6注入（不对bit0, bit1, bit7注入）
    print("Test 5.6: bit7冗余 (bit2-6格雷码)，故障注入时只对bit2-6注入 + 故障注入")
    # 注意：编码/解码只应用到指定层（features.0），其他层使用标准二进制
    # 对于features.0层：故障注入只对bit2-6注入（不对bit0, bit1, bit7注入）
    # 对于其他层：正常按位宽做故障注入
    injector_redundancy_gray_bit2_6_only = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},  # 只对指定层进行编码/解码
        protection_scheme='redundancy_gray_bit2_6_only',
        debug_log_file=debug_log_file
    )
    injector_redundancy_gray_bit2_6_only.enable()
    accuracy_redundancy_gray_bit2_6_only = evaluate_model(model, test_loader, device)
    injector_redundancy_gray_bit2_6_only.disable()
    print(f"准确率: {accuracy_redundancy_gray_bit2_6_only:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_redundancy_gray_bit2_6_only:.2f}%")
    print(f"相对二进制改进: {accuracy_redundancy_gray_bit2_6_only - accuracy_binary:.2f}%")
    # 打印故障注入统计信息
    stats = injector_redundancy_gray_bit2_6_only.get_flip_statistics()
    print(f"故障注入统计信息:")
    if stats:
        for layer_name, layer_stats in stats.items():
            print(f"  {layer_name}:")
            print(f"    翻转bit数: {layer_stats['flipped_bits']}/{layer_stats['total_bits']} ({layer_stats['flip_ratio']:.2f}%)")
            print(f"    受影响参数: {layer_stats['affected_params']}/{layer_stats['total_params']} ({layer_stats['affected_ratio']:.2f}%)")
            if layer_name == args.layer:
                print(f"    ⭐ 这是受保护的层 (bit7冗余编码/解码，故障注入时只对bit2-6注入)")
    else:
        print(f"  ⚠️  警告：没有统计信息！可能故障注入没有生效")
    print()
    
    # Test 5.7: 新方案（bit1备份bit7，bit0备份bit6；检测bit7!=bit1则置零，否则按bit7纠正bit6）+ 故障注入
    print("Test 5.7: bit1备份bit7 + bit0备份bit6（检测置零/纠正bit6）+ 故障注入")
    injector_backup_b7b6 = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},  # 只对指定层进行编码/解码
        protection_scheme='backup_b7b6_detect_zero',
        debug_log_file=debug_log_file
    )
    injector_backup_b7b6.enable()
    accuracy_backup_b7b6 = evaluate_model(model, test_loader, device)
    injector_backup_b7b6.disable()
    print(f"准确率: {accuracy_backup_b7b6:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_backup_b7b6:.2f}%")
    print(f"相对二进制改进: {accuracy_backup_b7b6 - accuracy_binary:.2f}%")
    print()
    
    # Test 6: 混合OLM编码 + 故障注入
    print("Test 6: 混合OLM编码 (bit7冗余 + bit2-6 OLM) + 故障注入")
    # 注意：编码/解码只应用到指定层（features.0），其他层使用标准二进制
    # 但故障注入应用到所有层
    injector_hybrid = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},
        protection_scheme='hybrid_olm',
        debug_log_file=debug_log_file
    )
    injector_hybrid.enable()
    accuracy_hybrid = evaluate_model(model, test_loader, device)
    injector_hybrid.disable()
    print(f"准确率: {accuracy_hybrid:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_hybrid:.2f}%")
    print(f"相对二进制改进: {accuracy_hybrid - accuracy_binary:.2f}%")
    print()
    
    # Test 6.5: 混合OLM编码，但故障注入时只对bit2-6注入（不对bit0, bit1, bit7注入）
    print("Test 6.5: 混合OLM编码 (bit7冗余 + bit2-6 OLM)，故障注入时只对bit2-6注入 + 故障注入")
    injector_hybrid_bit2_6_only = HybridOLMFaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        seed=args.seed,
        enable_statistics=True,
        hybrid_olm_mappings={args.layer: layer_mapping},
        protection_scheme='hybrid_olm_bit2_6_only',
        debug_log_file=debug_log_file
    )
    injector_hybrid_bit2_6_only.enable()
    accuracy_hybrid_bit2_6_only = evaluate_model(model, test_loader, device)
    injector_hybrid_bit2_6_only.disable()
    print(f"准确率: {accuracy_hybrid_bit2_6_only:.2f}%")
    print(f"相对Baseline下降: {accuracy_baseline - accuracy_hybrid_bit2_6_only:.2f}%")
    print(f"相对二进制改进: {accuracy_hybrid_bit2_6_only - accuracy_binary:.2f}%")
    print(f"相对混合OLM(全bit注入)改进: {accuracy_hybrid_bit2_6_only - accuracy_hybrid:.2f}%")
    print()
    
    # 总结
    print("="*80)
    print("测试总结")
    print("="*80)
    print(f"Baseline准确率:                    {accuracy_baseline:.2f}%")
    # print(f"bit7冗余编码/解码(无故障)准确率:   {accuracy_redundancy_clean:.2f}% (变化 {accuracy_redundancy_clean - accuracy_baseline:.2f}%)")
    print(f"二进制编码准确率:                  {accuracy_binary:.2f}% (下降 {accuracy_baseline - accuracy_binary:.2f}%)")
    print(f"bit7冗余+bit2-6二进制准确率:       {accuracy_redundancy_binary:.2f}% (下降 {accuracy_baseline - accuracy_redundancy_binary:.2f}%)")
    # print(f"bit7冗余+bit2-6格雷码准确率:       {accuracy_redundancy_gray:.2f}% (下降 {accuracy_baseline - accuracy_redundancy_gray:.2f}%)")
    print(f"bit7冗余+bit2-6二进制(仅bit2-6注入): {accuracy_redundancy_binary_bit2_6_only:.2f}% (下降 {accuracy_baseline - accuracy_redundancy_binary_bit2_6_only:.2f}%)")
    print(f"bit7冗余+bit2-6格雷码(仅bit2-6注入): {accuracy_redundancy_gray_bit2_6_only:.2f}% (下降 {accuracy_baseline - accuracy_redundancy_gray_bit2_6_only:.2f}%)")
    print(f"bit1备份bit7+bit0备份bit6(检测置零/纠正bit6): {accuracy_backup_b7b6:.2f}% (下降 {accuracy_baseline - accuracy_backup_b7b6:.2f}%)")
    print(f"全部格雷码准确率:                  {accuracy_full_gray:.2f}% (下降 {accuracy_baseline - accuracy_full_gray:.2f}%)")
    print(f"混合OLM编码准确率:                 {accuracy_hybrid:.2f}% (下降 {accuracy_baseline - accuracy_hybrid:.2f}%)")
    print(f"混合OLM编码(仅bit2-6注入)准确率:     {accuracy_hybrid_bit2_6_only:.2f}% (下降 {accuracy_baseline - accuracy_hybrid_bit2_6_only:.2f}%)")
    print()
    print("相对二进制改进:")
    print(f"  bit7冗余+bit2-6二进制:            {accuracy_redundancy_binary - accuracy_binary:.2f}%")
    # print(f"  bit7冗余+bit2-6格雷码:            {accuracy_redundancy_gray - accuracy_binary:.2f}%")
    print(f"  bit7冗余+bit2-6二进制(仅bit2-6注入): {accuracy_redundancy_binary_bit2_6_only - accuracy_binary:.2f}%")
    print(f"  bit7冗余+bit2-6格雷码(仅bit2-6注入): {accuracy_redundancy_gray_bit2_6_only - accuracy_binary:.2f}%")
    print(f"  bit1备份bit7+bit0备份bit6(检测置零/纠正bit6): {accuracy_backup_b7b6 - accuracy_binary:.2f}%")
    print(f"  全部格雷码:                      {accuracy_full_gray - accuracy_binary:.2f}%")
    print(f"  混合OLM编码:                     {accuracy_hybrid - accuracy_binary:.2f}%")
    print(f"  混合OLM编码(仅bit2-6注入):         {accuracy_hybrid_bit2_6_only - accuracy_binary:.2f}%")
    print()
    print("仅bit2-6注入 vs 全bit注入对比:")
    print(f"  bit7冗余+bit2-6二进制(仅bit2-6注入) vs (全bit注入): {accuracy_redundancy_binary_bit2_6_only - accuracy_redundancy_binary:.2f}%")
    print(f"  bit7冗余+bit2-6格雷码(仅bit2-6注入) vs (全bit注入): {accuracy_redundancy_gray_bit2_6_only - accuracy_redundancy_binary:.2f}%")
    print(f"  混合OLM(仅bit2-6注入) vs (全bit注入):              {accuracy_hybrid_bit2_6_only - accuracy_hybrid:.2f}%")
    print("="*80)
    
    # 关闭调试日志文件
    debug_log_file.close()
    print(f"\n调试日志已保存到: {args.debug_log}")


if __name__ == '__main__':
    main()

