"""
Optimal Label Mapping (OLM) Encoder

Implements the OLM encoding scheme that minimizes the average quantization error
caused by bit-flips. This is more robust than Gray Code when quantization values
are not uniformly distributed.

Reference: The method optimizes the mapping between 2^B quantized values and 2^B
binary codes to minimize: LRobust = Σ_i Σ_{j∈Hamming_1(i)} (c_map(i) - c_map(j))^2
"""

import itertools
import random
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from quan.func import QuanConv2d, QuanLinear


def collect_quantized_value_distribution(
    model: nn.Module,
    layer_name: str,
    num_samples: int = 1000
) -> Dict[int, int]:
    """
    收集指定层的量化值分布
    
    Args:
        model: 量化模型
        layer_name: 层名称
        num_samples: 采样数量（用于加速，如果为-1则使用全部权重）
        
    Returns:
        量化值到出现频率的映射 {quantized_value: frequency}
    """
    module = dict(model.named_modules()).get(layer_name)
    if module is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    if not isinstance(module, (QuanConv2d, QuanLinear)):
        raise ValueError(f"Layer {layer_name} is not a quantized layer")
    
    if not hasattr(module, 'quan_w_fn') or module.quan_w_fn is None:
        raise ValueError(f"Layer {layer_name} has no quantization function")
    
    # 检查是否有量化配置（支持 bits 和 fixed_bits）
    wbits = None
    if hasattr(module, 'bits') and module.bits is not None:
        wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
    
    if wbits is None:
        raise ValueError(f"Layer {layer_name} has no bit-width configuration (neither bits nor fixed_bits)")
    
    if isinstance(wbits, torch.Tensor):
        wbits = int(wbits.item())
    else:
        wbits = int(wbits)
    
    # 获取量化器
    quantizer = module.quan_w_fn
    scale = quantizer.get_scale(wbits, detach=True)
    
    # 计算量化阈值
    thd_neg = -(1 << (wbits - 1))
    thd_pos = (1 << (wbits - 1)) - 1
    
    # 获取量化后的权重
    with torch.no_grad():
        weight_q = quantizer(module.weight, wbits, is_activation=False)
        
        # 计算整数编码：code = round(weight_q / scale)
        if isinstance(scale, torch.Tensor):
            # 处理per-channel量化
            if scale.dim() > 0 and scale.numel() > 1:
                # 需要broadcast scale到weight的形状
                while scale.dim() < weight_q.dim():
                    scale = scale.unsqueeze(-1)
                code_f = torch.round(weight_q / scale)
            else:
                code_f = torch.round(weight_q / scale.item())
        else:
            code_f = torch.round(weight_q / scale)
        
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        code = code_f.int().cpu()
        
        # 采样（如果指定了采样数量）
        if num_samples > 0 and code.numel() > num_samples:
            flat_code = code.view(-1)
            indices = torch.randperm(flat_code.numel())[:num_samples]
            sampled_code = flat_code[indices]
        else:
            sampled_code = code.view(-1)
        
        # 统计频率
        distribution = {}
        for val in sampled_code.tolist():
            distribution[val] = distribution.get(val, 0) + 1
    
    return distribution


def compute_lrobust(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    k: int
) -> float:
    """
    计算LRobust损失
    
    Args:
        value_to_code: 量化值到编码的映射
        code_to_value: 编码到量化值的映射
        distribution: 量化值分布（频率）
        k: 位宽
        
    Returns:
        LRobust值
    """
    total_loss = 0.0
    total_weight = 0
    
    # 对于每个量化值
    for value, freq in distribution.items():
        if value not in value_to_code:
            continue
        
        code = value_to_code[value]
        
        # 找到所有Hamming距离为1的编码
        for bit_pos in range(k):
            neighbor_code = code ^ (1 << bit_pos)
            
            if neighbor_code in code_to_value:
                neighbor_value = code_to_value[neighbor_code]
                # 计算欧氏距离的平方
                error_sq = (value - neighbor_value) ** 2
                total_loss += error_sq * freq
                total_weight += freq
    
    return total_loss / total_weight if total_weight > 0 else float('inf')


def optimize_olm_mapping(
    distribution: Dict[int, int],
    k: int,
    method: str = 'greedy',
    max_iterations: int = 1000
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    优化OLM编码映射
    
    Args:
        distribution: 量化值分布 {value: frequency}
        k: 位宽
        method: 优化方法 ('greedy' 或 'simulated_annealing')
        max_iterations: 最大迭代次数
        
    Returns:
        (value_to_code, code_to_value, best_lrobust)
    """
    n_levels = 1 << k
    thd_neg = -(1 << (k - 1))
    thd_pos = (1 << (k - 1)) - 1
    
    # 获取所有可能的量化值（按频率排序）
    sorted_values = sorted(distribution.keys(), key=lambda x: -distribution.get(x, 0))
    
    # 初始化：将最常见的值映射到相邻的编码
    if method == 'greedy':
        return _greedy_search(distribution, k, sorted_values, n_levels, thd_neg, thd_pos)
    elif method == 'simulated_annealing':
        return _simulated_annealing(distribution, k, sorted_values, n_levels, thd_neg, thd_pos, max_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")


def _greedy_search(
    distribution: Dict[int, int],
    k: int,
    sorted_values: List[int],
    n_levels: int,
    thd_neg: int,
    thd_pos: int
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """贪婪搜索：将高频值映射到相邻编码"""
    value_to_code = {}
    code_to_value = {}
    used_codes = set()
    
    # 从最常见的值开始，尝试映射到相邻编码
    for value in sorted_values:
        if len(used_codes) >= n_levels:
            break
        
        # 如果已经有映射，跳过
        if value in value_to_code:
            continue
        
        # 尝试找到一个未使用的编码，优先选择与已映射编码相邻的
        best_code = None
        best_score = float('inf')
        
        for code in range(n_levels):
            if code in used_codes:
                continue
            
            # 计算与已映射编码的相邻度
            score = 0
            for mapped_code in used_codes:
                # 计算Hamming距离
                hamming_dist = bin(code ^ mapped_code).count('1')
                if hamming_dist == 1:
                    mapped_value = code_to_value[mapped_code]
                    # 如果量化值接近，优先选择
                    score += abs(value - mapped_value)
            
            if score < best_score:
                best_score = score
                best_code = code
        
        # 如果没找到相邻的，随机选择一个未使用的
        if best_code is None:
            for code in range(n_levels):
                if code not in used_codes:
                    best_code = code
                    break
        
        if best_code is not None:
            value_to_code[value] = best_code
            code_to_value[best_code] = value
            used_codes.add(best_code)
    
    # 填充未映射的值（使用随机映射）
    all_values = set(range(thd_neg, thd_pos + 1))
    unmapped_values = all_values - set(value_to_code.keys())
    unmapped_codes = set(range(n_levels)) - used_codes
    
    for value, code in zip(unmapped_values, unmapped_codes):
        value_to_code[value] = code
        code_to_value[code] = value
    
    # 计算LRobust
    lrobust = compute_lrobust(value_to_code, code_to_value, distribution, k)
    
    return value_to_code, code_to_value, lrobust


def _simulated_annealing(
    distribution: Dict[int, int],
    k: int,
    sorted_values: List[int],
    n_levels: int,
    thd_neg: int,
    thd_pos: int,
    max_iterations: int
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """模拟退火优化"""
    # 初始化：随机映射
    all_values = list(range(thd_neg, thd_pos + 1))
    all_codes = list(range(n_levels))
    
    random.shuffle(all_codes)
    value_to_code = {val: code for val, code in zip(all_values, all_codes)}
    code_to_value = {code: val for val, code in value_to_code.items()}
    
    current_lrobust = compute_lrobust(value_to_code, code_to_value, distribution, k)
    best_mapping = (dict(value_to_code), dict(code_to_value), current_lrobust)
    
    # 模拟退火参数
    initial_temp = 100.0
    final_temp = 0.1
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iterations)
    temp = initial_temp
    
    for iteration in range(max_iterations):
        # 随机交换两个编码的映射
        val1, val2 = random.sample(all_values, 2)
        code1, code2 = value_to_code[val1], value_to_code[val2]
        
        # 交换
        value_to_code[val1], value_to_code[val2] = code2, code1
        code_to_value[code1], code_to_value[code2] = val2, val1
        
        # 计算新损失
        new_lrobust = compute_lrobust(value_to_code, code_to_value, distribution, k)
        
        # 接受或拒绝
        delta = new_lrobust - current_lrobust
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_lrobust = new_lrobust
            if new_lrobust < best_mapping[2]:
                best_mapping = (dict(value_to_code), dict(code_to_value), new_lrobust)
        else:
            # 回退
            value_to_code[val1], value_to_code[val2] = code1, code2
            code_to_value[code1], code_to_value[code2] = val1, val2
        
        # 降温
        temp *= cooling_rate
    
    return best_mapping


def create_olm_encoder(
    model: nn.Module,
    layer_name: str,
    method: str = 'greedy',
    num_samples: int = 1000
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    为指定层创建OLM编码器
    
    Args:
        model: 量化模型
        layer_name: 层名称
        method: 优化方法 ('greedy' 或 'simulated_annealing')
        num_samples: 采样数量
        
    Returns:
        (value_to_code, code_to_value, lrobust)
    """
    # 收集分布
    distribution = collect_quantized_value_distribution(model, layer_name, num_samples)
    
    # 获取位宽
    module = dict(model.named_modules())[layer_name]
    wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
    if isinstance(wbits, torch.Tensor):
        wbits = int(wbits.item())
    else:
        wbits = int(wbits)
    
    # 优化映射
    value_to_code, code_to_value, lrobust = optimize_olm_mapping(
        distribution, wbits, method=method
    )
    
    return value_to_code, code_to_value, lrobust

