"""
改进的OLM编码器 V2：在方法2（多bit翻转）基础上的数学改进

改进点：
1. 基于频率的自适应权重（高频值不一定最重要）
2. 考虑量化值的分布特性（方差、偏度）
3. 考虑编码的局部性（相邻编码应该映射到相似值）
4. 自适应调整不同Hamming距离的权重
5. 考虑翻转后的累积影响（不只是单次误差）
"""

import torch
import torch.nn as nn
import math
import random
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from util.olm_encoder import compute_lrobust

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _comb(n: int, k: int) -> int:
    """计算组合数 C(n, k)"""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _get_codes_with_hamming_dist(code: int, hamming_dist: int, k: int) -> List[int]:
    """
    获取与给定编码的Hamming距离为hamming_dist的所有编码
    """
    if hamming_dist == 0:
        return [code]
    if hamming_dist > k:
        return []
    
    codes = []
    for bit_positions in combinations(range(k), hamming_dist):
        neighbor_code = code
        for bit_pos in bit_positions:
            neighbor_code ^= (1 << bit_pos)
        codes.append(neighbor_code)
    
    return codes


def compute_value_importance_by_distribution(
    distribution: Dict[int, int],
    k: int
) -> Dict[int, float]:
    """
    基于分布特性计算量化值的重要性
    
    改进点：
    1. 考虑频率的相对重要性（不是绝对频率）
    2. 考虑量化值的分布特性（方差、偏度）
    3. 考虑极端值的重要性（接近量化边界的值）
    
    Args:
        distribution: 量化值分布 {value: frequency}
        k: 位宽
    
    Returns:
        {value: importance} 字典，importance在[0, 1]范围内
    """
    if not distribution:
        return {}
    
    values = list(distribution.keys())
    frequencies = list(distribution.values())
    
    # 1. 频率归一化（相对重要性）
    max_freq = max(frequencies) if frequencies else 1
    freq_importance = {v: f / max_freq for v, f in distribution.items()}
    
    # 2. 极端值重要性（接近量化边界的值更重要）
    thd_neg = -(1 << (k - 1))
    thd_pos = (1 << (k - 1)) - 1
    
    extreme_importance = {}
    for v in values:
        # 计算到边界的距离
        dist_to_neg = abs(v - thd_neg)
        dist_to_pos = abs(v - thd_pos)
        min_dist = min(dist_to_neg, dist_to_pos)
        max_dist = max(abs(thd_neg), abs(thd_pos))
        
        # 越接近边界，重要性越高（归一化到[0.5, 1.0]）
        extreme_importance[v] = 0.5 + 0.5 * (1 - min_dist / max_dist) if max_dist > 0 else 1.0
    
    # 3. 分布方差重要性（如果某个值周围的值频率差异大，说明这个值特殊）
    variance_importance = {}
    for v in values:
        # 计算相邻值的频率方差
        neighbor_freqs = []
        for neighbor in [v-1, v+1]:
            if neighbor in distribution:
                neighbor_freqs.append(distribution[neighbor])
        
        if neighbor_freqs:
            neighbor_freqs.append(distribution[v])
            variance = np.var(neighbor_freqs) if len(neighbor_freqs) > 1 else 0
            # 归一化方差重要性
            variance_importance[v] = min(1.0, variance / (max_freq ** 2)) if max_freq > 0 else 0.5
        else:
            variance_importance[v] = 0.5
    
    # 4. 综合重要性（加权平均）
    importance = {}
    for v in values:
        # 权重：频率(0.4) + 极端值(0.3) + 方差(0.3)
        importance[v] = (
            0.4 * freq_importance[v] +
            0.3 * extreme_importance[v] +
            0.3 * variance_importance[v]
        )
    
    # 归一化到[0, 1]
    max_imp = max(importance.values()) if importance else 1.0
    if max_imp > 0:
        importance = {v: imp / max_imp for v, imp in importance.items()}
    
    return importance


def compute_hamming_distance_weights(
    k: int,
    ber: float,
    max_hamming_dist: int
) -> Dict[int, float]:
    """
    计算不同Hamming距离的自适应权重
    
    改进点：
    1. 不仅考虑概率，还考虑实际影响
    2. 高BER时，多bit翻转的权重应该更高
    3. 考虑Hamming距离的累积影响
    
    Args:
        k: 位宽
        ber: Bit-error-rate
        max_hamming_dist: 最大Hamming距离
    
    Returns:
        {hamming_dist: weight} 字典
    """
    weights = {}
    
    # 计算每种Hamming距离的概率
    probs = {}
    for d in range(1, max_hamming_dist + 1):
        if d > k:
            probs[d] = 0.0
        else:
            comb = _comb(k, d)
            prob = comb * (ber ** d) * ((1 - ber) ** (k - d))
            probs[d] = prob
    
    # 归一化概率
    total_prob = sum(probs.values())
    if total_prob > 0:
        probs = {d: p / total_prob for d, p in probs.items()}
    
    # 计算权重：概率 + 影响因子
    for d in range(1, max_hamming_dist + 1):
        prob = probs.get(d, 0.0)
        
        # 影响因子：Hamming距离越大，单次翻转的影响可能越大
        # 但也要考虑概率，所以使用 log(1 + d) 作为影响因子
        impact_factor = math.log(1 + d) / math.log(1 + max_hamming_dist)
        
        # 综合权重：概率(0.7) + 影响因子(0.3)
        weights[d] = 0.7 * prob + 0.3 * impact_factor
    
    # 归一化权重
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {d: w / total_weight for d, w in weights.items()}
    
    return weights


def compute_local_consistency_penalty(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    k: int
) -> float:
    """
    计算编码的局部一致性惩罚
    
    改进点：
    1. 相邻的编码应该映射到相似的量化值
    2. 如果相邻编码映射到差异很大的值，应该惩罚
    
    Args:
        value_to_code: 量化值到编码的映射
        code_to_value: 编码到量化值的映射
        k: 位宽
    
    Returns:
        局部一致性惩罚值（越小越好）
    """
    penalty = 0.0
    count = 0
    
    # 对于每个编码，检查其单bit邻居
    for code in range(1 << k):
        if code not in code_to_value:
            continue
        
        value = code_to_value[code]
        
        # 找到所有单bit邻居
        neighbors = _get_codes_with_hamming_dist(code, 1, k)
        
        for neighbor_code in neighbors:
            if neighbor_code in code_to_value:
                neighbor_value = code_to_value[neighbor_code]
                # 计算量化值的差异
                value_diff = abs(value - neighbor_value)
                # 惩罚：差异越大，惩罚越大
                penalty += value_diff ** 2
                count += 1
    
    return penalty / count if count > 0 else 0.0


def compute_lrobust_improved_v2(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    k: int,
    ber: float = 1e-2,
    consider_multi_bit: bool = True,
    max_hamming_dist: int = 3,
    use_value_importance: bool = True,
    use_local_consistency: bool = True,
    local_consistency_weight: float = 0.1
) -> float:
    """
    改进的LRobust V2：在方法2基础上的数学改进
    
    改进点：
    1. 基于分布特性的值重要性（而不是梯度）
    2. 自适应Hamming距离权重
    3. 局部一致性惩罚
    
    Args:
        value_to_code: 量化值到编码的映射
        code_to_value: 编码到量化值的映射
        distribution: 量化值分布
        k: 位宽
        ber: Bit-error-rate
        consider_multi_bit: 是否考虑多bit翻转
        max_hamming_dist: 最大Hamming距离
        use_value_importance: 是否使用值重要性
        use_local_consistency: 是否使用局部一致性惩罚
        local_consistency_weight: 局部一致性权重
    
    Returns:
        改进的LRobust值
    """
    total_loss = 0.0
    total_weight = 0.0
    
    # 1. 计算值重要性（如果启用）
    value_importance = None
    if use_value_importance:
        value_importance = compute_value_importance_by_distribution(distribution, k)
    else:
        value_importance = {v: 1.0 for v in distribution.keys()}
    
    # 2. 计算Hamming距离权重
    hamming_weights = compute_hamming_distance_weights(k, ber, max_hamming_dist)
    
    # 3. 计算多bit翻转的概率
    def prob_d_bit_flip(d: int, k: int, ber: float) -> float:
        if d == 0:
            return (1 - ber) ** k
        elif d > k:
            return 0.0
        else:
            comb = _comb(k, d)
            return comb * (ber ** d) * ((1 - ber) ** (k - d))
    
    # 4. 对于每个量化值，计算损失
    for value, freq in distribution.items():
        if value not in value_to_code:
            continue
        
        code = value_to_code[value]
        importance = value_importance.get(value, 1.0)
        
        # 考虑不同Hamming距离的翻转
        for hamming_dist in range(1, max_hamming_dist + 1 if consider_multi_bit else 2):
            # 计算概率
            prob = prob_d_bit_flip(hamming_dist, k, ber)
            
            if prob < 1e-10:
                continue
            
            # 获取Hamming距离权重
            hamming_weight = hamming_weights.get(hamming_dist, prob)
            
            # 找到所有Hamming距离为hamming_dist的编码
            neighbor_codes = _get_codes_with_hamming_dist(code, hamming_dist, k)
            
            for neighbor_code in neighbor_codes:
                if neighbor_code in code_to_value:
                    neighbor_value = code_to_value[neighbor_code]
                    # 计算量化误差的平方
                    error_sq = (value - neighbor_value) ** 2
                    
                    # 加权损失：error² * freq * importance * prob * hamming_weight
                    weight = freq * importance * prob * hamming_weight
                    total_loss += error_sq * weight
                    total_weight += weight
    
    # 5. 计算局部一致性惩罚（如果启用）
    local_penalty = 0.0
    if use_local_consistency:
        local_penalty = compute_local_consistency_penalty(value_to_code, code_to_value, k)
    
    # 6. 综合损失
    lrobust = (total_loss / total_weight if total_weight > 0 else float('inf'))
    total_loss_with_penalty = lrobust + local_consistency_weight * local_penalty
    
    return total_loss_with_penalty


def optimize_olm_mapping_improved_v2(
    distribution: Dict[int, int],
    k: int,
    method: str = 'simulated_annealing',
    max_iterations: int = 3000,
    ber: float = 1e-2,
    consider_multi_bit: bool = True,
    max_hamming_dist: int = 3,
    use_value_importance: bool = True,
    use_local_consistency: bool = True,
    local_consistency_weight: float = 0.1,
    # 遗传算法参数
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    改进的OLM映射优化 V2：在方法2基础上的数学改进
    
    Args:
        distribution: 量化值分布
        k: 位宽
        method: 优化方法
        max_iterations: 最大迭代次数
        ber: Bit-error-rate
        consider_multi_bit: 是否考虑多bit翻转
        max_hamming_dist: 最大Hamming距离
        use_value_importance: 是否使用值重要性
        use_local_consistency: 是否使用局部一致性
        local_consistency_weight: 局部一致性权重
    
    Returns:
        (value_to_code, code_to_value, best_lrobust)
    """
    import random
    import math
    
    # 获取所有量化值
    values = sorted(distribution.keys())
    n_values = len(values)
    n_codes = 1 << k
    
    if n_values > n_codes:
        # 如果量化值数量超过编码数量，只使用频率最高的n_codes个
        sorted_values = sorted(values, key=lambda v: distribution[v], reverse=True)
        values = sorted_values[:n_codes]
        n_values = len(values)
    
    # 初始化映射（随机或贪婪）
    if method == 'greedy':
        # 贪婪初始化：频率高的值优先分配好编码
        codes = list(range(n_codes))
        random.shuffle(codes)
        
        # 按频率排序
        sorted_values = sorted(values, key=lambda v: distribution[v], reverse=True)
        
        value_to_code = {}
        code_to_value = {}
        
        for i, value in enumerate(sorted_values):
            if i < len(codes):
                code = codes[i]
                value_to_code[value] = code
                code_to_value[code] = value
        
        # 填充剩余的编码
        for code in range(n_codes):
            if code not in code_to_value:
                # 随机分配一个未使用的值
                unused_values = [v for v in values if v not in value_to_code]
                if unused_values:
                    value = random.choice(unused_values)
                    value_to_code[value] = code
                    code_to_value[code] = value
                else:
                    # 如果没有未使用的值，使用最近的值
                    closest_value = min(values, key=lambda v: abs(v - (code - (1 << (k-1)))))
                    code_to_value[code] = closest_value
    else:
        # 随机初始化
        codes = list(range(n_codes))
        random.shuffle(codes)
        
        value_to_code = {values[i]: codes[i] for i in range(min(n_values, n_codes))}
        code_to_value = {codes[i]: values[i] for i in range(min(n_values, n_codes))}
        
        # 填充剩余的编码
        for code in range(n_codes):
            if code not in code_to_value:
                closest_value = min(values, key=lambda v: abs(v - (code - (1 << (k-1)))))
                code_to_value[code] = closest_value
    
    # 计算初始损失
    best_lrobust = compute_lrobust_improved_v2(
        value_to_code, code_to_value, distribution, k,
        ber, consider_multi_bit, max_hamming_dist,
        use_value_importance, use_local_consistency, local_consistency_weight
    )
    best_mapping = (dict(value_to_code), dict(code_to_value), best_lrobust)
    
    if method == 'greedy':
        return best_mapping
    
    # 模拟退火
    temperature = 1.0
    cooling_rate = 0.99
    min_temperature = 0.01
    
    current_mapping = (dict(value_to_code), dict(code_to_value), best_lrobust)
    
    for iteration in range(max_iterations):
        # 随机交换两个值的编码
        if len(values) < 2:
            break
        
        v1, v2 = random.sample(values, 2)
        
        if v1 in value_to_code and v2 in value_to_code:
            code1 = value_to_code[v1]
            code2 = value_to_code[v2]
            
            # 交换编码
            new_value_to_code = dict(value_to_code)
            new_code_to_value = dict(code_to_value)
            
            new_value_to_code[v1] = code2
            new_value_to_code[v2] = code1
            new_code_to_value[code1] = v2
            new_code_to_value[code2] = v1
            
            # 计算新损失
            new_lrobust = compute_lrobust_improved_v2(
                new_value_to_code, new_code_to_value, distribution, k,
                ber, consider_multi_bit, max_hamming_dist,
                use_value_importance, use_local_consistency, local_consistency_weight
            )
            
            # 接受或拒绝
            delta = new_lrobust - current_mapping[2]
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_mapping = (new_value_to_code, new_code_to_value, new_lrobust)
                
                if new_lrobust < best_lrobust:
                    best_lrobust = new_lrobust
                    best_mapping = current_mapping
        
        # 降温
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
    
    return best_mapping

