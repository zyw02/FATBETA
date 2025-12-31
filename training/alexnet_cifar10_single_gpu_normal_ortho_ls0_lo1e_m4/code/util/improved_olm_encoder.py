"""
改进的OLM编码器：考虑梯度信息和多bit翻转

改进点：
1. 基于梯度信息衡量权重的重要性
2. 考虑多bit翻转（高BER情况下）
3. 使用加权LRobust
"""

import torch
import torch.nn as nn
import math
import random
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from util.olm_encoder import compute_lrobust


def compute_weight_importance_by_gradient(
    model: nn.Module,
    layer_name: str,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    num_samples: int = 100
) -> Dict[int, float]:
    """
    通过梯度信息计算权重的重要性
    
    Args:
        model: 模型
        layer_name: 层名称
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        num_samples: 采样数量
    
    Returns:
        {quantized_value: importance_weight} 字典
    """
    model.eval()
    module = dict(model.named_modules())[layer_name]
    
    # 获取量化器
    if not hasattr(module, 'quan_w_fn') or module.quan_w_fn is None:
        raise ValueError(f"Layer {layer_name} has no quantization function")
    
    quantizer = module.quan_w_fn
    
    # 获取位宽
    wbits = None
    if hasattr(module, 'bits') and module.bits is not None:
        wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
    elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
        wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
    
    if wbits is None:
        raise ValueError(f"Layer {layer_name} has no bit-width configuration")
    
    if isinstance(wbits, torch.Tensor):
        wbits = int(wbits.item())
    else:
        wbits = int(wbits)
    
    # 收集梯度信息
    gradient_sum = {}
    gradient_count = {}
    
    sample_count = 0
    for inputs, targets in dataloader:
        if sample_count >= num_samples:
            break
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 获取权重梯度
        if module.weight.grad is not None:
            weight_grad = module.weight.grad.data.abs()
            
            # 量化权重
            with torch.no_grad():
                weight_q = quantizer(module.weight, wbits, is_activation=False)
                scale = quantizer.get_scale(wbits, detach=True)
                
                # 计算量化值
                if isinstance(scale, torch.Tensor):
                    code_f = torch.round(weight_q / scale)
                else:
                    code_f = torch.round(weight_q / scale.item())
                
                code_f = torch.clamp(code_f, -(1 << (wbits - 1)), (1 << (wbits - 1)) - 1)
                code_int = code_f.int()
                
                # 统计每个量化值的梯度
                flat_grad = weight_grad.view(-1)
                flat_code = code_int.view(-1)
                
                for grad_val, code_val in zip(flat_grad.cpu().numpy(), flat_code.cpu().numpy()):
                    if code_val not in gradient_sum:
                        gradient_sum[code_val] = 0.0
                        gradient_count[code_val] = 0
                    gradient_sum[code_val] += float(grad_val)
                    gradient_count[code_val] += 1
        
        sample_count += inputs.size(0)
    
    # 计算平均梯度（作为重要性权重）
    importance = {}
    for code_val in gradient_sum:
        if gradient_count[code_val] > 0:
            importance[code_val] = gradient_sum[code_val] / gradient_count[code_val]
    
    # 归一化到[0, 1]范围
    if importance:
        max_importance = max(importance.values())
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}
    
    return importance


def compute_lrobust_weighted(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    k: int,
    weight_importance: Optional[Dict[int, float]] = None,
    ber: float = 1e-2,
    consider_multi_bit: bool = True,
    max_hamming_dist: int = 3
) -> float:
    """
    改进的LRobust计算：考虑权重重要性和多bit翻转
    
    Args:
        value_to_code: 量化值到编码的映射
        code_to_value: 编码到量化值的映射
        distribution: 量化值分布（频率）
        k: 位宽
        weight_importance: 权重重要性字典 {value: importance}
        ber: Bit-error-rate
        consider_multi_bit: 是否考虑多bit翻转
        max_hamming_dist: 最大Hamming距离（考虑多bit翻转时）
    
    Returns:
        改进的LRobust值
    """
    total_loss = 0.0
    total_weight = 0
    
    # 如果没有提供权重重要性，使用均匀权重
    if weight_importance is None:
        weight_importance = {v: 1.0 for v in distribution.keys()}
    
    # 计算多bit翻转的概率
    # P(d bit flip) = C(k, d) * ber^d * (1-ber)^(k-d)
    def prob_d_bit_flip(d: int, k: int, ber: float) -> float:
        if d == 0:
            return (1 - ber) ** k
        elif d > k:
            return 0.0
        else:
            # 组合数 C(k, d)
            comb = math.comb(k, d) if hasattr(math, 'comb') else _comb(k, d)
            return comb * (ber ** d) * ((1 - ber) ** (k - d))
    
    # 对于每个量化值
    for value, freq in distribution.items():
        if value not in value_to_code:
            continue
        
        code = value_to_code[value]
        importance = weight_importance.get(value, 1.0)  # 默认重要性为1.0
        
        # 考虑不同Hamming距离的翻转
        for hamming_dist in range(1, max_hamming_dist + 1 if consider_multi_bit else 2):
            # 计算该Hamming距离的翻转概率
            prob = prob_d_bit_flip(hamming_dist, k, ber)
            
            if prob < 1e-10:  # 忽略概率过小的翻转
                continue
            
            # 找到所有Hamming距离为hamming_dist的编码
            neighbor_codes = _get_codes_with_hamming_dist(code, hamming_dist, k)
            
            for neighbor_code in neighbor_codes:
                if neighbor_code in code_to_value:
                    neighbor_value = code_to_value[neighbor_code]
                    # 计算欧氏距离的平方
                    error_sq = (value - neighbor_value) ** 2
                    # 加权损失：error^2 * freq * importance * prob
                    total_loss += error_sq * freq * importance * prob
                    total_weight += freq * importance * prob
    
    return total_loss / total_weight if total_weight > 0 else float('inf')


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
    
    Args:
        code: 原始编码
        hamming_dist: Hamming距离
        k: 位宽
    
    Returns:
        所有满足Hamming距离的编码列表
    """
    if hamming_dist == 0:
        return [code]
    if hamming_dist > k:
        return []
    
    codes = []
    # 生成所有可能的bit位置组合
    for bit_positions in combinations(range(k), hamming_dist):
        neighbor_code = code
        for bit_pos in bit_positions:
            neighbor_code ^= (1 << bit_pos)
        codes.append(neighbor_code)
    
    return codes


def optimize_olm_mapping_improved(
    distribution: Dict[int, int],
    k: int,
    method: str = 'simulated_annealing',
    max_iterations: int = 3000,
    weight_importance: Optional[Dict[int, float]] = None,
    ber: float = 1e-2,
    consider_multi_bit: bool = True,
    max_hamming_dist: int = 3
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    改进的OLM映射优化：考虑权重重要性和多bit翻转
    
    Args:
        distribution: 量化值分布
        k: 位宽
        method: 优化方法
        max_iterations: 最大迭代次数
        weight_importance: 权重重要性字典
        ber: Bit-error-rate
        consider_multi_bit: 是否考虑多bit翻转
        max_hamming_dist: 最大Hamming距离
    
    Returns:
        (value_to_code, code_to_value, best_lrobust)
    """
    from util.olm_encoder import optimize_olm_mapping
    
    # 如果提供了权重重要性，使用改进的LRobust
    if weight_importance is not None or consider_multi_bit:
        return _optimize_with_improved_lrobust(
            distribution, k, method, max_iterations,
            weight_importance, ber, consider_multi_bit, max_hamming_dist
        )
    else:
        # 否则使用原始方法
        return optimize_olm_mapping(distribution, k, method, max_iterations)


def _optimize_with_improved_lrobust(
    distribution: Dict[int, int],
    k: int,
    method: str,
    max_iterations: int,
    weight_importance: Optional[Dict[int, float]],
    ber: float,
    consider_multi_bit: bool,
    max_hamming_dist: int
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """使用改进的LRobust进行优化"""
    import random
    import math
    
    n_levels = 1 << k
    thd_neg = -(1 << (k - 1))
    thd_pos = (1 << (k - 1)) - 1
    
    # 获取所有可能的量化值
    all_values = list(range(thd_neg, thd_pos + 1))
    all_codes = list(range(n_levels))
    
    if method == 'greedy':
        # 贪婪搜索：优先考虑重要性高的值
        sorted_values = sorted(
            all_values,
            key=lambda x: -weight_importance.get(x, 1.0) * distribution.get(x, 0)
        )
        
        value_to_code = {}
        code_to_value = {}
        used_codes = set()
        
        for value in sorted_values:
            if len(used_codes) >= n_levels:
                break
            
            best_code = None
            best_score = float('inf')
            
            for code in range(n_levels):
                if code in used_codes:
                    continue
                
                # 计算与已映射编码的相邻度（考虑重要性）
                score = 0
                for mapped_code in used_codes:
                    hamming_dist = bin(code ^ mapped_code).count('1')
                    if hamming_dist == 1:
                        mapped_value = code_to_value[mapped_code]
                        mapped_importance = weight_importance.get(mapped_value, 1.0)
                        score += abs(value - mapped_value) * mapped_importance
                
                if score < best_score:
                    best_score = score
                    best_code = code
            
            if best_code is None:
                for code in range(n_levels):
                    if code not in used_codes:
                        best_code = code
                        break
            
            if best_code is not None:
                value_to_code[value] = best_code
                code_to_value[best_code] = value
                used_codes.add(best_code)
        
        # 填充未映射的值
        unmapped_values = set(all_values) - set(value_to_code.keys())
        unmapped_codes = set(range(n_levels)) - used_codes
        for value, code in zip(unmapped_values, unmapped_codes):
            value_to_code[value] = code
            code_to_value[code] = value
        
        best_lrobust = compute_lrobust_weighted(
            value_to_code, code_to_value, distribution, k,
            weight_importance, ber, consider_multi_bit, max_hamming_dist
        )
        
        return (value_to_code, code_to_value, best_lrobust)
    
    elif method == 'simulated_annealing':
        # 模拟退火
        random.shuffle(all_codes)
        value_to_code = {val: code for val, code in zip(all_values, all_codes)}
        code_to_value = {code: val for val, code in value_to_code.items()}
        
        current_lrobust = compute_lrobust_weighted(
            value_to_code, code_to_value, distribution, k,
            weight_importance, ber, consider_multi_bit, max_hamming_dist
        )
        best_mapping = (dict(value_to_code), dict(code_to_value), current_lrobust)
        
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
            new_lrobust = compute_lrobust_weighted(
                value_to_code, code_to_value, distribution, k,
                weight_importance, ber, consider_multi_bit, max_hamming_dist
            )
            
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
            
            temp *= cooling_rate
        
        return best_mapping
    else:
        raise ValueError(f"Unknown method: {method}")

