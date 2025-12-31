"""
混合目标函数的OLM优化器

同时考虑：
1. LRobust（量化误差，代理目标）
2. FI后的准确率损失（真实目标）

设计思路：
- 使用加权组合：Loss = α * LRobust + (1 - α) * Accuracy_Loss
- 支持不同的优化策略（贪心、模拟退火）
- 可以动态调整权重α
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from util.olm_encoder import (
    collect_quantized_value_distribution,
    compute_lrobust,
    optimize_olm_mapping
)


def compute_accuracy_loss(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    model: nn.Module,
    layer_name: str,
    dataloader,
    fault_injector,
    criterion: nn.Module,
    device: torch.device,
    num_samples: int = 100  # 使用较小的验证集以加速
) -> float:
    """
    计算故障注入后的准确率损失
    
    Args:
        value_to_code: 量化值到编码的映射
        code_to_value: 编码到量化值的映射
        model: 模型
        layer_name: 层名称
        dataloader: 数据加载器（用于评估）
        fault_injector: 故障注入器
        criterion: 损失函数
        device: 设备
        num_samples: 采样数量（用于加速）
    
    Returns:
        准确率损失（1 - accuracy）
    """
    # 临时更新FaultInjector的OLM映射
    original_olm_layers = fault_injector.olm_layers.copy()
    original_olm_code_to_value = fault_injector.olm_code_to_value.copy()
    
    fault_injector.update_olm_mappings(
        {layer_name: value_to_code},
        {layer_name: code_to_value}
    )
    
    # 在验证集上评估
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        sample_count = 0
        for inputs, targets in dataloader:
            if sample_count >= num_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)  # 自动使用故障注入
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item() * targets.size(0)
            sample_count += targets.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    accuracy_loss = 1.0 - accuracy  # 准确率损失
    
    # 恢复原始映射
    fault_injector.olm_layers = original_olm_layers
    fault_injector.olm_code_to_value = original_olm_code_to_value
    
    model.train()
    return accuracy_loss


def optimize_olm_mapping_hybrid(
    distribution: Dict[int, int],
    k: int,
    model: nn.Module,
    layer_name: str,
    dataloader,
    fault_injector,
    criterion: nn.Module,
    device: torch.device,
    method: str = 'greedy',
    max_iterations: int = 1000,
    alpha: float = 0.5,  # LRobust权重
    num_samples: int = 100,  # 用于准确率评估的采样数量
    use_cached_accuracy: bool = True  # 是否缓存准确率评估结果
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    使用混合目标函数优化OLM映射
    
    Loss = α * LRobust + (1 - α) * Accuracy_Loss
    
    Args:
        distribution: 量化值分布
        k: 位宽
        model: 模型
        layer_name: 层名称
        dataloader: 数据加载器
        fault_injector: 故障注入器
        criterion: 损失函数
        device: 设备
        method: 优化方法 ('greedy' 或 'simulated_annealing')
        max_iterations: 最大迭代次数
        alpha: LRobust权重（0-1）
        num_samples: 用于准确率评估的采样数量
        use_cached_accuracy: 是否缓存准确率评估结果
    
    Returns:
        (value_to_code, code_to_value, best_loss)
    """
    n_levels = 1 << k
    thd_neg = -(1 << (k - 1))
    thd_pos = (1 << (k - 1)) - 1
    
    # 准确率评估缓存
    accuracy_cache: Dict[Tuple[int, ...], float] = {}
    
    def compute_hybrid_loss(value_to_code: Dict[int, int], code_to_value: Dict[int, int]) -> float:
        """计算混合损失"""
        # 计算LRobust
        lrobust = compute_lrobust(value_to_code, code_to_value, distribution, k)
        
        # 计算准确率损失（如果alpha < 1）
        accuracy_loss = 0.0
        if alpha < 1.0:
            # 使用缓存键（映射的排序元组）来缓存准确率
            mapping_key = tuple(sorted(value_to_code.items()))
            if use_cached_accuracy and mapping_key in accuracy_cache:
                accuracy_loss = accuracy_cache[mapping_key]
            else:
                accuracy_loss = compute_accuracy_loss(
                    value_to_code, code_to_value, model, layer_name,
                    dataloader, fault_injector, criterion, device, num_samples
                )
                if use_cached_accuracy:
                    accuracy_cache[mapping_key] = accuracy_loss
        
        # 混合损失
        hybrid_loss = alpha * lrobust + (1 - alpha) * accuracy_loss
        return hybrid_loss
    
    if method == 'greedy':
        return _greedy_search_hybrid(
            distribution, k, compute_hybrid_loss, n_levels, thd_neg, thd_pos
        )
    elif method == 'simulated_annealing':
        return _simulated_annealing_hybrid(
            distribution, k, compute_hybrid_loss, n_levels, thd_neg, thd_pos, max_iterations
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _greedy_search_hybrid(
    distribution: Dict[int, int],
    k: int,
    compute_loss: Callable[[Dict[int, int], Dict[int, int]], float],
    n_levels: int,
    thd_neg: int,
    thd_pos: int
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    使用贪心算法优化混合目标函数
    """
    import random
    
    # 获取所有可能的量化值（按频率排序）
    sorted_values = sorted(distribution.keys(), key=lambda x: -distribution.get(x, 0))
    
    value_to_code = {}
    code_to_value = {}
    used_codes = set()
    
    # 从最常见的值开始，贪心选择最优编码
    for value in sorted_values:
        if len(used_codes) >= n_levels:
            break
        
        if value in value_to_code:
            continue
        
        best_code = None
        best_loss = float('inf')
        
        # 尝试所有未使用的编码
        for code in range(n_levels):
            if code in used_codes:
                continue
            
            # 临时映射
            temp_value_to_code = value_to_code.copy()
            temp_code_to_value = code_to_value.copy()
            temp_value_to_code[value] = code
            temp_code_to_value[code] = value
            
            # 计算损失
            loss = compute_loss(temp_value_to_code, temp_code_to_value)
            
            if loss < best_loss:
                best_loss = loss
                best_code = code
        
        # 如果没找到，随机选择一个未使用的编码
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
    all_values = set(range(thd_neg, thd_pos + 1))
    unmapped_values = all_values - set(value_to_code.keys())
    unmapped_codes = set(range(n_levels)) - used_codes
    
    for value, code in zip(unmapped_values, unmapped_codes):
        value_to_code[value] = code
        code_to_value[code] = value
    
    # 计算最终损失
    final_loss = compute_loss(value_to_code, code_to_value)
    
    return value_to_code, code_to_value, final_loss


def _simulated_annealing_hybrid(
    distribution: Dict[int, int],
    k: int,
    compute_loss: Callable[[Dict[int, int], Dict[int, int]], float],
    n_levels: int,
    thd_neg: int,
    thd_pos: int,
    max_iterations: int
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    使用模拟退火算法优化混合目标函数
    """
    import random
    import math
    
    # 初始化：随机映射
    all_values = list(range(thd_neg, thd_pos + 1))
    all_codes = list(range(n_levels))
    
    random.shuffle(all_codes)
    value_to_code = {val: code for val, code in zip(all_values, all_codes)}
    code_to_value = {code: val for val, code in value_to_code.items()}
    
    current_loss = compute_loss(value_to_code, code_to_value)
    best_mapping = (dict(value_to_code), dict(code_to_value), current_loss)
    
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
        new_loss = compute_loss(value_to_code, code_to_value)
        
        # 接受或拒绝
        delta = new_loss - current_loss
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_loss = new_loss
            if new_loss < best_mapping[2]:
                best_mapping = (dict(value_to_code), dict(code_to_value), new_loss)
        else:
            # 回退
            value_to_code[val1], value_to_code[val2] = code1, code2
            code_to_value[code1], code_to_value[code2] = val1, val2
        
        # 降温
        temp *= cooling_rate
    
    return best_mapping




