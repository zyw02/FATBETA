"""
增强的OLM编码器：基于Gemini建议的改进

改进点：
1. Hessian感知的加权目标函数（使用梯度平方和作为敏感度权重）
2. 多对一映射（利用空闲编码空间，为高频/高敏感值分配多个编码）
3. 支持遗传算法搜索（可选）

基于原始OLM方案逐步改进
"""

import torch
import torch.nn as nn
import math
import random
from typing import Dict, List, Tuple, Optional, Set
from util.olm_encoder import compute_lrobust, collect_quantized_value_distribution

# 确保math模块可用
if not hasattr(math, 'isinf'):
    def isinf(x):
        return x == float('inf') or x == float('-inf')
    math.isinf = isinf
if not hasattr(math, 'isnan'):
    def isnan(x):
        return x != x
    math.isnan = isnan


def collect_quantized_value_distribution_with_sensitivity(
    model: nn.Module,
    layer_name: str,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    num_samples: int = -1
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    收集量化值分布和敏感度（梯度的平方累计和）
    
    Args:
        model: 模型
        layer_name: 层名称
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        num_samples: 采样数量，-1表示使用整个训练集
    
    Returns:
        (distribution, sensitivity) 元组
        - distribution: {value: frequency} 量化值频率分布
        - sensitivity: {value: sensitivity} 量化值敏感度（梯度平方的累计和，用于排序重要度）
    """
    # 注意：我们需要在train模式下收集梯度，因为eval模式可能会影响某些层的梯度计算
    # 但为了统计分布，我们可以先用eval模式
    model.eval()
    module = dict(model.named_modules())[layer_name]
    
    # 获取量化器
    if not hasattr(module, 'quan_w_fn') or module.quan_w_fn is None:
        raise ValueError(f"Layer {layer_name} has no quantization function")
    
    quantizer = module.quan_w_fn
    
    # 临时启用所有参数的梯度计算（解除freezing限制）
    # 保存原始状态
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = True  # 临时启用梯度计算
    
    # 特别确保目标层的权重可以计算梯度
    if hasattr(module, 'weight') and module.weight is not None:
        module.weight.requires_grad = True
    
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
    
    # 第一步：统计完整的量化值分布（使用完整权重，不依赖梯度）
    # 这样可以确保统计到所有唯一值，与传统方法一致
    print(f"    第一步：统计完整量化值分布（使用所有权重）...")
    frequency = {}
    with torch.no_grad():
        weight = module.weight.data
        weight_q = quantizer(weight, wbits, is_activation=False)
        
        # 使用与传统方法相同的量化逻辑
        scale = quantizer.get_scale(wbits, detach=True)
        thd_neg = -(1 << (wbits - 1))
        thd_pos = (1 << (wbits - 1)) - 1
        
        # 计算整数编码：code = round(weight_q / scale)
        if isinstance(scale, torch.Tensor):
            if scale.dim() > 0 and scale.numel() > 1:
                while scale.dim() < weight_q.dim():
                    scale = scale.unsqueeze(-1)
                code_f = torch.round(weight_q / scale)
            else:
                code_f = torch.round(weight_q / scale.item())
        else:
            code_f = torch.round(weight_q / scale)
        
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        code = code_f.int().cpu()
        
        # 统计所有量化值的频率
        for val in code.view(-1).tolist():
            frequency[val] = frequency.get(val, 0) + 1
    
    print(f"    统计到 {len(frequency)} 个唯一量化值")
    
    # 第二步：收集梯度信息用于计算敏感度
    # 切换到train模式以确保梯度计算正常（特别是BatchNorm等层）
    model.train()
    print(f"    第二步：收集梯度信息用于计算敏感度（train模式）...")
    gradient_sq_sum = {}
    gradient_count = {}
    
    sample_count = 0
    for inputs, targets in dataloader:
        # 如果指定了采样数量且已达到，则停止
        if num_samples > 0 and sample_count >= num_samples:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播（现在所有参数都应该有梯度，因为我们已经启用了requires_grad）
        loss.backward()
        
        # 获取梯度（用于计算敏感度）
        # 注意：由于我们已经临时启用了所有参数的梯度，这里应该总是有梯度
        if module.weight.grad is not None:
            # 检查梯度是否全为0
            grad_norm = module.weight.grad.norm().item()
            if grad_norm == 0 and sample_count == 0:
                print(f"    ⚠️  警告：{layer_name}的梯度全为0（可能模型已收敛或损失函数无贡献）")
            grad = module.weight.grad.data
            weight = module.weight.data
            
            # 量化权重（用于对齐）
            with torch.no_grad():
                weight_q = quantizer(weight, wbits, is_activation=False)
                scale = quantizer.get_scale(wbits, detach=True)
                thd_neg = -(1 << (wbits - 1))
                thd_pos = (1 << (wbits - 1)) - 1
                
                if isinstance(scale, torch.Tensor):
                    if scale.dim() > 0 and scale.numel() > 1:
                        while scale.dim() < weight_q.dim():
                            scale = scale.unsqueeze(-1)
                        code_f = torch.round(weight_q / scale)
                    else:
                        code_f = torch.round(weight_q / scale.item())
                else:
                    code_f = torch.round(weight_q / scale)
                
                code_f = torch.clamp(code_f, thd_neg, thd_pos)
                code = code_f.int().cpu()
            
            grad_flat = grad.flatten().cpu()
            code_flat = code.view(-1).cpu()
            
            # 对齐长度
            min_len = min(len(code_flat), len(grad_flat))
            code_flat_aligned = code_flat[:min_len]
            grad_flat_aligned = grad_flat[:min_len]
            
            # 累加梯度平方和（用于计算敏感度）
            for code_val, g_val in zip(code_flat_aligned.tolist(), grad_flat_aligned.tolist()):
                # 累加梯度平方
                gradient_sq_sum[code_val] = gradient_sq_sum.get(code_val, 0.0) + (g_val ** 2)
                gradient_count[code_val] = gradient_count.get(code_val, 0) + 1
        
        sample_count += inputs.size(0)
        
        # 如果指定了采样数量且已达到，则停止
        if num_samples > 0 and sample_count >= num_samples:
            break
    
    # 统计有梯度的量化值数量
    values_with_gradient = len([v for v in frequency.keys() if v in gradient_sq_sum])
    print(f"    有梯度信息的量化值: {values_with_gradient}/{len(frequency)}")
    if values_with_gradient > 0:
        total_grad_sq_sum = sum(gradient_sq_sum.values())
        print(f"    总梯度平方和: {total_grad_sq_sum:.6f}")
        # 归一化后的敏感度范围
        normalized_sens = {v: g / total_grad_sq_sum for v, g in gradient_sq_sum.items()}
        if normalized_sens:
            max_sens = max(normalized_sens.values())
            min_sens = min(normalized_sens.values())
            print(f"    归一化敏感度范围: [{min_sens:.6f}, {max_sens:.6f}]")
    
    # 恢复原始梯度状态和模型模式
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad = original_requires_grad[name]
    model.eval()  # 恢复eval模式
    
    # 计算敏感度（使用累计梯度平方和，然后归一化）
    # 敏感度 = 每个量化值的梯度平方累计和 / 总梯度平方和
    sensitivity = {}
    total_grad_sq_sum = sum(gradient_sq_sum.values())
    
    if total_grad_sq_sum == 0:
        print(f"    ⚠️  警告：没有收集到任何梯度信息！")
        print(f"    可能原因：")
        print(f"      1. 模型参数被freeze（requires_grad=False）- 已临时启用")
        print(f"      2. 损失函数对目标层无贡献")
        print(f"      3. 模型已完全收敛，梯度为0")
        print(f"    将使用均匀敏感度（1/N）")
        # 使用均匀敏感度
        n_values = len(frequency)
        for value in frequency.keys():
            sensitivity[value] = 1.0 / n_values if n_values > 0 else 1.0
    else:
        # 归一化：每个值的敏感度 = 该值的梯度平方和 / 总梯度平方和
        for value in frequency.keys():
            if value in gradient_sq_sum:
                sensitivity[value] = gradient_sq_sum[value] / total_grad_sq_sum
            else:
                # 如果没有梯度信息，敏感度为0（该值不重要）
                sensitivity[value] = 0.0
    
    return frequency, sensitivity


def compute_lrobust_with_sensitivity(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    sensitivity: Dict[int, float],
    k: int
) -> float:
    """
    计算带敏感度权重的LRobust
    
    改进公式：
    L_Robust = Σ_v Σ_{j∈H_1(code(v))} (v - value(j))² * S(v)
    
    其中 S(v) 是量化值v的归一化敏感度（梯度平方累计和 / 总梯度平方和）
    
    Args:
        value_to_code: 量化值到编码的映射
        code_to_value: 编码到量化值的映射
        distribution: 量化值分布（频率，仅用于遍历）
        sensitivity: 量化值敏感度（归一化的梯度平方累计和，范围[0,1]）
        k: 位宽
    
    Returns:
        改进的LRobust值
    """
    total_loss = 0.0
    total_weight = 0.0
    
    # 对于每个量化值
    for value, freq in distribution.items():
        if value not in value_to_code:
            continue
        
        code = value_to_code[value]
        sens = sensitivity.get(value, 0.0)  # 默认敏感度为0.0（如果没梯度信息）
        
        # 找到所有Hamming距离为1的编码
        for bit_pos in range(k):
            neighbor_code = code ^ (1 << bit_pos)
            
            if neighbor_code in code_to_value:
                neighbor_value = code_to_value[neighbor_code]
                # 计算欧氏距离的平方
                error_sq = (value - neighbor_value) ** 2
                # 加权损失：error² * sensitivity（去掉频率）
                weight = sens
                total_loss += error_sq * weight
                total_weight += weight
    
    return total_loss / total_weight if total_weight > 0 else float('inf')


def _popcount(x: int) -> int:
    """Count set bits in an integer (Python 3.8+ compatible)."""
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


def compute_bsc_mse_with_weights(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    sensitivity: Dict[int, float],
    k: int,
    ber: float,
    weight_mode: str = "freq_sens",
    max_hamming: Optional[int] = None,
) -> float:
    """
    面向 BSC/BER 的期望失真目标（方案B核心）：最小化 bit-flip 信道下的期望 MSE。
    
    对每个原始值 v（其编码为 c=value_to_code[v]），在独立 bit-flip（BER=p）的 BSC 模型下，
    接收码为 c' = c XOR M，其中 M 的每一位独立以概率 p 翻转。
    解码输出为 v' = code_to_value[c']，目标是最小化 E[(v - v')^2]。
    
    加权方式：
    - weight_mode='sens'：仅用敏感度 S(v) 加权（与现有 LRobust 近似一致）
    - weight_mode='freq'：仅用频率 freq(v) 加权
    - weight_mode='freq_sens'：freq(v) * S(v) 加权（推荐：同时考虑出现概率与重要性）
    
    Args:
        value_to_code: 值到编码（注：应尽量为双射/置换，避免信息损失）
        code_to_value: 编码到值（建议覆盖全部 0..2^k-1）
        distribution: {value: frequency}
        sensitivity: {value: sensitivity}（建议已归一化到[0,1]）
        k: bit 数
        ber: bit error rate p（0..1）
        weight_mode: 权重模式
        max_hamming: 若指定，仅累计 Hamming<=max_hamming 的项（近似/加速；None 表示全考虑）
    
    Returns:
        加权期望 MSE（越小越好）
    """
    p = float(ber)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Invalid ber={ber}, expected in [0, 1].")
    
    n_codes = 1 << k
    # 预计算每个翻转 mask 的概率（由其汉明重量决定）
    mask_probs = [0.0] * n_codes
    for mask in range(n_codes):
        d = _popcount(mask)
        if max_hamming is not None and d > max_hamming:
            continue
        mask_probs[mask] = (p ** d) * ((1.0 - p) ** (k - d))
    
    total_loss = 0.0
    total_weight = 0.0
    
    for value, freq in distribution.items():
        if value not in value_to_code:
            continue
        code = int(value_to_code[value])
        sens = float(sensitivity.get(value, 0.0))
        
        if weight_mode == "sens":
            w = sens
        elif weight_mode == "freq":
            w = float(freq)
        elif weight_mode == "freq_sens":
            w = float(freq) * sens
        else:
            raise ValueError(f"Unknown weight_mode='{weight_mode}'")
        
        if w == 0.0:
            continue
        
        exp_err = 0.0
        for mask in range(n_codes):
            prob = mask_probs[mask]
            if prob == 0.0:
                continue
            decoded_value = code_to_value.get(code ^ mask, value)
            diff = float(value - decoded_value)
            exp_err += prob * (diff * diff)
        
        total_loss += w * exp_err
        total_weight += w
    
    return total_loss / total_weight if total_weight > 0 else float("inf")


def create_surjective_mapping(
    distribution: Dict[int, int],
    sensitivity: Dict[int, float],
    k: int,
    top_k_values: int = 10
) -> Dict[int, Set[int]]:
    """
    创建多对一映射（Surjective Mapping）
    
    为高频/高敏感的量化值分配多个编码，利用空闲编码空间
    
    Args:
        distribution: 量化值分布
        sensitivity: 量化值敏感度
        k: 位宽
        top_k_values: 为前k个最重要的值分配多个编码
    
    Returns:
        {value: set of codes} 多对一映射
    """
    n_codes = 1 << k
    n_values = len(distribution)
    n_free = n_codes - n_values
    
    # 如果只有一个或没有值，返回简单映射
    if n_values <= 1:
        if n_values == 1:
            value = list(distribution.keys())[0]
            # 所有编码都映射到这个值
            return {value: set(range(n_codes))}
        else:
            # 没有值，返回空映射
            return {}
    
    if n_free <= 0:
        # 没有空闲编码，返回一对一映射
        codes_list = list(range(n_codes))
        return {v: {codes_list[i]} for i, v in enumerate(distribution.keys())}
    
    # 计算每个值的重要性分数：只使用敏感度（归一化的梯度平方和）
    importance_scores = {}
    for value in distribution.keys():
        sens = sensitivity.get(value, 0.0)  # 默认敏感度为0.0
        importance_scores[value] = sens
    
    # 按重要性排序
    sorted_values = sorted(importance_scores.keys(), key=lambda v: importance_scores[v], reverse=True)
    
    # 为前top_k_values个值分配多个编码
    value_to_codes = {}
    used_codes = set()
    
    # 首先为所有值分配一个基础编码
    for i, value in enumerate(sorted_values):
        if i < n_codes:
            code = i
            value_to_codes[value] = {code}
            used_codes.add(code)
        else:
            # 如果值太多，使用最近的编码
            value_to_codes[value] = {min(used_codes, key=lambda c: abs(c - (value % n_codes)))}
    
    # 为前top_k_values个值分配额外的编码（从空闲编码中）
    free_codes = set(range(n_codes)) - used_codes
    free_codes_list = sorted(list(free_codes))
    
    # 为每个重要值分配额外的编码
    codes_per_value = max(1, n_free // top_k_values) if top_k_values > 0 else 1
    
    for i, value in enumerate(sorted_values[:top_k_values]):
        if len(free_codes_list) == 0:
            break
        
        # 为这个值分配额外的编码（优先选择Hamming距离为1的编码）
        base_code = list(value_to_codes[value])[0]
        additional_codes = set()
        
        # 找到与base_code的Hamming距离为1的编码
        for bit_pos in range(k):
            neighbor_code = base_code ^ (1 << bit_pos)
            if neighbor_code in free_codes_list:
                additional_codes.add(neighbor_code)
                free_codes_list.remove(neighbor_code)
                if len(additional_codes) >= codes_per_value:
                    break
        
        # 如果还不够，随机分配
        while len(additional_codes) < codes_per_value and len(free_codes_list) > 0:
            additional_codes.add(free_codes_list.pop(0))
        
        value_to_codes[value].update(additional_codes)
    
    return value_to_codes


def compute_lrobust_surjective(
    value_to_codes: Dict[int, Set[int]],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    sensitivity: Dict[int, float],
    k: int
) -> float:
    """
    计算多对一映射的LRobust
    
    对于每个量化值v，如果它对应多个编码，计算距离时取最近的距离
    
    Args:
        value_to_codes: {value: set of codes} 多对一映射
        code_to_value: 编码到量化值的映射（用于解码）
        distribution: 量化值分布
        sensitivity: 量化值敏感度
        k: 位宽
    
    Returns:
        改进的LRobust值
    """
    total_loss = 0.0
    total_weight = 0.0
    
    # 对于每个量化值
    for value, freq in distribution.items():
        if value not in value_to_codes:
            continue
        
        codes = value_to_codes[value]
        sens = sensitivity.get(value, 1.0)
        
        # 对于每个编码，找到所有Hamming距离为1的邻居
        for code in codes:
            for bit_pos in range(k):
                neighbor_code = code ^ (1 << bit_pos)
                
                if neighbor_code in code_to_value:
                    neighbor_value = code_to_value[neighbor_code]
                    
                    # 如果邻居编码也属于同一个值，误差为0
                    if neighbor_value == value:
                        continue
                    
                    # 计算误差
                    error_sq = (value - neighbor_value) ** 2
                    weight = freq * sens / len(codes)  # 平均分配到多个编码
                    total_loss += error_sq * weight
                    total_weight += weight
    
    # 如果total_weight为0，说明所有邻居编码都映射到同一个值（完美情况）
    # 这发生在只有一个量化值且使用多对一映射时
    if total_weight == 0:
        return 0.0  # 返回0而不是inf，表示没有误差（所有编码都映射到同一个值）
    
    return total_loss / total_weight


def enforce_bijective_mapping(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    k: int
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    强制修复映射，确保双射性
    
    Args:
        value_to_code: 值到编码的映射
        code_to_value: 编码到值的映射
        distribution: 量化值分布
        k: 位宽
    
    Returns:
        (修复后的value_to_code, 修复后的code_to_value)
    """
    import random
    
    n_codes = 1 << k
    all_values = sorted(distribution.keys())
    
    # 创建新的映射，确保双射性
    new_vtc = {}
    new_ctv = {}
    used_codes = set()
    
    # 首先，为每个值分配一个唯一的编码
    available_codes = list(range(n_codes))
    random.shuffle(available_codes)
    
    code_idx = 0
    for value in all_values:
        # 如果该值已经有映射，检查编码是否可用
        if value in value_to_code:
            code = value_to_code[value]
            # 如果编码未被使用，且编码在有效范围内
            if code not in used_codes and 0 <= code < n_codes:
                new_vtc[value] = code
                new_ctv[code] = value
                used_codes.add(code)
                continue
        
        # 否则，分配一个未使用的编码
        while code_idx < len(available_codes):
            candidate_code = available_codes[code_idx]
            if candidate_code not in used_codes:
                new_vtc[value] = candidate_code
                new_ctv[candidate_code] = value
                used_codes.add(candidate_code)
                code_idx += 1
                break
            code_idx += 1
    
    # 验证双射性
    assert len(new_vtc) == len(new_ctv), "映射数量不一致"
    assert len(set(new_vtc.values())) == len(new_vtc), "编码不唯一"
    assert len(set(new_ctv.values())) == len(new_ctv), "值不唯一"
    
    # 验证双向一致性
    for value, code in new_vtc.items():
        assert new_ctv[code] == value, f"双向不一致: value_to_code[{value}] = {code}, 但 code_to_value[{code}] = {new_ctv[code]}"
    
    return new_vtc, new_ctv


def enforce_bijective_mapping(
    value_to_code: Dict[int, int],
    code_to_value: Dict[int, int],
    distribution: Dict[int, int],
    k: int
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    强制修复映射，确保双射性
    
    Args:
        value_to_code: 值到编码的映射
        code_to_value: 编码到值的映射
        distribution: 量化值分布
        k: 位宽
    
    Returns:
        (修复后的value_to_code, 修复后的code_to_value)
    """
    import random
    
    n_codes = 1 << k
    all_values = sorted(distribution.keys())
    
    # 创建新的映射，确保双射性
    new_vtc = {}
    new_ctv = {}
    used_codes = set()
    
    # 首先，为每个值分配一个唯一的编码
    available_codes = list(range(n_codes))
    random.shuffle(available_codes)
    
    code_idx = 0
    for value in all_values:
        # 如果该值已经有映射，检查编码是否可用
        if value in value_to_code:
            code = value_to_code[value]
            # 如果编码未被使用，且编码在有效范围内，且双向一致
            if (code not in used_codes and 
                0 <= code < n_codes and 
                code in code_to_value and 
                code_to_value[code] == value):
                new_vtc[value] = code
                new_ctv[code] = value
                used_codes.add(code)
                continue
        
        # 否则，分配一个未使用的编码
        while code_idx < len(available_codes):
            candidate_code = available_codes[code_idx]
            if candidate_code not in used_codes:
                new_vtc[value] = candidate_code
                new_ctv[candidate_code] = value
                used_codes.add(candidate_code)
                code_idx += 1
                break
            code_idx += 1
    
    # 验证双射性
    assert len(new_vtc) == len(new_ctv), f"映射数量不一致: {len(new_vtc)} vs {len(new_ctv)}"
    assert len(set(new_vtc.values())) == len(new_vtc), "编码不唯一"
    assert len(set(new_ctv.values())) == len(new_ctv), "值不唯一"
    
    # 验证双向一致性
    for value, code in new_vtc.items():
        assert new_ctv[code] == value, f"双向不一致: value_to_code[{value}] = {code}, 但 code_to_value[{code}] = {new_ctv[code]}"
    
    return new_vtc, new_ctv


def optimize_olm_mapping_enhanced(
    distribution: Dict[int, int],
    sensitivity: Dict[int, float],
    k: int,
    method: str = 'genetic',
    max_iterations: int = 1000,
    use_surjective: bool = False,
    top_k_values: int = 10,
    # 目标函数选择
    objective: str = "lrobust_h1",  # 'lrobust_h1' or 'bsc_mse'
    ber: Optional[float] = None,
    weight_mode: str = "freq_sens",
    max_hamming: Optional[int] = None,
    # 遗传算法参数
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    增强的OLM映射优化
    
    Args:
        distribution: 量化值分布
        sensitivity: 量化值敏感度
        k: 位宽
        method: 优化方法 ('greedy', 'simulated_annealing', 'genetic')
        max_iterations: 最大迭代次数
        use_surjective: 是否使用多对一映射
        top_k_values: 为前k个值分配多个编码（仅当use_surjective=True时有效）
    
    Returns:
        (value_to_code, code_to_value, best_lrobust)
    """
    n_levels = 1 << k
    thd_neg = -(1 << (k - 1))
    thd_pos = (1 << (k - 1)) - 1
    
    # 获取所有可能的量化值（按重要性排序：只使用敏感度）
    importance_scores = {}
    for value in distribution.keys():
        sens = sensitivity.get(value, 0.0)  # 默认敏感度为0.0
        importance_scores[value] = sens
    
    sorted_values = sorted(importance_scores.keys(), key=lambda x: -importance_scores[x])
    
    # 检查分布大小
    if len(distribution) <= 1:
        # 只有一个或没有值，多对一映射没有意义，使用简单映射
        n_codes = 1 << k
        if len(distribution) == 1:
            value = list(distribution.keys())[0]
            value_to_code = {value: 0}
            code_to_value = {0: value}
            # 填充剩余编码
            for code in range(1, n_codes):
                code_to_value[code] = value
            best_lrobust = 0.0  # 只有一个值，LRobust为0（所有编码映射到同一个值，没有误差）
        else:
            value_to_code = {}
            code_to_value = {}
            default_value = 0
            for code in range(n_codes):
                code_to_value[code] = default_value
            best_lrobust = float('inf')
        return value_to_code, code_to_value, best_lrobust
    
    if use_surjective:
        # 使用多对一映射
        value_to_codes = create_surjective_mapping(distribution, sensitivity, k, top_k_values)
        
        # 转换为code_to_value（取第一个编码）
        code_to_value = {}
        value_to_code = {}
        for value, codes in value_to_codes.items():
            # 选择第一个编码作为主编码
            main_code = sorted(codes)[0]
            value_to_code[value] = main_code
            code_to_value[main_code] = value
            
            # 其他编码也映射到同一个值
            for code in codes:
                if code != main_code:
                    code_to_value[code] = value
        
        # 使用多对一映射的LRobust
        best_lrobust = compute_lrobust_surjective(
            value_to_codes, code_to_value, distribution, sensitivity, k
        )
        
        # 检查结果是否有效
        if math.isinf(best_lrobust) or math.isnan(best_lrobust):
            print(f"    ⚠️  警告：多对一映射的LRobust为{best_lrobust}，回退到一对一映射")
            # 回退到一对一映射
            use_surjective = False
        else:
            # 强制修复双射性
            value_to_code, code_to_value = enforce_bijective_mapping(
                value_to_code, code_to_value, distribution, k
            )
            return value_to_code, code_to_value, best_lrobust
    else:
        # 使用传统的一对一映射
        from util.olm_encoder import optimize_olm_mapping
        
        # 目标函数（默认 LRobust-H1；方案B：BSC/BER 期望MSE）
        def compute_loss(vtc, ctv):
            if objective == "lrobust_h1":
                return compute_loss(vtc, ctv)
            if objective == "bsc_mse":
                if ber is None:
                    raise ValueError("objective='bsc_mse' requires ber to be provided.")
                return compute_bsc_mse_with_weights(
                    vtc, ctv, distribution, sensitivity, k, ber=float(ber),
                    weight_mode=weight_mode, max_hamming=max_hamming
                )
            raise ValueError(f"Unknown objective='{objective}'")
        
        # 简单的贪婪搜索
        if method == 'greedy':
            value_to_code = {}
            code_to_value = {}
            used_codes = set()
            
            # 按重要性分配编码
            for value in sorted_values:
                if len(used_codes) >= n_levels:
                    break
                
                # 找到最好的编码（与已分配编码的Hamming距离最小）
                best_code = None
                best_score = float('inf')
                
                for code in range(n_levels):
                    if code in used_codes:
                        continue
                    
                    # 计算与已分配编码的最小Hamming距离
                    min_hamming = float('inf')
                    for used_code in used_codes:
                        hamming = bin(code ^ used_code).count('1')
                        min_hamming = min(min_hamming, hamming)
                    
                    if min_hamming < best_score:
                        best_score = min_hamming
                        best_code = code
                
                if best_code is None:
                    best_code = len(used_codes)
                
                value_to_code[value] = best_code
                code_to_value[best_code] = value
                used_codes.add(best_code)
            
            # 填充剩余的值
            all_values = set(range(thd_neg, thd_pos + 1))
            unmapped_values = all_values - set(value_to_code.keys())
            unmapped_codes = set(range(n_levels)) - used_codes
            
            for value, code in zip(unmapped_values, unmapped_codes):
                value_to_code[value] = code
                code_to_value[code] = value
            
            best_lrobust = compute_loss(value_to_code, code_to_value)
            return value_to_code, code_to_value, best_lrobust
        
        elif method == 'genetic':
            # 使用遗传算法（自定义适应度函数）
            import random
            import math
            
            # 获取所有量化值
            values = sorted(distribution.keys())
            n_values = len(values)
            n_codes = 1 << k
            
            if n_values > n_codes:
                # 如果量化值数量超过编码数量，只选择前n_codes个高频值
                # 但需要警告用户
                sorted_by_freq = sorted(values, key=lambda v: distribution[v], reverse=True)
                values = sorted_by_freq[:n_codes]
                n_values = len(values)
                print(f"    ⚠️  警告：量化值数量({len(distribution)})超过编码数量({n_codes})")
                print(f"    只映射前{n_codes}个高频量化值，其余值将使用最近邻映射")
            
            # 创建格雷码映射（确保双射：每个值对应唯一编码，每个编码对应唯一值）
            def create_gray_code_mapping():
                """创建格雷码映射：value -> gray_code（确保双射）"""
                vtc = {}
                ctv = {}
                used_codes = set()
                
                # 获取所有实际出现的量化值（从distribution中）
                all_distribution_values = sorted(distribution.keys())
                
                for value in all_distribution_values:
                    # 将量化值转换为二进制编码（0到n_codes-1）
                    code_binary = value - thd_neg
                    if 0 <= code_binary < n_codes:
                        # 转换为格雷码：G = B ^ (B >> 1)
                        code_gray = code_binary ^ (code_binary >> 1)
                        
                        # 如果编码已被使用，找一个未使用的编码
                        if code_gray in used_codes:
                            # 找一个未使用的编码（优先选择相邻的）
                            for offset in range(1, n_codes):
                                candidate_code = (code_gray + offset) % n_codes
                                if candidate_code not in used_codes:
                                    code_gray = candidate_code
                                    break
                            # 如果还是找不到，随机选择一个未使用的
                            if code_gray in used_codes:
                                available_codes = [c for c in range(n_codes) if c not in used_codes]
                                if available_codes:
                                    code_gray = random.choice(available_codes)
                                else:
                                    continue  # 没有可用编码，跳过这个值
                        
                        vtc[value] = code_gray
                        ctv[code_gray] = value
                        used_codes.add(code_gray)
                
                return vtc, ctv
            
            # 创建个体（随机或格雷码）
            # 注意：需要映射所有distribution中的值，不仅仅是values列表中的
            def create_individual(use_gray_code=False):
                if use_gray_code:
                    return create_gray_code_mapping()
                else:
                    # 获取所有实际出现的量化值
                    all_distribution_values = sorted(distribution.keys())
                    codes = list(range(n_codes))
                    random.shuffle(codes)
                    
                    # 如果值数量超过编码数量，只映射前n_codes个高频值
                    values_to_map = all_distribution_values[:n_codes]
                    
                    vtc = {values_to_map[i]: codes[i] for i in range(min(len(values_to_map), n_codes))}
                    ctv = {codes[i]: values_to_map[i] for i in range(min(len(values_to_map), n_codes))}
                    
                    # 不填充未映射的编码，保持未映射状态
                    # 这样在解码时，未映射的编码会使用identity映射（保持原值）
                    return vtc, ctv
            
            # 评估适应度
            def evaluate_fitness(vtc, ctv):
                return compute_loss(vtc, ctv)
            
            # 交叉（强制保持双射性）
            def crossover(p1, p2):
                vtc1, ctv1 = p1
                vtc2, ctv2 = p2
                
                # 如果只有一个或没有值，直接返回parent1
                if len(values) <= 1:
                    return dict(vtc1), dict(ctv1)
                
                # 随机选择交叉点（至少1个，最多len(values)-1个）
                cp = random.randint(1, max(1, len(values) - 1))
                selected = random.sample(values, min(cp, len(values)))
                new_vtc = dict(vtc1)
                new_ctv = dict(ctv1)
                for v in selected:
                    if v in vtc1 and v in vtc2:
                        c1, c2 = vtc1[v], vtc2[v]
                        if c1 != c2 and c1 in new_ctv and c2 in new_ctv:
                            old_v1, old_v2 = new_ctv[c1], new_ctv[c2]
                            new_vtc[v] = c2
                            new_vtc[old_v2] = c1
                            new_ctv[c1] = old_v2
                            new_ctv[c2] = v
                
                # 强制修复双射性
                new_vtc, new_ctv = enforce_bijective_mapping(new_vtc, new_ctv, distribution, k)
                return new_vtc, new_ctv
            
            # 变异（强制保持双射性）- 增强版：支持多对交换和随机重启
            def mutate(ind, mutation_strength='normal'):
                vtc, ctv = ind
                # 如果少于2个值，无法swap，直接返回
                if len(values) < 2:
                    return vtc, ctv
                
                # 确保有足够的值可以交换
                available_values = [v for v in values if v in vtc]
                if len(available_values) < 2:
                    return vtc, ctv
                
                new_vtc = dict(vtc)
                new_ctv = dict(ctv)
                
                if mutation_strength == 'strong':
                    # 强变异：交换多对值（3-5对）
                    num_swaps = random.randint(3, min(5, len(available_values) // 2))
                elif mutation_strength == 'extreme':
                    # 极端变异：交换大量值对（5-10对）或部分随机重映射
                    if random.random() < 0.3:  # 30%概率进行部分随机重映射
                        # 随机选择20-40%的值进行重新映射
                        num_to_remap = max(2, int(len(available_values) * random.uniform(0.2, 0.4)))
                        values_to_remap = random.sample(available_values, min(num_to_remap, len(available_values)))
                        codes_to_remap = [new_vtc[v] for v in values_to_remap if v in new_vtc]
                        random.shuffle(codes_to_remap)
                        for v, c in zip(values_to_remap, codes_to_remap):
                            if v in new_vtc and c in new_ctv:
                                old_c = new_vtc[v]
                                old_v = new_ctv[c]
                                new_vtc[v] = c
                                new_vtc[old_v] = old_c
                                new_ctv[c] = v
                                new_ctv[old_c] = old_v
                    else:
                        # 交换多对值
                        num_swaps = random.randint(5, min(10, len(available_values) // 2))
                        for _ in range(num_swaps):
                            if len(available_values) < 2:
                                break
                            v1, v2 = random.sample(available_values, 2)
                            c1, c2 = new_vtc.get(v1), new_vtc.get(v2)
                            if c1 is not None and c2 is not None and c1 != c2:
                                new_vtc[v1] = c2
                                new_vtc[v2] = c1
                                new_ctv[c1] = v2
                                new_ctv[c2] = v1
                else:  # 'normal'
                    # 正常变异：交换1-3对值
                    num_swaps = random.randint(1, min(3, len(available_values) // 2))
                    for _ in range(num_swaps):
                        if len(available_values) < 2:
                            break
                        v1, v2 = random.sample(available_values, 2)
                        c1, c2 = new_vtc.get(v1), new_vtc.get(v2)
                        if c1 is not None and c2 is not None and c1 != c2:
                            new_vtc[v1] = c2
                            new_vtc[v2] = c1
                            new_ctv[c1] = v2
                            new_ctv[c2] = v1
                
                # 强制修复双射性
                new_vtc, new_ctv = enforce_bijective_mapping(new_vtc, new_ctv, distribution, k)
                return new_vtc, new_ctv
            
            # 初始化种群：增加多样性
            # 策略：只有第一个个体使用Gray code，其他个体使用随机初始化
            population = []
            gray_vtc, gray_ctv = create_individual(use_gray_code=True)
            print(f"    初始化种群（population_size={population_size}）:")
            print(f"      - 个体0: Gray code映射")
            print(f"      - 个体1-{population_size-1}: 随机映射（增加多样性）")
            
            # 第一个个体使用Gray code
            population.append((dict(gray_vtc), dict(gray_ctv)))
            
            # 其他个体使用随机初始化
            for i in range(1, population_size):
                random_vtc, random_ctv = create_individual(use_gray_code=False)
                population.append((random_vtc, random_ctv))
            
            fitness = [evaluate_fitness(ind[0], ind[1]) for ind in population]
            
            best_idx = fitness.index(min(fitness))
            best_individual = population[best_idx]
            best_fitness = fitness[best_idx]
            
            # 记录没有改进的代数
            generations_without_improvement = 0
            max_stagnation = 1000  # 如果1000代没有改进，进行随机重启
            
            # 进化
            for generation in range(max_iterations):
                new_population = []
                # 精英保留（但只保留前elite_size-1个，最后一个位置用于探索）
                elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[:max(1, elite_size-1)]
                for idx in elite_indices:
                    new_population.append(population[idx])
                
                # 生成新个体
                while len(new_population) < population_size:
                    # 锦标赛选择
                    def tournament_select():
                        ts = 3
                        ti = random.sample(range(len(population)), min(ts, len(population)))
                        tf = [fitness[i] for i in ti]
                        return population[ti[tf.index(min(tf))]]
                    
                    p1 = tournament_select()
                    p2 = tournament_select()
                    
                    if random.random() < crossover_rate:
                        child = crossover(p1, p2)
                    else:
                        child = p1 if random.random() < 0.5 else p2
                    
                    # 自适应变异：如果长时间没有改进，增加变异强度
                    mutation_strength = 'normal'
                    if generations_without_improvement > 500:
                        mutation_strength = 'strong'
                    if generations_without_improvement > 800:
                        mutation_strength = 'extreme'
                    
                    # 应用变异
                    if random.random() < mutation_rate:
                        child = mutate(child, mutation_strength=mutation_strength)
                    # 额外的小概率大幅变异（跳出局部最优）
                    elif random.random() < 0.1:  # 10%的概率进行额外变异
                        child = mutate(child, mutation_strength='strong')
                    
                    new_population.append(child)
                
                # 如果长时间没有改进，添加一个完全随机的个体（随机重启）
                if generations_without_improvement > max_stagnation:
                    random_vtc, random_ctv = create_individual(use_gray_code=False)
                    new_population[-1] = (random_vtc, random_ctv)  # 替换最后一个个体
                    generations_without_improvement = 0  # 重置计数器
                    print(f"    ⚠️  第 {generation + 1} 代：{max_stagnation}代无改进，执行随机重启")
                
                population = new_population[:population_size]
                fitness = [evaluate_fitness(ind[0], ind[1]) for ind in population]
                
                current_best_idx = fitness.index(min(fitness))
                current_best_fitness = fitness[current_best_idx]
                
                if current_best_fitness < best_fitness - 1e-6:  # 允许小的数值误差
                    best_fitness = current_best_fitness
                    best_individual = population[current_best_idx]
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                if (generation + 1) % 100 == 0:
                    avg_fitness = sum(fitness) / len(fitness)
                    min_fitness = min(fitness)
                    max_fitness = max(fitness)
                    print(f"    第 {generation + 1}/{max_iterations} 代: 最优={best_fitness:.4f}, 当前最优={current_best_fitness:.4f}, 平均={avg_fitness:.4f}, 范围=[{min_fitness:.4f}, {max_fitness:.4f}], 停滞={generations_without_improvement}代")
            
            # 返回前强制修复双射性
            final_vtc, final_ctv = enforce_bijective_mapping(
                best_individual[0], best_individual[1], distribution, k
            )
            final_fitness = evaluate_fitness(final_vtc, final_ctv)
            return final_vtc, final_ctv, final_fitness
        
        else:
            # 使用模拟退火
            from util.olm_encoder import optimize_olm_mapping
            value_to_code, code_to_value, _ = optimize_olm_mapping(
                distribution, k, method, max_iterations
            )
            # 强制修复双射性
            value_to_code, code_to_value = enforce_bijective_mapping(
                value_to_code, code_to_value, distribution, k
            )
            best_lrobust = compute_loss(value_to_code, code_to_value)
            return value_to_code, code_to_value, best_lrobust

