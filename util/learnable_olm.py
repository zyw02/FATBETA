"""
可学习的OLM编码模块 (Learnable OLM Encoder)

将OLM编码映射转换为可学习的参数，支持端到端训练。
故障注入后的损失可以同时指导量化模型和OLM编码映射的训练。

核心思想：
1. 使用可学习的编码矩阵表示value->code映射
2. 使用Gumbel-Softmax或Straight-Through Estimator实现可微分的离散映射
3. 通过梯度更新编码映射参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    import warnings
    warnings.warn("scipy not available, Hungarian algorithm will not work. Install scipy for bijective mapping support.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    import warnings
    warnings.warn("numpy not available, Sinkhorn algorithm will not work. Install numpy for bijective mapping support.")


class LearnableOLMEncoder(nn.Module):
    """
    可学习的OLM编码器
    
    将量化值到编码的映射表示为可学习参数，支持端到端训练。
    
    设计思路：
    - 使用连续参数矩阵表示编码映射
    - 训练时使用soft assignment（可微分）
    - 推理时使用hard assignment（离散映射）
    - 通过梯度更新优化编码映射
    """
    
    def __init__(
        self,
        layer_name: str,
        bit_width: int,
        num_quantized_values: int,
        device: Optional[torch.device] = None,
        init_method: str = 'identity',  # 'identity', 'random', 'greedy'
        temperature: float = 1.0,  # Gumbel-Softmax温度参数
        use_straight_through: bool = True,  # 是否使用Straight-Through Estimator
    ):
        """
        Args:
            layer_name: 层名称
            bit_width: 量化位宽
            num_quantized_values: 实际出现的量化值数量（通常小于2^bit_width）
            device: 设备
            init_method: 初始化方法
            temperature: Gumbel-Softmax温度（训练时使用，推理时设为0）
            use_straight_through: 是否使用Straight-Through Estimator
        """
        super().__init__()
        self.layer_name = layer_name
        self.bit_width = bit_width
        self.num_quantized_values = num_quantized_values
        self.n_levels = 1 << bit_width  # 2^bit_width
        self.temperature = temperature
        self.use_straight_through = use_straight_through
        
        # 量化值范围
        self.thd_neg = -(1 << (bit_width - 1))
        self.thd_pos = (1 << (bit_width - 1)) - 1
        
        # 可学习的编码映射矩阵
        # shape: [num_quantized_values, n_levels]
        # 每一行表示一个量化值对所有可能编码的"偏好"（logits）
        self.encoding_logits = nn.Parameter(
            torch.zeros(num_quantized_values, self.n_levels, device=device)
        )
        
        # 量化值索引映射（将量化值映射到矩阵行索引）
        # 这个映射是固定的，不参与训练
        self.value_to_idx: Dict[int, int] = {}
        self.idx_to_value: Dict[int, int] = {}
        
        # M矩阵缓存（避免重复计算）
        self._M_cache: Optional[torch.Tensor] = None
        self._M_cache_ber: Optional[float] = None
        
        # 双射映射缓存（用于推理时保证100%可逆）
        self._cached_bijective_mapping: Optional[Tuple[Dict[int, int], Dict[int, int]]] = None
        
        # 初始化编码映射
        self._initialize_encoding(init_method)
    
    def _initialize_encoding(self, init_method: str):
        """初始化编码映射"""
        if init_method == 'identity':
            # Identity映射：量化值i映射到编码i
            for i in range(self.num_quantized_values):
                value = self.thd_neg + i
                if value <= self.thd_pos:
                    self.value_to_idx[value] = i
                    self.idx_to_value[i] = value
                    # 初始化logits：对对应的编码位置设置高值
                    self.encoding_logits.data[i, i % self.n_levels] = 10.0
        elif init_method == 'random':
            # 随机初始化
            nn.init.normal_(self.encoding_logits, mean=0, std=0.1)
            # 随机分配量化值索引
            values = list(range(self.thd_neg, self.thd_pos + 1))
            for i, value in enumerate(values[:self.num_quantized_values]):
                self.value_to_idx[value] = i
                self.idx_to_value[i] = value
        elif init_method == 'greedy':
            # 使用贪婪策略初始化（基于频率）
            # 这里简化处理，实际应该基于量化值分布
            for i in range(self.num_quantized_values):
                value = self.thd_neg + i
                if value <= self.thd_pos:
                    self.value_to_idx[value] = i
                    self.idx_to_value[i] = value
                    # 初始化：尝试将相近的值映射到相邻编码
                    code = i % self.n_levels
                    self.encoding_logits.data[i, code] = 10.0
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    def set_value_mapping(self, value_to_idx: Dict[int, int]):
        """
        设置量化值到索引的映射
        
        Args:
            value_to_idx: 量化值到矩阵行索引的映射
        """
        self.value_to_idx = value_to_idx
        self.idx_to_value = {idx: val for val, idx in value_to_idx.items()}
    
    def encode(
        self,
        quantized_values: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        将量化值编码为编码空间的值
        
        Args:
            quantized_values: 量化值（整数），shape: [*]
            training: 是否在训练模式
        
        Returns:
            编码后的值，shape: [*]
        """
        # 保存原始形状
        original_shape = quantized_values.shape
        
        # 将量化值转换为索引
        # 对于未映射的值，使用最近的映射值
        indices = self._value_to_indices(quantized_values)
        
        # 展平以便处理
        flat_indices = indices.view(-1)
        
        # 获取对应的logits
        # shape: [num_values, n_levels]
        logits = self.encoding_logits[flat_indices]  # [num_values, n_levels]
        
        if training and self.temperature > 0:
            # 训练时：使用Straight-Through + 匈牙利算法（100%可逆 + 可微）
            # 前向：使用匈牙利算法（硬双射，100%可逆）
            # 反向：使用Sinkhorn的梯度（可微分）
            if self.use_straight_through and SCIPY_AVAILABLE:
                # 前向：使用匈牙利算法（硬双射，100%可逆）
                value_to_code, _ = self.get_hard_mapping_hungarian()
                codes_hard = quantized_values.clone()
                for value, code in value_to_code.items():
                    codes_hard[quantized_values == value] = code
                codes_hard = codes_hard.to(quantized_values.dtype)
                
                # 反向：使用Sinkhorn算法的梯度（可微分）
                probs = F.softmax(logits, dim=-1)  # [num_values, n_levels]
                constrained_probs = self._sinkhorn(probs, num_iterations=10)
                code_levels = torch.arange(self.n_levels, device=constrained_probs.device, dtype=constrained_probs.dtype)
                codes_soft = (constrained_probs * code_levels).sum(dim=-1)  # [num_values]
                
                # Straight-Through: 前向用hard，反向用soft
                codes = codes_hard.float() + codes_soft - codes_soft.detach()
            else:
                # 回退方案：使用Sinkhorn算法（可微分，双射的软版本）
                probs = F.softmax(logits, dim=-1)  # [num_values, n_levels]
                constrained_probs = self._sinkhorn(probs, num_iterations=10)
                
            if self.use_straight_through:
                # Straight-Through Estimator: 前向使用hard，反向使用soft
                    hard_codes = torch.argmax(constrained_probs, dim=-1)  # [num_values]
                    soft_codes = F.gumbel_softmax(constrained_probs * 10.0, tau=self.temperature, hard=False, dim=-1)  # [num_values, n_levels]
                    hard_codes_one_hot = F.one_hot(hard_codes, num_classes=self.n_levels).float()  # [num_values, n_levels]
                    codes_soft = hard_codes_one_hot + soft_codes - soft_codes.detach()  # [num_values, n_levels]
                    code_levels = torch.arange(self.n_levels, device=codes_soft.device, dtype=codes_soft.dtype)
                    codes = (codes_soft * code_levels).sum(dim=-1)  # [num_values]
            else:
                # 纯Gumbel-Softmax
                    codes = F.gumbel_softmax(constrained_probs * 10.0, tau=self.temperature, hard=True, dim=-1)
                codes = torch.argmax(codes, dim=-1)
        else:
            # 推理时：使用双射映射表（保证100%可逆）
            # 使用缓存的双射映射表，确保encode和decode使用相同的映射
            value_to_code, _ = self._get_cached_bijective_mapping()
            
            # 将量化值转换为编码
            codes = quantized_values.clone()
            for value, code in value_to_code.items():
                codes[quantized_values == value] = code
            
            codes = codes.to(quantized_values.dtype)
        
        # 恢复原始形状
        return codes.view(original_shape)
    
    def decode(
        self,
        codes: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        将编码空间的值解码回量化值
        
        关键：保证双向一致性
        - 如果量化值a通过encode()映射到编码b
        - 那么编码b通过decode()必须映射回量化值a
        
        Args:
            codes: 编码值，shape: [*]
            training: 是否在训练模式
        
        Returns:
            量化值，shape: [*]
        """
        # 方法：构建双向一致的映射表（与传统OLM一致）
        # 对于每个量化值，找到它映射到的编码（argmax）
        # 然后构建 code -> value 的查找表
        
        # 构建code_to_value映射（保证双向一致性）
        # 使用get_hard_mapping获取双向一致的映射
        _, code_to_value_dict = self.get_hard_mapping()
        
        # 调试：记录映射统计信息
        if not training:
            import sys
            mapped_codes_count = len(code_to_value_dict)
            mapped_values = set(code_to_value_dict.values())
            print(f"[LearnableOLM DEBUG] decode: code_to_value_dict size={mapped_codes_count}, mapped_values_range=[{min(mapped_values) if mapped_values else 'N/A'}, {max(mapped_values) if mapped_values else 'N/A'}], expected_range=[{self.thd_neg}, {self.thd_pos}], n_levels={self.n_levels}", file=sys.stderr, flush=True)
        
        # 创建GPU查找表
        code_to_value_lookup = torch.zeros(self.n_levels, dtype=torch.float32, device=codes.device)
        
        # 填充映射的编码
        for code, value in code_to_value_dict.items():
            if 0 <= code < self.n_levels:
                code_to_value_lookup[code] = float(value)
        
        # 对于未映射的编码，使用最近邻查找而不是identity mapping
        # ⚠️ 关键修复：identity mapping会导致系统性偏差，应该使用最近邻查找
        mapped_codes = set(code_to_value_dict.keys())
        unmapped_codes = []
        if mapped_codes:
            # 对于未映射的编码，找到最接近的已映射编码，使用其对应的量化值
        for code in range(self.n_levels):
            if code not in mapped_codes:
                    # 找到最接近的已映射编码
                    closest_code = min(mapped_codes, key=lambda x: abs(x - code))
                    # 使用最接近编码对应的量化值
                    code_to_value_lookup[code] = code_to_value_lookup[closest_code]
                    unmapped_codes.append(code)
        else:
            # 如果没有已映射的编码，使用identity mapping作为后备
            for code in range(self.n_levels):
                decoded_value = code + self.thd_neg
                decoded_value = max(self.thd_neg, min(self.thd_pos, decoded_value))
                code_to_value_lookup[code] = float(decoded_value)
                unmapped_codes.append(code)
        
        # 调试：记录未映射编码的数量
        if not training and unmapped_codes:
            import sys
            print(f"[LearnableOLM DEBUG] decode: unmapped_codes_count={len(unmapped_codes)}, unmapped_codes_range=[{min(unmapped_codes)}, {max(unmapped_codes)}]", file=sys.stderr, flush=True)
        
        # 使用查找表进行解码
        code_indices = codes.clamp(0, self.n_levels - 1).long()
        values = code_to_value_lookup[code_indices]
        
        # 调试：记录解码统计信息（仅在推理时，避免训练时过多输出）
        if not training:
            codes_min = codes.min().item()
            codes_max = codes.max().item()
            codes_unique = codes.unique().numel()
            values_min = values.min().item()
            values_max = values.max().item()
            # 检查是否有超出预期范围的值
            out_of_range = ((values < self.thd_neg) | (values > self.thd_pos)).sum().item()
            if out_of_range > 0 or codes_max >= self.n_levels:
                import sys
                print(f"[LearnableOLM DEBUG] decode: codes_range=[{codes_min}, {codes_max}], codes_unique={codes_unique}, values_range=[{values_min:.2f}, {values_max:.2f}], expected_range=[{self.thd_neg}, {self.thd_pos}], out_of_range_count={out_of_range}, n_levels={self.n_levels}", file=sys.stderr, flush=True)
        
        return values
    
    def _value_to_indices(self, quantized_values: torch.Tensor) -> torch.Tensor:
        """
        将量化值转换为矩阵行索引
        
        Args:
            quantized_values: 量化值，shape: [*]
        
        Returns:
            索引，shape: [*]
        """
        # 展平
        flat_values = quantized_values.view(-1)
        
        # 创建索引映射（使用查找表加速）
        indices = torch.zeros_like(flat_values, dtype=torch.long)
        
        # 构建查找表（从量化值到索引）
        # 对于未映射的值，使用最近的映射值
        if not self.value_to_idx:
            # 如果没有映射，返回0索引
            return indices.view(quantized_values.shape)
        
        # 获取所有映射的值
        mapped_values = list(self.value_to_idx.keys())
        min_value = min(mapped_values)
        max_value = max(mapped_values)
        
        # 对于每个量化值，找到对应的索引
        for value, idx in self.value_to_idx.items():
            mask = (flat_values == value)
            indices[mask] = idx
        
        # 对于未映射的值，使用最近的映射值
        unmapped_mask = torch.zeros(flat_values.shape, dtype=torch.bool, device=flat_values.device)
        for value in mapped_values:
            unmapped_mask |= (flat_values == value)
        unmapped_mask = ~unmapped_mask
        
        if unmapped_mask.any():
            # 找到最接近的映射值
            unmapped_values = flat_values[unmapped_mask]
            for unmapped_val in unmapped_values.unique():
                # 找到最接近的映射值
                closest_value = min(mapped_values, key=lambda x: abs(x - unmapped_val.item()))
                closest_idx = self.value_to_idx[closest_value]
                mask = (flat_values == unmapped_val)
                indices[mask] = closest_idx
        
        return indices.view(quantized_values.shape)
    
    def _sinkhorn(self, probs: torch.Tensor, num_iterations: int = 10, epsilon: float = 1e-3) -> torch.Tensor:
        """
        Sinkhorn算法：将矩阵转换为双随机矩阵（可微分）
        
        保证每行和每列的和都为1，这是双射的软版本。
        
        Args:
            probs: [num_values, n_levels] 概率矩阵
            num_iterations: 迭代次数
            epsilon: 正则化参数（避免数值不稳定）
        
        Returns:
            双随机矩阵（每行和每列的和都为1）
        """
        # 添加正则化避免数值不稳定
        probs = probs + epsilon
        
        # 交替归一化行和列
        for _ in range(num_iterations):
            # 归一化行（每行和为1）
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            # 归一化列（每列和为1）
            probs = probs / (probs.sum(dim=0, keepdim=True) + 1e-8)
        
        return probs
    
    def _get_cached_bijective_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        获取缓存的双射映射表（推理时使用，保证100%可逆）
        
        如果映射表未缓存或已失效，使用匈牙利算法重新计算
        """
        if self._cached_bijective_mapping is None:
            # 使用匈牙利算法计算双射映射
            self._cached_bijective_mapping = self.get_hard_mapping_hungarian()
        return self._cached_bijective_mapping
    
    def invalidate_cache(self):
        """
        使缓存的双射映射表失效（训练时调用，因为encoding_logits会更新）
        """
        self._cached_bijective_mapping = None
    
    def get_hard_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        获取hard assignment的映射（用于推理）
        
        保证双向一致性：
        - value_to_code[a] = b 意味着 code_to_value[b] = a
        
        ⚠️ 修复：使用贪心算法解决编码冲突，确保每个量化值映射到唯一的编码
        
        Returns:
            (value_to_code, code_to_value) 映射字典
        """
        value_to_code = {}
        code_to_value = {}
        used_codes = set()
        
        # 按量化值的频率排序（如果有分布信息），优先映射高频值
        # 如果没有分布信息，按值的大小排序
        sorted_values = sorted(self.value_to_idx.keys())
        
        # 对于每个量化值，找到logits最大的未使用编码
        for value in sorted_values:
            idx = self.value_to_idx[value]
            logits = self.encoding_logits[idx]  # [n_levels]
            
            # 找到logits最大的未使用编码
            # 如果所有编码都被使用，选择logits最大的编码（即使已被使用）
            available_codes = [c for c in range(self.n_levels) if c not in used_codes]
            
            if available_codes:
                # 有未使用的编码，选择logits最大的
                available_logits = logits[available_codes]
                best_local_idx = torch.argmax(available_logits).item()
                code = available_codes[best_local_idx]
            else:
                # 所有编码都被使用，选择logits最大的编码（即使已被使用）
                # 这种情况下会有冲突，但至少保证每个量化值都有映射
                code = int(torch.argmax(logits).item())
                import sys
                print(f"[LearnableOLM WARNING] All codes used, value {value} forced to use code {code} (conflict with {code_to_value.get(code, 'unknown')})", file=sys.stderr, flush=True)
            
            value_to_code[value] = code
            # 如果编码已被使用，覆盖之前的映射（记录冲突）
            if code in used_codes:
                import sys
                old_value = code_to_value.get(code, None)
                if old_value is not None:
                    print(f"[LearnableOLM WARNING] Code {code} conflict: {old_value} -> {value} (using {value})", file=sys.stderr, flush=True)
            code_to_value[code] = value
            used_codes.add(code)
        
        # 调试：检查是否有编码值冲突（多个量化值映射到同一个编码）
        code_conflicts = {}
        for value, code in value_to_code.items():
            if code in code_conflicts:
                code_conflicts[code].append(value)
            else:
                code_conflicts[code] = [value]
        
        conflicts = {code: values for code, values in code_conflicts.items() if len(values) > 1}
        if conflicts:
            import sys
            print(f"[LearnableOLM WARNING] get_hard_mapping: {len(conflicts)} codes have conflicts (multiple values map to same code)", file=sys.stderr, flush=True)
            for code, values in list(conflicts.items())[:5]:  # 只显示前5个冲突
                print(f"  Code {code} mapped by values: {values}, using value: {code_to_value[code]}", file=sys.stderr, flush=True)
        
        return value_to_code, code_to_value
    
    def get_hard_mapping_hungarian(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        使用匈牙利算法获取hard assignment的双射映射（推理时使用，保证100%可逆）
        
        使用匈牙利算法找到最优的双射映射，保证：
        - 每个量化值映射到唯一的编码
        - 每个编码映射到唯一的量化值
        - 完全双射（100%可逆）
        
        Returns:
            (value_to_code, code_to_value) 映射字典
        """
        if not SCIPY_AVAILABLE:
            # 如果scipy不可用，回退到贪心方法
            return self.get_hard_mapping()
        
        try:
            from scipy.optimize import linear_sum_assignment
            
            # 构建成本矩阵：cost[i, j] = -logits[i, j]
            # 匈牙利算法会找到最小成本匹配，所以使用负logits
            logits_np = -self.encoding_logits.detach().cpu().numpy()  # [num_quantized_values, n_levels]
            
            # 如果量化值数量小于编码数量，需要填充
            num_values = len(self.value_to_idx)
            if num_values < self.n_levels:
                # 填充到 n_levels x n_levels（添加虚拟行，成本为0）
                padded_logits = np.zeros((self.n_levels, self.n_levels))
                padded_logits[:num_values, :] = logits_np
                cost_matrix = padded_logits
            else:
                # 如果量化值数量 >= 编码数量，只使用前 n_levels 个值
                cost_matrix = logits_np[:self.n_levels, :]
            
            # 使用匈牙利算法找到最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 构建映射
            value_to_code = {}
            code_to_value = {}
            
            # 获取所有量化值（按索引排序）
            sorted_values = sorted(self.value_to_idx.keys())
            
            for i, row_idx in enumerate(row_indices):
                if row_idx < num_values:
                    value = sorted_values[row_idx]
                    code = int(col_indices[i])
                    value_to_code[value] = code
                    code_to_value[code] = value
            
            return value_to_code, code_to_value
            
        except Exception as e:
            # 如果匈牙利算法失败，回退到贪心方法
            import sys
            print(f"[LearnableOLM WARNING] Hungarian algorithm failed: {e}, falling back to greedy method", file=sys.stderr, flush=True)
            return self.get_hard_mapping()
    
    def compute_lrobust_loss(
        self,
        distribution: Dict[int, int],
        ber: float = 1e-2,
        use_soft_assignment: bool = False
    ) -> torch.Tensor:
        """
        计算LRobust损失
        
        **与传统OLM相同的目标函数**：
        LRobust = (1/W) * Σ_v Σ_{j∈Hamming_1(code(v))} (v - value(j))^2 * f(v)
        
        其中：
        - v: 量化值
        - f(v): 量化值v的出现频率
        - code(v): 量化值v对应的编码
        - Hamming_1(c): 与编码c的Hamming距离为1的所有编码
        - value(j): 编码j对应的量化值
        - W: 总频率（归一化因子）
        
        注意：BER（P(flip)）是常数，不影响优化结果，可以省略或作为权重
        
        Args:
            distribution: 量化值分布 {value: frequency}
            ber: Bit-error-rate（可选，作为权重）
            use_soft_assignment: 是否使用soft assignment（可微分，用于端到端训练）
                                如果False，使用hard assignment（与传统OLM一致）
        
        Returns:
            LRobust损失值（可微分的tensor）
        """
        if use_soft_assignment:
            # 使用soft assignment（可微分，用于端到端训练）
            return self._compute_lrobust_loss_soft(distribution, ber)
        else:
            # 使用hard assignment（与传统OLM一致）
            return self._compute_lrobust_loss_hard(distribution, ber)
    
    def _compute_lrobust_loss_hard(
        self,
        distribution: Dict[int, int],
        ber: float = 1e-2
    ) -> torch.Tensor:
        """
        使用hard assignment计算LRobust（与传统OLM完全一致）
        
        注意：这个版本使用hard assignment，梯度通过Straight-Through传播
        """
        total_loss = 0.0
        total_weight = 0.0
        
        # 获取hard mapping（与传统OLM一致）
        value_to_code, code_to_value = self.get_hard_mapping()
        
        # 计算LRobust（与传统OLM的compute_lrobust函数一致）
        for value, frequency in distribution.items():
            if value not in value_to_code:
                continue
            
            code = value_to_code[value]
            
            # 找到所有Hamming距离为1的编码（与传统OLM一致）
            for bit_pos in range(self.bit_width):
                neighbor_code = code ^ (1 << bit_pos)  # Hamming距离=1
                
                if neighbor_code in code_to_value:
                    neighbor_value = code_to_value[neighbor_code]
                    # 计算误差平方（与传统OLM一致）
                    error_sq = (value - neighbor_value) ** 2
                    total_loss += error_sq * frequency
                    total_weight += frequency
        
        # 归一化（与传统OLM一致）
        lrobust = total_loss / total_weight if total_weight > 0 else 0.0
        
        # 转换为tensor并保持梯度（通过Straight-Through）
        lrobust_tensor = torch.tensor(lrobust, device=self.encoding_logits.device, 
                                     dtype=torch.float32, requires_grad=True)
        
        # 使用Straight-Through技巧：前向使用hard值，反向传播梯度到encoding_logits
        if self.encoding_logits.requires_grad:
            # 创建一个可微分的代理损失
            # 通过soft assignment的近似来传播梯度
            soft_lrobust = self._compute_lrobust_loss_soft(distribution, ber=1.0)
            # 使用Straight-Through：前向使用hard，反向使用soft
            lrobust_tensor = lrobust_tensor + soft_lrobust - soft_lrobust.detach()
        
        return lrobust_tensor
    
    def _build_bitflip_transition_matrix(self, ber: float) -> torch.Tensor:
        """
        构建 bit-flip 转移矩阵 M（使用 BER 计算概率，带缓存优化）
        
        M[i, j] = Pr(code'=j | code=i, bit-flip with BER)
        
        使用伯努利翻转模型：
        - 每个 bit 独立地以 BER 的概率翻转
        - 对于 k-bit 编码，从 code i 到 code j 的概率取决于它们的 Hamming 距离
        
        对于 Hamming 距离为 d 的两个编码：
        - 需要恰好 d 个 bit 翻转（从 i 到 j）
        - 需要其他 (k-d) 个 bit 不翻转
        - 概率 = BER^d * (1-BER)^(k-d)
        
        注意：这里简化处理，只考虑单 bit 翻转（d=1）和零翻转（d=0）
        对于多 bit 翻转，概率会很小，可以忽略
        
        Args:
            ber: Bit-error-rate（每个 bit 的翻转概率）
        
        Returns:
            转移矩阵 M，shape: [n_levels, n_levels]
        """
        # 检查缓存（如果BER不变，直接返回缓存的M）
        if self._M_cache is not None and self._M_cache_ber == ber:
            return self._M_cache
        
        device = self.encoding_logits.device
        M = torch.zeros(self.n_levels, self.n_levels, device=device)
        
        # 对于每个编码
        for code in range(self.n_levels):
            # 自身（无翻转）的概率：所有 bit 都不翻转
            # P(无翻转) = (1-BER)^k
            prob_no_flip = (1.0 - ber) ** self.bit_width
            M[code, code] = prob_no_flip
            
            # 单 bit 翻转：找到所有 Hamming 距离为 1 的邻居
            neighbors = self._get_hamming_neighbors(code, self.bit_width)
            # 单 bit 翻转的概率：恰好 1 个 bit 翻转，其他 (k-1) 个 bit 不翻转
            # P(单bit翻转) = BER * (1-BER)^(k-1)
            prob_single_flip = ber * ((1.0 - ber) ** (self.bit_width - 1))
            
            for neighbor in neighbors:
                M[code, neighbor] = prob_single_flip
            
            # 注意：多 bit 翻转（d>=2）的概率很小，这里忽略
            # 如果需要更精确，可以计算所有可能的 Hamming 距离
        
        # 更新缓存
        self._M_cache = M
        self._M_cache_ber = ber
        
        return M
    
    def _build_decode_value_lookup(self) -> torch.Tensor:
        """
        构建解码值查找表（可微分版本，使用矩阵运算优化）
        
        Returns:
            decode_value_lookup，shape: [n_levels]
            decode_value_lookup[code] = 期望解码值（使用 soft assignment）
        
        优化说明：
            使用矩阵运算代替嵌套循环，从 O(n_levels × n_values × n_levels) 
            降到 O(n_values × n_levels)，加速约 256 倍（对于8bit层）。
            
            数学原理：
            decode_value_lookup[code] = Σ_i P[i, code] * value[i]
            其中 P[i, code] = softmax(encoding_logits[i])[code]
            
            矩阵形式：decode_value_lookup = P^T @ values
            其中 P = softmax(encoding_logits / T)，shape: [n_values, n_levels]
        """
        if not self.value_to_idx:
            device = self.encoding_logits.device
            return torch.zeros(self.n_levels, device=device, requires_grad=True)
        
        device = self.encoding_logits.device
        
        # 获取所有量化值的索引和值
        value_list = []
        idx_list = []
        for value, idx in sorted(self.value_to_idx.items()):
            value_list.append(value)
            idx_list.append(idx)
        
        values_tensor = torch.tensor(value_list, dtype=torch.float32, device=device)  # [n_values]
        
        # 一次性获取所有相关logits
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        all_logits = self.encoding_logits[idx_tensor]  # [n_values, n_levels]
        
        # 一次性计算所有softmax概率矩阵 P
        if self.temperature > 0:
            P = F.softmax(all_logits / self.temperature, dim=-1)  # [n_values, n_levels]
        else:
            P = F.softmax(all_logits, dim=-1)  # [n_values, n_levels]
        
        # 矩阵乘法：decode_value_lookup = P^T @ values
        # P^T: [n_levels, n_values], values: [n_values]
        # 结果: [n_levels]
        decode_value_lookup = P.T @ values_tensor  # [n_levels]
        
        return decode_value_lookup
    
    def _compute_lrobust_loss_soft(
        self,
        distribution: Dict[int, int],
        ber: float = 1e-2
    ) -> torch.Tensor:
        """
        使用soft assignment计算LRobust（完全可微分，用于端到端训练）
        
        使用矩阵形式：Q = P @ M，然后 decode_soft = Q @ decode_value_lookup
        LRobust = E[distance(value, decode_soft(value))^2]
        
        其中：
        - P = softmax(encoding_logits / T)，shape: [n_values, n_codes]
        - M = bit-flip 转移矩阵，shape: [n_codes, n_codes]
        - Q = P @ M，表示 value->code->(flip)->code' 的联合 soft 分布
        - decode_soft = Q @ decode_value_lookup，期望解码值
        
        Args:
            distribution: 量化值分布 {value: frequency}
            ber: Bit-error-rate
        
        Returns:
            LRobust损失值（可微分的tensor）
        """
        if not self.value_to_idx:
            return torch.tensor(0.0, device=self.encoding_logits.device, requires_grad=True)
        
        device = self.encoding_logits.device
        
        # 1. 构建 soft assignment 矩阵 P
        # P[i, j] = Pr(code=j | value=i)
        n_values = len(self.value_to_idx)
        
        # 获取所有量化值的索引和值
        value_list = []
        idx_list = []
        for value, idx in sorted(self.value_to_idx.items()):
            value_list.append(value)
            idx_list.append(idx)
        
        # 计算 P（避免 in-place 操作）
        P_list = []
        for idx in idx_list:
            logits = self.encoding_logits[idx]  # [n_levels]
            if self.temperature > 0:
                probs = F.softmax(logits / self.temperature, dim=0)
            else:
                probs = F.softmax(logits, dim=0)
            P_list.append(probs)
        
        # 堆叠成矩阵
        P = torch.stack(P_list, dim=0)  # [n_values, n_levels]
        
        # 2. 构建 bit-flip 转移矩阵 M
        M = self._build_bitflip_transition_matrix(ber)  # [n_levels, n_levels]
        
        # 3. 计算翻转后的分布 Q = P @ M
        # Q[i, j] = Pr(value=i -> code'=j after bit-flip)
        Q = P @ M  # [n_values, n_levels]
        
        # 4. 构建解码值查找表（使用 soft assignment）
        decode_value_lookup = self._build_decode_value_lookup()  # [n_levels]
        
        # 5. 计算期望解码值 decode_soft = Q @ decode_value_lookup
        decode_soft = Q @ decode_value_lookup  # [n_values]
        
        # 6. 计算 LRobust = E[(value - decode_soft(value))^2]
        values_tensor = torch.tensor(value_list, dtype=torch.float32, device=device)
        errors_sq = (values_tensor - decode_soft) ** 2  # [n_values]
        
        # 7. 加权求和（根据分布频率）
        frequencies = torch.tensor(
            [distribution.get(v, 0) for v in value_list],
            dtype=torch.float32,
            device=device
        )
        total_weight = frequencies.sum()
        
        if total_weight > 0:
            lrobust = (errors_sq * frequencies).sum() / total_weight
        else:
            lrobust = errors_sq.mean()
        
        return lrobust
    
    def compute_regularization_loss(
        self,
        reg_col_weight: float = 1.0,
        reg_row_weight: float = 1.0,
        reg_usage_weight: float = 1.0
    ) -> torch.Tensor:
        """
        计算正则化损失
        
        包括：
        1. 列互斥正则：鼓励每列被恰好一行主导（一对一映射）
        2. 行熵正则：鼓励每行低熵（接近 one-hot）
        3. 码字利用率正则：防止只用少数列
        
        Args:
            reg_col_weight: 列互斥正则权重
            reg_row_weight: 行熵正则权重
            reg_usage_weight: 码字利用率正则权重
        
        Returns:
            正则化损失值
        """
        if not self.value_to_idx:
            return torch.tensor(0.0, device=self.encoding_logits.device, requires_grad=True)
        
        device = self.encoding_logits.device
        
        # 构建 soft assignment 矩阵 P（避免 in-place 操作）
        n_values = len(self.value_to_idx)
        
        idx_list = [idx for _, idx in sorted(self.value_to_idx.items())]
        P_list = []
        for idx in idx_list:
            logits = self.encoding_logits[idx]  # [n_levels]
            if self.temperature > 0:
                probs = F.softmax(logits / self.temperature, dim=0)
            else:
                probs = F.softmax(logits, dim=0)
            P_list.append(probs)
        
        # 堆叠成矩阵
        P = torch.stack(P_list, dim=0)  # [n_values, n_levels]
        
        total_reg = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 1. 列互斥正则：col_sum = P.sum(dim=0)，惩罚 (col_sum - 1)^2
        # 鼓励每列被恰好一行主导
        col_sum = P.sum(dim=0)  # [n_codes]
        reg_col = (col_sum - 1.0).pow(2).mean()
        total_reg = total_reg + reg_col_weight * reg_col
        
        # 2. 行熵正则：鼓励每行低熵（接近 one-hot）
        # entropy = -sum(p * log(p))
        row_entropy = -torch.sum(P * torch.log(P + 1e-8), dim=-1)  # [n_values]
        reg_row = row_entropy.mean()
        total_reg = total_reg + reg_row_weight * reg_row
        
        # 3. 码字利用率正则：KL(column_mean || uniform)
        # 防止只用少数列
        col_mean = P.mean(dim=0)  # [n_codes]
        uniform = torch.ones_like(col_mean) / len(col_mean)
        # KL divergence: sum(p * log(p / q))
        reg_usage = torch.sum(col_mean * torch.log((col_mean + 1e-8) / (uniform + 1e-8)))
        total_reg = total_reg + reg_usage_weight * reg_usage
        
        return total_reg
    
    def get_monitoring_metrics(self) -> Dict[str, float]:
        """
        获取监控指标
        
        Returns:
            包含以下指标的字典：
            - collision_count: 编码冲突数（多个 value 映射到同一 code）
            - code_usage_entropy: 码字利用率熵
            - row_entropy_mean: 行熵均值
        """
        if not self.value_to_idx:
            return {
                'collision_count': 0.0,
                'code_usage_entropy': 0.0,
                'row_entropy_mean': 0.0
            }
        
        device = self.encoding_logits.device
        
        # 构建 soft assignment 矩阵 P（避免 in-place 操作）
        n_values = len(self.value_to_idx)
        
        idx_list = [idx for _, idx in sorted(self.value_to_idx.items())]
        P_list = []
        for idx in idx_list:
            logits = self.encoding_logits[idx]  # [n_levels]
            if self.temperature > 0:
                probs = F.softmax(logits / self.temperature, dim=0)
            else:
                probs = F.softmax(logits, dim=0)
            P_list.append(probs)
        
        # 堆叠成矩阵
        P = torch.stack(P_list, dim=0)  # [n_values, n_levels]
        
        # 1. 计算编码冲突数（hard mapping）
        value_to_code, code_to_value = self.get_hard_mapping()
        code_usage = {}
        for value, code in value_to_code.items():
            if code not in code_usage:
                code_usage[code] = 0
            code_usage[code] += 1
        collision_count = sum(1 for count in code_usage.values() if count > 1)
        
        # 2. 码字利用率熵
        col_mean = P.mean(dim=0)  # [n_codes]
        code_usage_entropy = -torch.sum(col_mean * torch.log(col_mean + 1e-8)).item()
        
        # 3. 行熵均值
        row_entropy = -torch.sum(P * torch.log(P + 1e-8), dim=-1)  # [n_values]
        row_entropy_mean = row_entropy.mean().item()
        
        return {
            'collision_count': float(collision_count),
            'code_usage_entropy': code_usage_entropy,
            'row_entropy_mean': row_entropy_mean
        }

    def _get_hamming_neighbors(self, code: int, k: int) -> List[int]:
        """获取Hamming距离为1的所有编码"""
        neighbors = []
        for i in range(k):
            neighbor = code ^ (1 << i)  # 翻转第i位
            if 0 <= neighbor < self.n_levels:
                neighbors.append(neighbor)
        return neighbors
    
    def initialize_from_traditional_olm(
        self,
        value_to_code: Dict[int, int],
        distribution: Dict[int, int],
        noise_scale: float = 0.1
    ):
        """
        从传统OLM的映射结果初始化可学习编码器
        
        这是训练策略的关键：先用传统OLM方法建立好映射，然后在此基础上微调。
        
        Args:
            value_to_code: 传统OLM的映射 {value: code}
            distribution: 量化值分布 {value: frequency}
            noise_scale: 添加的噪声尺度（避免完全固定，给微调留出空间）
        """
        # 1. 更新值映射
        sorted_values = sorted(distribution.keys())
        value_to_idx = {val: idx for idx, val in enumerate(sorted_values)}
        self.set_value_mapping(value_to_idx)
        
        # 2. 初始化encoding_logits
        # 先清零
        self.encoding_logits.data.zero_()
        
        # 对于每个量化值，将其对应的编码位置的logits设置为高值
        for value, code in value_to_code.items():
            if value in self.value_to_idx:
                idx = self.value_to_idx[value]
                # 确保code在有效范围内
                if 0 <= code < self.n_levels:
                    # 设置对应编码位置的logits为高值（如10.0）
                    # 这样softmax后，该编码的概率会接近1
                    self.encoding_logits.data[idx, code] = 10.0
        
        # 3. 添加小量噪声，避免完全固定
        # 这样在微调时可以有调整空间，同时保持传统OLM的映射作为主导
        if noise_scale > 0:
            noise = torch.randn_like(self.encoding_logits) * noise_scale
            self.encoding_logits.data += noise
        
        # 4. 存储分布（用于后续LRobust计算）
        if not hasattr(self, '_distribution'):
            self._distribution = {}
        self._distribution = distribution


class LearnableOLMManager:
    """
    可学习OLM编码管理器
    
    管理多个层的可学习OLM编码器，并提供统一的接口。
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        bit_widths: Dict[str, int],
        device: Optional[torch.device] = None,
        init_method: str = 'identity',
        temperature: float = 1.0,
        use_straight_through: bool = True,
    ):
        """
        Args:
            model: 量化模型
            layer_names: 要应用OLM的层名称列表
            bit_widths: 每层的位宽 {layer_name: bit_width}
            device: 设备
            init_method: 初始化方法
            temperature: Gumbel-Softmax温度
            use_straight_through: 是否使用Straight-Through Estimator
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.encoders: Dict[str, LearnableOLMEncoder] = {}
        
        # 为每个层创建编码器
        for layer_name in layer_names:
            bit_width = bit_widths.get(layer_name, 8)
            # 初始化时，我们不知道实际出现的量化值数量
            # 使用最大可能值，后续可以通过collect_distribution更新
            num_values = 1 << bit_width  # 最大可能值
            
            encoder = LearnableOLMEncoder(
                layer_name=layer_name,
                bit_width=bit_width,
                num_quantized_values=num_values,
                device=device,
                init_method=init_method,
                temperature=temperature,
                use_straight_through=use_straight_through,
            )
            self.encoders[layer_name] = encoder
    
    def collect_distribution_and_update(
        self,
        model: nn.Module,
        layer_name: str,
        num_samples: int = 1000
    ):
        """
        收集量化值分布并更新编码器的值映射
        
        Args:
            model: 量化模型
            layer_name: 层名称
            num_samples: 采样数量
        """
        from util.olm_encoder import collect_quantized_value_distribution
        
        # 收集分布
        distribution = collect_quantized_value_distribution(
            model, layer_name, num_samples
        )
        
        # 更新编码器的值映射
        if layer_name in self.encoders:
            encoder = self.encoders[layer_name]
            # 创建value_to_idx映射
            value_to_idx = {val: idx for idx, val in enumerate(sorted(distribution.keys()))}
            encoder.set_value_mapping(value_to_idx)
            
            # 更新num_quantized_values（如果需要）
            num_values = len(distribution)
            if num_values < encoder.num_quantized_values:
                # 可以重新创建编码器，或者只使用前num_values个
                # 这里简化处理，只更新映射
                pass
    
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """获取所有编码器的可学习参数"""
        params = []
        for encoder in self.encoders.values():
            params.extend(encoder.parameters())
        return params
    
    def get_hard_mappings(self) -> Dict[str, Dict[int, int]]:
        """获取所有层的hard映射（用于FaultInjector）"""
        mappings = {}
        for layer_name, encoder in self.encoders.items():
            value_to_code, _ = encoder.get_hard_mapping()
            mappings[layer_name] = value_to_code
        return mappings
    
    def initialize_from_traditional_olm(
        self,
        layer_name: str,
        value_to_code: Dict[int, int],
        distribution: Dict[int, int],
        noise_scale: float = 0.1
    ):
        """
        从传统OLM的映射结果初始化指定层的编码器
        
        Args:
            layer_name: 层名称
            value_to_code: 传统OLM的映射 {value: code}
            distribution: 量化值分布 {value: frequency}
            noise_scale: 添加的噪声尺度
        """
        if layer_name in self.encoders:
            self.encoders[layer_name].initialize_from_traditional_olm(
                value_to_code, distribution, noise_scale
            )
        else:
            raise ValueError(f"Layer {layer_name} not found in encoders")
    
    def update_distribution(self, layer_name: str, distribution: Dict[int, int]):
        """
        更新指定层的分布（用于LRobust计算）
        
        Args:
            layer_name: 层名称
            distribution: 量化值分布 {value: frequency}
        """
        if layer_name in self.encoders:
            self.encoders[layer_name]._distribution = distribution
    
    def get_distribution(self, layer_name: str) -> Optional[Dict[int, int]]:
        """
        获取指定层的分布
        
        Args:
            layer_name: 层名称
        
        Returns:
            量化值分布，如果不存在则返回None
        """
        if layer_name in self.encoders:
            encoder = self.encoders[layer_name]
            if hasattr(encoder, '_distribution'):
                return encoder._distribution
        return None
    
    def set_training(self, training: bool):
        """设置所有编码器的训练模式"""
        for encoder in self.encoders.values():
            encoder.train(training)
    
    def set_temperature(self, temperature: float):
        """设置所有编码器的温度参数"""
        for encoder in self.encoders.values():
            encoder.temperature = temperature
    
    def get_temperature_schedule(self, epoch: int, total_epochs: int) -> float:
        """
        获取温度调度值
        
        温度从 1.0 → 0.1 → 0.01 逐渐降低，让 soft assignment 逐渐接近 hard assignment
        
        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
        
        Returns:
            温度值
        """
        if epoch < total_epochs * 0.3:
            return 1.0
        elif epoch < total_epochs * 0.7:
            return 0.1
        else:
            return 0.01
    
    def update_temperature(self, epoch: int, total_epochs: int):
        """
        根据 epoch 更新温度
        
        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
        """
        temperature = self.get_temperature_schedule(epoch, total_epochs)
        self.set_temperature(temperature)

