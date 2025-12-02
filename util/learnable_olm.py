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
        # 将量化值转换为索引
        # 对于未映射的值，使用最近的映射值
        indices = self._value_to_indices(quantized_values)
        
        # 获取对应的logits
        # shape: [num_values, n_levels]
        logits = self.encoding_logits[indices]  # [num_values, n_levels]
        
        if training and self.temperature > 0:
            # 训练时：使用Gumbel-Softmax实现可微分的离散采样
            if self.use_straight_through:
                # Straight-Through Estimator: 前向使用hard，反向使用soft
                hard_codes = torch.argmax(logits, dim=-1)  # [num_values]
                soft_codes = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
                # 使用straight-through技巧
                codes = hard_codes + soft_codes - soft_codes.detach()
            else:
                # 纯Gumbel-Softmax
                codes = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                codes = torch.argmax(codes, dim=-1)
        else:
            # 推理时：使用hard assignment
            codes = torch.argmax(logits, dim=-1)  # [num_values]
        
        return codes
    
    def decode(
        self,
        codes: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        将编码空间的值解码回量化值
        
        Args:
            codes: 编码值，shape: [*]
            training: 是否在训练模式
        
        Returns:
            量化值，shape: [*]
        """
        # 对于解码，我们需要反向映射：code -> value
        # 由于一个编码可能对应多个量化值，我们选择logits最大的那个
        
        # 找到每个编码对应的最大logits的量化值索引
        # shape: [n_levels]
        max_indices = torch.argmax(self.encoding_logits, dim=0)  # [n_levels]
        
        # 将编码映射到量化值索引
        code_indices = codes.clamp(0, self.n_levels - 1).long()
        value_indices = max_indices[code_indices]
        
        # 将索引转换为量化值（使用查找表加速）
        values = torch.zeros_like(codes, dtype=torch.float32)
        
        # 构建索引到值的查找表
        if self.idx_to_value:
            # 创建查找表（tensor）
            max_idx = max(self.idx_to_value.keys())
            lookup_table = torch.zeros(max_idx + 1, dtype=torch.float32, device=codes.device)
            for idx, value in self.idx_to_value.items():
                lookup_table[idx] = float(value)
            
            # 使用查找表
            valid_mask = (value_indices <= max_idx)
            values[valid_mask] = lookup_table[value_indices[valid_mask]]
            
            # 对于无效索引，使用默认值（最近的映射值）
            if not valid_mask.all():
                default_value = float(list(self.idx_to_value.values())[0])
                values[~valid_mask] = default_value
        
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
    
    def get_hard_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        获取hard assignment的映射（用于推理）
        
        Returns:
            (value_to_code, code_to_value) 映射字典
        """
        value_to_code = {}
        code_to_value = {}
        
        # 对于每个量化值，找到logits最大的编码
        for value, idx in self.value_to_idx.items():
            code = int(torch.argmax(self.encoding_logits[idx]).item())
            value_to_code[value] = code
            if code not in code_to_value:
                code_to_value[code] = value
        
        return value_to_code, code_to_value
    
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
    
    def _compute_lrobust_loss_soft(
        self,
        distribution: Dict[int, int],
        ber: float = 1e-2
    ) -> torch.Tensor:
        """
        使用soft assignment计算LRobust（完全可微分，用于端到端训练）
        
        这是可学习OLM的版本，使用soft assignment实现完全可微分。
        通过期望值计算，使得梯度可以传播到encoding_logits。
        """
        total_loss = torch.tensor(0.0, device=self.encoding_logits.device, requires_grad=True)
        total_weight = torch.tensor(0.0, device=self.encoding_logits.device)
        
        # 对于每个量化值
        for value, frequency in distribution.items():
            if value not in self.value_to_idx:
                continue
            
            idx = self.value_to_idx[value]
            freq_tensor = torch.tensor(float(frequency), device=self.encoding_logits.device)
            
            # 获取该量化值对所有编码的logits
            logits = self.encoding_logits[idx]  # [n_levels]
            
            # 使用softmax得到soft assignment（可微分）
            if self.temperature > 0:
                code_probs = F.softmax(logits / self.temperature, dim=0)  # [n_levels]
            else:
                code_probs = F.softmax(logits, dim=0)  # [n_levels]
            
            # 对于每个可能的编码，计算其对LRobust的贡献
            for code in range(self.n_levels):
                code_prob = code_probs[code]
                
                # 找到所有Hamming距离为1的编码
                hamming_neighbors = self._get_hamming_neighbors(code, self.bit_width)
                
                for neighbor_code in hamming_neighbors:
                    # 计算邻居编码对应的期望量化值（使用soft assignment）
                    # 对于每个量化值，计算它映射到neighbor_code的概率
                    neighbor_value_expected = torch.tensor(0.0, device=self.encoding_logits.device)
                    
                    for neighbor_value, neighbor_idx in self.value_to_idx.items():
                        neighbor_logits = self.encoding_logits[neighbor_idx]  # [n_levels]
                        if self.temperature > 0:
                            neighbor_probs = F.softmax(neighbor_logits / self.temperature, dim=0)
                        else:
                            neighbor_probs = F.softmax(neighbor_logits, dim=0)
                        neighbor_value_expected += neighbor_probs[neighbor_code] * float(neighbor_value)
                    
                    # 计算误差平方（可微分）
                    value_tensor = torch.tensor(float(value), device=self.encoding_logits.device)
                    error_sq = (value_tensor - neighbor_value_expected) ** 2
                    total_loss = total_loss + error_sq * code_prob * freq_tensor
                    total_weight = total_weight + code_prob * freq_tensor
        
        # 归一化
        lrobust = total_loss / (total_weight + 1e-8)
        return lrobust


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

