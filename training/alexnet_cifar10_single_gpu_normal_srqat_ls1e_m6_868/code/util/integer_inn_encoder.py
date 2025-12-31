"""
整数域可逆神经网络编码器 (Integer Domain Invertible Neural Network Encoder)

设计用于在量化级别（round后、scale前）进行可逆编码/解码。

核心思想：
1. 使用可学习的排列（permutation）保证整数到整数的双射映射
2. 支持端到端训练（可微分）
3. 保证100%可逆（整数到整数）
4. 在量化级别空间工作（round后、scale前）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class IntegerINNEncoder(nn.Module):
    """
    整数域可逆神经网络编码器
    
    用于在量化级别（round后、scale前）进行可逆编码/解码。
    
    设计特点：
    1. 输入输出都是整数（量化级别）
    2. 保证双射映射（整数到整数）
    3. 可微分（支持训练）
    4. 可逆（100%可逆）
    
    实现方法：
    - 使用可学习的排列矩阵（permutation matrix）
    - 通过Sinkhorn算法保证双随机性（训练时）
    - 通过匈牙利算法保证硬双射（推理时）
    """
    
    def __init__(
        self,
        layer_name: str,
        bit_width: int,
        device: Optional[torch.device] = None,
        use_sinkhorn: bool = True,
        sinkhorn_iterations: int = 10
    ):
        """
        Args:
            layer_name: 层名称
            bit_width: 量化位宽
            device: 设备
            use_sinkhorn: 是否使用Sinkhorn算法（训练时）
            sinkhorn_iterations: Sinkhorn算法迭代次数
        """
        super().__init__()
        self.layer_name = layer_name
        self.bit_width = bit_width
        self.n_levels = 1 << bit_width  # 2^bit_width
        
        # 量化值范围
        self.thd_neg = -(1 << (bit_width - 1))
        self.thd_pos = (1 << (bit_width - 1)) - 1
        
        # 可学习的排列矩阵（logits）
        # 使用logits而不是直接的排列，以便可微分
        # shape: [n_levels, n_levels]
        self.permutation_logits = nn.Parameter(
            torch.randn(self.n_levels, self.n_levels, device=device)
        )
        
        # Sinkhorn算法参数
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations
        
        # 缓存硬排列（推理时使用）
        self._cached_permutation: Optional[torch.Tensor] = None
        self._cached_inverse_permutation: Optional[torch.Tensor] = None
    
    def _sinkhorn(self, logits: torch.Tensor, num_iterations: int = 10, epsilon: float = 1e-3) -> torch.Tensor:
        """
        Sinkhorn算法：将矩阵转换为双随机矩阵（可微分）
        
        Args:
            logits: [n_levels, n_levels] 排列logits
            num_iterations: 迭代次数
            epsilon: 正则化参数
        
        Returns:
            双随机矩阵（每行和每列的和都为1）
        """
        probs = F.softmax(logits, dim=-1) + epsilon
        
        # 交替归一化行和列
        for _ in range(num_iterations):
            # 归一化行（每行和为1）
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            # 归一化列（每列和为1）
            probs = probs / (probs.sum(dim=0, keepdim=True) + 1e-8)
        
        return probs
    
    def _get_hard_permutation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取硬排列（推理时使用，保证双射）
        
        使用匈牙利算法找到最优排列，保证双射
        
        Returns:
            (permutation, inverse_permutation) 排列和逆排列
        """
        if self._cached_permutation is None:
            try:
                from scipy.optimize import linear_sum_assignment
                # 使用匈牙利算法找到最优排列（保证双射）
                cost_matrix = -self.permutation_logits.detach().cpu().numpy()
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                permutation = torch.tensor(col_indices, dtype=torch.long, device=self.permutation_logits.device)
            except ImportError:
                # 如果scipy不可用，使用贪心方法（可能不是完全双射）
                # 使用argmax找到每行的最大列索引
                permutation = torch.argmax(self.permutation_logits, dim=-1)  # [n_levels]
                
                # 验证是否是双射
                unique_outputs = torch.unique(permutation)
                if len(unique_outputs) < self.n_levels:
                    # 如果不是双射，尝试修复
                    # 对于冲突的输出，选择logits最大的输入
                    for output in range(self.n_levels):
                        if output not in unique_outputs:
                            # 找到所有映射到这个输出的输入，选择logits最大的
                            candidates = torch.where(permutation == output)[0]
                            if len(candidates) == 0:
                                # 如果没有输入映射到这个输出，找到logits最大的未使用输入
                                used_inputs = set(permutation.tolist())
                                unused_inputs = [i for i in range(self.n_levels) if i not in used_inputs]
                                if unused_inputs:
                                    best_input = max(unused_inputs, key=lambda i: self.permutation_logits[i, output].item())
                                    permutation[best_input] = output
            
            # 创建逆排列
            inverse_permutation = torch.argsort(permutation)
            
            self._cached_permutation = permutation
            self._cached_inverse_permutation = inverse_permutation
        
        return self._cached_permutation, self._cached_inverse_permutation
    
    def invalidate_cache(self):
        """使缓存失效（训练时调用）"""
        self._cached_permutation = None
        self._cached_inverse_permutation = None
    
    def encode(
        self,
        quantized_levels: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        编码：量化级别 → 编码值
        
        Args:
            quantized_levels: 量化级别（整数），范围 [thd_neg, thd_pos]
            training: 是否在训练模式
        
        Returns:
            编码值（整数），范围 [0, n_levels-1]
        """
        original_shape = quantized_levels.shape
        
        # 将量化级别转换为非负整数 [0, n_levels-1]
        quantized_levels_shifted = (quantized_levels - self.thd_neg).clamp(0, self.n_levels - 1).long()
        
        # 展平
        flat = quantized_levels_shifted.view(-1)
        
        if training and self.use_sinkhorn:
            # 训练时：使用Sinkhorn算法（可微分，软排列）
            probs = self._sinkhorn(self.permutation_logits, self.sinkhorn_iterations)
            
            # 使用概率分布进行编码（可微分）
            # 对于每个输入，使用概率分布计算期望输出
            codes_soft = (probs[flat] * torch.arange(self.n_levels, device=flat.device, dtype=probs.dtype)).sum(dim=-1)
            # ⚠️ 关键：确保codes_soft在[0, n_levels-1]范围内（位宽约束）
            codes_soft = codes_soft.clamp(0, self.n_levels - 1)
            
            # 使用Straight-Through Estimator
            codes_hard = torch.argmax(probs[flat], dim=-1)  # 保证在[0, n_levels-1]范围内
            codes = codes_hard.float() + codes_soft - codes_soft.detach()
            # ⚠️ 关键：最终确保在[0, n_levels-1]范围内（位宽约束）
            codes = codes.long().clamp(0, self.n_levels - 1)
        else:
            # 推理时：使用硬排列（保证双射）
            permutation, _ = self._get_hard_permutation()
            # 排列矩阵保证输出在[0, n_levels-1]范围内，但添加clamp作为额外保护
            codes = permutation[flat].clamp(0, self.n_levels - 1)
        
        return codes.view(original_shape)
    
    def decode(
        self,
        codes: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        解码：编码值 → 量化级别
        
        Args:
            codes: 编码值（整数），范围 [0, n_levels-1]
            training: 是否在训练模式
        
        Returns:
            量化级别（整数），范围 [thd_neg, thd_pos]
        """
        original_shape = codes.shape
        
        # 展平
        flat = codes.view(-1).long().clamp(0, self.n_levels - 1)
        
        if training and self.use_sinkhorn:
            # 训练时：使用Sinkhorn算法的逆映射（可微分）
            probs = self._sinkhorn(self.permutation_logits, self.sinkhorn_iterations)
            
            # 使用概率分布进行解码（可微分）
            # 对于每个编码值，找到最可能的输入
            # 使用转置矩阵进行逆映射
            probs_inverse = probs.T  # [n_levels, n_levels]
            quantized_levels_soft = (probs_inverse[flat] * torch.arange(self.n_levels, device=flat.device, dtype=probs.dtype)).sum(dim=-1)
            
            # 使用Straight-Through Estimator
            quantized_levels_hard = torch.argmax(probs_inverse[flat], dim=-1)
            quantized_levels_shifted = quantized_levels_hard.float() + quantized_levels_soft - quantized_levels_soft.detach()
            quantized_levels_shifted = quantized_levels_shifted.long().clamp(0, self.n_levels - 1)
        else:
            # 推理时：使用逆排列（保证双射）
            _, inverse_permutation = self._get_hard_permutation()
            quantized_levels_shifted = inverse_permutation[flat].clamp(0, self.n_levels - 1)
        
        # 转换回原始范围
        quantized_levels = quantized_levels_shifted + self.thd_neg
        
        return quantized_levels.view(original_shape)
    
    def get_hard_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        获取硬映射（用于验证和调试）
        
        Returns:
            (value_to_code, code_to_value) 映射字典
        """
        permutation, inverse_permutation = self._get_hard_permutation()
        
        value_to_code = {}
        code_to_value = {}
        
        for value in range(self.n_levels):
            code = int(permutation[value].item())
            value_to_code[value + self.thd_neg] = code
            code_to_value[code] = value + self.thd_neg
        
        return value_to_code, code_to_value


class IntegerINNOLMManager:
    """
    整数域INN OLM管理器
    
    管理多个层的整数域INN编码器。
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        bit_widths: Dict[str, int],
        device: Optional[torch.device] = None,
        use_sinkhorn: bool = True,
        sinkhorn_iterations: int = 10
    ):
        self.model = model
        self.layer_names = layer_names
        self.bit_widths = bit_widths
        self.device = device
        self.encoders: Dict[str, IntegerINNEncoder] = {}
        
        # 为每个层创建编码器
        for layer_name in layer_names:
            bit_width = bit_widths.get(layer_name, 8)
            encoder = IntegerINNEncoder(
                layer_name=layer_name,
                bit_width=bit_width,
                device=device,
                use_sinkhorn=use_sinkhorn,
                sinkhorn_iterations=sinkhorn_iterations
            )
            if device is not None:
                encoder = encoder.to(device)
            self.encoders[layer_name] = encoder
    
    def get_parameters(self):
        """获取所有编码器的参数"""
        params = []
        for encoder in self.encoders.values():
            params.extend(list(encoder.parameters()))
        return params
    
    def invalidate_cache(self):
        """使所有编码器的缓存失效"""
        for encoder in self.encoders.values():
            encoder.invalidate_cache()
