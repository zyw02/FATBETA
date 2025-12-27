"""
Progressive BER Scheduler for Curriculum Learning

借鉴对抗训练中的渐进式攻击强度调度，实现渐进式 BER 训练。
从低 BER 开始，逐步增加到高 BER，帮助模型更好地学习恢复策略。
"""

import numpy as np
from typing import Optional


class ProgressiveBERScheduler:
    """
    渐进式 BER 调度器
    
    支持多种调度策略：
    - linear: 线性增长
    - cosine: 余弦增长（开始慢，后期快）
    - exponential: 指数增长
    - step: 阶梯式增长
    """
    
    def __init__(
        self,
        ber_min: float = 2e-2,
        ber_max: float = 1e-1,
        total_epochs: int = 200,
        schedule_type: str = 'cosine',
        warmup_epochs: int = 10,
    ):
        """
        Args:
            ber_min: 最小 BER（训练开始时的 BER）
            ber_max: 最大 BER（训练结束时的 BER）
            total_epochs: 总训练轮数
            schedule_type: 调度类型 ('linear', 'cosine', 'exponential', 'step')
            warmup_epochs: 预热轮数（在 warmup 期间保持 ber_min）
        """
        self.ber_min = ber_min
        self.ber_max = ber_max
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        
        # 计算有效训练轮数（排除 warmup）
        self.training_epochs = total_epochs - warmup_epochs
    
    def get_ber(self, epoch: int) -> float:
        """
        根据当前 epoch 获取对应的 BER
        
        Args:
            epoch: 当前 epoch（从 0 开始）
        
        Returns:
            ber: 当前 epoch 应该使用的 BER
        """
        if epoch < self.warmup_epochs:
            # Warmup 期间使用最小 BER
            return self.ber_min
        
        # 计算进度（0.0 到 1.0）
        progress = (epoch - self.warmup_epochs) / max(1, self.training_epochs)
        progress = np.clip(progress, 0.0, 1.0)
        
        # 根据调度类型计算 BER
        if self.schedule_type == 'linear':
            ber = self.ber_min + (self.ber_max - self.ber_min) * progress
        elif self.schedule_type == 'cosine':
            # 余弦调度：开始慢，后期快
            ber = self.ber_min + (self.ber_max - self.ber_min) * (1 - np.cos(progress * np.pi / 2))
        elif self.schedule_type == 'exponential':
            # 指数调度：开始快，后期慢
            ber = self.ber_min + (self.ber_max - self.ber_min) * (np.exp(progress * np.log(2)) - 1)
        elif self.schedule_type == 'step':
            # 阶梯式调度：分阶段增加
            if progress < 0.3:
                ber = self.ber_min + (self.ber_max - self.ber_min) * 0.2
            elif progress < 0.6:
                ber = self.ber_min + (self.ber_max - self.ber_min) * 0.5
            elif progress < 0.9:
                ber = self.ber_min + (self.ber_max - self.ber_min) * 0.8
            else:
                ber = self.ber_max
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
        
        return float(np.clip(ber, self.ber_min, self.ber_max))
    
    def get_schedule_info(self) -> str:
        """获取调度信息（字符串格式）"""
        return f"Type: {self.schedule_type}, Min BER: {self.ber_min:.2e}, Max BER: {self.ber_max:.2e}, Total Epochs: {self.total_epochs}, Warmup Epochs: {self.warmup_epochs}"

