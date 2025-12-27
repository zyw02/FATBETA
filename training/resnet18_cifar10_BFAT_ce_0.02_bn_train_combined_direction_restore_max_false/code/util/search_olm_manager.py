"""
基于搜索的OLM管理器（用于FAT训练）

特点：
- 使用贪心/模拟退火算法直接优化LRobust
- 支持定期更新（每N个batch）
- 支持混合目标函数（LRobust + FI后的准确率损失）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from util.olm_encoder import (
    collect_quantized_value_distribution,
    optimize_olm_mapping
)


class SearchOLMManager:
    """
    基于搜索的OLM管理器（用于FAT训练）
    
    特点：
    - 使用贪心/模拟退火算法直接优化LRobust
    - 支持定期更新（每N个batch）
    - 可以后期改进目标函数
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        bit_widths: Dict[str, int],
        update_freq: int = 10,  # 每N个epoch更新一次（现在表示epoch数，而不是batch数）
        method: str = 'greedy',  # 'greedy' 或 'simulated_annealing'
        num_samples: int = 1000,  # 收集分布时的采样数量
        max_iterations: int = 1000,  # 模拟退火的最大迭代次数
        device: Optional[torch.device] = None,
        # 混合目标函数相关参数
        alpha: float = 1.0,  # LRobust权重（1.0表示只优化LRobust，0.0表示只优化准确率）
        dataloader=None,  # 用于准确率评估的数据加载器
        fault_injector=None,  # 故障注入器
        criterion=None,  # 损失函数
        use_hybrid: bool = False  # 是否使用混合目标函数
    ):
        """
        Args:
            model: 量化模型
            layer_names: 需要应用OLM的层名称列表
            bit_widths: 每层的位宽 {layer_name: bit_width}
            update_freq: 更新频率（每N个epoch更新一次）
            method: 优化方法 ('greedy' 或 'simulated_annealing')
            num_samples: 收集分布时的采样数量
            max_iterations: 模拟退火的最大迭代次数
            device: 设备
        """
        self.model = model
        self.layer_names = layer_names
        self.bit_widths = bit_widths
        self.update_freq = update_freq
        self.method = method
        self.num_samples = num_samples
        self.max_iterations = max_iterations
        self.device = device
        
        # 混合目标函数相关
        self.alpha = alpha
        self.dataloader = dataloader
        self.fault_injector = fault_injector
        self.criterion = criterion
        self.use_hybrid = use_hybrid
        
        # OLM映射缓存
        self.olm_mappings: Dict[str, Dict[int, int]] = {}
        self.olm_code_to_value: Dict[str, Dict[int, int]] = {}
        
        # 分布缓存（用于快速更新）
        self.distribution_cache: Dict[str, Dict[int, int]] = {}
        
        # 更新计数器
        self.batch_count = 0
        
        # 初始化：为所有层生成初始映射
        mode_str = "混合目标函数" if use_hybrid else "LRobust"
        print(f"初始化SearchOLMManager: {len(layer_names)}层, 更新频率={update_freq} epochs, 方法={method}, 模式={mode_str}")
        if use_hybrid:
            print(f"  混合权重: α={alpha:.2f} (LRobust权重)")
        self.update_olm_mappings(force_update=True)
    
    def should_update(self) -> bool:
        """判断是否应该更新OLM映射"""
        return self.batch_count % self.update_freq == 0
    
    def get_current_alpha(self, epoch: int = 0, total_epochs: int = 100) -> float:
        """
        根据训练进度动态调整alpha
        
        策略：
        - 早期（0-20%）：只优化LRobust（α=1.0）
        - 中期（20-80%）：混合优化（α=0.5）
        - 后期（80-100%）：主要优化准确率（α=0.2）
        """
        if total_epochs == 0:
            return self.alpha
        
        progress = epoch / total_epochs
        
        if progress < 0.2:
            return 1.0  # 早期：只优化LRobust
        elif progress < 0.8:
            return 0.5  # 中期：混合优化
        else:
            return 0.2  # 后期：主要优化准确率
    
    def generate_top_k_candidates(
        self,
        layer_name: str,
        top_k: int = 5,
        num_candidates: int = 20  # 生成更多候选，然后选出top-k
    ) -> List[Tuple[Dict[int, int], Dict[int, int], float]]:
        """
        为指定层生成top-k候选映射（按LRobust排序）
        
        Args:
            layer_name: 层名称
            top_k: 返回的top-k候选数量
            num_candidates: 生成的候选总数（应该 >= top_k）
            
        Returns:
            [(value_to_code, code_to_value, lrobust), ...] 按LRobust升序排序
        """
        from util.olm_encoder import compute_lrobust
        
        # 收集量化值分布
        distribution = collect_quantized_value_distribution(
            self.model, layer_name, num_samples=self.num_samples
        )
        
        # 获取位宽，处理列表/元组情况
        bit_width_raw = self.bit_widths.get(layer_name, 8)
        if isinstance(bit_width_raw, (list, tuple)):
            bit_width = int(bit_width_raw[0])
        elif isinstance(bit_width_raw, torch.Tensor):
            bit_width = int(bit_width_raw.item())
        else:
            bit_width = int(bit_width_raw)
        
        candidates = []
        
        if self.method == 'simulated_annealing':
            # 使用不同的随机种子生成多个候选
            import random
            for seed in range(num_candidates):
                random.seed(seed)
                value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                    distribution, bit_width, 
                    method='simulated_annealing', 
                    max_iterations=self.max_iterations
                )
                candidates.append((value_to_code, code_to_value, lrobust))
        else:
            # 贪心算法：只生成一个候选
            value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                distribution, bit_width, 
                method=self.method, 
                max_iterations=self.max_iterations
            )
            candidates.append((value_to_code, code_to_value, lrobust))
        
        # 按LRobust排序，返回top-k
        candidates.sort(key=lambda x: x[2])  # 按LRobust升序排序（越小越好）
        return candidates[:top_k]
    
    def update_olm_mappings(
        self,
        dataloader=None,
        force_update: bool = False,
        epoch: int = 0,
        total_epochs: int = 100
    ):
        """
        更新所有层的OLM映射（支持混合目标函数）
        
        Args:
            dataloader: 数据加载器（可选，用于收集真实数据分布）
            force_update: 是否强制更新（忽略更新频率）
            epoch: 当前epoch（用于动态调整alpha）
            total_epochs: 总epoch数（用于动态调整alpha）
        """
        if not force_update and not self.should_update():
            return
        
        # 动态调整alpha（如果使用混合目标函数）
        current_alpha = self.get_current_alpha(epoch, total_epochs) if self.use_hybrid else self.alpha
        
        mode_str = f"混合目标函数 (α={current_alpha:.2f})" if self.use_hybrid else "LRobust"
        print(f"  🔄 更新OLM映射 (batch {self.batch_count}, {mode_str})...")
        
        for layer_name in self.layer_names:
            try:
                # 收集量化值分布
                if dataloader is not None:
                    # 使用真实数据收集分布（更准确，但较慢）
                    distribution = collect_quantized_value_distribution(
                        self.model, layer_name, num_samples=self.num_samples
                    )
                    self.distribution_cache[layer_name] = distribution
                else:
                    # 使用当前权重收集分布（快速）
                    distribution = collect_quantized_value_distribution(
                        self.model, layer_name, num_samples=self.num_samples
                    )
                    self.distribution_cache[layer_name] = distribution
                
                # 获取位宽，处理列表/元组情况
                bit_width_raw = self.bit_widths.get(layer_name, 8)
                if isinstance(bit_width_raw, (list, tuple)):
                    bit_width = int(bit_width_raw[0])
                elif isinstance(bit_width_raw, torch.Tensor):
                    bit_width = int(bit_width_raw.item())
                else:
                    bit_width = int(bit_width_raw)
                
                # 根据是否使用混合目标函数选择优化方法
                if self.use_hybrid and current_alpha < 1.0:
                    # 使用高效的混合目标函数优化（两阶段策略）
                    from util.efficient_hybrid_olm_optimizer import optimize_olm_mapping_efficient_hybrid
                    value_to_code, code_to_value, loss = optimize_olm_mapping_efficient_hybrid(
                        distribution, bit_width,
                        self.model, layer_name, self.dataloader,
                        self.fault_injector, self.criterion, self.device,
                        method='simulated_annealing',  # 混合优化使用模拟退火
                        max_iterations=self.max_iterations,
                        alpha=current_alpha,
                        num_samples=50,  # 使用小样本集（50个样本）
                        top_k_candidates=5  # 只评估top-5候选
                    )
                    print(f"    ✅ {layer_name}: Hybrid_Loss={loss:.4f} (α={current_alpha:.2f})")
                else:
                    # 只优化LRobust（快速）
                    value_to_code, code_to_value, lrobust = optimize_olm_mapping(
                        distribution, bit_width, method=self.method, max_iterations=self.max_iterations
                    )
                    print(f"    ✅ {layer_name}: LRobust={lrobust:.4f}, 映射大小={len(value_to_code)}")
                
                # 更新映射
                self.olm_mappings[layer_name] = value_to_code
                self.olm_code_to_value[layer_name] = code_to_value
                
            except Exception as e:
                print(f"    ⚠️  {layer_name}: 更新失败 - {e}")
                # 如果更新失败，使用缓存的映射或默认映射
                if layer_name not in self.olm_mappings:
                    self.olm_mappings[layer_name] = {}
                    self.olm_code_to_value[layer_name] = {}
    
    def increment_batch_count(self):
        """增加batch计数（在训练循环中调用）"""
        self.batch_count += 1
    
    def get_olm_mappings(self) -> Dict[str, Dict[int, int]]:
        """获取OLM映射（value_to_code）"""
        return self.olm_mappings
    
    def get_olm_code_to_value(self) -> Dict[str, Dict[int, int]]:
        """获取OLM反向映射（code_to_value）"""
        return self.olm_code_to_value
    
    def reset_batch_count(self):
        """重置batch计数（用于新epoch）"""
        self.batch_count = 0

