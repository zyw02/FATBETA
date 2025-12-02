"""
Fault Injection Tool for Retraining-Free Quantization

This module provides a fault injection tool specifically designed for the 
retraining-free-quantization framework. It supports:
- BER (Bit-Error-Rate) mode fault injection
- Automatic reading of quantization bit-width configurations from layers
- Training and inference mode support
- GPU-accelerated bit-flip operations
- Gradient-preserving fault injection
- Loading bit-width configurations from JSON files for mixed-precision models

Isolation: This tool uses hook-based wrapping that can be enabled/disabled 
without affecting the original model behavior.
"""

import math
import json
import os
import hashlib
from typing import Optional, Literal, Dict, Any, Tuple, List
import torch
import torch.nn as nn
from prettytable import PrettyTable
from quan.func import QuanConv2d, QuanLinear
from .qat import set_bit_width, get_quantized_layers


Mode = Literal["ber"]


class FaultInjector:
    """
    Fault injector for quantized weights in retraining-free-quantization models.
    
    ⚠️ **重要说明：故障类型范围**
    - 当前只模拟**数据位翻转**（Data Bit Flips）：权重存储器的SEU
      - 适用于：FPGA BRAM、ASIC SRAM、GPU内存中的权重数据
      - 影响：数据值改变，但电路结构不变
    - 不模拟**配置位翻转**（Configuration Bit Flips）：FPGA配置存储器的SEU
      - 配置位翻转会导致电路路由错误、逻辑功能改变
      - 可能导致电路完全失效
      - 需要FPGA比特流信息和重新部署，不在当前模拟范围内
    
    Features:
    - Works for both inference and training (preserves gradients)
    - BER mode: per-bit Bernoulli flips with probability `ber`
    - Automatically reads quantization bit-width from layer configuration
    - Only applies to weights; activation and other params are untouched
    - Isolation: enable()/disable() wraps and restores forward methods
    
    Args:
        model: The quantized model (should contain QuanConv2d/QuanLinear layers)
        mode: Injection mode, currently only "ber" is supported
        ber: Bit-error-rate probability (0.0 to 1.0)
        device: Device for fault injection (default: model's device)
        enable_in_training: If True, enable fault injection during training
        enable_in_inference: If True, enable fault injection during inference
        seed: Random seed for reproducibility
    
    See also:
        - FPGA_FAULT_TYPES.md: 详细说明FPGA中配置位翻转 vs 数据位翻转的区别
        - SPACEBORNE_FAULT_MODEL.md: 星载平台故障模型分析
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        mode: Mode = "ber",
        ber: Optional[float] = None,
        device: Optional[torch.device] = None,
        enable_in_training: bool = True,
        enable_in_inference: bool = True,
        seed: Optional[int] = None,
        use_position_based_mask: bool = False,
        seed_list: Optional[List[int]] = None,
        skip_first_last: bool = False,
        use_random_flip_in_training: bool = False,
        enable_statistics: bool = False,  # 是否启用统计功能（默认关闭以提升性能）
        whitelist_layer: Optional[str] = None,  # 仅针对特定层进行故障注入（用于敏感度分析）
        gray_code_layers: Optional[List[str]] = None,  # 使用格雷码编码的层列表（None表示不使用格雷码）
        olm_layers: Optional[Dict[str, Dict[int, int]]] = None,  # 使用OLM编码的层映射 {layer_name: {value: code}}
    ) -> None:
        self.model = model
        self.mode = mode
        self.ber = ber
        self.device = device
        self.enable_in_training = enable_in_training
        self.enable_in_inference = enable_in_inference
        self.seed = seed
        self.use_position_based_mask = use_position_based_mask  # 是否使用基于位置的固定掩码
        self.seed_list = seed_list  # 固定的seed列表，训练时轮询使用，验证时从中随机采样
        self.use_random_flip_in_training = use_random_flip_in_training  # 训练时是否使用完全随机化的bit-flip（不使用base_seed+hash）
        self._current_seed_index = 0  # 当前使用的seed索引（用于训练时轮询，验证时按顺序使用）
        self._current_forward_seed = None  # 当前forward使用的base_seed（训练时，同一个forward中所有层使用相同的base_seed）
        
        # 统计每个seed的使用频率
        self._seed_usage_count = {}  # {seed: count}
        if self.seed_list is not None:
            for s in self.seed_list:
                self._seed_usage_count[s] = 0
        
        self._wrapped: Dict[int, Any] = {}
        self._enabled = False
        self._training_state = None
        self._wrap_logged = False  # 标记是否已打印包装日志
        self._layer_name_map: Dict[int, str] = {}  # 存储每个module的layer名称，key是id(module.quan_w_fn)（仅当use_position_based_mask=True时使用）
        self.skip_first_last = skip_first_last  # 是否跳过第一层和最后一层
        self.enable_statistics = enable_statistics  # 是否启用统计功能
        self.whitelist_layer = whitelist_layer  # 白名单层
        self.gray_code_layers = set(gray_code_layers) if gray_code_layers else set()  # 使用格雷码的层集合
        self.olm_layers = olm_layers if olm_layers else {}  # 使用OLM编码的层映射 {layer_name: {value: code}}
        # 为OLM创建反向映射（code -> value）以加速查找
        self.olm_code_to_value: Dict[str, Dict[int, int]] = {}
        for layer_name, value_to_code in self.olm_layers.items():
            self.olm_code_to_value[layer_name] = {code: value for value, code in value_to_code.items()}
        
        # 统计信息：记录实际翻转的bit数和总bit数
        self._flip_stats: Dict[str, Dict[str, int]] = {}  # {layer_name: {'flipped_bits': int, 'total_bits': int, 'injections': int, 'total_params': int, 'affected_params': int}}
        # 延迟统计：累积flip_mask的sum，避免每次GPU-CPU同步
        # 使用异步方式：存储flip_mask的sum tensor（不立即同步到CPU）
        self._pending_stats: List[Tuple[torch.Tensor, int, int, str]] = []  # [(flip_mask_sum_tensor, total_bits, total_params, layer_name), ...]
        
        # Validate and convert BER to float if needed
        if self.mode == "ber":
            if self.ber is None:
                raise ValueError("BER mode requires ber parameter")
            # Convert string to float if needed (e.g., "1e-2" from YAML)
            if isinstance(self.ber, str):
                self.ber = float(self.ber)
            else:
                self.ber = float(self.ber)
            # Validate range
            if self.ber < 0 or self.ber > 1:
                raise ValueError(f"BER mode requires 0 <= ber <= 1, got {self.ber}")
        
        # Validate seed_list
        if self.seed_list is not None:
            if not isinstance(self.seed_list, (list, tuple)) or len(self.seed_list) == 0:
                raise ValueError("seed_list must be a non-empty list or tuple of integers")
            self.seed_list = [int(s) for s in self.seed_list]
            # If seed_list is provided, use the first seed as default
            if self.seed is None:
                self.seed = self.seed_list[0]
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
        # Debug trace: print per-layer flip ratio once when enabled via env
        import os
        self._trace_once = os.environ.get('FI_TRACE_ONCE', '0') == '1'
        self._traced_layers = set()
    
    def enable(self) -> None:
        """Enable fault injection by wrapping layer forward methods."""
        if self._enabled:
            return
        self._wrap_modules()
        self._enabled = True
    
    def disable(self) -> None:
        """Disable fault injection by restoring original forward methods."""
        if not self._enabled:
            return
        self._restore_modules()
        self._enabled = False
        # Reset forward seed (but keep _current_seed_index to continue round-robin)
        # This ensures that each forward pass uses a different seed from seed_list
        # instead of always starting from seed_list[0] = 42
        self._current_forward_seed = None
        # NOTE: Do NOT reset _current_seed_index here, as it would cause all forwards
        # to use seed=42. Instead, let it continue from where it left off.
        
        # 处理延迟统计：在disable时批量处理pending的统计信息
        # 只在启用统计功能时才处理，避免不必要的开销
        if self.enable_statistics and self._pending_stats:
            self._process_pending_statistics()
    
    def _process_pending_statistics(self) -> None:
        """
        批量处理延迟统计信息，避免在每次注入时都进行GPU-CPU同步。
        这个方法应该在disable()时或需要统计时调用。
        使用批量同步，减少GPU-CPU同步次数，提升性能。
        """
        if not self._pending_stats:
            return
        
        # 批量处理所有pending的统计信息
        # 先收集所有需要同步的tensor，然后一次性同步到CPU
        flip_mask_sums = [item[0] for item in self._pending_stats]
        affected_params_sums = [item[4] for item in self._pending_stats]  # 新增：受影响的参数数量
        
        if flip_mask_sums:
            # 批量同步：将所有tensor的sum结果一次性同步到CPU
            # 这样可以减少同步次数，提升性能
            # 如果tensor是标量，需要先unsqueeze
            processed_sums = []
            for s in flip_mask_sums:
                if s.dim() == 0:
                    processed_sums.append(s.unsqueeze(0))
                else:
                    processed_sums.append(s)
            if processed_sums:
                flipped_bits_counts = torch.cat(processed_sums).cpu().tolist()
            else:
                flipped_bits_counts = []
        else:
            flipped_bits_counts = []
        
        if affected_params_sums:
            # 批量同步受影响的参数数量
            processed_affected = []
            for s in affected_params_sums:
                if s.dim() == 0:
                    processed_affected.append(s.unsqueeze(0))
                else:
                    processed_affected.append(s)
            if processed_affected:
                affected_params_counts = torch.cat(processed_affected).cpu().tolist()
            else:
                affected_params_counts = []
        else:
            affected_params_counts = []
        
        # 更新统计信息
        for idx, (_, total_bits, total_params, stats_key, __) in enumerate(self._pending_stats):
            flipped_bits_count = int(flipped_bits_counts[idx]) if idx < len(flipped_bits_counts) else 0
            affected_params_count = int(affected_params_counts[idx]) if idx < len(affected_params_counts) else 0
            
            if stats_key not in self._flip_stats:
                self._flip_stats[stats_key] = {
                    'flipped_bits': 0, 
                    'total_bits': 0, 
                    'injections': 0,
                    'total_params': 0,
                    'affected_params': 0
                }
            self._flip_stats[stats_key]['flipped_bits'] += flipped_bits_count
            self._flip_stats[stats_key]['total_bits'] += total_bits
            self._flip_stats[stats_key]['total_params'] += total_params
            self._flip_stats[stats_key]['affected_params'] += affected_params_count
            self._flip_stats[stats_key]['injections'] += 1
        
        # 清空pending列表
        self._pending_stats.clear()
    
    def get_seed_usage_stats(self) -> dict:
        """Get statistics about seed usage frequency."""
        total = sum(self._seed_usage_count.values())
        stats = {}
        for seed, count in self._seed_usage_count.items():
            stats[seed] = {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0.0
            }
        return stats
    
    def get_flip_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取故障注入统计信息。
        
        Returns:
            字典，格式为 {layer_name: {
                'flipped_bits': int,      # 翻转的bit数
                'total_bits': int,        # 总bit数
                'injections': int,         # 注入次数
                'flip_ratio': float,       # 总翻转比例 (%)
                'avg_flip_ratio': float,  # 平均每次注入的翻转比例 (%)
                'total_params': int,      # 总参数数量
                'affected_params': int,   # 受影响的参数数量（至少有一个bit被翻转）
                'affected_ratio': float,  # 受影响参数比例 (%)
                'avg_affected_ratio': float  # 平均每次注入的受影响参数比例 (%)
            }}
        """
        # 处理pending的统计信息
        if self._pending_stats:
            self._process_pending_statistics()
        stats = {}
        for layer_name, data in self._flip_stats.items():
            flipped = data['flipped_bits']
            total = data['total_bits']
            injections = data['injections']
            total_params = data.get('total_params', 0)
            affected_params = data.get('affected_params', 0)
            
            flip_ratio = (flipped / total * 100) if total > 0 else 0.0
            avg_flip_ratio = flip_ratio / injections if injections > 0 else 0.0
            affected_ratio = (affected_params / total_params * 100) if total_params > 0 else 0.0
            avg_affected_ratio = affected_ratio / injections if injections > 0 else 0.0
            
            stats[layer_name] = {
                'flipped_bits': flipped,
                'total_bits': total,
                'injections': injections,
                'flip_ratio': flip_ratio,  # 总翻转比例 (%)
                'avg_flip_ratio': avg_flip_ratio,  # 平均每次注入的翻转比例 (%)
                'total_params': total_params,  # 总参数数量
                'affected_params': affected_params,  # 受影响的参数数量（至少有一个bit被翻转）
                'affected_ratio': affected_ratio,  # 受影响参数比例 (%)
                'avg_affected_ratio': avg_affected_ratio,  # 平均每次注入的受影响参数比例 (%)
            }
        return stats
    
    def reset_flip_statistics(self) -> None:
        """重置故障注入统计信息。"""
        self._flip_stats.clear()
        self._pending_stats.clear()
    
    def print_flip_statistics(self, verbose: bool = True) -> None:
        """
        打印故障注入统计信息。
        
        Args:
            verbose: 如果为True，打印每层的详细信息；否则只打印汇总信息
        """
        stats = self.get_flip_statistics()
        if not stats:
            print("故障注入统计：暂无数据（可能尚未进行故障注入）")
            return
        
        print("=" * 80)
        print("故障注入统计信息")
        print("=" * 80)
        print(f"配置BER: {self.ber:.2e} ({self.ber * 100:.2f}%)")
        print(f"统计层数: {len(stats)}")
        print("-" * 80)
        
        total_flipped = 0
        total_bits = 0
        total_injections = 0
        
        total_params = 0
        total_affected = 0
        
        if verbose:
            # 使用 PrettyTable 创建表格
            table = PrettyTable()
            table.field_names = [
                "层名称",
                "翻转bit数",
                "总bit数",
                "注入次数",
                "翻转比例",
                "平均翻转比例",
                "受影响参数",
                "总参数数",
                "受影响比例"
            ]
            # 设置对齐方式
            table.align["层名称"] = "l"
            table.align["翻转bit数"] = "r"
            table.align["总bit数"] = "r"
            table.align["注入次数"] = "r"
            table.align["翻转比例"] = "r"
            table.align["平均翻转比例"] = "r"
            table.align["受影响参数"] = "r"
            table.align["总参数数"] = "r"
            table.align["受影响比例"] = "r"
            # 使用简洁的表格风格
            table.set_style(12)  # MSWORD_FRIENDLY 风格
        
        for layer_name, data in sorted(stats.items()):
            flipped = data['flipped_bits']
            total = data['total_bits']
            injections = data['injections']
            flip_ratio = data['flip_ratio']
            avg_flip_ratio = data['avg_flip_ratio']
            affected_params = data.get('affected_params', 0)
            total_params_layer = data.get('total_params', 0)
            affected_ratio = data.get('affected_ratio', 0.0)
            
            total_flipped += flipped
            total_bits += total
            total_injections += injections
            total_params += total_params_layer
            total_affected += affected_params
            
            if verbose:
                table.add_row([
                    layer_name,
                    f"{flipped:,}",
                    f"{total:,}",
                    injections,
                    f"{flip_ratio:.4f}%",
                    f"{avg_flip_ratio:.4f}%",
                    f"{affected_params:,}",
                    f"{total_params_layer:,}",
                    f"{affected_ratio:.4f}%"
                ])
        
        # 打印汇总信息
        overall_ratio = (total_flipped / total_bits * 100) if total_bits > 0 else 0.0
        avg_overall_ratio = overall_ratio / total_injections if total_injections > 0 else 0.0
        overall_affected_ratio = (total_affected / total_params * 100) if total_params > 0 else 0.0
        
        if verbose:
            # 添加总计行
            table.add_row([
                "总计",
                f"{total_flipped:,}",
                f"{total_bits:,}",
                total_injections,
                f"{overall_ratio:.4f}%",
                f"{avg_overall_ratio:.4f}%",
                f"{total_affected:,}",
                f"{total_params:,}",
                f"{overall_affected_ratio:.4f}%"
            ])
            print(table)
        print("=" * 80)
        print(f"实际翻转比例: {overall_ratio:.4f}% (配置BER: {self.ber * 100:.2f}%)")
        if abs(overall_ratio - self.ber * 100) > 0.1:
            print(f"⚠️  警告：实际翻转比例与配置BER差异较大！")
        else:
            print(f"✓ 实际翻转比例与配置BER基本一致")
        print("=" * 80)
    
    def print_seed_usage_stats(self, logger=None):
        """Print seed usage statistics."""
        stats = self.get_seed_usage_stats()
        total = sum(self._seed_usage_count.values())
        
        if logger:
            logger.info("=" * 80)
            logger.info("📊 Seed Usage Statistics")
            logger.info("=" * 80)
            logger.info(f"Total forward passes with fault injection: {total}")
            logger.info("-" * 80)
            logger.info(f"{'Seed':<10} {'Count':<15} {'Percentage':<15}")
            logger.info("-" * 80)
            for seed in sorted(stats.keys()):
                count = stats[seed]['count']
                pct = stats[seed]['percentage']
                logger.info(f"{seed:<10} {count:<15} {pct:>6.2f}%")
            logger.info("=" * 80)
        else:
            print("=" * 80)
            print("📊 Seed Usage Statistics")
            print("=" * 80)
            print(f"Total forward passes with fault injection: {total}")
            print("-" * 80)
            print(f"{'Seed':<10} {'Count':<15} {'Percentage':<15}")
            print("-" * 80)
            for seed in sorted(stats.keys()):
                count = stats[seed]['count']
                pct = stats[seed]['percentage']
                print(f"{seed:<10} {count:<15} {pct:>6.2f}%")
            print("=" * 80)
    
    def reset_forward_seed(self) -> None:
        """
        Reset the current forward seed.
        
        This should be called at the beginning of each forward pass during training
        to ensure all layers in the same forward use the same base_seed.
        """
        self._current_forward_seed = None
    
    # --- Internal helpers ---
    
    def _wrap_modules(self) -> None:
        """
        Wrap quantizer forward methods to inject faults.
        
        This wraps ALL quantized layers, including:
        - Layers with dynamic bits (from search config)
        - Layers with fixed_bits (first/last layers, typically 8-bit)
        
        All these layers should receive fault injection according to their
        respective bit-width configurations.
        """
        # 如果已经包装过，直接返回（避免重复包装和日志）
        if len(self._wrapped) > 0:
            return
        
        # 调试信息：显示格雷码层配置
        if len(self.gray_code_layers) > 0:
            import sys
            print(f"[FaultInjector] Gray code layers configured: {self.gray_code_layers}", file=sys.stderr, flush=True)
        
        wrapped_count = 0
        fixed_bits_count = 0
        dynamic_bits_count = 0
        
        # Collect all quantized layers first to identify first and last
        all_quantized_layers = []
        for name, module in self.model.named_modules():
            if not isinstance(module, (QuanConv2d, QuanLinear)):
                continue
            if not self._has_quantization_enabled(module):
                continue
            all_quantized_layers.append((name, module))
        
        # Identify first and last layers
        first_layer_name = None
        last_layer_name = None
        if self.skip_first_last and len(all_quantized_layers) > 0:
            # First layer: first conv layer (usually features.0 or similar)
            for name, module in all_quantized_layers:
                if isinstance(module, QuanConv2d):
                    first_layer_name = name
                    break
            # Last layer: last linear layer (usually classifier.6 or similar)
            for name, module in reversed(all_quantized_layers):
                if isinstance(module, QuanLinear):
                    last_layer_name = name
                    break
        
        for name, module in all_quantized_layers:
            # Skip first and last layers if requested
            if self.skip_first_last:
                if name == first_layer_name or name == last_layer_name:
                    continue
            
            # If whitelist is provided, skip all other layers
            if self.whitelist_layer is not None:
                if name != self.whitelist_layer:
                    continue

            # Count layer types for debugging
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                fixed_bits_count += 1
            elif hasattr(module, 'bits') and module.bits is not None:
                dynamic_bits_count += 1
            
            # Wrap the weight quantizer
            if hasattr(module, 'quan_w_fn') and module.quan_w_fn is not None:
                key = id(module.quan_w_fn)
                if key in self._wrapped:
                    continue
                
                # Store layer name for this module (only if using position-based mask)
                if self.use_position_based_mask:
                    self._layer_name_map[key] = name
                
                # Store original forward method
                orig_quan_forward = module.quan_w_fn.forward
                
                def make_quan_wrapper(quantizer_instance, module_instance, orig_fn, layer_name_str, layer_key, use_pos_mask):
                    def wrapped_quan_forward(x, bits, is_activation=False, **kwargs):
                        # Only inject faults on weights, not activations
                        if is_activation or bits is None or bits >= 32:
                            return orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        # Determine if we should inject faults based on training mode
                        is_training = module_instance.training
                        should_inject = (
                            (is_training and self.enable_in_training) or
                            (not is_training and self.enable_in_inference)
                        )
                        
                        # Check if we're in restorer training mode (model in eval mode but fault injection enabled)
                        # This happens when training restorer: model is in eval mode, but we want random flips
                        is_restorer_training = (
                            not is_training and self.enable_in_inference and self.use_random_flip_in_training
                        )
                        
                        if not should_inject:
                            return orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        # 调试信息：对于格雷码层，添加调试输出
                        is_gray_layer = layer_name_str in self.gray_code_layers
                        if is_gray_layer:
                            import sys
                            print(f"[DEBUG] Processing gray code layer: {layer_name_str}, bits={bits}, shape={x.shape}", file=sys.stderr, flush=True)
                        
                        # Call original quantization
                        # Note: For fixed_bits layers, bits will be fixed_bits[0] (e.g., 8)
                        # For dynamic layers, bits will be from the search config (e.g., 3, 4, 5, etc.)
                        x_q = orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        if is_gray_layer:
                            import sys
                            print(f"[DEBUG] Quantization completed for {layer_name_str}, x_q shape={x_q.shape}", file=sys.stderr, flush=True)
                        
                        # Get scale (clip_value) from quantizer
                        # The bits parameter here is the actual bit-width for this layer
                        # (8 for fixed_bits layers, or the configured value for dynamic layers)
                        try:
                            scale = quantizer_instance.get_scale(bits, detach=True)
                            if scale is None:
                                return x_q
                            
                            # Select seed for this forward pass
                            # Priority logic:
                            # 1. If use_random_flip_in_training is True and we're in training mode,
                            #    use None (completely random, no seed) for maximum randomization
                            # 2. If not in training mode AND self.seed was explicitly set (e.g., from eval_with_fault_injection.py),
                            #    use it directly (for evaluation trials with different seeds).
                            #    This ensures each trial uses a different seed.
                            # 3. Otherwise, if seed_list is provided:
                            #    - Training: round-robin (轮询) through seed_list to ensure all seeds are used
                            #    - Inference: use seed from seed_list in order (for reproducibility)
                            # 4. Otherwise: use self.seed (original behavior)
                            # 
                            # Important: To ensure each layer gets a different mask even with the same base seed,
                            # we combine the base seed with a hash of the layer name.
                            # This ensures:
                            # 1. Determinism: same layer_name + same base_seed → same mask
                            # 2. Layer diversity: different layers get different masks even with same base_seed
                            
                            # For training with random flip: use None seed for complete randomization
                            # This applies both to normal training mode and restorer training mode
                            if (is_training and self.use_random_flip_in_training) or is_restorer_training:
                                selected_seed = None  # Completely random, no seed-based determinism
                            elif not is_training and self.seed is not None:
                                # Check if seed was explicitly set (not just from seed_list default)
                                # In eval_with_fault_injection.py, we pass seed=selected_seed explicitly,
                                # and seed_list is also passed. We want to use the explicit seed.
                                # Simple heuristic: if seed_list exists and seed equals seed_list[0] and
                                # _current_seed_index is 0, it might be from seed_list default.
                                # But if _current_seed_index is 0 and we're in eval, it's likely explicit.
                                # Actually, simpler: in eval mode, if seed is set, always use it directly.
                                base_seed = self.seed
                                # Use deterministic hash (hashlib.md5) instead of Python's hash() which may vary between runs
                                layer_hash = int(hashlib.md5(layer_name_str.encode()).hexdigest()[:8], 16) % (2**31)
                                selected_seed = base_seed + layer_hash
                            elif self.seed_list is not None:
                                if is_training:
                                    # Training: round-robin through seed_list to ensure all seeds are used
                                    # Each forward pass uses the next seed in the list, cycling through all seeds
                                    # Important: All layers in the same forward should use the same base_seed
                                    if self._current_forward_seed is None:
                                        # This is the first layer in this forward pass, select a new base_seed
                                        self._current_forward_seed = self.seed_list[self._current_seed_index % len(self.seed_list)]
                                        # 统计seed使用频率
                                        if self._current_forward_seed in self._seed_usage_count:
                                            self._seed_usage_count[self._current_forward_seed] += 1
                                        self._current_seed_index += 1
                                    base_seed = self._current_forward_seed
                                else:
                                    # Inference: use seed from seed_list in order
                                    base_seed = self.seed_list[self._current_seed_index % len(self.seed_list)]
                                    self._current_seed_index += 1
                                
                                # Combine base_seed with layer_name hash to ensure each layer gets different mask
                                # This ensures determinism: same layer_name + same base_seed → same mask
                                # Use deterministic hash (hashlib.md5) instead of Python's hash() which may vary between runs
                                layer_hash = int(hashlib.md5(layer_name_str.encode()).hexdigest()[:8], 16) % (2**31)
                                selected_seed = base_seed + layer_hash
                            else:
                                # If no seed_list, use self.seed but still combine with layer_name for diversity
                                if self.seed is not None:
                                    # Use deterministic hash (hashlib.md5) instead of Python's hash() which may vary between runs
                                    layer_hash = int(hashlib.md5(layer_name_str.encode()).hexdigest()[:8], 16) % (2**31)
                                    selected_seed = self.seed + layer_hash
                                else:
                                    selected_seed = None
                            
                            # Inject faults on quantized weights
                            # Fault injection respects the layer's bit-width:
                            # - Fixed_bits layers (first/last): 8-bit → flip bits in [-128, 127] range
                            # - Dynamic layers: their configured bit-width → flip bits in corresponding range
                            # Pass layer_name for:
                            # 1. Gray code layers (needed for gray code check)
                            # 2. OLM layers (needed for OLM encoding check)
                            # 3. Position-based mask (if enabled)
                            # 4. Statistics tracking (but use None for mask generation to avoid slow hash computation)
                            is_gray_layer = (self.gray_code_layers and layer_name_str in self.gray_code_layers)
                            is_olm_layer = (self.olm_layers and layer_name_str in self.olm_layers)
                            
                            if is_gray_layer or is_olm_layer:
                                # Gray code or OLM encoding needs layer_name for encoding/decoding
                                layer_name_for_mask = layer_name_str
                            elif self.use_position_based_mask:
                                layer_name_for_mask = layer_name_str  # Needed for position-based mask
                            else:
                                layer_name_for_mask = None  # Use fast random mask generation
                            
                            # Always pass layer_name for statistics (separate from mask generation)
                            layer_name_for_stats = layer_name_str
                            
                            if is_gray_layer:
                                import sys
                                print(f"[DEBUG] Calling _inject_on_quantized_tensor for {layer_name_str}...", file=sys.stderr, flush=True)
                            
                            x_faulted = self._inject_on_quantized_tensor(
                                x_q, int(bits), scale, layer_name=layer_name_for_mask, forward_seed=selected_seed, layer_name_for_stats=layer_name_for_stats
                            )
                            
                            if is_gray_layer:
                                import sys
                                print(f"[DEBUG] Fault injection completed for {layer_name_str}, preparing return...", file=sys.stderr, flush=True)
                            
                            # Optional debug: print flip ratio once per layer (跳过，避免阻塞)
                            # if self._trace_once and layer_name_str not in self._traced_layers:
                            #     try:
                            #         # Estimate flip ratio by regenerating mask with same parameters
                            #         N = (x_q.view(-1)).numel()
                            #         k_bits = int(bits)
                            #         mask = self._generate_flip_mask(N, k_bits, device=(x_q.device if self.device is None else self.device), layer_name=layer_name_arg, mask_seed=selected_seed)
                            #         flip_ratio = float(mask.float().mean().item())
                            #         print(f"[FaultInjector TRACE] layer={layer_name_str}, bits={k_bits}, ber={self.ber:.2e}, flip_ratio={flip_ratio:.4f}")
                            #     except Exception:
                            #         pass
                            #     self._traced_layers.add(layer_name_str)
                            
                            # Preserve gradients: forward uses faulted value, backward uses original
                            if is_gray_layer:
                                import sys
                                print(f"[DEBUG] Computing gradient-preserving return for {layer_name_str}...", file=sys.stderr, flush=True)
                                print(f"[DEBUG] x_faulted: shape={x_faulted.shape}, device={x_faulted.device}, dtype={x_faulted.dtype}, requires_grad={x_faulted.requires_grad}", file=sys.stderr, flush=True)
                                print(f"[DEBUG] x_q: shape={x_q.shape}, device={x_q.device}, dtype={x_q.dtype}, requires_grad={x_q.requires_grad}", file=sys.stderr, flush=True)
                            
                            # 简化梯度保留计算，避免可能的阻塞
                            # 确保所有张量在同一个设备上
                            if x_faulted.device != x_q.device:
                                if is_gray_layer:
                                    import sys
                                    print(f"[DEBUG] Device mismatch! Moving x_faulted from {x_faulted.device} to {x_q.device}", file=sys.stderr, flush=True)
                                x_faulted = x_faulted.to(x_q.device)
                            
                            # 在 eval 模式下，直接返回故障值，不需要梯度保留
                            # 这样可以避免不必要的计算图构建，提升性能
                            if not is_training:
                                result = x_faulted
                            else:
                                # 训练模式下，保留梯度：forward 使用故障值，backward 使用原始值
                                x_faulted_detached = x_faulted.detach()
                                x_q_detached = x_q.detach()
                                diff = x_q - x_q_detached
                                result = x_faulted_detached + diff
                            
                            if is_gray_layer:
                                import sys
                                print(f"[DEBUG] Return value computed: shape={result.shape}, device={result.device}, dtype={result.dtype}, requires_grad={result.requires_grad}", file=sys.stderr, flush=True)
                                print(f"[DEBUG] Returning from wrapped_quan_forward for {layer_name_str}...", file=sys.stderr, flush=True)
                            
                            return result
                        except Exception as e:
                            # On any failure, gracefully fall back
                            # 添加调试信息（仅在启用统计时打印，避免性能影响）
                            if self.enable_statistics and layer_name_str in self.gray_code_layers:
                                import sys
                                print(f"[FaultInjector ERROR] Layer {layer_name_str} failed: {e}", file=sys.stderr, flush=True)
                            return x_q
                    
                    return wrapped_quan_forward
                
                # Replace quantizer forward
                module.quan_w_fn.forward = make_quan_wrapper(
                    module.quan_w_fn, module, orig_quan_forward, name, key, self.use_position_based_mask
                )
                self._wrapped[key] = orig_quan_forward
                wrapped_count += 1
        
        # Log wrapped layers info (only once, never repeat)
        if wrapped_count > 0 and not self._wrap_logged:
            print(f"[FaultInjector] Wrapped {wrapped_count} layers for fault injection: "
                  f"{dynamic_bits_count} dynamic bits layers, {fixed_bits_count} fixed_bits layers "
                  f"(first/last layers will use their fixed bit-width, e.g., 8-bit)")
            self._wrap_logged = True
    
    def _restore_modules(self) -> None:
        """Restore original quantizer forward methods."""
        for module in self.model.modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if hasattr(module, 'quan_w_fn') and module.quan_w_fn is not None:
                    key = id(module.quan_w_fn)
                    if key in self._wrapped:
                        module.quan_w_fn.forward = self._wrapped[key]
        self._wrapped.clear()
    
    def _has_quantization_enabled(self, module: nn.Module) -> bool:
        """
        Check if module has quantization enabled.
        
        This includes layers with:
        - bits set (dynamic bit-width layers from search)
        - fixed_bits set (first/last layers with fixed 8-bit quantization)
        
        Both types of layers should be included in fault injection.
        """
        if not hasattr(module, 'quan_w_fn'):
            return False
        # Check if bits or fixed_bits is set
        # Both should be included in fault injection
        if hasattr(module, 'bits') and module.bits is not None:
            return True
        if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
            return True
        return False
    
    def _get_weight_bits(self, module: nn.Module) -> Optional[int]:
        """
        Get weight bit-width for a layer, considering bits and fixed_bits.
        
        Note: This method is kept for compatibility, but bit-width is typically
        obtained directly from the quantizer call in wrapped_quan_forward.
        """
        # Try to get from bits attribute
        if hasattr(module, 'bits') and module.bits is not None:
            wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
            if isinstance(wbits, torch.Tensor):
                wbits = int(wbits.item())
            else:
                wbits = int(wbits)
            return wbits
        
        # Try to get from fixed_bits attribute
        if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
            wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
            if isinstance(wbits, torch.Tensor):
                wbits = int(wbits.item())
            else:
                wbits = int(wbits)
            return wbits
        
        return None
    
    
    @staticmethod
    def _binary_to_gray(binary: torch.Tensor) -> torch.Tensor:
        """
        将二进制编码转换为格雷码（Gray Code）- JIT编译加速版本
        
        格雷码特点：相邻两个码字只有一位不同，单bit翻转只会跳到相邻值
        
        转换公式：G = B ^ (B >> 1)
        
        Args:
            binary: 二进制编码的整数张量
            
        Returns:
            格雷码编码的整数张量
        """
        # 向量化操作，GPU加速
        return binary ^ (binary >> 1)
    
    @staticmethod
    def _gray_to_binary(gray: torch.Tensor, k: int) -> torch.Tensor:
        """
        将格雷码转换回二进制编码（向量化优化版本）
        
        转换方法：向量化逐位异或
        B = G ^ (G >> 1) ^ (G >> 2) ^ ... ^ (G >> (k-1))
        
        Args:
            gray: 格雷码编码的整数张量
            k: 位宽
            
        Returns:
            二进制编码的整数张量
        """
        # 对于小位宽（k <= 8），使用循环是高效的
        # 使用原地操作和向量化优化性能
        binary = gray.clone()
        # 向量化转换：B = G ^ (G >> 1) ^ (G >> 2) ^ ... ^ (G >> (k-1))
        # 限制最大循环次数，通常 k <= 8
        max_shift = min(k, 16)
        for i in range(1, max_shift):
            binary ^= (gray >> i)
        return binary
    
    def _inject_on_quantized_tensor(
        self, x_q: torch.Tensor, k: int, scale: torch.Tensor, layer_name: Optional[str] = None, forward_seed: Optional[int] = None, layer_name_for_stats: Optional[str] = None
    ) -> torch.Tensor:
        """
        Inject bit-flip faults on a quantized tensor using LSQ quantization format.
        
        LSQ quantization: x_q = round(x / s) * s, where x is clamped to [thd_neg * s, thd_pos * s]
        Integer code: code = round(x_q / s), which ranges from [thd_neg, thd_pos]
        
        Args:
            x_q: Quantized tensor (float values after LSQ quantization)
            k: Bit-width for quantization
            scale: Scale parameter (s) from LSQ quantizer
            layer_name: Optional layer name for statistics tracking
            forward_seed: Optional seed for this forward pass
            
        Returns:
            Faulted tensor with same shape as x_q
        """
        device = x_q.device if self.device is None else self.device
        
        # 判断是否使用格雷码或OLM编码（优化：先检查长度）
        use_gray_code = (len(self.gray_code_layers) > 0 and 
                        layer_name is not None and 
                        layer_name in self.gray_code_layers)
        use_olm = (len(self.olm_layers) > 0 and 
                  layer_name is not None and 
                  layer_name in self.olm_layers)
        
        # 格雷码和OLM不能同时使用
        if use_gray_code and use_olm:
            raise ValueError(f"Layer {layer_name} cannot use both Gray Code and OLM encoding")
        
        # Handle scale as tensor or scalar
        if isinstance(scale, torch.Tensor):
            s = scale.to(device)
            # If per-channel, ensure proper shape for broadcasting
            if s.dim() > 0 and s.numel() > 1:
                while s.dim() < x_q.dim():
                    s = s.unsqueeze(-1)
        else:
            s = torch.tensor(float(scale), device=device, dtype=x_q.dtype)
        
        # Compute quantization thresholds based on bit-width
        # For weights: typically symmetric quantization
        # Signed k-bit: [-2^(k-1), 2^(k-1)-1]
        thd_neg = -(1 << (k - 1))  # -2^(k-1)
        thd_pos = (1 << (k - 1)) - 1  # 2^(k-1)-1
        
        # ============================================================
        # 正确的流程（符合用户要求）：
        # 1. 浮点权重 → 量化器 → 整数码（通过 round(x_q / s)）
        # 2. 整数码 → 格雷编码（如果使用格雷码）
        # 3. 格雷编码空间 → 注入故障（位翻转）
        # 4. 格雷编码 → 映射回整数（如果使用格雷码）
        # 5. 整数 → 反量化 → 浮点
        # ============================================================
        
        # Step 1: 从量化后的浮点数反推整数码
        # LSQ量化: x_q = round(x / s) * s
        # 整数码: code = round(x_q / s)，范围 [thd_neg, thd_pos]
        code_f = torch.round(x_q.to(device) / s)
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        
        # Shift to non-negative range [0, 2^k-1] for bit operations
        code_shifted = code_f - thd_neg  # Now in [0, 2^k-1]
        n_levels = (1 << k) - 1
        
        # Use compact integer dtype for efficiency
        code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
        code = code_shifted.to(code_dtype).clamp(0, n_levels)
        
        # Step 2: 如果使用格雷码或OLM，将整数码转换为编码空间
        if use_gray_code:
            import sys
            print(f"[DEBUG _inject] Step 2: Converting to gray code, code shape={code.shape}, device={code.device}, target device={device}", file=sys.stderr, flush=True)
            # 确保在正确的设备上操作
            if code.device != device:
                code = code.to(device)
            # 向量化：G = B ^ (B >> 1)
            code = code ^ (code >> 1)
            print(f"[DEBUG _inject] Step 2 completed: gray code shape={code.shape}, device={code.device}", file=sys.stderr, flush=True)
        elif use_olm:
            # OLM编码：将量化值映射到编码空间
            # 需要先将code_shifted（0到n_levels）转换回原始量化值范围（thd_neg到thd_pos）
            code_original = code_shifted + thd_neg  # 转换回原始量化值范围
            value_to_code = self.olm_layers[layer_name]
            # 使用向量化查找表进行映射（优化：使用GPU tensor作为查找表）
            # 创建查找表：将量化值范围映射到编码
            # 注意：对于未映射的值，使用原值（identity mapping）
            lookup_table = torch.arange(n_levels + 1, dtype=code_dtype, device=device)  # 默认identity映射
            for val, enc in value_to_code.items():
                idx = val - thd_neg  # 转换到[0, n_levels]范围
                if 0 <= idx <= n_levels:
                    lookup_table[idx] = enc
            # 向量化查找
            code = lookup_table[code_shifted.clamp(0, n_levels)]
        
        # Flatten for bit operations (GPU-accelerated)
        flat = code.view(-1)
        N = flat.numel()
        
        # Generate flip mask [N, k] using GPU
        # If layer_name is provided, generate fixed mask based on weight position
        # If forward_seed is provided (from seed_list), use it instead of self.seed
        mask_seed = forward_seed if forward_seed is not None else self.seed
        flip_mask = self._generate_flip_mask(N, k, device, layer_name=layer_name, mask_seed=mask_seed)
        
        # 统计实际翻转的bit数（延迟统计，避免GPU-CPU同步阻塞）
        # 默认关闭统计功能以提升性能，需要时可以通过enable_statistics=True启用
        if self.enable_statistics:
            total_bits = N * k
            total_params = N  # 参数总数（每个参数有k个bit）
            # Use layer_name_for_stats if provided, otherwise fall back to layer_name or "unknown"
            stats_key = layer_name_for_stats if layer_name_for_stats is not None else (layer_name if layer_name is not None else "unknown")
            # 延迟统计：只存储sum结果（tensor，不立即同步到CPU），稍后批量处理
            # 这样可以避免每次调用都触发GPU-CPU同步，提升性能
            flip_mask_sum = flip_mask.sum()  # 返回tensor，不调用.item()
            # 计算受影响的参数数量：至少有一个bit被翻转的参数
            # flip_mask形状为[N, k]，对每行求和，如果>0说明该参数至少有一个bit被翻转
            affected_params_sum = (flip_mask.sum(dim=1) > 0).sum()  # 返回tensor，不调用.item()
            self._pending_stats.append((flip_mask_sum, total_bits, total_params, stats_key, affected_params_sum))
        
        # Step 3: 在编码空间（二进制或格雷码）中进行位翻转（向量化优化）
        # 预计算位权重，避免重复计算，提升性能
        bit_positions = torch.arange(k, device=device, dtype=torch.int64)
        bit_weights = (1 << bit_positions).to(torch.int64)  # 预计算位权重，避免重复位移
        
        # 向量化位提取和翻转（减少类型转换，确保设备一致）
        flat_int64 = flat.to(torch.int64)
        if flat_int64.device != device:
            flat_int64 = flat_int64.to(device)
        bits = ((flat_int64.unsqueeze(-1) >> bit_positions) & 1).to(torch.bool)
        flipped_bits = bits ^ flip_mask
        
        # 向量化重建编码值（使用预计算的权重，避免重复位移操作）
        flat_faulted = (flipped_bits.to(torch.int64) * bit_weights).sum(-1)
        flat_faulted = flat_faulted.clamp(0, n_levels).to(code_dtype)
        # 确保在正确的设备上
        if flat_faulted.device != device:
            flat_faulted = flat_faulted.to(device)
        
        # Step 4: 如果使用格雷码或OLM，将编码转换回二进制整数码
        if use_gray_code:
            import sys
            print(f"[DEBUG _inject] Step 4: Converting gray to binary, k={k}, flat_faulted shape={flat_faulted.shape}, device={flat_faulted.device}, target device={device}", file=sys.stderr, flush=True)
            # 确保在正确的设备上操作
            if flat_faulted.device != device:
                flat_faulted = flat_faulted.to(device)
            # 向量化转换：B = G ^ (G >> 1) ^ (G >> 2) ^ ... ^ (G >> (k-1))
            # 需要保留原始值，但使用更高效的方式：累积异或，避免多次内存分配
            gray_orig = flat_faulted  # 保存原始格雷码值
            # 对于常见的小位宽（2-8 bit），直接展开循环
            binary = gray_orig
            if k >= 2:
                binary = binary ^ (gray_orig >> 1)
            if k >= 3:
                binary = binary ^ (gray_orig >> 2)
            if k >= 4:
                binary = binary ^ (gray_orig >> 3)
            if k >= 5:
                binary = binary ^ (gray_orig >> 4)
            if k >= 6:
                binary = binary ^ (gray_orig >> 5)
            if k >= 7:
                binary = binary ^ (gray_orig >> 6)
            if k >= 8:
                binary = binary ^ (gray_orig >> 7)
            # 对于更大的位宽（很少见），使用循环
            if k > 8:
                for i in range(8, min(k, 16)):
                    binary = binary ^ (gray_orig >> i)
            flat_faulted = binary
            print(f"[DEBUG _inject] Step 4 completed: binary shape={flat_faulted.shape}, device={flat_faulted.device}", file=sys.stderr, flush=True)
        elif use_olm:
            # OLM解码：将编码映射回量化值
            code_to_value = self.olm_code_to_value[layer_name]
            # 使用向量化查找表进行映射（优化：使用GPU tensor作为查找表）
            # 创建反向查找表：将编码映射回量化值
            max_code = max(code_to_value.keys()) if code_to_value else n_levels
            # 注意：对于未映射的编码，使用原值（identity mapping）
            reverse_lookup = torch.arange(max_code + 1, dtype=code_dtype, device=device)  # 默认identity映射
            for enc, val in code_to_value.items():
                if 0 <= enc <= max_code:
                    reverse_lookup[enc] = val - thd_neg  # 转换到[0, n_levels]范围
            # 向量化查找
            flat_faulted = reverse_lookup[flat_faulted.clamp(0, max_code)].clamp(0, n_levels)
        
        # Reshape back to original shape
        code_faulted = flat_faulted.view_as(code)
        
        # Step 5: 将整数码转换回量化范围并反量化回浮点数
        # Shift back to original range [thd_neg, thd_pos]
        code_faulted_shifted = code_faulted.to(x_q.dtype) + thd_neg
        
        # De-quantize back to float: x_faulted = code_faulted * s
        x_faulted = code_faulted_shifted * s
        
        # Numerical safety: clamp to reasonable range
        if torch.is_floating_point(x_faulted):
            max_range = torch.abs(s) * (thd_pos + 1)
            x_faulted = torch.clamp(x_faulted, -max_range, max_range)
        
        return x_faulted
    
    def _generate_flip_mask(self, N: int, k: int, device: torch.device, layer_name: Optional[str] = None, mask_seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate a boolean tensor of shape [N, k] indicating which bits to flip.
        Uses GPU-accelerated random generation.
        
        If layer_name is provided, generates fixed mask based on weight position,
        ensuring the same weight position always gets the same mask (for reproducibility
        between training and validation).
        
        Args:
            N: Number of elements
            k: Bit-width
            device: Device for tensor generation
            layer_name: Optional layer name for generating position-based fixed mask
            mask_seed: Optional seed for generating mask (if None, use self.seed or random)
            
        Returns:
            Boolean tensor [N, k] where True indicates bit should be flipped
        """
        if self.mode == "ber":
            p = float(self.ber or 0.0)
            
            # Use mask_seed if provided, otherwise use self.seed
            seed_to_use = mask_seed if mask_seed is not None else self.seed
            
            # Only use position-based mask if explicitly enabled AND layer_name is provided
            # Otherwise, use fast random mask generation (even if layer_name is provided for statistics)
            if layer_name is not None and self.use_position_based_mask:
                # Generate fixed mask based on weight position
                # This ensures the same weight position always gets the same mask
                # across different forward passes (training vs validation)
                import hashlib
                
                # Use hash function to generate fixed mask (GPU-friendly, deterministic)
                # For each weight position i and bit j, compute hash and map to [0, 1]
                # This is more efficient than using torch.Generator in a loop
                mask = torch.zeros((N, k), dtype=torch.bool, device=device)
                
                # Vectorized approach: generate all position indices
                i_indices = torch.arange(N, device=device, dtype=torch.int64)
                j_indices = torch.arange(k, device=device, dtype=torch.int64)
                
                # Create meshgrid for all (i, j) combinations
                i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing='ij')
                i_flat = i_grid.flatten()
                j_flat = j_grid.flatten()
                
                # Generate hash-based random values for all positions at once
                # Convert to CPU for hash computation (hashlib doesn't support GPU)
                i_cpu = i_flat.cpu().numpy()
                j_cpu = j_flat.cpu().numpy()
                
                # Compute hash for each position and map to [0, 1]
                hash_values = []
                for idx in range(len(i_cpu)):
                    i_val, j_val = int(i_cpu[idx]), int(j_cpu[idx])
                    # Create unique identifier for this weight position and bit
                    # Use seed_to_use instead of self.seed
                    identifier = f"{layer_name}_{i_val}_{j_val}_{seed_to_use}"
                    # Compute hash
                    hash_int = int(hashlib.md5(identifier.encode()).hexdigest()[:8], 16)
                    # Map to [0, 1] range
                    hash_val = (hash_int % 1000000) / 1000000.0
                    hash_values.append(hash_val)
                
                # Convert to tensor and reshape
                hash_tensor = torch.tensor(hash_values, device=device, dtype=torch.float32)
                mask_flat = hash_tensor < p
                mask = mask_flat.reshape(N, k)
                
                return mask
            else:
                # Generate random mask using seed_to_use
                if seed_to_use is not None:
                    # Use generator with specific seed for reproducibility
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed_to_use)
                    return torch.rand((N, k), generator=generator, device=device) < p
                else:
                    # Original behavior: generate random mask each time (no seed)
                    # GPU-accelerated: generate all random values at once
                    return torch.rand((N, k), device=device) < p
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def load_bit_width_config_from_json(json_path: str, config_index: int = 0) -> Tuple[List[int], List[int]]:
    """
    Load bit-width configuration from JSON file generated by search.
    
    Args:
        json_path: Path to JSON configuration file (e.g., from search output)
        config_index: Index of configuration to use (default: 0, use first configuration)
        
    Returns:
        Tuple of (weight_bits, act_bits) lists for each quantized layer
        
    Example:
        >>> weight_bits, act_bits = load_bit_width_config_from_json(
        ...     "search/resnet18_cifar10_single_gpu_search_bit_width_config.json"
        ... )
        >>> set_bit_width(model, weight_bits, act_bits)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Bit-width config file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        config_data = json.load(f)
    
    if 'configurations' not in config_data:
        raise ValueError(f"Invalid config file format: 'configurations' key not found")
    
    if len(config_data['configurations']) == 0:
        raise ValueError(f"No configurations found in config file")
    
    if config_index >= len(config_data['configurations']):
        raise ValueError(
            f"Config index {config_index} out of range. "
            f"File contains {len(config_data['configurations'])} configurations."
        )
    
    config = config_data['configurations'][config_index]
    
    weight_bits = config['weight_bits']
    act_bits = config['act_bits']
    
    # Convert to Python int if needed
    weight_bits = [int(b) for b in weight_bits]
    act_bits = [int(b) for b in act_bits]
    
    if len(weight_bits) != len(act_bits):
        raise ValueError(
            f"Mismatch in bit-width lists: "
            f"weight_bits has {len(weight_bits)} elements, "
            f"act_bits has {len(act_bits)} elements"
        )
    
    return weight_bits, act_bits


def setup_model_with_bit_width_config(
    model: torch.nn.Module,
    json_path: str,
    config_index: int = 0,
    verbose: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Load bit-width configuration from JSON and set it on the model.
    
    This is a convenience function that combines loading and setting bit-widths.
    It should be called before enabling fault injection for mixed-precision models.
    
    Args:
        model: The quantized model
        json_path: Path to JSON configuration file from search
        config_index: Index of configuration to use (default: 0)
        verbose: Whether to print information about loaded configuration
        
    Returns:
        Tuple of (weight_bits, act_bits) that were set on the model
        
    Example:
        >>> setup_model_with_bit_width_config(
        ...     model,
        ...     "search/resnet18_cifar10_single_gpu_search_bit_width_config.json"
        ... )
        >>> injector = FaultInjector(model, mode="ber", ber=1e-6)
        >>> injector.enable()
    """
    weight_bits, act_bits = load_bit_width_config_from_json(json_path, config_index)
    
    # Get quantized layers, excluding those with fixed_bits (first/last layers)
    # The config file only contains bit-widths for layers without fixed_bits
    quantized_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            # Skip layers with fixed_bits (typically first conv and last linear)
            # These layers are not included in the search config
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                continue
            quantized_layers.append(module)
    
    # Handle layer count mismatch more gracefully
    if len(quantized_layers) != len(weight_bits):
        if len(weight_bits) > len(quantized_layers):
            # Config has more layers than model - use only the first N layers from config
            if verbose:
                print(f"  Warning: Config specifies {len(weight_bits)} layers, but model has {len(quantized_layers)} layers.")
                print(f"  Using first {len(quantized_layers)} layers from config.")
            weight_bits = weight_bits[:len(quantized_layers)]
            act_bits = act_bits[:len(quantized_layers)]
        else:
            # Config has fewer layers than model - only set first M layers, warn about the rest
            if verbose:
                print(f"  Warning: Model has {len(quantized_layers)} layers, but config specifies {len(weight_bits)} layers.")
                print(f"  Only setting bit-widths for first {len(weight_bits)} layers. Remaining layers will keep their current bit-widths.")
            # We'll only set the first len(weight_bits) layers
    
    # Set bit-widths on the model (only for layers without fixed_bits)
    # We need to also get BN layers for switching
    from .qat import get_quantized_layers
    
    # First, set bits on all layers so get_quantized_layers can find them
    # We'll set them directly first, then use set_bit_width to also update BN layers
    config_idx = 0
    layers_set = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            # Skip layers with fixed_bits
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                continue
            # Set bits for this layer (only if we have config for it)
            if config_idx < len(weight_bits):
                module.bits = (weight_bits[config_idx], act_bits[config_idx])
                layers_set += 1
                config_idx += 1
    
    # Now get quantized layers and BN layers to update BN
    try:
        layers, bns = get_quantized_layers(model)
        # Update BN layers for the layers we actually set
        for idx in range(min(layers_set, len(layers))):
            if idx < len(bns) and bns[idx] is not None:
                if hasattr(bns[idx], 'switch_bn'):
                    bns[idx].switch_bn(layers[idx].bits)
    except Exception as e:
        # If get_quantized_layers fails (e.g., output_size not set), continue anyway
        # The bits are already set on the layers
        if verbose:
            print(f"  Warning: Could not update BN layers: {e}")
    
    # Return the actual bits that were set (may be truncated if config had more layers)
    actual_weight_bits = weight_bits[:layers_set] if layers_set < len(weight_bits) else weight_bits
    actual_act_bits = act_bits[:layers_set] if layers_set < len(act_bits) else act_bits
    
    if verbose:
        print(f"Loaded bit-width configuration from: {json_path}")
        print(f"  Configuration index: {config_index}")
        print(f"  Set bit-widths on {layers_set} out of {len(quantized_layers)} layers")
        if layers_set < len(quantized_layers):
            print(f"  Note: {len(quantized_layers) - layers_set} layers were not configured (keeping current bit-widths)")
        if actual_weight_bits:
            print(f"  Weight bits range: {min(actual_weight_bits)}-{max(actual_weight_bits)}")
            print(f"  Activation bits range: {min(actual_act_bits)}-{max(actual_act_bits)}")
            print(f"  Sample weight bits (first 5 layers): {actual_weight_bits[:5]}")
            print(f"  Sample act bits (first 5 layers): {actual_act_bits[:5]}")
    
    return actual_weight_bits, actual_act_bits

