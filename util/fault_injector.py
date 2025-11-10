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
from quan.func import QuanConv2d, QuanLinear
from .qat import set_bit_width, get_quantized_layers


Mode = Literal["ber"]


class FaultInjector:
    """
    Fault injector for quantized weights in retraining-free-quantization models.
    
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
        # Reset seed index for next enable
        self._current_seed_index = 0
        self._current_forward_seed = None
    
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
        
        wrapped_count = 0
        fixed_bits_count = 0
        dynamic_bits_count = 0
        
        for name, module in self.model.named_modules():
            # Only wrap QuanConv2d and QuanLinear layers
            if not isinstance(module, (QuanConv2d, QuanLinear)):
                continue
            
            # Check if layer has quantization enabled
            # This includes both bits (dynamic) and fixed_bits (first/last layers)
            if not self._has_quantization_enabled(module):
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
                        
                        if not should_inject:
                            return orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        # Call original quantization
                        # Note: For fixed_bits layers, bits will be fixed_bits[0] (e.g., 8)
                        # For dynamic layers, bits will be from the search config (e.g., 3, 4, 5, etc.)
                        x_q = orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        # Get scale (clip_value) from quantizer
                        # The bits parameter here is the actual bit-width for this layer
                        # (8 for fixed_bits layers, or the configured value for dynamic layers)
                        try:
                            scale = quantizer_instance.get_scale(bits, detach=True)
                            if scale is None:
                                return x_q
                            
                            # Select seed for this forward pass
                            # Priority logic:
                            # 1. If not in training mode AND self.seed was explicitly set (e.g., from eval_with_fault_injection.py),
                            #    use it directly (for evaluation trials with different seeds).
                            #    This ensures each trial uses a different seed.
                            # 2. Otherwise, if seed_list is provided:
                            #    - Training: round-robin (轮询) through seed_list to ensure all seeds are used
                            #    - Inference: use seed from seed_list in order (for reproducibility)
                            # 3. Otherwise: use self.seed (original behavior)
                            # 
                            # Important: To ensure each layer gets a different mask even with the same base seed,
                            # we combine the base seed with a hash of the layer name.
                            # This ensures:
                            # 1. Determinism: same layer_name + same base_seed → same mask
                            # 2. Layer diversity: different layers get different masks even with same base_seed
                            
                            # For evaluation: if seed was explicitly set and we're not in training mode,
                            # use it directly (this handles eval_with_fault_injection.py case where each trial
                            # creates a new FaultInjector with a different seed)
                            if not is_training and self.seed is not None:
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
                            # Pass layer_name only if using position-based mask (for code isolation)
                            layer_name_arg = layer_name_str if use_pos_mask else None
                            x_faulted = self._inject_on_quantized_tensor(
                                x_q, int(bits), scale, layer_name=layer_name_arg, forward_seed=selected_seed
                            )
                            # Preserve gradients: forward uses faulted value, backward uses original
                            return x_faulted.detach() + (x_q - x_q.detach())
                        except Exception:
                            # On any failure, gracefully fall back
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
    
    
    def _inject_on_quantized_tensor(
        self, x_q: torch.Tensor, k: int, scale: torch.Tensor, layer_name: Optional[str] = None, forward_seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Inject bit-flip faults on a quantized tensor using LSQ quantization format.
        
        LSQ quantization: x_q = round(x / s) * s, where x is clamped to [thd_neg * s, thd_pos * s]
        Integer code: code = round(x_q / s), which ranges from [thd_neg, thd_pos]
        
        Args:
            x_q: Quantized tensor (float values after LSQ quantization)
            k: Bit-width for quantization
            scale: Scale parameter (s) from LSQ quantizer
            
        Returns:
            Faulted tensor with same shape as x_q
        """
        device = x_q.device if self.device is None else self.device
        
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
        
        # Map quantized float back to integer code: code = round(x_q / s)
        # This should be in [thd_neg, thd_pos] range
        code_f = torch.round(x_q.to(device) / s)
        code_f = torch.clamp(code_f, thd_neg, thd_pos)
        
        # Shift to non-negative range [0, 2^k-1] for bit operations
        code_shifted = code_f - thd_neg  # Now in [0, 2^k-1]
        n_levels = (1 << k) - 1
        
        # Use compact integer dtype for efficiency
        code_dtype = torch.int16 if n_levels <= 32767 else torch.int32
        code = code_shifted.to(code_dtype).clamp(0, n_levels)
        
        # Flatten for bit operations (GPU-accelerated)
        flat = code.view(-1)
        N = flat.numel()
        
        # Generate flip mask [N, k] using GPU
        # If layer_name is provided, generate fixed mask based on weight position
        # If forward_seed is provided (from seed_list), use it instead of self.seed
        mask_seed = forward_seed if forward_seed is not None else self.seed
        flip_mask = self._generate_flip_mask(N, k, device, layer_name=layer_name, mask_seed=mask_seed)
        
        # Expand to bit planes and apply XOR (vectorized on GPU)
        bit_positions = torch.arange(k, device=device, dtype=torch.int64)
        bits = ((flat.unsqueeze(-1).to(torch.int64) >> bit_positions) & 1).to(torch.bool)
        flipped_bits = bits ^ flip_mask
        flat_faulted = (flipped_bits.to(torch.int64) * (1 << bit_positions)).sum(-1)
        flat_faulted = flat_faulted.clamp(0, n_levels).to(code_dtype)
        
        # Reshape back to original shape
        code_faulted = flat_faulted.view_as(code)
        
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
            
            if layer_name is not None:
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
    
    if len(quantized_layers) != len(weight_bits):
        raise ValueError(
            f"Model has {len(quantized_layers)} quantized layers (excluding fixed_bits layers), "
            f"but config specifies {len(weight_bits)} layers. "
            f"This mismatch suggests the config file is for a different model architecture. "
            f"Note: Layers with fixed_bits (first/last layers) are excluded from the config."
        )
    
    # Set bit-widths on the model (only for layers without fixed_bits)
    # We need to also get BN layers for switching
    from .qat import get_quantized_layers
    
    # First, set bits on all layers so get_quantized_layers can find them
    # We'll set them directly first, then use set_bit_width to also update BN layers
    config_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            # Skip layers with fixed_bits
            if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                continue
            # Set bits for this layer
            if config_idx < len(weight_bits):
                module.bits = (weight_bits[config_idx], act_bits[config_idx])
                config_idx += 1
    
    # Now get quantized layers and BN layers to update BN
    try:
        layers, bns = get_quantized_layers(model)
        if len(layers) == len(weight_bits):
            # Update BN layers
            for idx, (layer, bn) in enumerate(zip(layers, bns)):
                if hasattr(bn, 'switch_bn'):
                    bn.switch_bn(layer.bits)
    except Exception as e:
        # If get_quantized_layers fails (e.g., output_size not set), continue anyway
        # The bits are already set on the layers
        if verbose:
            print(f"  Warning: Could not update BN layers: {e}")
    
    if verbose:
        print(f"Loaded bit-width configuration from: {json_path}")
        print(f"  Configuration index: {config_index}")
        print(f"  Number of layers: {len(weight_bits)}")
        print(f"  Weight bits range: {min(weight_bits)}-{max(weight_bits)}")
        print(f"  Activation bits range: {min(act_bits)}-{max(act_bits)}")
        print(f"  Sample weight bits (first 5 layers): {weight_bits[:5]}")
        print(f"  Sample act bits (first 5 layers): {act_bits[:5]}")
    
    return weight_bits, act_bits

