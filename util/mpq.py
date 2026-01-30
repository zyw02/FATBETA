"""
Highly Optimized Mixed-Precision Quantization Utilities

Key optimizations for maximum training speed:
- Pre-cached layer references (built once per model)
- Direct attribute setting without method calls
- Fast max/min sampling with single-pass layer updates
- Eliminated all CPU-GPU synchronization
- Minimized Python function call overhead
"""

from quan.func import SwithableBatchNorm, QuanConv2d, QuanLinear
from timm.utils import unwrap_model
import torch
import random

# Global layer cache - populated once per model
_layer_cache = {}


def _get_model_id(model):
    """Get unique identifier for model to support multiple models"""
    return id(unwrap_model(model))


def get_cached_layers(model, configs):
    """
    Get cached quantized layer references.
    Caches: layer refs, bn refs, conv-to-bn mapping, and bit info.
    Only called once per model - all subsequent calls use cache.
    """
    model_id = _get_model_id(model)
    
    if model_id not in _layer_cache:
        quan_layers = []  # List of (module, name, layer_type) tuples
        bn_layers = []    # List of SwithableBatchNorm modules
        conv_to_bn = {}   # Map conv layer index to corresponding BN index
        
        quan_scheduler = configs.quan
        next_bn_for_conv_idx = -1
        
        for name, module in unwrap_model(model).named_modules():
            if isinstance(module, QuanConv2d):
                if name not in quan_scheduler.excepts:
                    next_bn_for_conv_idx = len(quan_layers)
                    quan_layers.append((module, name, 'conv'))
                    
            elif isinstance(module, QuanLinear):
                if name not in quan_scheduler.excepts:
                    quan_layers.append((module, name, 'linear'))
                    
            elif isinstance(module, SwithableBatchNorm):
                if name not in quan_scheduler.excepts and next_bn_for_conv_idx >= 0:
                    bn_layers.append(module)
                    conv_to_bn[next_bn_for_conv_idx] = len(bn_layers) - 1
                    next_bn_for_conv_idx = -1
        
        # Pre-compute target bits as sorted tuple for fast access
        target_bits = tuple(sorted(configs.target_bits, reverse=True))
        max_bit = target_bits[0]
        min_bit = target_bits[-1]
        
        # Pre-compute weights for safe layers (lower bits get higher probability)
        # Weights: (12-b) -> 8:4, 7:5, 6:6, 5:7, 4:8, 3:9, 2:10 (Total 49)
        # Prob (8-bit) = 4/49 ≈ 8.2%
        safe_bits = [b for b in target_bits if b <= 8]
        safe_weights = [(12-b) for b in safe_bits]
        
        # Sensitive bits: [8, 7, 6, 5]
        sensitive_bits = [b for b in target_bits if b >= 5]
        
        _layer_cache[model_id] = {
            'quan_layers': quan_layers,
            'bn_layers': bn_layers,
            'conv_to_bn': conv_to_bn,
            'target_bits': target_bits,
            'max_bit': max_bit,
            'min_bit': min_bit,
            'safe_bits': safe_bits,
            'safe_weights': safe_weights,
            'sensitive_bits': sensitive_bits,
        }
    
    return _layer_cache[model_id]


def invalidate_cache(model=None):
    """Invalidate layer cache (call if model structure changes)"""
    global _layer_cache
    if model is None:
        _layer_cache = {}
    else:
        model_id = _get_model_id(model)
        if model_id in _layer_cache:
            del _layer_cache[model_id]


def switch_bit_width_bn(model, wbit, abits):
    """Switch BN layers to specific bit-width"""
    bit_pair = (wbit, abits)
    for name, module in model.named_modules():
        if isinstance(module, SwithableBatchNorm):
            module.switch_bn(bit_pair)


def switch_bit_width(model, quan_scheduler, wbit, abits):
    """Switch bit-width for all quantized layers"""
    bit_pair = (wbit, abits)
    for name, module in unwrap_model(model).named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if name not in quan_scheduler.excepts:
                if hasattr(module, 'bits'):
                    module.bits = bit_pair
    switch_bit_width_bn(model, wbit, abits)


def sample_max_cands(model, configs):
    """
    FAST PATH: Sample maximum bit-width configuration.
    Direct attribute setting - no method calls, no sampling logic.
    """
    cache = get_cached_layers(model, configs)
    max_bit = cache['max_bit']
    bit_pair = (max_bit, max_bit)
    
    # Direct iteration over cached layer list
    quan_layers = cache['quan_layers']
    bn_layers = cache['bn_layers']
    conv_to_bn = cache['conv_to_bn']
    
    weights = []
    act = []
    
    for layer_idx, (module, name, layer_type) in enumerate(quan_layers):
        # Direct attribute assignment - no method call overhead
        module.bits = bit_pair
        weights.append(max_bit)
        act.append(max_bit)
        
        # Update corresponding BN if exists
        if layer_idx in conv_to_bn:
            bn_layers[conv_to_bn[layer_idx]].switch_bn(bit_pair, is_sample_min=False)
    
    return weights, act, []


def sample_min_cands(model, configs):
    """
    FAST PATH: Sample minimum bit-width configuration.
    Direct attribute setting - no method calls, no sampling logic.
    """
    cache = get_cached_layers(model, configs)
    min_bit = cache['min_bit']
    bit_pair = (min_bit, min_bit)
    
    quan_layers = cache['quan_layers']
    bn_layers = cache['bn_layers']
    conv_to_bn = cache['conv_to_bn']
    
    weights = []
    act = []
    is_min = []
    
    for layer_idx, (module, name, layer_type) in enumerate(quan_layers):
        module.bits = bit_pair
        weights.append(min_bit)
        act.append(min_bit)
        is_min.append(layer_idx)
        
        if layer_idx in conv_to_bn:
            bn_layers[conv_to_bn[layer_idx]].switch_bn(bit_pair, is_sample_min=True)
    
    return weights, act, is_min


def sample_one_mixed_policy(model, configs, max_sample_bits=None, sample_max=False, 
                            sample_min=False, weight_cands=None, act_cands=None,
                            sensitive_indices=None):
    """
    OPTIMIZED: Sample one mixed-precision policy with sensitivity awareness.
    
    If sensitive_indices is provided:
    - Sensitive layers: bits from [8, 7, 6, 5] uniform random.
    - Safe layers: bits from [8..2] with weighted random (higher prob for lower bits).
    """
    # Fast path for max/min sampling
    if sample_max:
        return sample_max_cands(model, configs)
    if sample_min:
        return sample_min_cands(model, configs)
    
    cache = get_cached_layers(model, configs)
    quan_layers = cache['quan_layers']
    bn_layers = cache['bn_layers']
    conv_to_bn = cache['conv_to_bn']
    target_bits = cache['target_bits']
    min_bit = cache['min_bit']
    
    sensitive_bits = cache.get('sensitive_bits', [8, 7, 6, 5])
    safe_bits = cache.get('safe_bits', [8, 7, 6, 5, 4, 3, 2])
    safe_weights = cache.get('safe_weights', None)
    
    weights = []
    act = []
    is_min = []
    
    sensitive_set = set(sensitive_indices) if sensitive_indices is not None else None
    
    for layer_idx, (module, name, layer_type) in enumerate(quan_layers):
        if weight_cands is not None and act_cands is not None:
            wbit = weight_cands[layer_idx]
            abit = act_cands[layer_idx]
        else:
            if sensitive_set is None:
                # WARMUP: Uniform random for all layers
                wbit = random.choice(target_bits)
                abit = random.choice(target_bits)
            elif layer_idx in sensitive_set:
                # Sensitive policy: uniform random from sensitive_bits
                wbit = random.choice(sensitive_bits)
                abit = random.choice(sensitive_bits)
            else:
                # Safe policy: weighted random (lower bits preferred)
                if safe_weights:
                    wbit = random.choices(safe_bits, weights=safe_weights, k=1)[0]
                    abit = random.choices(safe_bits, weights=safe_weights, k=1)[0]
                else:
                    wbit = random.choice(target_bits)
                    abit = random.choice(target_bits)
        
        bit_pair = (wbit, abit)
        module.bits = bit_pair
        weights.append(wbit)
        act.append(abit)
        
        is_sample_min = (wbit == min_bit)
        if is_sample_min:
            is_min.append(layer_idx)
        
        if layer_idx in conv_to_bn:
            bn_layers[conv_to_bn[layer_idx]].switch_bn(bit_pair, is_sample_min=is_sample_min)
    
    return weights, act, is_min


def sample_batch_mixed_policies(model, configs, num_samples):
    """
    BATCH SAMPLING: Pre-compute multiple policies in one call.
    Useful for planning gradient accumulation strategies.
    Returns list of (weights, act, is_min) tuples.
    """
    cache = get_cached_layers(model, configs)
    target_bits = cache['target_bits']
    num_layers = len(cache['quan_layers'])
    
    policies = []
    min_bit = cache['min_bit']
    
    for _ in range(num_samples):
        weights = [random.choice(target_bits) for _ in range(num_layers)]
        act = [random.choice(target_bits) for _ in range(num_layers)]
        is_min = [i for i, w in enumerate(weights) if w == min_bit]
        policies.append((weights, act, is_min))
    
    return policies


def apply_policy(model, configs, weights, act):
    """
    Apply a pre-computed policy to the model.
    Used with sample_batch_mixed_policies for efficient batch updates.
    """
    cache = get_cached_layers(model, configs)
    quan_layers = cache['quan_layers']
    bn_layers = cache['bn_layers']
    conv_to_bn = cache['conv_to_bn']
    min_bit = cache['min_bit']
    
    for layer_idx, (module, name, layer_type) in enumerate(quan_layers):
        wbit = weights[layer_idx]
        abit = act[layer_idx]
        bit_pair = (wbit, abit)
        
        module.bits = bit_pair
        
        is_sample_min = (wbit == min_bit)
        if layer_idx in conv_to_bn:
            bn_layers[conv_to_bn[layer_idx]].switch_bn(bit_pair, is_sample_min=is_sample_min)