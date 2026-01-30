import torch
import torch.nn as nn
import numpy as np
from quan.quantizer import LsqQuan
from quan.quantizer.lsq import compute_thd
from quan.func import SwithableBatchNorm, QuanConv2d
from .utils import model_profiling
from .dist import logger_info

def get_quantized_layers(model):
    return model_profiling(model, return_layers=True)[-2], model_profiling(model, return_layers=True)[-1]

def set_bit_width(model, confw, confa):
    layers, bns = get_quantized_layers(model)
    for idx, (layer, bn) in enumerate(zip(layers, bns)):
        oldw, olda = layer.bits
        layer.bits = (confw[idx], confa[idx])
        if bn is not None:  # Some layers (e.g., QuanLinear) don't have BatchNorm
            bn.switch_bn(layer.bits)

def set_forward_hook_for_quantized_layers(model, features, is_max=False):
    if not is_max:
        def block_wise_feature_hook(module, inp, out):
            features.append(out if module.is_sample_min else None)
    else:
        def block_wise_feature_hook(module, inp, out):
            features.append(out.detach())
                
    hooks = []
    for module in model.modules():
        if isinstance(module, SwithableBatchNorm) and module.size > 1:
            hooks.append(module.register_forward_hook(hook=block_wise_feature_hook))
    
    return hooks

def add_hook_for_quantized_layers(model: torch.nn.Module, hook, container: list):
    hooks = []
    for module in model.modules():
        if isinstance(module, SwithableBatchNorm) and module.size > 1:
            hooks.append(module.register_forward_hook(hook=hook))
    
    return hooks

def remove_hook_for_quantized_layers(hooks):
    for h in hooks:
        h.remove()


def set_forward_hook_for_conv_linear_layers(model, activations_container: list):
    """
    为所有QuanConv2d和QuanLinear层注册forward hook，用于提取激活值。
    
    Args:
        model: 模型
        activations_container: 用于存储激活值的列表，每个元素是一个层的激活值
    
    Returns:
        hooks: hook列表，用于后续移除
    """
    def activation_hook(module, inp, out):
        # 保存激活值（使用detach避免梯度计算）
        activations_container.append(out.detach())
    
    hooks = []
    for name, module in model.named_modules():
        from quan.func import QuanConv2d, QuanLinear
        if isinstance(module, (QuanConv2d, QuanLinear)):
            hooks.append(module.register_forward_hook(activation_hook))
    
    return hooks

def link_conv_bn(model):
    """
    Find Conv-BN pairs and link them for BN-Folding Aware Fault Injection.
    Sets 'bn_layer' attribute on QuanConv2d layers.
    """
    print("Linking Conv-BN pairs for BN-Folding Aware Fault Injection...")
    linked_count = 0
    
    # We can reuse the logic from set_bit_width which uses model_profiling to find pairs
    try:
        # model_profiling returns (bitops, model_size, quantized_layers, bns)
        # Note: model_profiling needs to be called with return_layers=True
        # created just for this purpose to ensure exact matching logic
        from .utils import model_profiling
        _, _, layers, bns = model_profiling(model, return_layers=True)
        
        for layer, bn in zip(layers, bns):
            # layer is QuanConv2d or QuanLinear
            # bn is SwitchableBatchNorm or None
            
            if isinstance(layer, QuanConv2d) and bn is not None:
                layer.bn_layer = bn
                # Attach parent layer reference to quantizer so FaultInjector wrapper can access it
                if hasattr(layer, 'quan_w_fn'):
                    layer.quan_w_fn.parent_layer = layer
                linked_count += 1
                
    except Exception as e:
        print(f"Warning: Failed to use model_profiling for linking: {e}")
        # Fallback: Simple sequential scan
        params_set = 0
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, QuanConv2d):
                last_conv = module
            elif isinstance(module, SwithableBatchNorm) and last_conv is not None:
                # Assume this BN belongs to the last Conv
                # This is a heuristic and might be fragile for complex paths
                # But typically Conv is immediately followed by BN
                if last_conv.bn_layer is None:
                    last_conv.bn_layer = module
                    # Attach parent layer reference to quantizer so FaultInjector wrapper can access it
                    if hasattr(last_conv, 'quan_w_fn'):
                        last_conv.quan_w_fn.parent_layer = last_conv
                    linked_count += 1
                last_conv = None # Reset
            elif isinstance(module, (torch.nn.Linear, QuanLinear)):
                 last_conv = None # Optimization: BN usually doesn't cross other layers
        
    print(f"Linked {linked_count} Conv-BN pairs.")

def profile_layerwise_quantization_metric(model: nn.Module, proportion=0.25):
    num_shifed_ls = []

    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d) and module.fixed_bits is None:
            shifted_val = 0.

            for idx, bit_width in enumerate(module.weight_bit_cands):
                q_weights = module.quan_w_fn(module.weight, bit_width, is_activation=False).detach()
                step_quantizer_param = module.quan_w_fn.get_scale(bit_width)
                lower, upper = compute_thd(module.quan_w_fn, bit_width)
                if isinstance(lower, torch.Tensor):
                    lower = int(lower.cpu().item())
                    upper = int(upper.cpu().item())

                num_shifed = 0
                num_weights = 0

                for int_bin in range(lower, upper + 1):
                    fp_bin = int_bin * step_quantizer_param
                    fp_bin_weights = module.weight[q_weights == fp_bin]

                    if int_bin == lower:
                        shitfed_weights_right = fp_bin_weights[fp_bin_weights > step_quantizer_param * (int_bin +  .5 * (1 - proportion))]

                        num_shifed += shitfed_weights_right.shape[0]
                    elif int_bin == upper:
                        shitfed_weights_left = fp_bin_weights[fp_bin_weights < fp_bin - step_quantizer_param * .5 * (1 - proportion)]

                        num_shifed += shitfed_weights_left.shape[0]
                    else:
                        # decision_bound = fp_bin + step_quantizer_param * .5
                        shitfed_weights_left = fp_bin_weights[fp_bin_weights < fp_bin - step_quantizer_param * .5 * (1 - proportion)]
                        shitfed_weights_right = fp_bin_weights[fp_bin_weights > fp_bin + step_quantizer_param * .5 * (1 - proportion)]

                        num_shifed += shitfed_weights_left.shape[0] + shitfed_weights_right.shape[0]
                    
                    num_weights += fp_bin_weights.shape[0]
                
                shifted_val += (num_shifed / num_weights / (bit_width.cpu() if isinstance(bit_width, torch.Tensor) else bit_width))
            
            num_shifed_ls.append(shifted_val / len(module.weight_bit_cands))
    
    # all_bin_distance = torch.stack(all_bin_distance)
    return num_shifed_ls

def freeze_layers(metric, model, org_cands, ratio=0.25, point_wise_min=2, depth_wise_min=3, progressive=False, logger=None):
    sorted_layer_metric = np.argsort(-np.array(metric))
    freezed_layer_num = 0
    
    layer_name, layers = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d) and module.fixed_bits is None:
            layer_name.append(name)
            layers.append(module)

    freezed_layer_id = []
    assert len(sorted_layer_metric) == len(layers) == len(layer_name)

    for metric_id, layer_id in enumerate(sorted_layer_metric):
        if freezed_layer_num >= int(len(sorted_layer_metric)*ratio):
            break

        module = layers[layer_id]
        # if layer_idx in freezed_layer_idx:
        old_cands = module.weight_bit_cands.cpu()
        cands = old_cands.tolist()

        # avoid hungry
        # if (module.weight.shape[1] == 1 and min(cands) >= 4) or \
        # (module.weight.shape[1] > 1 and min(cands) >= 3):
        #     continue

        new_cands = cands
        
        if not ((module.weight.shape[1] == 1 and min(cands) > depth_wise_min) or \
        (module.weight.shape[1] > 1 and min(cands) > point_wise_min)): 
            new_cands.sort()
            new_cands.reverse()
            new_cands = new_cands[:-1]

        module.set_bit_cands(new_cands)
        freezed_layer_num += 1
        freezed_layer_id.append(layer_id)

        logger_info(logger=logger, msg=f"[Freezing] layer {layer_name[layer_id]}, {metric[layer_id]}, {old_cands.tolist()} -> {module.weight_bit_cands.cpu().tolist()}")
       
    for layer_id in sorted_layer_metric:
        if layer_id not in freezed_layer_id:
            module = layers[layer_id]

            old_cands = module.weight_bit_cands.cpu().tolist()
            new_cands = tuple(org_cands[:-1]) if module.weight.shape[1] == 1 else tuple(org_cands)

            if progressive:
                diff = len(new_cands) - len(old_cands)
                assert diff <= 2
                new_cands = new_cands if diff <= 1 else new_cands[:-1]

            logger_info(logger=logger, msg=f"[Activating] layer {layer_name[layer_id]}, {metric[layer_id]}, {old_cands} -> {new_cands}")
            
            module.set_bit_cands(new_cands)

# Global cache for auxiliary_quantized_loss
_auxiliary_loss_cache = {}

def _get_quanconv_layers(model):
    """Get cached list of QuanConv2d layers with LsqQuan quantizers."""
    model_id = id(model.module if hasattr(model, 'module') else model)
    
    if model_id not in _auxiliary_loss_cache:
        layers = []
        unwrapped = model.module if hasattr(model, 'module') else model
        for _, module in unwrapped.named_modules():
            if isinstance(module, QuanConv2d):
                w_quantizer = module.quan_w_fn
                if isinstance(w_quantizer, LsqQuan):
                    layers.append(module)
        _auxiliary_loss_cache[model_id] = layers
    
    return _auxiliary_loss_cache[model_id]


def auxiliary_quantized_loss(model, conf=None, fairness_regularization=False, quantization_error_minimization=False):
    """
    OPTIMIZED: Uses cached layer list to avoid module traversal overhead.
    """
    QE_loss, distribution_loss = 0, 0
    
    # Use cached layers instead of iterating model.named_modules()
    layers = _get_quanconv_layers(model)
    
    for quantizer_idx, module in enumerate(layers):
        if module.bits is None:
            continue
        
        weights = module.weight
        current_wbits = module.bits[0] if conf is None else conf[quantizer_idx]
        w_quantizer = module.quan_w_fn
        
        step_size = w_quantizer.get_scale(current_wbits, detach=False)
        
        is_computed_clipped_weights = False
        if fairness_regularization:
            if current_wbits == 2:
                lower_bound, upper_bound = w_quantizer.weight_bound(bits=current_wbits)
                clipped_weights = torch.clamp(weights, min=lower_bound, max=upper_bound)
                distribution_loss += 1/2 * clipped_weights.pow(2).sum()
                is_computed_clipped_weights = True

        if quantization_error_minimization:
            if not is_computed_clipped_weights:
                lower_bound, upper_bound = w_quantizer.weight_bound(bits=current_wbits)
                clipped_weights = torch.clamp(weights, min=lower_bound, max=upper_bound)

            q_weights = w_quantizer(weights, bits=current_wbits).detach()
            bit_wise_distance = 2**(current_wbits - min(module.weight_bit_cands))

            if bit_wise_distance != 1:
                step_size = step_size.detach()
                thd_neg_min, thd_pos_min = compute_thd(w_quantizer, min(module.weight_bit_cands))
                bit_wise_distance_mapping = [ele*bit_wise_distance*step_size for ele in range(thd_neg_min, thd_pos_min+1)]

                idx = q_weights == bit_wise_distance_mapping[0]
                for cod in bit_wise_distance_mapping[1:]:
                    idx |= (q_weights == cod)
                
                latent_weights = clipped_weights.detach()
                q_weights = torch.where(idx, q_weights, latent_weights)
            
            QE_loss += ((clipped_weights - q_weights) ** 2).sum(0).mean()
                
    if conf is not None:
        assert quantizer_idx + 1 == len(conf)
    
    return QE_loss, distribution_loss