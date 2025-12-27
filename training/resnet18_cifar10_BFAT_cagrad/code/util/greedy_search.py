from .utils import model_profiling, reset_batchnorm_stats
import torch
import torch.distributed as dist
import math
from datetime import datetime
from .qat import get_quantized_layers
from .mpq import switch_bit_width

def get_timestamp():
    """Get current timestamp in HH:MM:SS format"""
    return datetime.now().strftime("%H:%M:%S")

def dist_all_reduce_tensor(tensor):
    """ Reduce to all ranks """
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        tensor.div_(world_size)
    return tensor


def forward_loss(
        model, criterion, input, target, meter, soft_target=None,
        soft_criterion=None, return_soft_target=False, return_acc=False, eval_mode=False):
    
    """forward model and return loss"""
    if eval_mode:
        model.eval()
    topk = (1, 5)
    output = model(input)

    loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in topk:
        correct_k.append(correct[:k].float().sum(0))
    tensor = torch.cat([loss.view(1)] + correct_k, dim=0)
    # allreduce
    tensor = dist_all_reduce_tensor(tensor)
    # cache to meter
    tensor = tensor.cpu().detach().numpy()
    bs = (tensor.size-1)//2
    for i, k in enumerate(topk):
        error_list = list(1.-tensor[1+i*bs:1+(i+1)*bs])
        if return_acc and k == 1:
            top1_error = sum(error_list) / len(error_list)
            return loss, top1_error
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(tensor[0])
    if return_soft_target:
        return loss, torch.nn.functional.softmax(output, dim=1)
    return loss


def forward_loss_with_fault(
        model, criterion, input, target, fault_injector=None, 
        return_acc=False, eval_mode=False):
    """
    Forward model with both normal and fault injection evaluation.
    
    Returns:
        - If fault_injector is None: returns (loss_normal, acc_normal) or loss_normal
        - If fault_injector is provided: returns (loss_normal, loss_fault, acc_normal, acc_fault) or (loss_normal, loss_fault)
    """
    if eval_mode:
        model.eval()
    
    topk = (1, 5)
    
    # Normal forward pass
    if fault_injector is not None:
        fault_injector.disable()
    output_normal = model(input)
    loss_normal = torch.mean(criterion(output_normal, target))
    
    # Calculate normal accuracy
    _, pred_normal = output_normal.topk(max(topk))
    pred_normal = pred_normal.t()
    correct_normal = pred_normal.eq(target.view(1, -1).expand_as(pred_normal))
    correct_k_normal = []
    for k in topk:
        correct_k_normal.append(correct_normal[:k].float().sum(0))
    
    tensor_normal = torch.cat([loss_normal.view(1)] + correct_k_normal, dim=0)
    tensor_normal = dist_all_reduce_tensor(tensor_normal)
    tensor_normal = tensor_normal.cpu().detach().numpy()
    bs = (tensor_normal.size-1)//2
    
    # Extract normal accuracy
    error_list_normal = list(1. - tensor_normal[1+0*bs:1+(0+1)*bs])
    acc_normal = 1.0 - (sum(error_list_normal) / len(error_list_normal))
    
    # If no fault injector, return normal results only
    if fault_injector is None:
        if return_acc:
            return loss_normal, 1.0 - acc_normal  # Return error for consistency
        return loss_normal
    
    # Fault injection forward pass
    fault_injector.enable()
    output_fault = model(input)
    loss_fault = torch.mean(criterion(output_fault, target))
    
    # Calculate fault accuracy
    _, pred_fault = output_fault.topk(max(topk))
    pred_fault = pred_fault.t()
    correct_fault = pred_fault.eq(target.view(1, -1).expand_as(pred_fault))
    correct_k_fault = []
    for k in topk:
        correct_k_fault.append(correct_fault[:k].float().sum(0))
    
    tensor_fault = torch.cat([loss_fault.view(1)] + correct_k_fault, dim=0)
    tensor_fault = dist_all_reduce_tensor(tensor_fault)
    tensor_fault = tensor_fault.cpu().detach().numpy()
    
    error_list_fault = list(1. - tensor_fault[1+0*bs:1+(0+1)*bs])
    acc_fault = 1.0 - (sum(error_list_fault) / len(error_list_fault))
    
    if return_acc:
        return loss_normal, loss_fault, 1.0 - acc_normal, 1.0 - acc_fault  # Return errors
    return loss_normal, loss_fault



def adjust_one_layer_bit_width(layer, bn, next_bits:int, reduce: bool, tensor: str):
    if not reduce:
        if tensor == 'weight':
            layer.bits = (next_bits, layer.bits[1])
            if bn is not None:
                bn.switch_bn(layer.bits)
            return layer.bits[0]
        else:
            layer.bits = (layer.bits[0], next_bits)
            if bn is not None:
                bn.switch_bn(layer.bits)
            return layer.bits[1]
    else:
        if tensor == 'weight':
            layer.bits = (next_bits, layer.bits[1])
            if bn is not None:
                bn.switch_bn(layer.bits)
            return layer.bits[0]
        else:
            layer.bits = (layer.bits[0], next_bits)
            if bn is not None:
                bn.switch_bn(layer.bits)
            return layer.bits[1]

def get_layer_wise_conf(layers, tensor):

    return [l.bits[0] if tensor=='weight' else l.bits[1] for l in layers]  


def reset_bit_cands(model: torch.nn.Module, reset=True):
    from quan.func import QuanConv2d
    
    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d):
            if reset:
                print(name, 'bits_cands to', module.reset_bits_cands())
            else:
                print(name, module.weight_bit_cands, module.act_bit_cands)


def search(loader, model, criterion, metrics, cfgs, epoch=0, start_bits=6, init_w=None, init_a=None, fault_injector=None):
    constraint_type, target_size = metrics
    quan_scheduler = cfgs.quan
    target_bits = cfgs.target_bits

    if init_a and init_w:
        from .qat import set_bit_width
        set_bit_width(model, init_w, init_a)
        print("init from", init_w, init_a)
    assert constraint_type in ['model_size', 'bitops']

    # Only set epoch if sampler has set_epoch method (DistributedSampler)
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    def get_next_batch(iterator_ref):
        """Get next batch from iterator, recreate if exhausted"""
        try:
            return next(iterator_ref[0])
        except StopIteration:
            # Recreate iterator if exhausted
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)
            iterator_ref[0] = iter(loader)
            return next(iterator_ref[0])
    
    iterator_ref = [iter(loader)]
    iterator = iterator_ref[0]

    data_for_bn_rest = []
    for _ in range(3):
        data_for_bn_rest.append(get_next_batch(iterator_ref))

    model.eval()
    reset_batchnorm_stats(model)
    
    quantized_layers, bn = get_quantized_layers(model)
    target_size: list

    lut, lut_complexity = [], []
    for _ in range(2):
        for layer in quantized_layers:
            lut.append(0)
            
            lut_complexity.append(0)

    print("searching...")
    configs = []
    model.eval()
    
    bitops, model_size = model_profiling(model)
    start_complexity = bitops if constraint_type == 'bitops' else model_size

    # Determine smallest bit width based on target BitOPs
    # For ImageNet scale (target_size >= 5.0), use 3-bit minimum to avoid severe accuracy drop
    # For CIFAR10 scale (target_size < 5.0), use 2-bit minimum
    # Note: For CIFAR10 even with higher target BitOPs (e.g., 6.0), we should still allow 2-bit
    # You can force smallest_bit_width = 2 if needed for CIFAR10
    smallest_bit_width = 3 if max(target_size) >= 5.0 else 2
    # smallest_bit_width = 2  # Uncomment this line to always allow 2-bit for CIFAR10 
    model.train()

    # ==================== Fault-Aware Search Configuration ====================
    # 故障感知搜索配置（方案3：故障感知的贪婪搜索）
    use_fault_aware_search = False
    fault_aware_search_config = None
    search_fault_injector = None
    
    if fault_injector is None:
        # Check if fault_aware_search is enabled in configs
        fault_aware_search_config = getattr(cfgs, 'fault_aware_search', None)
        if fault_aware_search_config is not None:
            use_fault_aware_search = getattr(fault_aware_search_config, 'enabled', False)
            
            if use_fault_aware_search:
                # Get search-specific fault injection config
                fault_injection_config = getattr(fault_aware_search_config, 'fault_injection', {})
                search_ber = getattr(fault_injection_config, 'ber', 2e-2)  # Default BER for search
                
                # Get weights for multi-objective optimization
                weights_config = getattr(fault_aware_search_config, 'weights', {})
                alpha = getattr(weights_config, 'alpha', 0.5)  # Normal accuracy weight
                beta = getattr(weights_config, 'beta', 0.3)    # Fault tolerance weight
                
                # Create fault injector for search
                from util.fault_injector import FaultInjector
                training_model = model.module if hasattr(model, 'module') else model
                search_fault_injector = FaultInjector(
                    model=training_model,
                    mode="ber",
                    ber=float(search_ber),
                    enable_in_training=False,  # Only use in evaluation during search
                    enable_in_inference=True,
                    seed=getattr(cfgs, 'seed', 42)
                )
                
                print("=" * 80)
                print("🔍 FAULT-AWARE SEARCH (Scheme 3) - ENABLED")
                print("=" * 80)
                print(f"  ✅ FaultInjector initialized for search")
                print(f"  ✅ Search BER: {search_ber}")
                print(f"  ✅ Multi-objective weights: alpha={alpha} (normal), beta={beta} (fault)")
                print(f"  ✅ Optimization: score = α * acc_normal + β * acc_fault - γ * bitops")
                print("=" * 80)
            else:
                print("=" * 80)
                print("⚠️  FAULT-AWARE SEARCH - DISABLED")
                print("  (Using standard search without fault injection)")
                print("=" * 80)
    else:
        # If fault_injector is provided, use it (for backward compatibility)
        search_fault_injector = fault_injector
        use_fault_aware_search = True
        # Default weights if not in config
        fault_aware_search_config = getattr(cfgs, 'fault_aware_search', None)
        if fault_aware_search_config is not None:
            weights_config = getattr(fault_aware_search_config, 'weights', {})
            alpha = getattr(weights_config, 'alpha', 0.5)
            beta = getattr(weights_config, 'beta', 0.3)
        else:
            alpha = 0.5
            beta = 0.3

    print('smallest bits', smallest_bit_width)

    def bops_map_to_bits(bops, arch='resnet18'):
        if 'mobilenetv2' in arch:
            
            if 5.0 <= bops <= 5.8:
                return 4
            
            if 3.3 <= bops <= 3.8:
                return 3
            
            return 4
        elif 'efficientnet' in arch:
            if 6.3 <= bops <= 7.1:
                return 4
            
            if 3.3 <= bops <= 4.5:
                return 3
            
            return 3
        elif 'alexnet' in arch:
            # CIFAR10 scale BitOPs for AlexNet (32x32 input)
            # AlexNet has more parameters than ResNet18, so BitOPs will be higher
            if 8.0 <= bops <= 12.0:  # CIFAR10: moderate BitOPs target for AlexNet
                return 4
            if 5.0 <= bops < 8.0:    # CIFAR10: lower BitOPs target
                return 3
            if bops < 5.0:           # CIFAR10: very low BitOPs target
                return 2
            # Default fallback: use 4-bit for higher BitOPs targets
            return 4
        elif 'resnet18' in arch:
            # ImageNet scale BitOPs (224x224 input)
            if 31 <= bops <= 36:
                return 4
            
            if 20 < bops <= 23.9:
                return 3
            
            if bops <= 20:
                return 2
            
            # CIFAR10 scale BitOPs (32x32 input, much smaller)
            # CIFAR10 BitOPs are roughly 0.02x of ImageNet BitOPs
            # For target BitOPs around 5.68, start from 4-bit weights
            if 5.0 <= bops <= 10.0:  # CIFAR10: moderate BitOPs target
                return 4
            
            if 3.0 <= bops < 5.0:    # CIFAR10: lower BitOPs target
                return 3
            
            if bops < 3.0:           # CIFAR10: very low BitOPs target
                return 2
            
            # Default fallback: use 4-bit for higher BitOPs targets
            return 4

    done_w, done_a = False, False
    w_init, a_init = False, False
    while True:
        input, target = get_next_batch(iterator_ref)
        input, target = input.cuda(), target.cuda()

        print(f"[{get_timestamp()}] current bitops {round(bitops, 2)}")
        metric = bitops if constraint_type == 'bitops' else model_size

        acc_scale = 1.5

        # sc = 1.1 # for W2A3-ResNet18
        sc = .95
        
        if not done_w and not w_init:
            # ==================== Initial Bit-Width Selection ====================
            # 初始bit-width选择逻辑
            bits = bops_map_to_bits(max(target_size), cfgs.arch)
            
            # 故障感知搜索：使用更高的初始bit-width，以探索更好的容错结构
            # 即使target_bops很小，也从更高的bit开始搜索，这样可以：
            # 1. 在更大的搜索空间中探索容错性更好的配置
            # 2. 有更多的bit分配选择（如某些层4-bit，某些层3-bit，某些层2-bit）
            # 3. 即使最终BitOPs会稍高一些，但容错性可能更好
            if use_fault_aware_search and search_fault_injector is not None:
                # 获取故障感知搜索的初始bit-width配置
                initial_bit_config = getattr(fault_aware_search_config, 'initial_weight_bit_width', None)
                if initial_bit_config is not None:
                    # 如果配置中指定了初始bit-width，使用配置的值
                    bits_fault_aware = int(initial_bit_config)
                    print(f"[{get_timestamp()}] Fault-Aware Search: Using configured initial weight bit-width: {bits_fault_aware}")
                else:
                    # 否则，使用一个更保守的策略：从bops_map_to_bits的结果向上调整
                    # 例如：如果bops_map_to_bits返回2-bit，我们从4-bit开始
                    # 如果返回3-bit，我们从4-bit开始
                    # 如果返回4-bit，我们从5-bit开始（如果target_bits中有5-bit）
                    if bits <= 2:
                        # 如果bops_map_to_bits返回2-bit或更小，从4-bit开始（如果target_bits中有4-bit）
                        bits_fault_aware = min(4, max(target_bits)) if 4 in target_bits else min(3, max(target_bits))
                    elif bits == 3:
                        # 如果返回3-bit，从4-bit开始
                        bits_fault_aware = min(4, max(target_bits)) if 4 in target_bits else 3
                    else:
                        # 如果返回4-bit或更大，从更高一位开始（如果target_bits中有）
                        bits_fault_aware = min(bits + 1, max(target_bits)) if (bits + 1) in target_bits else bits
                    
                    print(f"[{get_timestamp()}] Fault-Aware Search: Adjusted initial weight bit-width from {bits} (bops_map_to_bits) to {bits_fault_aware} (for better fault tolerance exploration)")
                
                # 确保选择的bit在target_bits范围内
                if bits_fault_aware not in target_bits:
                    # 选择target_bits中最接近且不超过bits_fault_aware的值
                    bits_fault_aware = max([b for b in target_bits if b <= bits_fault_aware], default=min(target_bits))
                    print(f"[{get_timestamp()}] Fault-Aware Search: Adjusted initial weight bit-width to {bits_fault_aware} (closest valid value in target_bits)")
                
                bits = bits_fault_aware  # 使用调整后的bit-width
            # ==================== End Initial Bit-Width Selection ====================
            
            print(f"[{get_timestamp()}] Current BitOPs: {bitops:.2f}, Target BitOPs: {max(target_size):.2f}, Initial weight bit-width: {bits}, start_bit_width config: {start_bits}")
            print(bitops, model_size)

            # Set initial bit-width based on bops_limits (bits from bops_map_to_bits) or fault-aware adjusted value
            switch_bit_width(model, quan_scheduler, wbit=bits, abits=start_bits)
            w_init = True
            w_target_bitops, _ = model_profiling(model) # only support bops now
            # ⚠️ 修复：保持使用bits而不是start_bits，因为bits是根据bops_limits计算的正确初始值（或故障感知调整后的值）
            switch_bit_width(model, quan_scheduler, wbit=bits, abits=start_bits)
            w_target_bitops *= sc
            print(f"[{get_timestamp()}] w_target_bitops (after scaling): {w_target_bitops:.2f}")

            # ==================== Fault-Aware Search: Initialize Fault Tolerance Target ====================
            # 故障感知搜索：初始化容错性目标
            w_target_fault_tolerance = None
            if use_fault_aware_search and search_fault_injector is not None:
                # 评估初始配置的容错性（作为基准）
                with torch.no_grad():
                    model.eval()
                    # 使用一个batch评估初始容错性
                    eval_input, eval_target = get_next_batch(iterator_ref)
                    eval_input, eval_target = eval_input.cuda(), eval_target.cuda()
                    
                    # 评估正常和故障准确率
                    _, _, top1_error_normal_init, top1_error_fault_init = forward_loss_with_fault(
                        model, criterion, eval_input, eval_target,
                        fault_injector=search_fault_injector,
                        return_acc=True, eval_mode=True
                    )
                    
                    acc_normal_init = 1.0 - top1_error_normal_init
                    acc_fault_init = 1.0 - top1_error_fault_init
                    fault_tolerance_init = acc_fault_init / acc_normal_init if acc_normal_init > 0 else 0.0
                    
                    # 设置容错性目标：初始容错性的某个百分比（如80%或90%）
                    # 或者使用配置文件中的目标值
                    fault_tolerance_target_config = getattr(fault_aware_search_config, 'fault_tolerance_target', None)
                    if fault_tolerance_target_config is not None:
                        # 如果配置了绝对目标值（如0.8表示80%）
                        w_target_fault_tolerance = float(fault_tolerance_target_config)
                    else:
                        # 否则使用相对目标：初始容错性的某个百分比（默认90%）
                        fault_tolerance_ratio = getattr(fault_aware_search_config, 'fault_tolerance_ratio', 0.9)
                        w_target_fault_tolerance = fault_tolerance_init * fault_tolerance_ratio
                    
                    print(f"[{get_timestamp()}] Fault-Aware Search: Initial fault tolerance = {fault_tolerance_init:.4f} "
                          f"(acc_normal={acc_normal_init:.4f}, acc_fault={acc_fault_init:.4f})")
                    print(f"[{get_timestamp()}] Fault-Aware Search: Target fault tolerance = {w_target_fault_tolerance:.4f}")
                    
                    model.train()  # 恢复训练模式
            # ==================== End Fault-Aware Search: Initialize Fault Tolerance Target ====================

            if init_a and init_w:
                set_bit_width(model, init_w, init_a)
            # metric = bitops if constraint_type == 'bitops' else model_size

        # Check if all weight bit-widths have reached minimum, force switch to activation bit-width adjustment
        # This check should happen BEFORE the metric check, so it takes priority
        if not done_w:
            all_weights_at_min = all(layer.bits[0] <= smallest_bit_width for layer in quantized_layers)
            if all_weights_at_min:
                print(f"[{get_timestamp()}] All weight bit-widths have reached minimum ({smallest_bit_width}), switching to activation bit-width adjustment")
                done_w = True
                lut = [0 for _ in range(len(quantized_layers) * 2)]
                lut_complexity = [0 for _ in range(len(quantized_layers) * 2)]
                # Note: done_a remains False to allow activation bit-width adjustment to proceed
                print(f"[{get_timestamp()}] Starting activation bit-width adjustment (current BitOPs: {metric:.2f}, target: {max(target_size):.2f})")
        
        # ==================== Fault-Aware Search: Check Activation Search Termination ====================
        # 故障感知搜索：检查激活值搜索结束条件
        # 当所有激活值bit-width都达到最小值时，结束激活值搜索（避免无效的继续搜索）
        # 注意：此检查仅在故障感知搜索模式下执行，保持代码隔离
        if use_fault_aware_search and done_w and not done_a:
            all_activations_at_min = all(layer.bits[1] <= smallest_bit_width for layer in quantized_layers)
            if all_activations_at_min:
                print(f"[{get_timestamp()}] Fault-Aware Search: All activation bit-widths have reached minimum ({smallest_bit_width}), ending activation bit-width adjustment")
                done_a = True
                lut = [0 for _ in range(len(quantized_layers) * 2)]
                lut_complexity = [0 for _ in range(len(quantized_layers) * 2)]
                print(f"[{get_timestamp()}] Fault-Aware Search: Activation search complete: all activations are at minimum bit-width")
                # Note: When both done_w and done_a are True, we should save the current configuration
                # and move to the next target_size (if any) or end the search
        # ==================== End Fault-Aware Search: Check Activation Search Termination ====================
        
        # ==================== Weight Search Termination Condition ====================
        # 结束weight搜索的条件
        should_end_weight_search = False
        reason = ""
        
        # 标准条件：BitOPs达到目标
        if metric <= w_target_bitops and not done_w:
            # 对于故障感知搜索，还需要检查容错性目标
            if use_fault_aware_search and search_fault_injector is not None and w_target_fault_tolerance is not None:
                # 评估当前配置的容错性
                with torch.no_grad():
                    model.eval()
                    eval_input, eval_target = get_next_batch(iterator_ref)
                    eval_input, eval_target = eval_input.cuda(), eval_target.cuda()
                    
                    # 评估正常和故障准确率
                    _, _, top1_error_normal, top1_error_fault = forward_loss_with_fault(
                        model, criterion, eval_input, eval_target,
                        fault_injector=search_fault_injector,
                        return_acc=True, eval_mode=True
                    )
                    
                    acc_normal = 1.0 - top1_error_normal
                    acc_fault = 1.0 - top1_error_fault
                    current_fault_tolerance = acc_fault / acc_normal if acc_normal > 0 else 0.0
                    
                    # 检查容错性是否达到目标
                    if current_fault_tolerance >= w_target_fault_tolerance:
                        # 容错性达到目标，可以结束weight搜索
                        should_end_weight_search = True
                        reason = f"BitOPs target reached ({metric:.2f} <= {w_target_bitops:.2f}) AND fault tolerance target reached (fault_tol={current_fault_tolerance:.4f} >= target={w_target_fault_tolerance:.4f})"
                    else:
                        # 容错性未达到目标，继续搜索以改善容错性
                        # 但设置一个最低阈值：如果容错性达到初始的80%，即使未达到目标，也可以结束（避免无限搜索）
                        min_fault_tolerance_threshold = w_target_fault_tolerance * 0.8  # 目标值的80%作为最低阈值
                        if current_fault_tolerance >= min_fault_tolerance_threshold:
                            # 容错性达到最低阈值，可以结束（避免无限搜索）
                            should_end_weight_search = True
                            reason = f"BitOPs target reached ({metric:.2f} <= {w_target_bitops:.2f}) AND fault tolerance reached minimum threshold (fault_tol={current_fault_tolerance:.4f} >= min_threshold={min_fault_tolerance_threshold:.4f}, target={w_target_fault_tolerance:.4f})"
                        else:
                            # 容错性未达到最低阈值，继续搜索
                            print(f"[{get_timestamp()}] BitOPs target reached ({metric:.2f} <= {w_target_bitops:.2f}), "
                                  f"but fault tolerance ({current_fault_tolerance:.4f}) < min threshold ({min_fault_tolerance_threshold:.4f}) < target ({w_target_fault_tolerance:.4f}), "
                                  f"continuing weight search to improve fault tolerance...")
                            should_end_weight_search = False
                    
                    model.train()  # 恢复训练模式
            else:
                # 标准搜索：BitOPs达到目标即可结束
                should_end_weight_search = True
                reason = f"BitOPs target reached ({metric:.2f} <= {w_target_bitops:.2f})"
        
        # 执行结束weight搜索的逻辑
        if should_end_weight_search and not done_w:
            done_w = True
            lut = [0 for _ in range(len(quantized_layers) * 2)]
            lut_complexity = [0 for _ in range(len(quantized_layers) * 2)]
            print(f"[{get_timestamp()}] Weight search done: {reason}")
        # ==================== End Weight Search Termination Condition ====================
        
        # Note: When done_w is True and done_a is False, activation bit-width adjustment will proceed
        # done_a will be set to True when:
        # 1. All activation bit-widths reach minimum (故障感知搜索模式，checked above)
        # 2. BitOPs reaches target and we save the configuration (checked below)
        
        # Save configuration when BitOPs reaches target
        # For fault-aware search: also save when both weight and activation searches are complete
        should_save_config = False
        save_reason = ""
        
        if metric < max(target_size):
            should_save_config = True
            save_reason = f"BitOPs target reached ({metric:.2f} < {max(target_size):.2f})"
        elif use_fault_aware_search and done_w and done_a:
            # 故障感知搜索：当权重和激活值搜索都完成时，也保存配置
            should_save_config = True
            save_reason = "Both weight and activation searches complete (fault-aware search)"
        
        if should_save_config:
            print(f"[{get_timestamp()}] {save_reason}, saving configuration")
            
            configs.append((max(target_size), get_layer_wise_conf(quantized_layers, tensor='weight'), get_layer_wise_conf(quantized_layers, tensor='act')))
            target_size.remove(max(target_size))
            done_w, done_a, w_init = False, False, False
            # After reset, check if all weights are already at minimum, if so, immediately switch to activation adjustment
            all_weights_at_min = all(layer.bits[0] <= smallest_bit_width for layer in quantized_layers)
            if all_weights_at_min:
                print(f"[{get_timestamp()}] After reset: All weight bit-widths are already at minimum ({smallest_bit_width}), switching to activation bit-width adjustment")
                done_w = True
                lut = [0 for _ in range(len(quantized_layers) * 2)]
                lut_complexity = [0 for _ in range(len(quantized_layers) * 2)]
                # 故障感知搜索：如果激活值也已经达到最小值，直接结束激活值搜索
                if use_fault_aware_search:
                    all_activations_at_min = all(layer.bits[1] <= smallest_bit_width for layer in quantized_layers)
                    if all_activations_at_min:
                        print(f"[{get_timestamp()}] Fault-Aware Search: After reset, all activation bit-widths are already at minimum ({smallest_bit_width}), ending activation search")
                        done_a = True
        
        if len(target_size) == 0:
            break

        for idx, layer in enumerate(quantized_layers):
            wbits, abits = layer.bits

            for mode in ['+', '-']:
                
                if mode == '+':
                    overall_idx  = idx * 2 + 1
                    # continue
                else:
                    overall_idx  = idx * 2 
                lut[overall_idx] = 0.
                lut_complexity[overall_idx] = 0.

                if mode == '-':
                    if (wbits <= smallest_bit_width and not done_w):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue
                else:
                    if (wbits >= max(target_bits) and not done_w):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue

                if mode == '-':
                    if abits <= smallest_bit_width and (done_w and not done_a):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue
                else:
                    if abits >= max(target_bits) and (done_w and not done_a):
                        lut[overall_idx] = math.inf
                        lut_complexity[overall_idx] = math.inf
                        continue

                if mode == '-':
                    if wbits > smallest_bit_width and not done_w:
                        next_wbits_index = target_bits.index(wbits) + 1

                        # Use weight_bit_cands property to handle both split_aw_cands modes
                        if wbits == min(layer.weight_bit_cands):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_wbits = target_bits[next_wbits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_wbits, reduce=False, tensor='weight')
                    
                    if abits > smallest_bit_width and (done_w and not done_a):
                        next_abits_index = target_bits.index(abits) + 1

                        # Use act_bit_cands property to handle both split_aw_cands modes
                        if abits == min(layer.act_bit_cands):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_abits = target_bits[next_abits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_abits, reduce=False, tensor='act')
                else:
                    if wbits < max(target_bits) and not done_w:
                        next_wbits_index = target_bits.index(wbits) - 1

                        if wbits == max(target_bits):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_wbits = target_bits[next_wbits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_wbits, reduce=False, tensor='weight')
                    
                    if abits < max(target_bits) and (done_w and not done_a):
                        next_abits_index = target_bits.index(abits) - 1

                        if abits == max(target_bits):
                            lut[overall_idx] = math.inf
                            lut_complexity[overall_idx] = math.inf
                            continue

                        next_abits = target_bits[next_abits_index]
                        adjust_one_layer_bit_width(layer, bn[idx], next_abits, reduce=False, tensor='act')
            
                with torch.no_grad():
                    # calibrate_batchnorm_state(model, loader, 15, reset=True, distributed_training=False)

                    # calibrate_batchnorm_state(model, loader=data_for_bn_rest, reset=True, distributed_training=True)
                    
                    # ==================== Fault-Aware Evaluation ====================
                    if use_fault_aware_search and search_fault_injector is not None:
                        # Use fault-aware evaluation: evaluate both normal and fault cases
                        _, _, top1_error_normal, top1_error_fault = forward_loss_with_fault(
                            model, criterion, input, target, 
                            fault_injector=search_fault_injector,
                            return_acc=True, eval_mode=False
                        )
                        
                        # Calculate fault tolerance: use accuracy (1 - error) instead of error
                        acc_normal = 1.0 - top1_error_normal
                        acc_fault = 1.0 - top1_error_fault
                        
                        # Combined metric: α * acc_normal + β * acc_fault
                        # But we store error in lut, so we need to convert back
                        # We'll store a combined error: 1 - (α * acc_normal + β * acc_fault)
                        combined_acc = alpha * acc_normal + beta * acc_fault
                        combined_error = 1.0 - combined_acc
                        
                        lut[overall_idx] = combined_error
                    else:
                        # Standard evaluation: only normal accuracy
                        _, top1_error = forward_loss(model, criterion, input, target, None, return_acc=True, eval_mode=False)
                        lut[overall_idx] = top1_error
                    # ==================== End Fault-Aware Evaluation ====================

                tmp_bitops, tmp_model_size = model_profiling(model)
                comp = tmp_bitops if constraint_type == 'bitops' else tmp_model_size
                # lut_complexity[overall_idx] += (start_complexity - comp)
                lut_complexity[overall_idx] += -comp

                if mode == '-':
                    if wbits > smallest_bit_width and not done_w:
                        adjust_one_layer_bit_width(layer, bn[idx], wbits, reduce=True, tensor='weight')

                    if abits > smallest_bit_width and (done_w and not done_a):
                        adjust_one_layer_bit_width(layer, bn[idx], abits, reduce=True, tensor='act')
                else:
                    if wbits >= smallest_bit_width and not done_w:
                        adjust_one_layer_bit_width(layer, bn[idx], wbits, reduce=True, tensor='weight')

                    if abits >= smallest_bit_width and (done_w and not done_a):
                        adjust_one_layer_bit_width(layer, bn[idx], abits, reduce=True, tensor='act')
    
        # if wbits > smallest_bit_width:
        #     adjust_one_layer_bit_width(layerw, bn[idxw], wbits, reduce=True, tensor='weight')
            # print(f"top-1 error {top1_error}")
        
        tmp_lut = []
        max_acc, max_comp = 0, 0
        min_acc, min_comp = 0, 0
        for acc, comp in zip(lut, lut_complexity):
            max_acc = acc if (acc > max_acc and acc is not math.inf) else max_acc
            min_acc = acc if (acc < min_acc and acc is not math.inf) else min_acc

            max_comp = comp if (comp > max_comp and comp is not math.inf) else max_comp
            min_comp = comp if (comp < min_comp and comp is not math.inf) else min_comp
        
        for acc, comp in zip(lut, lut_complexity):
            if acc == math.inf:
                tmp_lut.append(math.inf)
                continue

            tmp_lut.append(acc_scale*((acc-min_acc)/(max_acc-min_acc)) - (comp-min_comp)/(max_comp-min_comp))

        # Count valid and invalid candidates
        valid_count = sum(1 for x in tmp_lut if x != math.inf)
        invalid_count = sum(1 for x in tmp_lut if x == math.inf)
        print(f"[{get_timestamp()}] Candidates: {valid_count} valid, {invalid_count} invalid out of {len(tmp_lut)} total")
        
        if valid_count == 0:
            print(f"[{get_timestamp()}] WARNING: All candidates are invalid (inf)!")
            print(f"[{get_timestamp()}] This usually means all layers have reached bit-width boundaries.")
            print(f"[{get_timestamp()}] Current layer states (first 5 layers):")
            for idx in range(min(5, len(quantized_layers))):
                layer = quantized_layers[idx]
                wbits, abits = layer.bits
                print(f"  Layer {idx}: w={wbits}, a={abits}, w_cands={layer.weight_bit_cands.cpu().tolist()}, a_cands={layer.act_bit_cands.cpu().tolist()}")
        
        best_idx = tmp_lut.index(min(tmp_lut))
        min_metric = min(tmp_lut)
        min_error = lut[best_idx]
        
        print(f"[{get_timestamp()}] current optim metric", min_metric, 'min top-1 error', min_error)
        
        if min_metric != math.inf:
            print(f"[{get_timestamp()}] Selected: layer_idx={best_idx//2}, mode={'-' if best_idx%2==0 else '+'}, "
                  f"metric={tmp_lut[best_idx]:.4f}, error={lut[best_idx]:.4f}, bops_delta={lut_complexity[best_idx]:.4f}")
            # Show top 5 valid candidates
            valid_candidates = [(i, tmp_lut[i], lut[i], lut_complexity[i]) for i in range(len(tmp_lut)) if tmp_lut[i] != math.inf]
            valid_candidates.sort(key=lambda x: x[1])  # Sort by metric
            print(f"[{get_timestamp()}] Top 5 valid candidates:")
            for i, (idx, metric, error, comp) in enumerate(valid_candidates[:5]):
                print(f"  {i+1}. Layer {idx//2}, mode={'-' if idx%2==0 else '+'}, metric={metric:.4f}, error={error:.4f}, comp={comp:.4f}")
        
        assert best_idx is not math.inf

        if not done_w:
            best_layer_index_w = best_idx // 2
            best_layer_wbits = quantized_layers[best_layer_index_w].bits[0]
            offset = 1 if best_idx % 2 == 0 else -1

            if best_layer_wbits > smallest_bit_width:
                
                next_w_bit_index = target_bits.index(best_layer_wbits) + offset
                next_w_bit = target_bits[next_w_bit_index]
                wnew_bit_width = adjust_one_layer_bit_width(quantized_layers[best_layer_index_w], bn[best_layer_index_w], next_w_bit, reduce=False, tensor='weight')
                print(f"[{get_timestamp()}] layer {best_layer_index_w} weight: bit-width {best_layer_wbits} -> {wnew_bit_width}")
            else:
                print(f"[{get_timestamp()}] layer {best_layer_index_w} weight bit-width {best_layer_wbits} not change")
        
        if done_w and not done_a:
            best_layer_index_a = best_idx // 2
            best_layer_abits = quantized_layers[best_layer_index_a].bits[1]
            offset = 1 if best_idx % 2 == 0 else -1

            if best_layer_abits > smallest_bit_width:
                next_a_bit_index = target_bits.index(best_layer_abits) + offset
                next_a_bit = target_bits[next_a_bit_index]
                anew_bit_width = adjust_one_layer_bit_width(quantized_layers[best_layer_index_a], bn[best_layer_index_a], next_a_bit, reduce=False, tensor='act')
                print(f"[{get_timestamp()}] layer {best_layer_index_a} act: bit-width {best_layer_abits} -> {anew_bit_width}")
            else:
                print(f"[{get_timestamp()}] layer {best_layer_index_a} act bit-width {best_layer_abits} not change")
        
        print("")
        print(f"[{get_timestamp()}] weight bit-width assignment", get_layer_wise_conf(quantized_layers, tensor='weight'))
        print(f"[{get_timestamp()}] activs bit-width assignment", get_layer_wise_conf(quantized_layers, tensor='act'))
        print('-'*50)

        epoch += 1
        
        bitops, model_size = model_profiling(model)

    return configs
        

