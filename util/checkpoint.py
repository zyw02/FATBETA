import logging
import os

import torch
import torch.nn as nn
from quan import QuanConv2d
from timm.utils import unwrap_model, get_state_dict
from util.dist import master_only

logger = logging.getLogger()

def filter_optimizer_state(optimizer, optimizer_state, model):
    """Filter optimizer state to remove parameters with shape mismatch.
    
    Args:
        optimizer: Current optimizer instance
        optimizer_state: Optimizer state dict from checkpoint
        model: Current model to get parameter shapes
    
    Returns:
        Filtered optimizer state dict
    """
    if optimizer_state is None:
        return None
    
    # Create a mapping from parameter to its name for current model
    param_to_name = {}
    model_state_dict = model.state_dict()
    for name, param in model.named_parameters():
        param_to_name[param] = name
    
    # Build mapping: checkpoint param_id -> checkpoint param_state
    checkpoint_state = optimizer_state.get('state', {})
    
    # Build mapping: current param -> checkpoint param_id (by matching shapes)
    # We'll match parameters by their position in param_groups and shape
    filtered_state = {
        'state': {},
        'param_groups': []
    }
    
    # Match parameters by position in param_groups and shape
    if 'param_groups' in optimizer_state:
        checkpoint_param_groups = optimizer_state['param_groups']
        for group_idx, current_group in enumerate(optimizer.param_groups):
            if group_idx < len(checkpoint_param_groups):
                checkpoint_group = checkpoint_param_groups[group_idx]
                checkpoint_param_ids = checkpoint_group.get('params', [])
                
                # Match current params with checkpoint params by position and shape
                current_param_ids = []
                for param_idx, current_param in enumerate(current_group['params']):
                    current_param_id = id(current_param)
                    current_param_ids.append(current_param_id)
                    
                    # Try to find matching checkpoint param by position
                    if param_idx < len(checkpoint_param_ids):
                        checkpoint_param_id = checkpoint_param_ids[param_idx]
                        if checkpoint_param_id in checkpoint_state:
                            param_state = checkpoint_state[checkpoint_param_id]
                            
                            # Check shape compatibility
                            if current_param in param_to_name:
                                param_name = param_to_name[current_param]
                                if param_name in model_state_dict:
                                    model_shape = model_state_dict[param_name].shape
                                    
                                    # Check if shapes match by looking at exp_avg or exp_avg_sq
                                    shape_matches = True
                                    for state_key in ['exp_avg', 'exp_avg_sq']:
                                        if state_key in param_state:
                                            checkpoint_shape = param_state[state_key].shape
                                            if checkpoint_shape != model_shape:
                                                shape_matches = False
                                                # Log for quantizer parameters
                                                if any(pattern in param_name for pattern in ['quan_w_fn.s', 'quan_a_fn.s']):
                                                    logger.info(f"Skipping optimizer state for '{param_name}' "
                                                               f"due to shape mismatch: checkpoint {checkpoint_shape} vs model {model_shape}")
                                                break
                                    
                                    if shape_matches:
                                        filtered_state['state'][current_param_id] = param_state
                
                # Copy param group with updated param IDs
                new_group = current_group.copy()
                new_group['params'] = current_param_ids
                filtered_state['param_groups'].append(new_group)
            else:
                # No matching checkpoint group, use current group as-is
                new_group = current_group.copy()
                new_group['params'] = [id(p) for p in current_group['params']]
                filtered_state['param_groups'].append(new_group)
    
    return filtered_state

@master_only
def save_checkpoint(epoch, arch, model, target_model, optimizer, extras=None, is_best=None, name=None, output_dir='.', lr_scheduler=None, lr_scheduler_q=None, optimizer_q=None, output_corrector=None, corrector_optimizer=None, sensitive_restorer=None, sensitive_optimizer=None, sensitive_lr_scheduler=None):
    """Save a pyTorch training checkpoint
    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pyTorch model
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        output_dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(output_dir):
        raise IOError('Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    filepath = os.path.join(output_dir, filename)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    filepath_best = os.path.join(output_dir, filename_best)

    checkpoint = {
        'epoch': epoch,
        'state_dict': get_state_dict(model), 
        'arch': arch,
        'extras': extras,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
        'lr_scheduler_q': lr_scheduler_q.state_dict() if lr_scheduler_q else None,
        'optimizer_q': optimizer_q.state_dict() if optimizer_q else None,
    }

    if target_model is not None:
        # Check if target_model has ema attribute (for EMA models)
        if hasattr(target_model, 'ema') and target_model.ema is not None:
            checkpoint['state_dict_ema'] = get_state_dict(unwrap_model(target_model.ema))
        else:
            checkpoint['state_dict_ema'] = None
    else:
        checkpoint['state_dict_ema'] = None

    if output_corrector is not None:
        checkpoint['output_corrector'] = output_corrector.state_dict()
    if corrector_optimizer is not None:
        checkpoint['corrector_optimizer'] = corrector_optimizer.state_dict()
    if sensitive_restorer is not None:
        checkpoint['sensitive_restorer'] = sensitive_restorer.state_dict()
    if sensitive_optimizer is not None:
        checkpoint['sensitive_optimizer'] = sensitive_optimizer.state_dict()
    if sensitive_lr_scheduler is not None:
        checkpoint['sensitive_lr_scheduler'] = sensitive_lr_scheduler.state_dict()

    msg = '([%d] Epoch) Saving checkpoint to:\n' % epoch
    msg += '             Current: %s\n' % filepath
    torch.save(checkpoint, filepath)
    if is_best:
        msg += '                Best: %s\n' % filepath_best
        torch.save(checkpoint, filepath_best)
    logger.info(msg)


def load_checkpoint(model:nn.Module, chkp_file, model_device=None, strict=True, lean=False, optimizer=None, override_optim=False, lr_scheduler=None, lr_scheduler_q=None, optimizer_q=None, output_corrector=None, corrector_optimizer=None, sensitive_restorer=None, sensitive_optimizer=None):
    """Load a pyTorch training checkpoint.
    Args:
        model: the pyTorch model to which we will load the parameters.  You can
        specify model=None if the checkpoint contains enough metadata to infer
        the model.  The order of the arguments is misleading and clunky, and is
        kept this way for backward compatibility.
        chkp_file: the checkpoint file
        lean: if set, read into model only 'state_dict' field
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, optimizer, start_epoch
    """
    if not os.path.isfile(chkp_file):
        raise IOError('Cannot find a checkpoint at', chkp_file)

    checkpoint = torch.load(chkp_file, map_location=lambda storage, loc: storage)

    if 'state_dict' not in checkpoint:
        raise ValueError('Checkpoint must contain model parameters')

    extras = checkpoint.get('extras', None)
    arch = checkpoint.get('arch', '_nameless_')

    # Use unwrap_model to handle both DDP and non-DDP cases
    # Need to unwrap early so we can use it for optimizer state filtering
    unwrapped_model = unwrap_model(model)

    # optimizer_state = checkpoint.get('optimizer', None)
    # if optimizer is not None and optimizer_state is not None and not override_optim:
    #     optimizer.load_state_dict(optimizer_state)

    optimizer_q_state = checkpoint.get('optimizer_q', None)
    if optimizer_q is not None and optimizer_q_state is not None and not override_optim:
        # Filter optimizer state to remove parameters with shape mismatch
        # This is necessary when model configuration changed (e.g., classifier.1 from dynamic to fixed bit)
        filtered_optimizer_q_state = filter_optimizer_state(optimizer_q, optimizer_q_state, unwrapped_model)
        try:
            optimizer_q.load_state_dict(filtered_optimizer_q_state)
        except Exception as e:
            logger.warning(f"Failed to load optimizer_q state: {e}. Continuing with fresh optimizer state.")
    
    lr_scheduler_state = checkpoint.get('lr_scheduler', None)
    if lr_scheduler is not None and lr_scheduler_state is not None:
        lr_scheduler.load_state_dict(lr_scheduler_state)

    lr_scheduler_q_state = checkpoint.get('lr_scheduler_q', None)
    if lr_scheduler_q is not None and lr_scheduler_q_state is not None:
        lr_scheduler_q.load_state_dict(lr_scheduler_q_state)
    
    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
    
    for name, module in unwrapped_model.named_modules():
        if isinstance(module, QuanConv2d) and hasattr(module, 'current_bit_cands') and name + '.' + 'current_bit_cands' in checkpoint['state_dict']:
            module.current_bit_cands = torch.ones(len(checkpoint['state_dict'][name + '.' + 'current_bit_cands']), device=module.weight.device, dtype=torch.int32)
        
        if isinstance(module, QuanConv2d) and hasattr(module, 'current_bit_cands_w') and name + '.' + 'current_bit_cands_w' in checkpoint['state_dict']:
            module.current_bit_cands_w = torch.ones(len(checkpoint['state_dict'][name + '.' + 'current_bit_cands_w']), device=module.weight.device, dtype=torch.int32) 
        
        if isinstance(module, QuanConv2d) and hasattr(module, 'current_bit_cands_a') and name + '.' + 'current_bit_cands_a' in checkpoint['state_dict']:
            module.current_bit_cands_a = torch.ones(len(checkpoint['state_dict'][name + '.' + 'current_bit_cands_a']), device=module.weight.device, dtype=torch.int32) 
            
        # if isinstance(module, QuanConv2d) 

    # Filter out parameters with shape mismatch before loading
    # This is necessary because load_state_dict with strict=False still raises errors on shape mismatch
    filtered_state_dict = {}
    model_state_dict = unwrapped_model.state_dict()
    
    for key, value in checkpoint['state_dict'].items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                # For quantizer parameters (quan_*.s, init_state), allow shape mismatch
                # They will be auto-initialized during forward pass if needed
                is_quantizer_param = any(pattern in key for pattern in ['quan_w_fn.s', 'quan_a_fn.s', 'quan_w_fn.init_state', 'quan_a_fn.init_state'])
                if is_quantizer_param:
                    logger.info(f"Skipping quantizer parameter '{key}' due to shape mismatch: "
                               f"checkpoint shape {value.shape} vs model shape {model_state_dict[key].shape} "
                               f"(will be auto-initialized during forward pass)")
                else:
                    logger.warning(f"Skipping parameter '{key}' due to shape mismatch: "
                                 f"checkpoint shape {value.shape} vs model shape {model_state_dict[key].shape}")
        else:
            # Key not in model, skip it
            pass
    
    # Load filtered state dict
    missing_keys, unexpected_keys = unwrapped_model.load_state_dict(filtered_state_dict, strict=False)
    
    # Filter out optional buffers from missing_keys (e.g., current_bit_cands added for QuanLinear)
    # These buffers are OK to be missing from old checkpoints
    # Also allow quantizer parameters (quan_w_fn.s, quan_a_fn.s, init_state) to be missing
    # This happens when a layer was previously fixed_bits (excepts) but is now dynamic bits
    # Example: classifier.1 in AlexNet V2 was fixed 8-bit in old checkpoint, now dynamic bits
    optional_buffer_patterns = [
        'current_bit_cands', 
        'current_bit_cands_w', 
        'current_bit_cands_a',
        'quan_w_fn.s',  # Weight quantizer scale parameter
        'quan_w_fn.init_state',  # Weight quantizer init state
        'quan_a_fn.s',  # Activation quantizer scale parameter
        'quan_a_fn.init_state'  # Activation quantizer init state
    ]
    if missing_keys:
        missing_keys_filtered = [k for k in missing_keys if not any(pattern in k for pattern in optional_buffer_patterns)]
        if len(missing_keys_filtered) < len(missing_keys):
            ignored_count = len(missing_keys) - len(missing_keys_filtered)
            ignored_keys = [k for k in missing_keys if any(pattern in k for pattern in optional_buffer_patterns)]
            logger.info(f"Ignoring {ignored_count} missing optional buffers/parameters that were added after checkpoint was saved:")
            for key in ignored_keys[:5]:  # Show first 5
                logger.info(f"  - {key}")
            if len(ignored_keys) > 5:
                logger.info(f"  ... and {len(ignored_keys) - 5} more")
            logger.info(f"Note: Missing quantizer parameters (quan_*.s, init_state) will be auto-initialized during forward pass")
        missing_keys = missing_keys_filtered
    
    anomalous_keys = (missing_keys, unexpected_keys)

    if strict:
        if anomalous_keys:
            missing_keys, unexpected_keys = anomalous_keys
            if unexpected_keys:
                logger.warning("The loaded checkpoint (%s) contains %d unexpected state keys" %
                            (chkp_file, len(unexpected_keys)))
            if missing_keys:
                print(missing_keys)
                raise ValueError("The loaded checkpoint (%s) is missing %d state keys" %
                                (chkp_file, len(missing_keys)))
            

    model.cuda()

    if output_corrector is not None and 'output_corrector' in checkpoint:
        try:
            output_corrector.load_state_dict(checkpoint['output_corrector'])
            logger.info("Loaded output corrector state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load output corrector state: {e}. Continuing with fresh corrector state.")

    if corrector_optimizer is not None and 'corrector_optimizer' in checkpoint and not override_optim:
        try:
            corrector_optimizer.load_state_dict(checkpoint['corrector_optimizer'])
            logger.info("Loaded corrector optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load corrector optimizer state: {e}. Continuing with fresh optimizer state.")

    if sensitive_restorer is not None and 'sensitive_restorer' in checkpoint:
        try:
            sensitive_restorer.load_state_dict(checkpoint['sensitive_restorer'])
            logger.info("Loaded sensitive restorer state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load sensitive restorer state: {e}. Continuing with fresh state.")

    if sensitive_optimizer is not None and 'sensitive_optimizer' in checkpoint and not override_optim:
        try:
            sensitive_optimizer.load_state_dict(checkpoint['sensitive_optimizer'])
            logger.info("Loaded sensitive optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load sensitive optimizer state: {e}. Continuing with fresh optimizer state.")

    if lean:
        logger.info("Loaded checkpoint %s model (next epoch %d) from %s", arch, 0, chkp_file)
        return model, 0, None
    else:
        logger.info("Loaded checkpoint %s model (next epoch %d) from %s", arch, start_epoch, chkp_file)
        return model, start_epoch, extras
