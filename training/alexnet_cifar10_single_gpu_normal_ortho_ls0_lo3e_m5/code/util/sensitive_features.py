import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from quan.func import QuanConv2d, QuanLinear
from tqdm import tqdm


def _get_training_module(model):
    return model.module if hasattr(model, "module") else model


def collect_gradient_sensitivity(
    model,
    train_loader,
    criterion,
    device,
    topk_per_layer: int = 8,
    max_batches: int = 100,
    output_path: str = None,
):
    training_model = _get_training_module(model)
    training_model.train()
    sensitivity = {}
    channel_sizes = {}

    target_modules = []
    for name, module in training_model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if not module.weight.requires_grad:
                continue
            target_modules.append((name, module))
            out_channels = module.weight.shape[0]
            sensitivity[name] = torch.zeros(out_channels, device=device)
            channel_sizes[name] = out_channels

    if not target_modules:
        raise RuntimeError("No quantized modules found for sensitivity collection.")

    # --- CRITICAL FIX 3.0: THE COMBINED, DEFINITIVE SOLUTION ---

    # 1. Set model to .train() mode. This is ESSENTIAL to get the differentiable 
    #    'fake_quant' forward path from the custom QAT modules.
    model.train()

    # 2. Selectively disable stochastic modules like Dropout for deterministic analysis.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    # 3. Force-enable gradients on ALL parameters. The Stage 1 checkpoint is likely
    #    saved in a 'frozen' state (e.g., after bit-width searching or stabilization)
    #    where requires_grad was set to False. This loop overrides that state.
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    layer_grads = {}

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        training_model.zero_grad(set_to_none=True)
        outputs = training_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        with torch.no_grad():
            for name, module in target_modules:
                if module.weight.grad is None:
                    continue
                grad = module.weight.grad
                channel_grad = grad.view(grad.shape[0], -1).abs().mean(dim=1)
                sensitivity[name][: channel_grad.shape[0]] += channel_grad

    sensitive_channels = {}
    for name, scores in sensitivity.items():
        num_channels = scores.numel()
        k = min(topk_per_layer, num_channels)
        values, indices = torch.topk(scores, k, largest=True)
        sensitive_channels[name] = {
            "indices": indices.cpu().tolist(),
            "scores": values.cpu().tolist(),
        }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({"channels": sensitive_channels}, output_path)

    return sensitive_channels


def compute_activation_baseline(
    model,
    data_loader,
    sensitive_channels,
    device,
    max_batches: int = 50,
    output_path: str = None,
):
    """
    Computes the activation baseline for sensitive features.
    This function captures activations from a model's forward pass and
    computes their mean and standard deviation for each sensitive layer.

    Args:
        model: The PyTorch model.
        data_loader: A DataLoader object containing the training data.
        sensitive_channels: A dictionary mapping layer names to their sensitive channel indices.
        device: The device (e.g., 'cuda', 'cpu') to use for computation.
        max_batches: The maximum number of batches to process for baseline calculation.
        output_path: The path to save the computed baseline statistics.

    Returns:
        A dictionary containing the computed baseline statistics for each sensitive layer.
    """
    model.eval()
    
    hooks = []
    activations = {}
    def _make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    modules = dict(model.named_modules())
    for name in sensitive_channels.keys():
        if name in modules:
            hooks.append(modules[name].register_forward_hook(_make_hook(name)))

    # Dynamically determine feature dimensions by doing a single dummy forward pass
    stats = {}
    with torch.no_grad():
        dummy_input = next(iter(data_loader))[0].to(device)
        model(dummy_input) # This populates the 'activations' dict
        
        for layer_name, channel_info in sensitive_channels.items():
            if layer_name not in activations: continue
            
            idx = channel_info['indices']
            if not idx: continue
            
            act_dim = activations[layer_name][:, idx, ...].dim()
            num_features = 16 if act_dim == 4 else 4
            num_channels = len(idx)
            
            stats[layer_name] = {
                'sum': torch.zeros(num_channels, num_features, device=device),
                'sum_sq': torch.zeros(num_channels, num_features, device=device),
                'count': 0
            }
        # Clear activations for the real pass
        activations.clear()

    # Main loop for baseline computation
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(data_loader, desc="Computing Activation Baseline")):
            if i >= max_batches:
                break
            images = images.to(device)
            model(images)
            
            for layer_name, channel_info in sensitive_channels.items():
                if layer_name not in activations: continue
                idx = channel_info['indices']
                if not idx: continue
                
                act_sel = activations[layer_name][:, idx, ...].clone()

                # For convolutional layers, divide the activation map into a 2x2 grid
                if act_sel.dim() == 4:
                    B, C, H, W = act_sel.shape
                    grid_size = 2
                    if H < grid_size or W < grid_size: h_step, w_step = H, W
                    else: h_step, w_step = H // grid_size, W // grid_size

                    all_grid_stats = []
                    for r in range(grid_size):
                        for c in range(grid_size):
                            if H < grid_size or W < grid_size: grid = act_sel
                            else: grid = act_sel[:, :, r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]

                            energy = grid.pow(2).mean(dim=[2, 3])
                            mean_val = grid.mean(dim=[2, 3])
                            std_val = grid.std(dim=[2, 3])
                            max_val = grid.flatten(2).max(dim=2)[0]
                            grid_stats = torch.stack([energy, mean_val, std_val, max_val], dim=-1)
                            all_grid_stats.append(grid_stats)
                    
                    layer_stats = torch.cat(all_grid_stats, dim=-1) # Shape: [B, C, 16]
                
                else: # For linear layers, compute global stats
                    energy = act_sel.pow(2)
                    mean_val = act_sel
                    std_val = torch.zeros_like(mean_val)
                    max_val = act_sel
                    layer_stats = torch.stack([energy, mean_val, std_val, max_val], dim=-1) # Shape: [B, C, 4]
                
                # Accumulate stats for baseline calculation
                stats[layer_name]['sum'] += layer_stats.sum(dim=0)
                stats[layer_name]['sum_sq'] += (layer_stats ** 2).sum(dim=0)
                stats[layer_name]['count'] += images.shape[0] # Use B from images
            
            # Clear activations for next batch
            activations.clear()

    for hook in hooks:
        hook.remove()

    # Calculate final mean and std
    baseline_stats = {}
    for name, data in stats.items():
        count = data['count']
        if count == 0: continue
        mean = data['sum'] / count
        std = ((data['sum_sq'] / count) - mean**2).sqrt()
        # Ensure std is not zero to avoid division by zero later
        std[std < 1e-6] = 1.0 
        
        # Reshape to match the format expected by the collector [num_channels, num_features] -> [num_features, num_channels]
        baseline_stats[name] = {
            'mean': mean.T, 
            'std': std.T
        }

    # Save baseline stats to file
    torch.save(baseline_stats, output_path)
    print(f"Activation baseline stats saved to {output_path}")
    return baseline_stats

