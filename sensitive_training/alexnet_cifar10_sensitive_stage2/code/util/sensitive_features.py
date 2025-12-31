import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from quan.func import QuanConv2d, QuanLinear


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
    train_loader,
    sensitive_info: Dict[str, Dict[str, List[int]]],
    device,
    max_batches: int = 50,
    output_path: str = None,
):
    training_model = _get_training_module(model)
    training_model.eval()
    hooks = []
    buffers: Dict[str, torch.Tensor] = {}

    def _make_hook(name):
        def hook(module, input, output):
            buffers[name] = output.detach()
        return hook

    modules = dict(training_model.named_modules())
    for name in sensitive_info.keys():
        if name in modules:
            hooks.append(modules[name].register_forward_hook(_make_hook(name)))

    stats = {}
    total_count = 0
    for batch_idx, (inputs, _) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        buffers.clear()
        with torch.no_grad():
            training_model(inputs)
        batch_size = inputs.size(0)
        total_count += batch_size
        for name, info in sensitive_info.items():
            if name not in buffers:
                continue
            activations = buffers[name]
            idx = info["indices"]
            if not idx:
                continue
            act_sel = activations[:, idx]
            if act_sel.dim() == 4:
                energy = act_sel.pow(2).mean(dim=[2, 3])
            else:
                energy = act_sel.pow(2)
            if name not in stats:
                stats[name] = {
                    "sum": torch.zeros_like(energy.sum(dim=0)),
                    "sum_sq": torch.zeros_like(energy.sum(dim=0)),
                    "count": 0,
                }
            stats[name]["sum"] += energy.sum(dim=0)
            stats[name]["sum_sq"] += (energy ** 2).sum(dim=0)
            stats[name]["count"] += energy.size(0)

    for hook in hooks:
        hook.remove()

    baseline = {}
    for name, info in sensitive_info.items():
        idx = info["indices"]
        if not idx or name not in stats:
            continue
        stat = stats[name]
        count = max(stat["count"], 1)
        mean = stat["sum"] / count
        var = stat["sum_sq"] / count - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-6))
        baseline[name] = {
            "indices": idx,
            "mean": mean.cpu(),
            "std": std.cpu(),
        }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(baseline, output_path)

    return baseline

