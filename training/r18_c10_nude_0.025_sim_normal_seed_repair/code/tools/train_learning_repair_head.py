#!/usr/bin/env python3
"""
Train learning-based repair heads for the sensitive-layer restorer.

The script collects clean and fault activations for selected layers,
optimizes the learning repair head (MLP) to minimize MSE between repaired
fault activations and clean activations, and saves the resulting state dict.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector
from util.utils import preprocess_model
from util.mpq import switch_bit_width, switch_bit_width_bn
from util.sensitive_layer_restorer import create_sensitive_layer_restorer


class ActivationRecorder:
    def __init__(self, model: torch.nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.hooks = []
        self.outputs: Dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.outputs[name] = output.detach().clone()
        return hook

    def register(self):
        modules = dict(self.model.named_modules())
        for name in self.target_layers:
            module = modules.get(name)
            if module is None:
                continue
            self.hooks.append(module.register_forward_hook(self._make_hook(name)))

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def capture(self, inputs: torch.Tensor, device: torch.device):
        self.outputs.clear()
        self.register()
        with torch.no_grad():
            _ = self.model(inputs.to(device))
        self.remove()
        return {k: v.to(device) for k, v in self.outputs.items()}


def parse_args():
    parser = argparse.ArgumentParser(description='Train learning-based repair heads')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--stage1_ckpt', type=str, required=True)
    parser.add_argument('--layer_profile', type=str, required=True)
    parser.add_argument('--fault_layer_profile', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--target_layers', type=str, required=True,
                        help='Comma-separated layer names')
    parser.add_argument('--bit_width_config', type=str, default=None)
    parser.add_argument('--config_index', type=int, default=0)
    parser.add_argument('--force_w2a2', action='store_true')
    parser.add_argument('--force_w2a6', action='store_true')
    parser.add_argument('--skip_first_last', action='store_true')
    parser.add_argument('--ber', type=float, default=1e-1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batches_per_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--output', type=str, required=True, help='Path to save learning repair head state')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def apply_bitwidth(model, configs, args):
    if args.force_w2a2:
        switch_bit_width(model, configs.quan, wbit=2, abits=2)
        switch_bit_width_bn(model, 2, 2)
    elif args.force_w2a6:
        switch_bit_width(model, configs.quan, wbit=2, abits=6)
        switch_bit_width_bn(model, 2, 6)
    elif args.bit_width_config:
        from util.fault_injector import setup_model_with_bit_width_config
        setup_model_with_bit_width_config(model, args.bit_width_config, config_index=args.config_index, verbose=True)
    else:
        target_bits = configs.target_bits if hasattr(configs, 'target_bits') else [6, 5, 4, 3, 2]
        max_bit = max(target_bits)
        switch_bit_width(model, configs.quan, wbit=max_bit, abits=max_bit)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    original_argv = sys.argv.copy()
    sys.argv = ['train_learning_repair_head', args.config]
    try:
        configs = get_config(args.config)
    finally:
        sys.argv = original_argv

    configs.post_training_batchnorm_calibration = False

    device = torch.device(args.device)

    model = create_model(configs.arch, dataset=configs.dataloader.dataset)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model.to(device)
    load_checkpoint(model, args.stage1_ckpt, model_device=str(device), strict=False)
    apply_bitwidth(model, configs, args)
    model.eval()

    target_layers = [name.strip() for name in args.target_layers.split(',') if name.strip()]
    restorer = create_sensitive_layer_restorer(
        model=model,
        profile_path=args.layer_profile,
        fault_profile_path=args.fault_layer_profile,
        fault_profile_ber=args.ber,
        target_layers=target_layers,
        repair_mode='mlp',
    )

    params = restorer.learning_parameters()
    if not params:
        raise RuntimeError('No learnable parameters found. Ensure repair_mode=mlp.')
    optimizer = torch.optim.Adam(params, lr=args.lr)

    fault_injector = FaultInjector(
        model=model,
        mode='ber',
        ber=args.ber,
        device=device,
        enable_in_inference=True,
        skip_first_last=args.skip_first_last,
        seed=args.seed,
        seed_list=None,
    )

    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, configs.arch)
    recorder = ActivationRecorder(model, target_layers)

    for epoch in range(args.epochs):
        iterator = iter(train_loader)
        for batch_idx in range(args.batches_per_epoch):
            try:
                inputs, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, _ = next(iterator)

            inputs = inputs.to(device)

            clean_outputs = recorder.capture(inputs, device)

            fault_injector.enable()
            fault_outputs = recorder.capture(inputs, device)
            fault_injector.disable()

            restorer.set_ber(args.ber)

            loss = 0.0
            for name in target_layers:
                clean = clean_outputs[name]
                faulty = fault_outputs[name]
                repaired = restorer.repair_manual(name, faulty)
                loss = loss + F.mse_loss(repaired, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{args.batches_per_epoch}, Loss {loss.item():.6f}")

    restorer.save_learning_state(args.output)
    print(f"Saved learning repair head state to {args.output}")


if __name__ == '__main__':
    main()

