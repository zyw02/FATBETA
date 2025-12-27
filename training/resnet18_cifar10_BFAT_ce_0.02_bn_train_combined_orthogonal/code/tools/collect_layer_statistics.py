#!/usr/bin/env python3
"""
Layer Statistics Collector

Collects per-layer activation statistics under clean or fault-injected
conditions for downstream restorer design. Focuses on user-specified
target layers (typically the fault-sensitive ones).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from quan.func import QuanConv2d, QuanLinear
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
from util.mpq import switch_bit_width, switch_bit_width_bn
from util.utils import preprocess_model


class ChannelStats:
    """
    Streaming per-channel statistics with optional sample reservoir.
    """

    def __init__(self, sample_size: int = 4096, sample_stride: int = 512):
        self.sample_size = sample_size
        self.sample_stride = sample_stride

        self.count: Optional[torch.Tensor] = None
        self.sum: Optional[torch.Tensor] = None
        self.sum_sq: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None
        self.min_val: Optional[torch.Tensor] = None
        self.max_abs: Optional[torch.Tensor] = None

        self.sample_values: Optional[torch.Tensor] = None

    def _init_buffers(self, num_channels: int, device: torch.device):
        dtype = torch.float64
        zeros = torch.zeros(num_channels, dtype=dtype, device=device)
        self.count = zeros.clone()
        self.sum = zeros.clone()
        self.sum_sq = zeros.clone()
        self.max_val = torch.full((num_channels,), -torch.inf, dtype=dtype, device=device)
        self.min_val = torch.full((num_channels,), torch.inf, dtype=dtype, device=device)
        self.max_abs = torch.full((num_channels,), 0.0, dtype=dtype, device=device)

    def update(self, tensor: torch.Tensor):
        if tensor.dim() not in (2, 4):
            return

        with torch.no_grad():
            data = tensor.detach().to(dtype=torch.float32, device='cpu')
            if data.dim() == 4:
                b, c, h, w = data.shape
                reshaped = data.permute(1, 0, 2, 3).contiguous().view(c, -1)
            else:
                b, c = data.shape
                reshaped = data.transpose(0, 1).contiguous().view(c, -1)

            if reshaped.numel() == 0:
                return

            num_channels = reshaped.size(0)
            if self.count is None:
                self._init_buffers(num_channels, device=reshaped.device)

            chunk_count = reshaped.size(1)
            chunk_sum = reshaped.sum(dim=1, dtype=torch.float64)
            chunk_sq = (reshaped.to(dtype=torch.float64) ** 2).sum(dim=1)
            chunk_max = reshaped.max(dim=1).values.to(dtype=torch.float64)
            chunk_min = reshaped.min(dim=1).values.to(dtype=torch.float64)
            chunk_abs = reshaped.abs().max(dim=1).values.to(dtype=torch.float64)

            self.count += chunk_count
            self.sum += chunk_sum
            self.sum_sq += chunk_sq
            self.max_val = torch.maximum(self.max_val, chunk_max)
            self.min_val = torch.minimum(self.min_val, chunk_min)
            self.max_abs = torch.maximum(self.max_abs, chunk_abs)

            if self.sample_size > 0 and self.sample_stride > 0:
                flat = reshaped.view(-1)
                take = min(self.sample_stride, flat.numel())
                if take > 0:
                    idx = torch.randperm(flat.numel())[:take]
                    samples = flat[idx].to(dtype=torch.float32)
                    if self.sample_values is None:
                        self.sample_values = samples[: self.sample_size]
                    else:
                        remaining = self.sample_size - self.sample_values.numel()
                        if remaining > 0:
                            append = samples[:remaining]
                            self.sample_values = torch.cat([self.sample_values, append], dim=0)

    def to_dict(self) -> Dict:
        if self.count is None or (self.count == 0).all():
            return {}

        mean = self.sum / self.count.clamp(min=1)
        mean_sq = self.sum_sq / self.count.clamp(min=1)
        var = torch.clamp(mean_sq - mean ** 2, min=0.0)
        std = torch.sqrt(var)
        energy = mean_sq

        stats = {
            'channel_mean': mean.float(),
            'channel_std': std.float(),
            'channel_energy': energy.float(),
            'channel_max': self.max_val.float(),
            'channel_min': self.min_val.float(),
            'channel_max_abs': self.max_abs.float(),
            'num_elements_per_channel': self.count.clone(),
        }

        if self.sample_values is not None:
            stats['sample_values'] = self.sample_values.clone()

        return stats


class LayerStatisticsCollector:
    """
    Registers forward hooks on selected layers and aggregates statistics.
    """

    def __init__(self, model: nn.Module, target_layers: List[str], sample_size: int, sample_stride: int):
        self.model = model
        self.target_layers = set(target_layers)
        self.sample_size = sample_size
        self.sample_stride = sample_stride

        self.stats: Dict[str, ChannelStats] = defaultdict(
            lambda: ChannelStats(sample_size=self.sample_size, sample_stride=self.sample_stride)
        )
        self.hooks = []
        self.activation_dumper = None

    def register(self):
        modules = dict(self.model.named_modules())
        for name in self.target_layers:
            module = modules.get(name)
            if module is None:
                continue
            if not isinstance(module, (nn.Conv2d, nn.Linear, QuanConv2d, QuanLinear)):
                continue

            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            self.stats[layer_name].update(output)

        return hook

    def export(self) -> Dict[str, Dict]:
        result = {}
        for name, stat in self.stats.items():
            stat_dict = stat.to_dict()
            if stat_dict:
                result[name] = stat_dict
        return result


class ActivationDumper:
    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        save_dir: Path,
        mode: str,
        max_batches: int,
        dtype: str = 'fp16',
    ):
        self.model = model
        self.layers = set(layers)
        self.save_dir = save_dir
        self.mode = mode
        self.max_batches = max_batches
        self.dtype = dtype
        self.hooks = []
        self.current_batch = 0
        self.saved_counts = defaultdict(int)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_batch_index(self, batch_idx: int):
        self.current_batch = batch_idx

    def register(self):
        modules = dict(self.model.named_modules())
        for name in self.layers:
            module = modules.get(name)
            if module is None or not isinstance(module, (nn.Conv2d, nn.Linear, QuanConv2d, QuanLinear)):
                continue
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _convert_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dtype == 'fp32':
            return tensor.detach().cpu().float()
        if self.dtype == 'bf16':
            return tensor.detach().cpu().to(torch.bfloat16)
        return tensor.detach().cpu().to(torch.float16)

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if self.max_batches > 0 and self.saved_counts[layer_name] >= self.max_batches:
                return output
            tensor = self._convert_dtype(output)
            fname = f"{layer_name.replace('.', '_')}_{self.mode}_batch{self.current_batch:04d}.pt"
            torch.save(
                {
                    'layer': layer_name,
                    'mode': self.mode,
                    'batch': self.current_batch,
                    'tensor': tensor,
                },
                self.save_dir / fname,
            )
            self.saved_counts[layer_name] += 1
            return output

        return hook


def parse_args():
    parser = argparse.ArgumentParser(description='Collect per-layer activation statistics')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--stage1_ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--mode', type=str, default='clean', choices=['clean', 'fault'])
    parser.add_argument('--output', type=str, default='layer_profiles/clean_stats.pt', help='Output file path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_batches', type=int, default=100,
                        help='Number of batches to collect (0 for full dataset)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target_layers', type=str, default=None,
                        help='Comma-separated layer names to monitor')
    parser.add_argument('--sample_size', type=int, default=4096, help='Reservoir size per layer')
    parser.add_argument('--sample_stride', type=int, default=512, help='Samples per batch per layer')
    parser.add_argument('--use_train_split', action='store_true',
                        help='Force using training split even when not saving activations')

    # Bit-width options
    parser.add_argument('--bit_width_config', type=str, default=None)
    parser.add_argument('--config_index', type=int, default=0)
    parser.add_argument('--force_w2a2', action='store_true')
    parser.add_argument('--force_w2a6', action='store_true')
    parser.add_argument('--force_w6a6', action='store_true')
    parser.add_argument('--skip_first_last', action='store_true')

    # Fault injection options
    parser.add_argument('--ber', type=float, default=1e-1)

    # Activation dump options
    parser.add_argument('--save_activations', action='store_true',
                        help='Dump raw activations for target layers')
    parser.add_argument('--activations_dir', type=str, default=None,
                        help='Directory to store dumped activations')
    parser.add_argument('--activation_dtype', type=str, default='fp16',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='Precision of stored activations')
    parser.add_argument('--max_activation_batches', type=int, default=20,
                        help='Maximum batches per layer to dump activations (<=0 for no limit)')

    return parser.parse_args()


def build_config(args) -> torch.device:
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['collect_layer_statistics.py', args.config]
    try:
        configs = get_config(args.config)
    finally:
        sys.argv = original_argv

    defaults = {
        'local_rank': 0,
        'rank': 0,
        'world_size': 1,
        'enable_dynamic_bit_training': True,
        'split_aw_cands': False,
        'smoothing': 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(configs, key):
            setattr(configs, key, value)

    return configs


def resolve_target_layers(model: nn.Module, provided: Optional[str]) -> List[str]:
    if provided:
        return [name.strip() for name in provided.split(',') if name.strip()]

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, QuanConv2d, QuanLinear)):
            layers.append(name)
    return layers


def apply_bitwidth_overrides(model, configs, args):
    if args.force_w2a2:
        switch_bit_width(model, configs.quan, wbit=2, abits=2)
        switch_bit_width_bn(model, 2, 2)
        return 'force_w2a2'
    if args.force_w2a6:
        switch_bit_width(model, configs.quan, wbit=2, abits=6)
        switch_bit_width_bn(model, 2, 6)
        return 'force_w2a6'
    if args.force_w6a6:
        switch_bit_width(model, configs.quan, wbit=6, abits=6)
        switch_bit_width_bn(model, 6, 6)
        return 'force_w6a6'
    if args.bit_width_config:
        setup_model_with_bit_width_config(
            model,
            args.bit_width_config,
            config_index=args.config_index,
            verbose=True
        )
        return Path(args.bit_width_config).name

    target_bits = configs.target_bits if hasattr(configs, 'target_bits') else [6, 5, 4, 3, 2]
    max_bit = max(target_bits)
    switch_bit_width(model, configs.quan, wbit=max_bit, abits=max_bit)
    return f'max_{max_bit}bit'


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    configs = build_config(args)
    device = torch.device(args.device)

    model = create_model(configs.arch, dataset=configs.dataloader.dataset)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model.to(device)
    load_checkpoint(model, args.stage1_ckpt, model_device=str(device), strict=False)
    model.eval()

    bitwidth_tag = apply_bitwidth_overrides(model, configs, args)

    train_loader, val_loader, test_loader, _, _ = init_dataloader(configs.dataloader, configs.arch)
    if args.save_activations or args.use_train_split:
        data_loader = train_loader
        print("⚠️ 保存激活或指定 use_train_split 时，使用训练集以避免数据泄漏。")
    else:
        data_loader = test_loader

    target_layers = resolve_target_layers(model, args.target_layers)
    if not target_layers:
        raise ValueError('No target layers specified or discovered.')

    collector = LayerStatisticsCollector(
        model=model,
        target_layers=target_layers,
        sample_size=args.sample_size,
        sample_stride=args.sample_stride,
    )
    collector.register()
    activation_dumper = None
    if args.save_activations:
        dump_dir = Path(args.activations_dir) if args.activations_dir else Path('activation_dumps')
        dump_dir = dump_dir / args.mode
        max_batches = args.max_activation_batches
        if max_batches is None:
            max_batches = 0
        if max_batches <= 0:
            max_batches = 0
        activation_dumper = ActivationDumper(
            model=model,
            layers=target_layers,
            save_dir=dump_dir,
            mode=args.mode,
            max_batches=max_batches,
            dtype=args.activation_dtype,
        )
        activation_dumper.register()
        collector.activation_dumper = activation_dumper

    injector = None
    if args.mode == 'fault':
        injector = FaultInjector(
            model=model,
            mode='ber',
            ber=args.ber,
            device=device,
            enable_in_inference=True,
            skip_first_last=args.skip_first_last,
            seed=args.seed,
            seed_list=None,
        )
        injector.enable()

    processed_batches = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if args.num_batches and processed_batches >= args.num_batches:
                break

            inputs = inputs.to(device)
            if activation_dumper:
                activation_dumper.set_batch_index(processed_batches)
            _ = model(inputs)
            processed_batches += 1

            if processed_batches % 10 == 0:
                print(f'Processed {processed_batches} batches...')

    if injector is not None:
        injector.disable()

    collector.remove()
    if activation_dumper is not None:
        activation_dumper.remove()

    stats = collector.export()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 诊断信息：显示哪些层被成功收集
    collected_layers = set(stats.keys())
    missing_layers = set(target_layers) - collected_layers
    if missing_layers:
        print(f'[Warning] {len(missing_layers)} target layers were not collected: {sorted(missing_layers)}')
        print(f'[Warning] This may indicate that these layers were not called during forward pass or had no data.')
    print(f'[Stats] Successfully collected {len(collected_layers)} layers: {sorted(collected_layers)}')

    payload = {
        'meta': {
            'mode': args.mode,
            'ber': args.ber if args.mode == 'fault' else 0.0,
            'num_batches': processed_batches,
            'bitwidth': bitwidth_tag,
            'target_layers': target_layers,
            'stage1_ckpt': args.stage1_ckpt,
            'config': args.config,
        },
        'layers': stats,
    }

    torch.save(payload, output_path)
    print(f'[Stats] Saved to {output_path} (layers={len(stats)})')


if __name__ == '__main__':
    main()

