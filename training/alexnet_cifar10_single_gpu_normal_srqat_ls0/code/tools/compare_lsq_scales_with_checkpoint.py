#!/usr/bin/env python3
"""
Compare LSQ quantizer scale/init_state in a loaded model vs values saved in checkpoint.

Why:
- In this repo, LSQ (LsqQuan) can (re-)initialize `s` on the first forward if `init_state==0`.
- load_checkpoint() may skip quantizer params on shape mismatch.
- Different eval scripts may apply bit-width config before/after loading checkpoint, changing what gets loaded.

This tool helps you verify, layer-by-layer, whether `quan_w_fn.s` / `quan_w_fn.init_state`
match checkpoint values after loading, and whether they change after an optional forward pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.fault_injector import setup_model_with_bit_width_config


def _strip_module_prefix(k: str) -> str:
    return k[len("module.") :] if k.startswith("module.") else k


def _collect_quant_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    keys = []
    for k in state_dict.keys():
        # weight/act quantizer scale + init_state
        if any(
            pat in k
            for pat in (
                "quan_w_fn.s",
                "quan_w_fn.init_state",
                "quan_a_fn.s",
                "quan_a_fn.init_state",
            )
        ):
            keys.append(k)
    keys.sort()
    return keys


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff) as python floats."""
    a = a.detach().cpu()
    b = b.detach().cpu()
    diff = (a - b).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _compare_state_dicts(
    ckpt_sd: Dict[str, torch.Tensor],
    model_sd: Dict[str, torch.Tensor],
    *,
    title: str,
    max_lines: int = 80,
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    # Normalize keys: checkpoint may or may not have module. prefix
    ckpt_norm = { _strip_module_prefix(k): v for k, v in ckpt_sd.items() }
    model_norm = { _strip_module_prefix(k): v for k, v in model_sd.items() }

    keys = sorted(set(_collect_quant_keys(ckpt_norm)) | set(_collect_quant_keys(model_norm)))
    missing_in_model = []
    missing_in_ckpt = []
    shape_mismatch = []
    diffs = []

    for k in keys:
        in_ckpt = k in ckpt_norm
        in_model = k in model_norm
        if not in_model and in_ckpt:
            missing_in_model.append(k)
            continue
        if not in_ckpt and in_model:
            missing_in_ckpt.append(k)
            continue
        assert in_ckpt and in_model
        a = ckpt_norm[k]
        b = model_norm[k]
        if a.shape != b.shape:
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
            continue
        max_abs, mean_abs = _tensor_diff(a, b)
        diffs.append((k, max_abs, mean_abs))

    # Sort by max_abs diff desc
    diffs.sort(key=lambda x: x[1], reverse=True)

    print(f"Total quantizer keys (union): {len(keys)}")
    print(f"Missing in model: {len(missing_in_model)}")
    print(f"Missing in checkpoint: {len(missing_in_ckpt)}")
    print(f"Shape mismatch: {len(shape_mismatch)}")

    # Quick init_state sanity
    init_state_zero = 0
    init_state_total = 0
    for k, v in model_norm.items():
        if k.endswith("quan_w_fn.init_state") or k.endswith("quan_a_fn.init_state"):
            init_state_total += 1
            if v.detach().cpu().numel() > 0 and (v.detach().cpu() == 0).any().item():
                init_state_zero += 1
    print(f"Model init_state tensors containing zeros: {init_state_zero}/{init_state_total}")

    if missing_in_model:
        print("\n[Missing in model] (showing up to 20)")
        for k in missing_in_model[:20]:
            print(f"  - {k}")
        if len(missing_in_model) > 20:
            print(f"  ... +{len(missing_in_model) - 20}")

    if missing_in_ckpt:
        print("\n[Missing in checkpoint] (showing up to 20)")
        for k in missing_in_ckpt[:20]:
            print(f"  - {k}")
        if len(missing_in_ckpt) > 20:
            print(f"  ... +{len(missing_in_ckpt) - 20}")

    if shape_mismatch:
        print("\n[Shape mismatch] (showing up to 20)")
        for k, s_ckpt, s_model in shape_mismatch[:20]:
            print(f"  - {k}: ckpt={s_ckpt} vs model={s_model}")
        if len(shape_mismatch) > 20:
            print(f"  ... +{len(shape_mismatch) - 20}")

    print("\n[Top diffs] (showing up to %d)" % max_lines)
    shown = 0
    for k, max_abs, mean_abs in diffs:
        if max_abs == 0.0:
            continue
        print(f"  - {k}: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}")
        shown += 1
        if shown >= max_lines:
            break
    if shown == 0:
        print("  (All matching for keys with same shape.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config (used to build/quantize model).")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pth.tar).")
    ap.add_argument("--bit_width_config", type=str, default=None, help="Optional bit-width config JSON.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    ap.add_argument("--do_forward", action="store_true", help="Run one forward pass to see if s/init_state change.")
    args = ap.parse_args()

    # Load config via get_config (matches other tools)
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    try:
        cfg = get_config(default_file=args.config)
    finally:
        sys.argv = original_argv

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load raw checkpoint for ground truth
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_sd = ckpt.get("state_dict", {})
    if not isinstance(ckpt_sd, dict) or not ckpt_sd:
        raise RuntimeError("Checkpoint has no state_dict or it's empty.")

    # Build + quantize model
    model = create_model(cfg.arch, dataset=cfg.dataloader.dataset, pre_trained=getattr(cfg, "pre_trained", False))
    model = model.to(device)
    modules_to_replace = find_modules_to_quantize(model, cfg)
    replace_module_by_names(model, modules_to_replace)

    # Stage A: model just constructed (pre-load)
    _compare_state_dicts(ckpt_sd, model.state_dict(), title="Stage A: Before load_checkpoint()")

    # Stage B: load checkpoint
    load_checkpoint(model, args.ckpt, model_device=device)
    _compare_state_dicts(ckpt_sd, model.state_dict(), title="Stage B: After load_checkpoint()")

    # Stage C: apply bit-width config (optional)
    if args.bit_width_config:
        setup_model_with_bit_width_config(model, args.bit_width_config, verbose=False)
        _compare_state_dicts(
            ckpt_sd, model.state_dict(), title="Stage C: After setup_model_with_bit_width_config()"
        )

    # Stage D: run one forward (optional)
    if args.do_forward:
        _, _, test_loader, _, _ = init_dataloader(cfg.dataloader, cfg.arch)
        model.eval()
        with torch.no_grad():
            x, _y = next(iter(test_loader))
            x = x.to(device)
            _ = model(x)
        _compare_state_dicts(ckpt_sd, model.state_dict(), title="Stage D: After one forward pass (eval)")


if __name__ == "__main__":
    main()


