#!/usr/bin/env python3
"""
Export SR-QAT penalty-related statistics from sweep checkpoints.

Goal: quantify what SR-QAT was designed to control: sum of *active* LSQ weight scales squared.

We approximate the training-time SR-QAT numerator:
    sum_s2 = sum_over_layers (s_active(layer, wbits)^2).sum()

Because this is a checkpoint-only analysis (no model build), we infer:
- If a layer has a scalar/vector `*.quan_w_fn.s`:
  - If len==1: active scale is that scalar.
  - If len>1: active scale is chosen by `--eval_wbit` using the bit_list order (default [6,5,4,3,2]).

Outputs:
- out_dir/summary_active_s2.csv  (per exp)
- out_dir/layers_active_scale.csv (per exp, per layer)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def _infer_ckpt_path_from_cfg(cfg: Dict[str, Any]) -> Path:
    out_dir = str(cfg.get("output_dir", "training"))
    name = str(cfg["name"])
    return REPO_ROOT / out_dir / name / f"{name}_checkpoint.pth.tar"


def _load_generated_cfgs(out_config_dir: Path, glob_prefix: str) -> List[Path]:
    # e.g., alexnet_cifar10_single_gpu_normal_srqat_ls*.yaml
    return sorted(out_config_dir.glob(f"{glob_prefix}_ls*.yaml"))


def _find_layer_prefixes_with_scales(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    prefixes = []
    for k in state_dict.keys():
        if k.endswith(".quan_w_fn.s"):
            prefix = k[: -len(".quan_w_fn.s")]
            if f"{prefix}.weight" in state_dict:
                prefixes.append(prefix)
    return sorted(set(prefixes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_config",
        type=str,
        default="configs/training/train_alexnet_cifar10_single_gpu_normal_srqat.yaml",
        help="Base training YAML config (relative to repo root). Used to infer experiment name prefix.",
    )
    ap.add_argument(
        "--out_config_dir",
        type=str,
        default="configs/training/srqat_sweep",
        help="Directory containing generated sweep configs (relative to repo root).",
    )
    ap.add_argument(
        "--bit_list",
        type=int,
        nargs="+",
        default=[6, 5, 4, 3, 2],
        help="Bit-width candidates order used to index s vector for dynamic layers.",
    )
    ap.add_argument(
        "--eval_wbit",
        type=int,
        default=6,
        help="Weight bit-width assumed active during eval (used to pick s index for dynamic layers).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="training/srqat_scale_weight_analysis",
        help="Output directory (relative to repo root).",
    )
    args = ap.parse_args()

    base_cfg_path = (REPO_ROOT / args.base_config).resolve()
    out_cfg_dir = (REPO_ROOT / args.out_config_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _read_yaml(base_cfg_path)
    base_name = str(base_cfg.get("name", "exp"))
    glob_prefix = base_name  # configs are named f"{base_name}_ls{tag}.yaml"

    cfg_paths = _load_generated_cfgs(out_cfg_dir, glob_prefix=glob_prefix)
    if not cfg_paths:
        raise FileNotFoundError(
            f"No generated sweep configs found under {out_cfg_dir} with prefix {glob_prefix}_ls*.yaml.\n"
            f"Hint: run tools/sweep_srqat.py once to generate configs, or pass correct --out_config_dir."
        )

    bit_to_idx = {b: i for i, b in enumerate(args.bit_list)}
    if args.eval_wbit not in bit_to_idx:
        raise ValueError(f"--eval_wbit={args.eval_wbit} not in --bit_list={args.bit_list}")

    summary_rows: List[Dict[str, object]] = []
    layer_rows: List[Dict[str, object]] = []

    for cfg_path in cfg_paths:
        cfg = _read_yaml(cfg_path)
        exp_name = str(cfg["name"])
        lam = float(cfg.get("scale_penalty", {}).get("lambda_scale", 0.0))
        ckpt_path = _infer_ckpt_path_from_cfg(cfg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd: Dict[str, torch.Tensor] = ckpt.get("state_dict", {})
        if not isinstance(sd, dict) or not sd:
            raise RuntimeError(f"Checkpoint state_dict missing/empty: {ckpt_path}")

        prefixes = _find_layer_prefixes_with_scales(sd)
        sum_s2_active_all = 0.0
        sum_s2_active_dynamic = 0.0
        n_layers = 0
        n_dynamic = 0

        for layer in prefixes:
            s = sd[f"{layer}.quan_w_fn.s"].detach().float().view(-1)
            if s.numel() == 1:
                s_active = s[0]
                is_dynamic = False
            else:
                s_active = s[bit_to_idx[args.eval_wbit]]
                is_dynamic = True

            s_active_val = float(s_active.item())
            s2 = float((s_active * s_active).item())
            sum_s2_active_all += s2
            if is_dynamic:
                sum_s2_active_dynamic += s2
                n_dynamic += 1
            n_layers += 1

            layer_rows.append(
                {
                    "exp_name": exp_name,
                    "lambda_scale": lam,
                    "layer": layer,
                    "is_dynamic_s": int(is_dynamic),
                    "s_len": int(s.numel()),
                    "s_active_wbit": int(args.eval_wbit) if is_dynamic else "",
                    "s_active": s_active_val,
                    "s_active_sq": s2,
                    "ckpt_path": str(ckpt_path.relative_to(REPO_ROOT)),
                }
            )

        summary_rows.append(
            {
                "exp_name": exp_name,
                "lambda_scale": lam,
                "eval_wbit": int(args.eval_wbit),
                "num_layers_with_s": n_layers,
                "num_dynamic_s_layers": n_dynamic,
                "sum_s2_active_all_layers": sum_s2_active_all,
                "sum_s2_active_dynamic_layers": sum_s2_active_dynamic,
                "ckpt_path": str(ckpt_path.relative_to(REPO_ROOT)),
            }
        )

    # Write outputs
    summary_path = out_dir / "summary_active_s2.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    layers_path = out_dir / "layers_active_scale.csv"
    with layers_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        w.writerows(layer_rows)

    print(f"[OK] Wrote: {summary_path.relative_to(REPO_ROOT)}")
    print(f"[OK] Wrote: {layers_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()


