#!/usr/bin/env python3
"""
Analyze SR-QAT sweep checkpoints: LSQ weight scales (quan_w_fn.s) and weight norms.

Input:
- A sweep results CSV produced by tools/sweep_srqat.py (contains exp_name, lambda_scale, ckpt_path, etc.)

Output:
- Per-layer CSV with weight norms and scale stats
- Per-experiment summary CSV aggregated across layers

This script is checkpoint/state_dict based (does not require building the model),
so it works even when configs differ (e.g., classifier.1 fixed vs dynamic) and
will surface shape mismatches via s/init_state shapes.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def _to_float(x) -> float:
    return float(x) if x is not None else float("nan")


def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float().cpu()
    return {
        "len": int(t.numel()),
        "min": float(t.min().item()) if t.numel() else float("nan"),
        "max": float(t.max().item()) if t.numel() else float("nan"),
        "mean": float(t.mean().item()) if t.numel() else float("nan"),
        "std": float(t.std(unbiased=False).item()) if t.numel() else float("nan"),
    }


def _weight_norms(w: torch.Tensor) -> Dict[str, float]:
    w = w.detach().float().cpu()
    return {
        "numel": int(w.numel()),
        "l2": float(torch.norm(w, p=2).item()),
        "linf": float(torch.norm(w, p=float("inf")).item()),
        "mean_abs": float(w.abs().mean().item()),
    }


def _read_sweep_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns exp_name -> one representative row (ckpt_path, lambda_scale, etc).
    The input CSV has multiple rows per exp (one per BER); we de-duplicate by exp_name.
    """
    exp_rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row["exp_name"]
            if exp not in exp_rows:
                exp_rows[exp] = row
    return exp_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep_csv",
        type=str,
        default="training/srqat_sweep_results.csv",
        help="CSV produced by tools/sweep_srqat.py --eval (relative to repo root).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="training/srqat_scale_weight_analysis",
        help="Output directory (relative to repo root).",
    )
    ap.add_argument(
        "--include_act_scales",
        action="store_true",
        help="Also include quan_a_fn.s/init_state stats (default: only weight scales).",
    )
    args = ap.parse_args()

    sweep_csv = (REPO_ROOT / args.sweep_csv).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_rows = _read_sweep_csv(sweep_csv)
    if not exp_rows:
        raise RuntimeError(f"No rows found in {sweep_csv}")

    layer_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for exp, row in sorted(exp_rows.items()):
        lam = row.get("lambda_scale", "")
        ckpt_rel = row.get("ckpt_path", "")
        ckpt_path = (REPO_ROOT / ckpt_rel).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {exp}: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd: Dict[str, torch.Tensor] = ckpt.get("state_dict", {})
        if not isinstance(sd, dict) or not sd:
            raise RuntimeError(f"Checkpoint state_dict missing/empty: {ckpt_path}")

        # Find layer prefixes that have weight + weight scale
        prefixes = []
        for k in sd.keys():
            if k.endswith(".quan_w_fn.s"):
                prefix = k[: -len(".quan_w_fn.s")]
                if f"{prefix}.weight" in sd:
                    prefixes.append(prefix)
        prefixes = sorted(set(prefixes))

        # Aggregate stats across layers (weighted by numel where appropriate)
        total_w_numel = 0
        sum_w_l2 = 0.0
        sum_w_mean_abs_weighted = 0.0
        sum_s2 = 0.0
        sum_s_mean = 0.0
        s_count = 0

        for prefix in prefixes:
            w = sd[f"{prefix}.weight"]
            w_stats = _weight_norms(w)

            s = sd[f"{prefix}.quan_w_fn.s"]
            s_stats = _tensor_stats(s)

            init_key = f"{prefix}.quan_w_fn.init_state"
            init_stats = _tensor_stats(sd[init_key]) if init_key in sd else {"len": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}

            act_s_stats = None
            act_init_stats = None
            if args.include_act_scales:
                a_s_key = f"{prefix}.quan_a_fn.s"
                a_i_key = f"{prefix}.quan_a_fn.init_state"
                if a_s_key in sd:
                    act_s_stats = _tensor_stats(sd[a_s_key])
                if a_i_key in sd:
                    act_init_stats = _tensor_stats(sd[a_i_key])

            # Per-layer row
            out_row: Dict[str, object] = {
                "exp_name": exp,
                "lambda_scale": lam,
                "layer": prefix,
                "weight_numel": w_stats["numel"],
                "weight_l2": w_stats["l2"],
                "weight_linf": w_stats["linf"],
                "weight_mean_abs": w_stats["mean_abs"],
                "w_scale_len": s_stats["len"],
                "w_scale_min": s_stats["min"],
                "w_scale_max": s_stats["max"],
                "w_scale_mean": s_stats["mean"],
                "w_scale_std": s_stats["std"],
                "w_init_len": init_stats["len"],
                "w_init_min": init_stats["min"],
                "w_init_max": init_stats["max"],
                "w_init_mean": init_stats["mean"],
            }
            if args.include_act_scales:
                if act_s_stats is not None:
                    out_row.update(
                        {
                            "a_scale_len": act_s_stats["len"],
                            "a_scale_min": act_s_stats["min"],
                            "a_scale_max": act_s_stats["max"],
                            "a_scale_mean": act_s_stats["mean"],
                            "a_scale_std": act_s_stats["std"],
                        }
                    )
                if act_init_stats is not None:
                    out_row.update(
                        {
                            "a_init_len": act_init_stats["len"],
                            "a_init_min": act_init_stats["min"],
                            "a_init_max": act_init_stats["max"],
                            "a_init_mean": act_init_stats["mean"],
                        }
                    )

            layer_rows.append(out_row)

            # Aggregate
            total_w_numel += int(w_stats["numel"])
            sum_w_l2 += float(w_stats["l2"])
            sum_w_mean_abs_weighted += float(w_stats["mean_abs"]) * int(w_stats["numel"])
            sum_s2 += float((s.detach().float().cpu() ** 2).sum().item())
            sum_s_mean += float(s_stats["mean"])
            s_count += 1

        summary_rows.append(
            {
                "exp_name": exp,
                "lambda_scale": lam,
                "ckpt_path": ckpt_rel,
                "num_layers_with_w_scale": len(prefixes),
                "total_weight_numel": total_w_numel,
                "avg_layer_weight_l2": (sum_w_l2 / len(prefixes)) if prefixes else float("nan"),
                "global_weight_mean_abs": (sum_w_mean_abs_weighted / total_w_numel) if total_w_numel else float("nan"),
                "avg_layer_w_scale_mean": (sum_s_mean / s_count) if s_count else float("nan"),
                "sum_w_scale_sq": sum_s2,
            }
        )

    # Write per-layer CSV
    layer_csv = out_dir / "layers.csv"
    with layer_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(layer_rows[0].keys()) if layer_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in layer_rows:
            writer.writerow(r)

    # Write summary CSV
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"[OK] Wrote: {layer_csv.relative_to(REPO_ROOT)}")
    print(f"[OK] Wrote: {summary_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()




