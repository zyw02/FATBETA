#!/usr/bin/env python3
"""
Plot weight distribution violin chart from a checkpoint (AlexNet CIFAR10 / this repo format).

Default preset tries to match the user's example with 5 layers:
  Conv-1: features.0
  Conv-2: features.3
  Conv-3: features.6
  FC-1:   classifier.1
  FC-2:   classifier.4

Optionally, plot quantized weights at a given bit-width using LSQ parameters stored in the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def _compute_thd(bits: int, *, all_positive: bool, symmetric: bool) -> Tuple[int, int]:
    if all_positive:
        if symmetric:
            raise ValueError("Positive quantization cannot be symmetric")
        thd_neg = 0
        thd_pos = (1 << bits) - 1
    else:
        if symmetric:
            thd_neg = -(1 << (bits - 1)) + 1
            thd_pos = (1 << (bits - 1)) - 1
        else:
            thd_neg = -(1 << (bits - 1))
            thd_pos = (1 << (bits - 1)) - 1
    return int(thd_neg), int(thd_pos)


def _quantize_lsq(w: torch.Tensor, s: float, bits: int, *, all_positive: bool, symmetric: bool) -> torch.Tensor:
    thd_neg, thd_pos = _compute_thd(bits, all_positive=all_positive, symmetric=symmetric)
    ws = w / s
    ws = torch.clamp(ws, thd_neg, thd_pos)
    ws = torch.round(ws)
    return ws * s


def _sample_flatten(t: torch.Tensor, n: int, rng: np.random.Generator) -> np.ndarray:
    a = t.detach().float().cpu().view(-1).numpy()
    if n <= 0 or a.size <= n:
        return a
    idx = rng.choice(a.size, size=n, replace=False)
    return a[idx]


def _hist_violin_stats(t: torch.Tensor, bins: int) -> Dict[str, object]:
    """
    Compute histogram-based density stats from the FULL tensor (no sampling).
    Returns dict with:
      - y: bin centers (np.ndarray)
      - dens: density values (np.ndarray, scaled as PDF)
      - min, max, median: float
    """
    x = t.detach().float().cpu().view(-1)
    if x.numel() == 0:
        raise ValueError("Empty tensor")
    vmin = float(x.min().item())
    vmax = float(x.max().item())
    if vmin == vmax:
        # Expand a tiny bit to avoid histc issues
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps
    counts = torch.histc(x, bins=bins, min=vmin, max=vmax).double()
    edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(edges[1] - edges[0])
    total = float(counts.sum().item())
    dens = (counts.numpy() / (total * bin_w)) if total > 0 else np.zeros_like(centers)

    # Exact median (still full-data, but OK for tens of millions)
    median = float(torch.median(x).item())
    return {"y": centers, "dens": dens, "min": vmin, "max": vmax, "median": median}


def _draw_hist_violin(
    ax,
    x_pos: float,
    y: np.ndarray,
    dens: np.ndarray,
    *,
    max_width: float,
    facecolor: str = "#A6D6A8",
    edgecolor: str = "#6AAE6E",
    alpha: float = 0.7,
) -> None:
    if y.size == 0 or dens.size == 0:
        return
    dmax = float(dens.max()) if dens.size else 0.0
    if dmax <= 0:
        return
    w = (dens / dmax) * max_width
    ax.fill_betweenx(y, x_pos - w, x_pos + w, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: alongside ckpt, suffix _weight_violin.png).",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="alexnet5",
        choices=["alexnet5", "alexnet8", "all"],
        help=(
            "Layer preset: "
            "alexnet5 (3 conv + 2 fc, matches the example figure style), "
            "alexnet8 (AlexNet CIFAR: 5 conv + 3 fc), "
            "or all layers found in ckpt."
        ),
    )
    ap.add_argument(
        "--layers",
        type=str,
        nargs="+",
        default=None,
        help="Explicit layer prefixes (e.g., features.0 classifier.1). Overrides --preset.",
    )
    ap.add_argument(
        "--sample_per_layer",
        type=int,
        default=200_000,
        help="Random samples per layer for KDE violin plotting (0=all points; not recommended).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["kde", "hist"],
        help=(
            "Plot mode. 'kde' uses matplotlib.violinplot (may be slow for huge layers). "
            "'hist' uses full-data histogram to draw a violin silhouette (no sampling). "
            "Default: hist if --sample_per_layer=0 else kde."
        ),
    )
    ap.add_argument(
        "--hist_bins",
        type=int,
        default=400,
        help="Histogram bins for --mode hist (full-data).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    ap.add_argument(
        "--quantized",
        action="store_true",
        help="Plot quantized weights (LSQ) instead of raw float weights.",
    )
    ap.add_argument(
        "--wbit",
        type=int,
        default=6,
        help="Weight bit-width used when --quantized is set.",
    )
    ap.add_argument(
        "--bit_list",
        type=int,
        nargs="+",
        default=[6, 5, 4, 3, 2],
        help="Dynamic bit candidates order for indexing *.quan_w_fn.s vector.",
    )
    ap.add_argument(
        "--weight_all_positive",
        action="store_true",
        help="Use all_positive=True for weight quantization thresholds (default False).",
    )
    ap.add_argument(
        "--weight_symmetric",
        action="store_true",
        help="Use symmetric=True for weight quantization thresholds (default False, matches many configs here).",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix("").with_name(ckpt_path.stem + "_weight_violin.png")
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd: Dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)
    if not isinstance(sd, dict):
        raise RuntimeError("Unsupported checkpoint format (expected dict with state_dict)")

    # Select layers
    if args.layers:
        layers = list(args.layers)
        labels = list(args.layers)
    else:
        if args.preset == "alexnet5":
            layers = ["features.0", "features.3", "features.6", "classifier.1", "classifier.4"]
            labels = ["Conv-1", "Conv-2", "Conv-3", "FC-1", "FC-2"]
        elif args.preset == "alexnet8":
            # AlexNet CIFAR in this repo: 5 conv layers in features.{0,3,6,8,10} and 3 linear in classifier.{1,4,6}
            layers = ["features.0", "features.3", "features.6", "features.8", "features.10", "classifier.1", "classifier.4", "classifier.6"]
            labels = ["Conv-1", "Conv-2", "Conv-3", "Conv-4", "Conv-5", "FC-1", "FC-2", "FC-3"]
        else:
            layers = sorted({k[: -len(".weight")] for k in sd.keys() if k.endswith(".weight")})
            labels = layers

    rng = np.random.default_rng(args.seed)
    mode = args.mode
    if mode is None:
        mode = "hist" if int(args.sample_per_layer) == 0 else "kde"

    bit_to_idx = {b: i for i, b in enumerate(args.bit_list)}
    if args.quantized and args.wbit not in bit_to_idx:
        raise ValueError(f"--wbit={args.wbit} not in --bit_list={args.bit_list}")

    data: List[np.ndarray] = []  # for KDE mode
    hist_stats: List[Dict[str, object]] = []  # for hist mode
    used_labels: List[str] = []

    for layer, label in zip(layers, labels):
        w_key = f"{layer}.weight"
        if w_key not in sd:
            continue
        w = sd[w_key].detach().float()
        if args.quantized:
            s_key = f"{layer}.quan_w_fn.s"
            if s_key not in sd:
                # Fallback: can't quantize without scale; skip
                continue
            s_vec = sd[s_key].detach().float().view(-1)
            if s_vec.numel() == 1:
                s = float(s_vec[0].item())
            else:
                s = float(s_vec[bit_to_idx[args.wbit]].item())
            if s <= 0:
                continue
            w = _quantize_lsq(
                w, s, int(args.wbit), all_positive=bool(args.weight_all_positive), symmetric=bool(args.weight_symmetric)
            )
        if mode == "kde":
            sampled = _sample_flatten(w, int(args.sample_per_layer), rng)
            data.append(sampled)
            used_labels.append(label)
        else:
            stats = _hist_violin_stats(w, bins=int(args.hist_bins))
            hist_stats.append(stats)
            used_labels.append(label)

    if mode == "kde" and not data:
        raise RuntimeError("No layers found to plot. Check layer names / checkpoint keys.")
    if mode == "hist" and not hist_stats:
        raise RuntimeError("No layers found to plot. Check layer names / checkpoint keys.")

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(used_labels)
    fig_w = max(10.0, 1.7 * n_layers)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    if mode == "kde":
        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#A6D6A8")  # light green
            pc.set_edgecolor("#6AAE6E")
            pc.set_alpha(0.7)

        # Add min/max whisker and median dot
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            y_min = float(np.min(arr))
            y_max = float(np.max(arr))
            y_med = float(np.median(arr))
            ax.vlines(i, y_min, y_max, color="#1f77b4", linewidth=2.0)
            ax.hlines([y_min, y_max], i - 0.12, i + 0.12, color="#1f77b4", linewidth=2.0)
            ax.scatter([i], [y_med], color="#1f77b4", s=12, zorder=3)
    else:
        # Histogram-based violins using FULL weights
        max_width = 0.38
        for i, st in enumerate(hist_stats, start=1):
            _draw_hist_violin(ax, i, st["y"], st["dens"], max_width=max_width)
            y_min = float(st["min"])
            y_max = float(st["max"])
            y_med = float(st["median"])
            ax.vlines(i, y_min, y_max, color="#1f77b4", linewidth=2.0)
            ax.hlines([y_min, y_max], i - 0.12, i + 0.12, color="#1f77b4", linewidth=2.0)
            ax.scatter([i], [y_med], color="#1f77b4", s=12, zorder=3)

    ax.set_xticks(range(1, n_layers + 1))
    ax.set_xticklabels(used_labels, rotation=0)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Weight value" + (f" (LSQ {args.wbit}-bit quantized)" if args.quantized else ""))
    title = "Weight Distribution"
    if args.quantized:
        title = f"{title} (Quantized)"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # y-limits with small margin
    if mode == "kde":
        all_vals = np.concatenate([d for d in data if d.size > 0])
        y0, y1 = float(all_vals.min()), float(all_vals.max())
    else:
        y0 = float(min(float(st["min"]) for st in hist_stats))
        y1 = float(max(float(st["max"]) for st in hist_stats))
    pad = 0.05 * (y1 - y0 + 1e-12)
    ax.set_ylim(y0 - pad, y1 + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()


