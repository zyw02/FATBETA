#!/usr/bin/env python3
"""
SR-QAT sweep runner for this repo.

What it does:
- Reads a base training YAML config (main_normal.py/process_normal.py compatible).
- Generates multiple derived configs by varying `scale_penalty.lambda_scale`.
- Optionally runs training for each derived config.
- Optionally evaluates SEU fault injection robustness (BER sweep) and writes a CSV.

Assumptions (matches current training layout in this repo):
- Output directory is `${repo_root}/{output_dir}/{name}/`
- Checkpoint file is `${name}_checkpoint.pth.tar`
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _sanitize_lambda_tag(lam_str: str) -> str:
    # Make a filesystem-safe tag like: 1e-6 -> 1e_m6, 3e-7 -> 3e_m7
    s = lam_str.strip()
    s = s.replace("+", "")
    s = s.replace("-", "_m")
    s = s.replace(".", "p")
    return s


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _infer_ckpt_path(cfg: Dict[str, Any]) -> Path:
    out_dir = str(cfg.get("output_dir", "training"))
    name = str(cfg["name"])
    return REPO_ROOT / out_dir / name / f"{name}_checkpoint.pth.tar"


def _run_capture(cmd: List[str]) -> str:
    """Run command and capture combined stdout/stderr (used when we need to parse output)."""
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n\n{proc.stdout}")
    return proc.stdout


def _run_stream(cmd: List[str]) -> None:
    """Run command and stream combined stdout/stderr to console (used for training)."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        # forward child's output to our stdout
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def _parse_fault_injection_output(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse outputs of tools/test_fault_injection_baseline_resnet18.py.
    Returns (baseline_acc, fault_acc), may be None if not found.
    """
    baseline = None
    faulted = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Baseline准确率:"):
            # e.g. "Baseline准确率: 93.12%"
            try:
                baseline = float(line.split("Baseline准确率:")[1].strip().rstrip("%"))
            except Exception:
                pass
        if line.startswith("故障注入后准确率:"):
            try:
                faulted = float(line.split("故障注入后准确率:")[1].strip().rstrip("%"))
            except Exception:
                pass
    return baseline, faulted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/training/train_resnet18_cifar10_single_gpu_normal_srqat.yaml",
        help="Base training YAML config (relative to repo root).",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        nargs="+",
        default=["0", "1e-8", "3e-8", "1e-7", "3e-7", "1e-6", "3e-6", "1e-5"],
        help="Lambda values for scale_penalty.lambda_scale (strings).",
    )
    parser.add_argument(
        "--out_config_dir",
        type=str,
        default="configs/training/srqat_sweep",
        help="Directory to write generated configs (relative to repo root).",
    )
    parser.add_argument("--train", action="store_true", help="Run training for each generated config.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run SEU fault-injection evaluation and write CSV (requires checkpoints).",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="configs/eval/eval_resnet18_cifar10_single_gpu.yaml",
        help="Eval YAML config used by fault injection script.",
    )
    parser.add_argument(
        "--bers",
        type=str,
        nargs="+",
        default=["0", "1e-5", "1e-4", "1e-3", "1e-2", "2e-2", "5e-2", "1e-1"],
        help="BER values (strings). Includes 0 if you want baseline printed again (it will run, but no flips).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for eval script.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for eval script.")
    parser.add_argument(
        "--csv",
        type=str,
        default="training/srqat_sweep_results.csv",
        help="Where to write CSV results (relative to repo root).",
    )

    args = parser.parse_args()

    base_cfg_path = (REPO_ROOT / args.base_config).resolve()
    out_dir = (REPO_ROOT / args.out_config_dir).resolve()
    eval_cfg_path = (REPO_ROOT / args.eval_config).resolve()
    csv_path = (REPO_ROOT / args.csv).resolve()

    base_cfg = _read_yaml(base_cfg_path)
    base_name = str(base_cfg.get("name", "exp"))

    generated: List[Path] = []
    for lam_str in args.lambdas:
        cfg = deepcopy(base_cfg)
        tag = _sanitize_lambda_tag(lam_str)
        cfg["name"] = f"{base_name}_ls{tag}"
        cfg.setdefault("scale_penalty", {})
        cfg["scale_penalty"]["enabled"] = True
        cfg["scale_penalty"]["lambda_scale"] = float(lam_str)

        cfg_path = out_dir / f"{cfg['name']}.yaml"
        _write_yaml(cfg_path, cfg)
        generated.append(cfg_path)

    print(f"[SRQAT] Generated {len(generated)} configs under: {out_dir}")
    for p in generated:
        print(f"  - {p.relative_to(REPO_ROOT)}")

    if args.train:
        for cfg_path in generated:
            print(f"\n[SRQAT][TRAIN] Running: {cfg_path.name}")
            _run_stream(["python3", "main_normal.py", str(cfg_path.relative_to(REPO_ROOT))])

    if args.eval:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["exp_name", "lambda_scale", "ber", "baseline_acc", "fault_acc", "ckpt_path"],
            )
            writer.writeheader()

            for cfg_path in generated:
                cfg = _read_yaml(cfg_path)
                ckpt = _infer_ckpt_path(cfg)
                if not ckpt.exists():
                    raise FileNotFoundError(
                        f"Checkpoint not found for {cfg['name']}.\n"
                        f"Expected: {ckpt}\n"
                        f"Hint: run with --train first, or adjust output_dir/name conventions."
                    )

                lam = cfg.get("scale_penalty", {}).get("lambda_scale", None)
                for ber_str in args.bers:
                    ber = float(ber_str)
                    print(f"\n[SRQAT][EVAL] {cfg['name']} | lambda={lam} | BER={ber_str}")
                    out = _run_capture(
                        [
                            "python3",
                            "tools/test_fault_injection_baseline_resnet18.py",
                            "--config",
                            str(eval_cfg_path.relative_to(REPO_ROOT)),
                            "--ckpt",
                            str(ckpt.relative_to(REPO_ROOT)),
                            "--ber",
                            ber_str,
                            "--seed",
                            str(args.seed),
                            "--device",
                            args.device,
                        ]
                    )
                    baseline_acc, fault_acc = _parse_fault_injection_output(out)
                    writer.writerow(
                        {
                            "exp_name": cfg["name"],
                            "lambda_scale": lam,
                            "ber": ber,
                            "baseline_acc": baseline_acc,
                            "fault_acc": fault_acc,
                            "ckpt_path": str(ckpt.relative_to(REPO_ROOT)),
                        }
                    )
                    f.flush()

        print(f"\n[SRQAT] Wrote CSV: {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

