#!/usr/bin/env python3
"""
Evaluate multiple sensitive-layer combinations using the sensitive restorer.
"""

import argparse
import json
import subprocess
import shlex
import sys
from pathlib import Path

DEFAULT_BERS = ["1e-3", "1e-2", "5e-2", "1e-1"]


def normalize_ber(value: str) -> str:
    try:
        return f"{float(value):.0e}"
    except ValueError:
        return value


def parse_summary(path: Path):
    data = json.load(open(path, 'r'))
    ranking = [entry['layer'] for entry in data.get('ranking', [])]
    return data, ranking


def build_combos(ranking, combo_sizes, extra_combos):
    combos = []
    for size in combo_sizes:
        size = int(size)
        if size <= 0 or size > len(ranking):
            continue
        combo = tuple(ranking[:size])
        combos.append(combo)
    for combo in extra_combos:
        layers = tuple(layer.strip() for layer in combo.split(',') if layer.strip())
        if layers:
            combos.append(layers)
    # remove duplicates while preserving order
    seen = set()
    unique = []
    for combo in combos:
        if combo not in seen:
            unique.append(combo)
            seen.add(combo)
    return unique


def run_evaluation(base_cmd, combo_layers, log_dir):
    combo_name = "_".join(layer.replace('.', '-') for layer in combo_layers)
    log_path = log_dir / f"combo_{combo_name}.log"
    cmd = base_cmd + ['--sensitive_layers', ",".join(combo_layers)]
    print(f"[Eval] Layers={combo_layers} -> log: {log_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        print(f"  Command failed (exit {result.returncode}), see log for details.")
        return None
    return parse_comparison_table(result.stdout)


def parse_comparison_table(stdout: str):
    capture = False
    entries = {}
    for line in stdout.splitlines():
        if "Comparison: With Restorer vs Baseline" in line:
            capture = True
            continue
        if capture:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            if line.startswith("BER") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 4:
                break
            ber = normalize_ber(parts[0])
            try:
                baseline = float(parts[1])
                restorer = float(parts[2])
                improvement = float(parts[3].replace('%', ''))
            except ValueError:
                continue
            entries[ber] = {
                'baseline': baseline,
                'restorer': restorer,
                'improvement': improvement,
            }
    return entries if entries else None


def print_summary(results, ber_values):
    if not results:
        print("No successful evaluations.")
        return
    print("\n=== Sensitive Layer Combination Summary ===")
    canonical_bers = [normalize_ber(b) for b in ber_values]
    header = ["Combo"] + [f"{ber} Δ(%)" for ber in canonical_bers]
    print("{:<40} {}".format(header[0], "  ".join(f"{h:<10}" for h in header[1:])))
    print("-" * 80)
    for combo, metrics in results.items():
        label = ",".join(combo)
        row = []
        for ber in canonical_bers:
            if ber in metrics:
                row.append(f"{metrics[ber]['improvement']:+.2f}")
            else:
                row.append("N/A")
        print("{:<40} {}".format(label, "  ".join(f"{val:<10}" for val in row)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple sensitive-layer combinations")
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--layer_profile', required=True)
    parser.add_argument('--summary_json', required=True, help='Summary JSON from analyze_multi_bit_sensitivity')
    parser.add_argument('--bit_width_config', default=None)
    parser.add_argument('--config_index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--restorer_script', default='tools/eval_gradient_statistics_restorer.py')
    parser.add_argument('--combo_sizes', nargs='+', default=['2', '3', '4'])
    parser.add_argument('--combo', action='append', default=[], help='Additional combos e.g. features.0,features.3')
    parser.add_argument('--ber_values', nargs='+', default=DEFAULT_BERS)
    parser.add_argument('--log_dir', default='logs/sensitive_layer_sets')
    parser.add_argument('--restorer_mode', default='sensitive')
    parser.add_argument('--sensitive_args', default='', help='Extra args passed to restorer script')

    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    summary, ranking = parse_summary(summary_path)
    if not ranking:
        raise ValueError("Summary does not contain ranking information.")

    combos = build_combos(ranking, args.combo_sizes, args.combo)
    if not combos:
        raise ValueError("No valid combinations to evaluate.")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, args.restorer_script,
        '--config', args.config,
        '--stage1_ckpt', args.ckpt,
        '--device', args.device,
        '--layer_profile', args.layer_profile,
        '--restorer_mode', args.restorer_mode,
        '--ber_values'
    ] + [str(ber) for ber in args.ber_values]

    if args.bit_width_config:
        base_cmd += ['--bit_width_config', args.bit_width_config, '--config_index', str(args.config_index)]
    if args.sensitive_args:
        base_cmd += shlex.split(args.sensitive_args)

    results = {}
    for combo in combos:
        metrics = run_evaluation(base_cmd, combo, log_dir)
        if metrics:
            results[combo] = metrics

    print_summary(results, args.ber_values)


if __name__ == '__main__':
    main()

