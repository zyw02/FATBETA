#!/usr/bin/env python3
"""
Sweep SR-QAT lambda_scale values for ResNet18 NUDE training.
Trains models with different lambda values and evaluates fault tolerance.
"""

import os
import sys
import subprocess
import yaml
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Sweep SR-QAT lambda_scale for ResNet18')
    parser.add_argument('--base_config', type=str, 
                       default='configs/training/train_resnet18_cifar10_single_gpu_nude_srqat.yaml',
                       help='Base config file')
    parser.add_argument('--lambdas', nargs='+', type=float, 
                       default=[0, 1e-7, 1e-6, 1e-5, 1e-4],
                       help='Lambda values to sweep')
    parser.add_argument('--bers', nargs='+', type=float,
                       default=[1e-2, 2e-2, 3e-2, 4e-2, 5e-2],
                       help='BER values for evaluation')
    parser.add_argument('--output_dir', type=str, default='training',
                       help='Output directory')
    parser.add_argument('--skip_train', action='store_true',
                       help='Skip training, only evaluate')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation')
    return parser.parse_args()


def create_config(base_config_path, lambda_scale, output_name):
    """Create a config file with the specified lambda_scale."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['name'] = output_name
    config['scale_penalty']['lambda_scale'] = lambda_scale
    config['scale_penalty']['enabled'] = lambda_scale > 0
    
    config_dir = PROJECT_ROOT / 'configs' / 'training' / 'srqat_sweep_resnet18'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f'{output_name}.yaml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    
    return str(config_path)


def train_model(config_path):
    """Train a model with the given config."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'main_nude.py'),
        config_path
    ]
    print(f"\n{'='*60}")
    print(f"[TRAIN] Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT)
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode == 0


def evaluate_model(ckpt_path, config_path, bers):
    """Evaluate model with fault injection at different BERs."""
    results = {'baseline': None, 'faulted': {}}
    
    for ber in bers:
        cmd = [
            sys.executable, str(PROJECT_ROOT / 'tools' / 'test_fault_injection_baseline_resnet18.py'),
            '--config', config_path,
            '--ckpt', ckpt_path,
            '--ber', str(ber),
            '--seed', '42'
        ]
        
        print(f"\n[EVAL] BER={ber}: Running...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        # Parse output
        for line in result.stdout.split('\n'):
            if 'Baseline准确率' in line:
                try:
                    results['baseline'] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            if '故障注入后准确率' in line:
                try:
                    results['faulted'][ber] = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
    
    return results


def main():
    args = parse_args()
    
    base_config = PROJECT_ROOT / args.base_config
    if not base_config.exists():
        print(f"Error: Base config not found: {base_config}")
        return
    
    all_results = []
    
    for lam in args.lambdas:
        # Create config
        lam_str = f'{lam:.0e}'.replace('-', 'm').replace('+', 'p').replace('.', '_') if lam > 0 else '0'
        output_name = f'resnet18_cifar10_nude_srqat_ls{lam_str}'
        
        config_path = create_config(str(base_config), lam, output_name)
        ckpt_path = str(PROJECT_ROOT / args.output_dir / output_name / f'{output_name}_checkpoint.pth.tar')
        
        print(f"\n{'#'*60}")
        print(f"# Lambda = {lam}")
        print(f"# Config: {config_path}")
        print(f"# Checkpoint: {ckpt_path}")
        print(f"{'#'*60}")
        
        # Train
        if not args.skip_train:
            if not os.path.exists(ckpt_path):
                success = train_model(config_path)
                if not success:
                    print(f"[WARN] Training failed for lambda={lam}")
                    continue
            else:
                print(f"[INFO] Checkpoint exists, skipping training: {ckpt_path}")
        
        # Evaluate
        if not args.skip_eval:
            if os.path.exists(ckpt_path):
                results = evaluate_model(ckpt_path, config_path, args.bers)
                results['lambda'] = lam
                results['name'] = output_name
                all_results.append(results)
                
                print(f"\n[RESULTS] Lambda={lam}")
                print(f"  Baseline: {results['baseline']:.2f}%")
                for ber, acc in results['faulted'].items():
                    print(f"  BER={ber}: {acc:.2f}%")
            else:
                print(f"[WARN] Checkpoint not found: {ckpt_path}")
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Lambda':<12} {'Baseline':<10}", end='')
        for ber in args.bers:
            print(f"BER={ber:<8}", end='')
        print()
        print('-' * 80)
        
        for r in all_results:
            print(f"{r['lambda']:<12.0e} {r['baseline']:<10.2f}", end='')
            for ber in args.bers:
                acc = r['faulted'].get(ber, 0)
                print(f"{acc:<14.2f}", end='')
            print()
        
        # Save to CSV
        csv_path = PROJECT_ROOT / args.output_dir / 'resnet18_srqat_sweep_results.csv'
        with open(csv_path, 'w') as f:
            f.write('lambda,baseline,' + ','.join([f'ber_{ber}' for ber in args.bers]) + '\n')
            for r in all_results:
                f.write(f"{r['lambda']},{r['baseline']},")
                f.write(','.join([str(r['faulted'].get(ber, '')) for ber in args.bers]))
                f.write('\n')
        print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    main()


