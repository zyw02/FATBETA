import subprocess
import argparse
import os
import sys

# 使用 argparse 允许从命令行指定位宽
parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int, default=6, help='Dynamic bits (e.g., 2, 4, 6)')
parser.add_argument('--arch', type=str, default='mobilenetv2', help='Architecture name')
args = parser.parse_args()

bits = args.bits
arch = args.arch

# 定义 BER 扫描范围
bers = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]

# 设置路径
if arch == 'mobilenetv2':
    ckpt = "/root/autodl-tmp/FATBETA/training/mobilenetv2_cifar10_w2to6_a2to6/mobilenetv2_cifar10_w2to6_a2to6_checkpoint.pth.tar"
    config = "/root/autodl-tmp/FATBETA/training/mobilenetv2_cifar10_w2to6_a2to6/configs.yaml"
else:
    ckpt = f"/root/autodl-tmp/FATBETA/training/{arch}_cifar10_checkpoint.pth.tar"
    config = f"configs/training/{arch}_c10.yaml"

if not os.path.exists(ckpt):
    print(f"Error: Checkpoint not found at {ckpt}")
    sys.exit(1)

# ANSI 颜色转义码
C_RESET = "\033[0m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"

print(f"{C_CYAN}{arch.upper()} W{bits}A{bits} BER Sweep - {os.path.basename(ckpt)}{C_RESET}")
print(f"{'BER':<10} | {'All Bits Acc':<15} | {'Skip MSB Acc':<15} | {'Only MSB Acc':<15}")
print("-" * 65)

for ber in bers:
    results = {}
    modes = [
        ("all", ""),
        ("skip_msb", "--skip_msb"),
        ("only_msb", "--only_msb")
    ]
    
    for mode_name, extra_flag in modes:
        cmd = (
            f"python tools/test_fault_injection_baseline_resnet18.py "
            f"--config {config} --ckpt {ckpt} --ber {ber} "
            f"--skip_baseline --dynamic_bits {bits} {extra_flag}"
        )
        
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        acc = "N/A"
        if res.returncode != 0:
            print(f"\n[ERROR] Command failed for BER={ber}, Mode={mode_name}")
            print(f"Stderr: {res.stderr}")
        else:
            for line in res.stdout.split('\n'):
                # 匹配包含 "故障注入后准确率" 的行
                if "故障注入后准确率" in line:
                    try:
                        acc = line.split(':')[-1].split('%')[0].strip()
                    except Exception:
                        acc = "ParseErr"
        
        results[mode_name] = acc

    # 打印结果行
    print(f"{ber:<10} | {C_RED}{results['all']:<15}{C_RESET} | {C_GREEN}{results['skip_msb']:<15}{C_RESET} | {C_YELLOW}{results['only_msb']:<15}{C_RESET}")
