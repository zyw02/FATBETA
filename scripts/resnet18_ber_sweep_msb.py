import subprocess
import argparse
import os

# 使用 argparse 允许从命令行指定位宽
parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int, default=6, help='Dynamic bits (4 or 6)')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ckpt', type=str, default="/root/autodl-tmp/FATBETA/training/r18_c10_nude_msb0022_lsb_002_orth_double_3x2080ti_rev/r18_c10_nude_msb0022_lsb_002_orth_double_3x2080ti_rev_checkpoint.pth.tar")
parser.add_argument('--config', type=str, default="configs/training/r18_c10_nude_standard_bfat_allbits_2.yaml")
parser.add_argument('--bers', type=str, default=None)
args = parser.parse_args()

bits = args.bits
seed_arg = f" --seed {args.seed}" if args.seed is not None else ""
if args.bers is not None:
    try:
        bers = [float(x) for x in args.bers.split(',')]
    except Exception:
        bers = [0,1e-6,1e-5,1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
else:
    bers = [0,1e-6,1e-5,1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
ckpt = args.ckpt
config = args.config

# colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_TITLE = "\033[36m"
C_COL1 = "\033[32m"
C_COL2 = "\033[33m"
C_COL3 = "\033[35m"

ckpt_name = os.path.basename(ckpt)
print(f"{C_BOLD}{C_TITLE}ResNet18 W{bits}A{bits} BER Sweep - {ckpt_name}{C_RESET}")
print(f"{'BER':<10} | {C_COL1}{'All Bits Acc':<15}{C_RESET} | {C_COL2}{'Skip MSB Acc':<15}{C_RESET} | {C_COL3}{'Only MSB Acc':<15}{C_RESET}")
print("-" * 65)

for ber in bers:
    cmd_all = f"python tools/test_fault_injection_baseline_resnet18.py --config {config} --ckpt {ckpt} --ber {ber} --skip_baseline --dynamic_bits {bits}{seed_arg}"
    res_all = subprocess.run(cmd_all, shell=True, capture_output=True, text=True)
    acc_all = next((l.split(':')[-1].replace('%', '').strip() for l in res_all.stdout.split('\n') if "故障注入后准确率" in l), None)
    if acc_all is None:
        print(f"[ALL] CMD: {cmd_all}")
        if res_all.returncode != 0:
            print(f"[ALL] Return code: {res_all.returncode}")
        if res_all.stderr:
            lines = res_all.stderr.splitlines()
            print("\n".join(lines[-20:]))
        if res_all.stdout:
            lines = res_all.stdout.splitlines()
            print("\n".join(lines[-20:]))
        acc_all = "ERR"
    cmd_skip = f"python tools/test_fault_injection_baseline_resnet18.py --config {config} --ckpt {ckpt} --ber {ber} --skip_msb --skip_baseline --dynamic_bits {bits}{seed_arg}"
    res_skip = subprocess.run(cmd_skip, shell=True, capture_output=True, text=True)
    acc_skip = next((l.split(':')[-1].replace('%', '').strip() for l in res_skip.stdout.split('\n') if "故障注入后准确率" in l), None)
    if acc_skip is None:
        print(f"[SKIP_MSB] CMD: {cmd_skip}")
        if res_skip.returncode != 0:
            print(f"[SKIP_MSB] Return code: {res_skip.returncode}")
        if res_skip.stderr:
            lines = res_skip.stderr.splitlines()
            print("\n".join(lines[-20:]))
        if res_skip.stdout:
            lines = res_skip.stdout.splitlines()
            print("\n".join(lines[-20:]))
        acc_skip = "ERR"
    cmd_only = f"python tools/test_fault_injection_baseline_resnet18.py --config {config} --ckpt {ckpt} --ber {ber} --only_msb --skip_baseline --dynamic_bits {bits}{seed_arg}"
    res_only = subprocess.run(cmd_only, shell=True, capture_output=True, text=True)
    acc_only = next((l.split(':')[-1].replace('%', '').strip() for l in res_only.stdout.split('\n') if "故障注入后准确率" in l), None)
    if acc_only is None:
        print(f"[ONLY_MSB] CMD: {cmd_only}")
        if res_only.returncode != 0:
            print(f"[ONLY_MSB] Return code: {res_only.returncode}")
        if res_only.stderr:
            lines = res_only.stderr.splitlines()
            print("\n".join(lines[-20:]))
        if res_only.stdout:
            lines = res_only.stdout.splitlines()
            print("\n".join(lines[-20:]))
        acc_only = "ERR"
    print(f"{ber:<10} | {C_COL1}{acc_all:<15}{C_RESET} | {C_COL2}{acc_skip:<15}{C_RESET} | {C_COL3}{acc_only:<15}{C_RESET}")
