import subprocess
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser(description='MobileNetV2 BER Sweep with EMA Weights')
parser.add_argument('--bits', type=int, default=6, help='Dynamic bits (e.g., 2, 4, 6)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ckpt', type=str, default="/workspace/FATBETA/training/mobilenetv2_cifar_nude_bfat_v2/mobilenetv2_cifar_nude_bfat_v2_checkpoint.pth.tar")
parser.add_argument('--config', type=str, default="configs/training/train_mobilenetv2_cifar_qat.yaml")
parser.add_argument('--bers', type=str, default=None)
parser.add_argument('--no_ema', action='store_true', help='Use standard weights instead of EMA')
args = parser.parse_args()

# BER List
if args.bers is not None:
    try:
        bers = [float(x) for x in args.bers.split(',')]
    except Exception:
        # bers = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
        bers = [1e-4, 1e-3, 2e-3, 4e-3, 8e-3,1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2e-2]
else:
    bers = [1e-4, 1e-3, 2e-3, 4e-3, 8e-3,1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2e-2]

# Configuration
bits = args.bits
seed_arg = f" --seed {args.seed}"
ckpt = args.ckpt
config = args.config

# Output Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_TITLE = "\033[36m"
C_COL1 = "\033[32m"
C_COL2 = "\033[33m"
C_COL3 = "\033[35m"

ckpt_name = os.path.basename(ckpt)
ema_status = "Standard" if args.no_ema else "EMA"
print(f"{C_BOLD}{C_TITLE}MobileNetV2 W{bits}A{bits} {ema_status} BER Sweep - {ckpt_name}{C_RESET}")
print(f"{'BER':<10} | {C_COL1}{'All Bits Acc':<15}{C_RESET} | {C_COL2}{'Skip MSB Acc':<15}{C_RESET} | {C_COL3}{'Only MSB Acc':<15}{C_RESET}")
print("-" * 65)

# Tool Path
tool_path = "tools/test_fault_injection_general.py"

for ber in bers:
    ema_arg = " --use_ema" if not args.no_ema else ""
    base_cmd = f"python {tool_path} --config {config} --ckpt {ckpt}{ema_arg} --ber {ber} --skip_baseline --dynamic_bits {bits}{seed_arg}"
    
    # 1. All Bits
    cmd_all = base_cmd
    res_all = subprocess.run(cmd_all, shell=True, capture_output=True, text=True)
    acc_all = next((l.split(':')[-1].replace('%', '').strip() for l in res_all.stdout.split('\n') if "故障注入后准确率" in l), None)
    if acc_all is None: acc_all = "ERR"
    
    # 2. Skip MSB
    cmd_skip = f"{base_cmd} --skip_msb"
    res_skip = subprocess.run(cmd_skip, shell=True, capture_output=True, text=True)
    acc_skip = next((l.split(':')[-1].replace('%', '').strip() for l in res_skip.stdout.split('\n') if "故障注入后准确率" in l), None)
    if acc_skip is None: acc_skip = "ERR"
    
    # 3. Only MSB
    cmd_only = f"{base_cmd} --only_msb"
    res_only = subprocess.run(cmd_only, shell=True, capture_output=True, text=True)
    acc_only = next((l.split(':')[-1].replace('%', '').strip() for l in res_only.stdout.split('\n') if "故障注入后准确率" in l), None)
    if acc_only is None: acc_only = "ERR"
    
    # Print Result Row
    print(f"{ber:<10} | {C_COL1}{acc_all:<15}{C_RESET} | {C_COL2}{acc_skip:<15}{C_RESET} | {C_COL3}{acc_only:<15}{C_RESET}")

