import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int, default=6)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ckpt', type=str, default="/root/autodl-tmp/retraining-free-quantization/training/r18_c10_nude_0.02_all_bits_orth_best_ever_result_for_nude/r18_c10_nude_0.02_all_bits_orth_checkpoint.pth.tar")
parser.add_argument('--config', type=str, default="configs/training/train_resnet18_cifar10_single_gpu.yaml")
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

print(f"ResNet18 W{bits}A{bits} BER Sweep - Baseline Checkpoint")
print(f"{'BER':<10} | {'All Bits Acc':<15} | {'Skip MSB Acc':<15} | {'Only MSB Acc':<15}")
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
    print(f"{ber:<10} | {acc_all:<15} | {acc_skip:<15} | {acc_only:<15}")
