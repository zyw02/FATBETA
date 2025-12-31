import subprocess
import argparse

# 使用 argparse 允许从命令行指定位宽
parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int, default=6, help='Dynamic bits (4 or 6)')
args = parser.parse_args()

bits = args.bits
bers = [1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
ckpt = "/root/autodl-tmp/retraining-free-quantization/training/resnet18_cifar10_BFAT_ce_0.01_bn_train_cagrad/resnet18_cifar10_BFAT_ce_0.01_bn_train_cagrad_checkpoint.pth.tar"
config = "configs/training/train_resnet18_cifar10_single_gpu.yaml"

print(f"ResNet18 W{bits}A{bits} BER Sweep - Baseline Checkpoint")
print(f"{'BER':<10} | {'All Bits Acc':<15} | {'Skip MSB Acc':<15} | {'Only MSB Acc':<15}")
print("-" * 65)

for ber in bers:
    # 1. 跑 All Bits
    cmd_all = f"python tools/test_fault_injection_baseline_resnet18.py --config {config} --ckpt {ckpt} --ber {ber} --skip_baseline --dynamic_bits {bits}"
    res_all = subprocess.run(cmd_all, shell=True, capture_output=True, text=True)
    acc_all = next((l.split(':')[-1].replace('%', '').strip() for l in res_all.stdout.split('\n') if "故障注入后准确率" in l), "N/A")
    
    # 2. 跑 Skip MSB
    cmd_skip = f"python tools/test_fault_injection_baseline_resnet18.py --config {config} --ckpt {ckpt} --ber {ber} --skip_msb --skip_baseline --dynamic_bits {bits}"
    res_skip = subprocess.run(cmd_skip, shell=True, capture_output=True, text=True)
    acc_skip = next((l.split(':')[-1].replace('%', '').strip() for l in res_skip.stdout.split('\n') if "故障注入后准确率" in l), "N/A")
    
    # 3. 跑 Only MSB
    cmd_only = f"python tools/test_fault_injection_baseline_resnet18.py --config {config} --ckpt {ckpt} --ber {ber} --only_msb --skip_baseline --dynamic_bits {bits}"
    res_only = subprocess.run(cmd_only, shell=True, capture_output=True, text=True)
    acc_only = next((l.split(':')[-1].replace('%', '').strip() for l in res_only.stdout.split('\n') if "故障注入后准确率" in l), "N/A")
    
    print(f"{ber:<10} | {acc_all:<15} | {acc_skip:<15} | {acc_only:<15}")
