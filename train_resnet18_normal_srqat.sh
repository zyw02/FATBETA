#!/bin/bash
# Train ResNet18 (CIFAR10) with main_normal.py/process_normal.py + SR-QAT (scale penalty)

set -euo pipefail

CONFIG="configs/training/train_resnet18_cifar10_single_gpu_normal_srqat.yaml"

python3 main_normal.py "$CONFIG"



