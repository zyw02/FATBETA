#!/bin/bash
# Clean AlexNet CIFAR10 training (120 epochs) using main.py
# - No FAT / TRADES / corrector
# - No KD / Distribution Loss

set -euo pipefail

python3 main.py configs/training/train_alexnet_cifar10_single_gpu_clean120.yaml



