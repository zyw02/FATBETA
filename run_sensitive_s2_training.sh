#!/bin/bash

set -e

STAGE2_CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage2.yaml"
STAGE1_CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"

if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "[ERROR] Stage 1 checkpoint not found at ${STAGE1_CKPT}"
  echo "Please run Stage 1 training first."
  exit 1
fi

echo "==== Stage 2: Training sensitive channel restorer (resumed) ===="
python train_sensitive_restorer.py --config "${STAGE2_CONFIG}" --stage1_ckpt "${STAGE1_CKPT}"