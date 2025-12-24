#!/bin/bash

set -e

STAGE1_CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
STAGE2_CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage2.yaml"
STAGE1_OUTPUT="training/alexnet_cifar10_sensitive_stage1"
STAGE1_CKPT="${STAGE1_OUTPUT}/checkpoint.pth.tar"

if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "==== Stage 1: Training main model (clean) ===="
  python main.py ${STAGE1_CONFIG}
fi

echo "==== Stage 2: Training sensitive channel restorer ===="
python train_sensitive_restorer.py --config ${STAGE2_CONFIG} --stage1_ckpt ${STAGE1_CKPT}
