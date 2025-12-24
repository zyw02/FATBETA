#!/bin/bash

# Stage 2 Training Script with RL Restorer and Enhanced Features
# This script trains the restorer using RL Restorer with enhanced feature extraction

set -e

# Configuration
STAGE2_CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage2.yaml"
STAGE1_CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"

# Check if Stage 1 checkpoint exists
if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "[ERROR] Stage 1 checkpoint not found at ${STAGE1_CKPT}"
  echo "Please run Stage 1 training first."
  exit 1
fi

# Check if config file exists
if [ ! -f "${STAGE2_CONFIG}" ]; then
  echo "[ERROR] Config file not found at ${STAGE2_CONFIG}"
  exit 1
fi

echo "============================================================"
echo "Stage 2: Training RL Restorer with Enhanced Features"
echo "============================================================"
echo "Config: ${STAGE2_CONFIG}"
echo "Stage1 Checkpoint: ${STAGE1_CKPT}"
echo ""

# Change to the training script directory
SCRIPT_DIR="sensitive_training/alexnet_cifar10_sensitive_stage2_transformer/code"
cd "${SCRIPT_DIR}" || exit 1

# Use RL Restorer config
RL_CONFIG="../../../configs/training/train_alexnet_cifar10_sensitive_stage2_rl_enhanced.yaml"

# Run training with RL Restorer and enhanced features
python train_sensitive_restorer.py \
    --config "${RL_CONFIG}" \
    --stage1_ckpt "../../../${STAGE1_CKPT}" \
    --device cuda

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"

