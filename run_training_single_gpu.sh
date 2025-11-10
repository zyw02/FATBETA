#!/bin/bash
# Single GPU training script for ResNet18 on 11GB 2080Ti
# This script runs training without distributed launcher

# Set environment variables for single GPU training
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Make sure only one GPU is visible
export CUDA_VISIBLE_DEVICES=0

# Run training (single GPU, no distributed)
python main.py configs/training/train_resnet18_w2to6_a2to6_single_gpu.yaml




