#!/bin/bash
# Single GPU training script for AlexNet on CIFAR10 with 11GB 2080Ti

# Set environment variables for single GPU training
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Make sure only one GPU is visible
export CUDA_VISIBLE_DEVICES=0

# Run training (single GPU, no distributed)
# Note: Dataset path is configured in the YAML config file (default: ./data/cifar10)
# If you need to change it, edit configs/training/train_alexnet_cifar10_single_gpu.yaml
python main.py configs/training/train_alexnet_cifar10_single_gpu.yaml

