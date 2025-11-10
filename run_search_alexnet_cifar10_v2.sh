#!/bin/bash
# Single GPU search script for AlexNet on CIFAR10 with 11GB 2080Ti
# V2: classifier.1 uses dynamic bits [2,3,4,5,6] instead of fixed 8-bit
# This script runs search without distributed launcher

# Set environment variables for single GPU training
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Make sure only one GPU is visible
export CUDA_VISIBLE_DEVICES=0

# Run search (single GPU, no distributed)
# Note: Checkpoint and dataset paths are configured in the YAML config file
# If you need to override them, edit configs/search/search_alexnet_cifar10_single_gpu_v2.yaml directly
python main.py configs/search/search_alexnet_cifar10_single_gpu_v2.yaml

