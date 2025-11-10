#!/bin/bash
# Single GPU search script for ResNet18 on CIFAR10 with 11GB 2080Ti
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
# If you need to override them, edit configs/search/search_resnet18_single_gpu.yaml directly
python main.py configs/search/search_resnet18_single_gpu.yaml


