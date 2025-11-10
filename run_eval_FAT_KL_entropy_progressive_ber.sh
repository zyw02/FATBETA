#!/bin/bash
# Evaluation script for alexnet_cifar10_FAT_KL_entropy_progressive_ber

# Set environment variables for single GPU evaluation
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Make sure only one GPU is visible
export CUDA_VISIBLE_DEVICES=0

# Run evaluation
cd /root/autodl-tmp/retraining-free-quantization
python main.py configs/eval/eval_alexnet_cifar10_FAT_KL_entropy_progressive_ber.yaml

