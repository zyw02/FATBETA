#!/bin/bash
# 从epoch 140的checkpoint恢复训练

# Set environment variables for single GPU training
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Make sure only one GPU is visible
export CUDA_VISIBLE_DEVICES=0

# 使用训练目录中的配置文件（已设置resume路径）
cd /root/autodl-tmp/retraining-free-quantization
python main.py training/alexnet_cifar10_FAT_KL_entropy_progressive_ber/configs.yaml

