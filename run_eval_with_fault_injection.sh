#!/bin/bash
# Single GPU evaluation script with fault injection for ResNet18 CIFAR10 mixed-precision model
# This script evaluates the model's robustness under different BER (Bit-Error-Rate) values

# Set environment variables for single GPU training
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Make sure only one GPU is visible
export CUDA_VISIBLE_DEVICES=0

# Path to your CIFAR10 dataset (should be the same as in config)
CIFAR10_PATH="./data/cifar10"

# Path to the evaluation config YAML file
EVAL_CONFIG="./configs/eval/eval_resnet18_cifar10_single_gpu.yaml"

# Path to the trained checkpoint (from training phase)
CHECKPOINT_PATH="./training/resnet18_cifar10_single_gpu/resnet18_cifar10_single_gpu_checkpoint.pth.tar"

# Path to the bit-width configuration JSON file (from search phase)
BIT_WIDTH_CONFIG="./search/resnet18_cifar10_single_gpu_search_bit_width_config.json"

# BER (Bit-Error-Rate) values to test (comma-separated)
# These represent the probability of bit-flip per bit
# Lower values = fewer faults, Higher values = more faults
BER_LIST="1e-4,1e-3,1e-2,2e-2,3e-2,4e-2,1e-1"

# Random seed for reproducibility
SEED=42

# Configuration index to use from JSON file (if multiple configurations exist)
CONFIG_INDEX=0

# Run evaluation with fault injection
python tools/eval_with_fault_injection.py \
    --config $EVAL_CONFIG \
    --bit_width_config $BIT_WIDTH_CONFIG \
    --ber_list "$BER_LIST" \
    --seed $SEED \
    --config_index $CONFIG_INDEX

