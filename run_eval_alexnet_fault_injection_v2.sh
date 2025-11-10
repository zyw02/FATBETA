#!/bin/bash
# Single GPU evaluation script with fault injection for AlexNet CIFAR10 mixed-precision model
# This script evaluates the NEW checkpoint (classifier.1 uses dynamic bits [2,3,4,5,6])

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

# Path to the evaluation config YAML file (V2 version)
EVAL_CONFIG="./configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml"

# Path to the trained checkpoint (from V2 training phase)
# Uncomment the one you want to evaluate:
# CHECKPOINT_PATH="./training/alexnet_cifar10_single_gpu_v2/alexnet_cifar10_single_gpu_v2_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_progressive/alexnet_cifar10_FAT_progressive_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT/alexnet_cifar10_FAT_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_a92b25/alexnet_cifar10_FAT_a92b25_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_KL_entropy/alexnet_cifar10_FAT_KL_entropy_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_KL_entropy_fixed_ber2e2_seed42/alexnet_cifar10_FAT_KL_entropy_fixed_ber2e2_seed42_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_KL_entropy_multi_seed/alexnet_cifar10_FAT_KL_entropy_multi_seed_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_KL_entropy_multi_seed_ber3e2/alexnet_cifar10_FAT_KL_entropy_multi_seed_ber3e2_checkpoint.pth.tar"
# CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_KL_entropy_progressive_ber/epoch_160_checkpoint.pth.tar_checkpoint.pth.tar"
CHECKPOINT_PATH="./training/alexnet_cifar10_FAT_KL_entropy_progressive_v2/alexnet_cifar10_FAT_KL_entropy_progressive_v2_checkpoint.pth.tar"

# Path to the bit-width configuration JSON file (from V2 search phase)
# Note: Both v2 and FAT use the same search config (bit-width=[2,2,2,2,2,2])
# If you want to evaluate FAT model, make sure to use FAT checkpoint above
# BIT_WIDTH_CONFIG="./search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json"
# BIT_WIDTH_CONFIG="./search/alexnet_cifar10_FAT_search_bit_width_config.json"
BIT_WIDTH_CONFIG="./search/alexnet_cifar10_FAT_a92b25search_bit_width_config.json"

# BER (Bit-Error-Rate) values to test (comma-separated)
# These represent the probability of bit-flip per bit
# Lower values = fewer faults, Higher values = more faults
# Testing multiple BER values: 2e-2, 3e-2, 4e-2, 5e-2, 1e-1
BER_LIST="2e-2,3e-2,4e-2,5e-2,1e-1"

# Random seed for reproducibility
SEED=42

# Number of trials for each BER value (each trial uses a different random seed)
# This samples different fault patterns, simulating the training distribution
# Using only 1 trial with seed=42 as requested
NUM_TRIALS=1

# Seed list used during training (should match the seed_list in training config)
# Training: each forward randomly selects from this list
# Evaluation: trials will sample from this list to ensure same fault patterns as training
# Using only seed=42 as requested
SEED_LIST="42"

# Configuration index to use from JSON file (if multiple configurations exist)
CONFIG_INDEX=0

# Run evaluation with fault injection
python tools/eval_with_fault_injection.py \
    --config $EVAL_CONFIG \
    --bit_width_config $BIT_WIDTH_CONFIG \
    --resume_path $CHECKPOINT_PATH \
    --ber_list "$BER_LIST" \
    --seed $SEED \
    --num_trials $NUM_TRIALS \
    --seed_list "$SEED_LIST" \
    --config_index $CONFIG_INDEX

