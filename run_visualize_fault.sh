#!/bin/bash
# 可视化 features.0 在故障注入后的激活值变化

CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_FAT_a92b25search_bit_width_config.json"
BIT_CONFIG_INDEX=0
LAYER="features.0"
BER=1e-1
NUM_SAMPLES=10
OUTPUT_DIR="visualizations/fault_activation_features_0_ber1e1"
DEVICE="cuda:0"
SEED=42

python tools/visualize_fault_activation.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --layer "${LAYER}" \
    --ber "${BER}" \
    --num_samples "${NUM_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --bit_width_config "${BIT_WIDTH_CONFIG}" \
    --config_index "${BIT_CONFIG_INDEX}" \
    --device "${DEVICE}" \
    --seed "${SEED}"

echo "Visualization complete! Check ${OUTPUT_DIR} for results."

