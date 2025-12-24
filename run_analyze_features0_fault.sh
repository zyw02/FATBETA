#!/bin/bash
# 分析只对 features.0 注入故障时的性能损失
# 使用 w6a6 配置，测试 BER=1e-3, 1e-2, 5e-2

CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
LAYER="features.0"
BERS="1e-3,1e-2,5e-2"
DEVICE="cuda:0"

python tools/analyze_single_layer_fault_impact.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --layer "${LAYER}" \
    --bers "${BERS}" \
    --device "${DEVICE}" \
    --force_w6a6

echo "Analysis complete! Check logs/single_layer_fault_features_0_w6a6.json for results."


