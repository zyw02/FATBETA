#!/bin/bash
# 测试ResNet18模型的SEU容错能力（Baseline，无编码保护）

# 设置参数
CONFIG="configs/eval/eval_resnet18_cifar10_single_gpu.yaml"
CKPT="training/resnet18_cifar10_single_gpu/resnet18_cifar10_single_gpu_checkpoint.pth.tar"
BER=1e-1
SEED=42
DEVICE="cuda"

# 可选：如果存在bit_width_config，可以指定
# BIT_WIDTH_CONFIG="search/resnet18_cifar10_single_gpu_search_bit_width_config.json"

echo "=========================================="
echo "ResNet18 SEU容错能力测试（Baseline）"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $CKPT"
echo "BER: $BER"
echo "随机种子: $SEED"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 运行测试
python tools/test_fault_injection_baseline_resnet18.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --ber "$BER" \
    --seed "$SEED" \
    --device "$DEVICE" \
    ${BIT_WIDTH_CONFIG:+--bit_width_config "$BIT_WIDTH_CONFIG"}

