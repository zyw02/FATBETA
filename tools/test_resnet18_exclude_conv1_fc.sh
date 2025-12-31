#!/bin/bash
# 测试ResNet18模型的SEU容错能力（排除conv1和fc层）

# 设置参数
CONFIG="configs/eval/eval_resnet18_cifar10_single_gpu.yaml"
CKPT="training/resnet18_cifar10_single_gpu/resnet18_cifar10_single_gpu_checkpoint.pth.tar"
BER=1e-1
SEED=42
DEVICE="cuda"

# 排除conv1和fc层
EXCLUDE_LAYERS="conv1 fc"

echo "=========================================="
echo "ResNet18 SEU容错能力测试（排除conv1和fc层）"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $CKPT"
echo "BER: $BER"
echo "随机种子: $SEED"
echo "设备: $DEVICE"
echo "排除层: $EXCLUDE_LAYERS"
echo "=========================================="
echo ""

# 运行测试
python tools/test_fault_injection_baseline_resnet18.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --ber "$BER" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --exclude_layers $EXCLUDE_LAYERS




