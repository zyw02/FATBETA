#!/bin/bash
# 训练 ResNet18 CIFAR10 - 原始版本（启用 Distribution Loss）
# 用于对比研究 Distribution Loss 对容错能力的影响

# 配置文件
CONFIG="configs/training/train_resnet18_cifar10_single_gpu.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi

echo "=========================================="
echo "ResNet18 CIFAR10 训练脚本"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "使用版本: main.py (原始版本，Distribution Loss 已启用)"
echo "=========================================="
echo ""
echo "ℹ️  注意: 此版本启用了 Distribution Loss (KL散度)"
echo "   用于对比研究 Distribution Loss 对容错能力的影响"
echo ""
echo "开始训练..."
echo ""

# 运行训练
python main.py "$CONFIG"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ 训练完成"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi


