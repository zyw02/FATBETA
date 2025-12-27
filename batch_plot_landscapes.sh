#!/bin/bash

# 批量绘图脚本：对 plot_model 文件夹下的所有模型进行 Loss Landscape 可视化
# 开启 --no_norm 以对比绝对锐度，暴露模型真实脆弱性

# 1. 设置基础路径
MODEL_DIR="plot_model"
CONFIG_FILE="configs/training/train_resnet18_cifar10_single_gpu.yaml"
PLOT_SCRIPT="visualize_loss_landscape.py"

# 2. 确保输出文件夹存在
mkdir -p exp/2d exp/3d

echo "Starting batch plotting for models in $MODEL_DIR..."
echo "Using config: $CONFIG_FILE"
echo "Mode: Absolute Sharpness (--no_norm)"
echo "----------------------------------------------------"

# 3. 遍历 plot_model 目录下的所有 .pth.tar 文件
for ckpt in "$MODEL_DIR"/*.pth.tar; do
    if [ -f "$ckpt" ]; then
        model_name=$(basename "$ckpt")
        echo "Processing model: $model_name"
        
        # 调用可视化脚本
        # 增加 --no_norm 以对比绝对锐度，并固定位宽为 6-bit
        python "$PLOT_SCRIPT" "$CONFIG_FILE" --checkpoint "$ckpt" --bit 6 --no_norm
        
        echo "Finished $model_name"
        echo "----------------------------------------------------"
    fi
done

echo "All sharpness plots completed! Check exp/2d and exp/3d folders."

