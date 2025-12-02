#!/bin/bash
# 测试OLM编码的故障注入效果（使用预生成的OLM映射JSON文件）

# 配置参数
CONFIG="configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml"
CHECKPOINT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json"
OLM_JSON="olm_encoding_features_0_classifier_1.json"

# BER值（可以修改为不同的值进行测试）
BER=0.1  # 1e-1

# 测试的层（从JSON文件中可以看到有 features.0 和 classifier.1）
LAYERS=("features.0" "classifier.1")

# 设备
DEVICE="cuda"

# 随机种子
SEED=42

echo "=========================================="
echo "OLM编码故障注入测试"
echo "=========================================="
echo "配置文件: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Bit-width配置: $BIT_WIDTH_CONFIG"
echo "OLM映射JSON: $OLM_JSON"
echo "BER: $BER"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$BIT_WIDTH_CONFIG" ]; then
    echo "错误: Bit-width配置文件不存在: $BIT_WIDTH_CONFIG"
    exit 1
fi

if [ ! -f "$OLM_JSON" ]; then
    echo "错误: OLM映射JSON文件不存在: $OLM_JSON"
    exit 1
fi

# 对每个层进行测试
for LAYER in "${LAYERS[@]}"; do
    echo "=========================================="
    echo "测试层: $LAYER"
    echo "=========================================="
    echo ""
    
    python tools/test_olm_fault_injection.py \
        --config "$CONFIG" \
        --ckpt "$CHECKPOINT" \
        --bit_width_config "$BIT_WIDTH_CONFIG" \
        --ber "$BER" \
        --layer "$LAYER" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --olm_json "$OLM_JSON"
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "=========================================="
echo "所有测试完成"
echo "=========================================="

