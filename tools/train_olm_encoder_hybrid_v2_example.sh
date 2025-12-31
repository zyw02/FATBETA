#!/bin/bash

# 混合保护OLM编码器训练示例脚本 V2
# 方案：bit7/bit0/bit1做冗余，bit6-2提取出来做OLM（一对多编码，一对一解码）

# 配置参数
CONFIG="configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json"
LAYER="features.0"
OUTPUT="olm_encoding_hybrid_v2_features_0.json"

# 训练参数
METHOD="genetic"
MAX_ITERATIONS=100000
USE_SENSITIVITY=true
GRADIENT_SAMPLES=-1  # -1表示使用整个训练集
POPULATION_SIZE=200
CROSSOVER_RATE=0.8
MUTATION_RATE=0.1
ELITE_SIZE=5
SEED=42

echo "=================================================================================="
echo "混合保护OLM编码器训练 V2"
echo "=================================================================================="
echo "配置: $CONFIG"
echo "Checkpoint: $CKPT"
echo "目标层: $LAYER"
echo "输出文件: $OUTPUT"
echo ""
echo "保护方案:"
echo "  - bit7, bit0, bit1: 做冗余（3倍冗余保护bit7）"
echo "  - bit6-2: 提取出来当做无符号数，做OLM训练（5位，32个值）"
echo ""
echo "映射关系:"
echo "  - 编码：一对多（同一个原始二进制数可以对应多个OLM码值）"
echo "  - 解码：一对一（每个OLM编码能且仅能对应唯一的一个二进制数）"
echo ""
echo "训练参数:"
echo "  - 方法: $METHOD"
echo "  - 最大迭代次数: $MAX_ITERATIONS"
echo "  - 使用敏感度权重: $USE_SENSITIVITY"
echo "  - 梯度样本数: $GRADIENT_SAMPLES (整个训练集)"
echo "  - 种群大小: $POPULATION_SIZE"
echo "  - 交叉率: $CROSSOVER_RATE"
echo "  - 变异率: $MUTATION_RATE"
echo "  - 精英数量: $ELITE_SIZE"
echo "  - 随机种子: $SEED"
echo "=================================================================================="
echo ""

# 构建命令
CMD="python3 tools/train_olm_encoder_hybrid_v2.py \
    --config $CONFIG \
    --ckpt $CKPT \
    --bit_width_config $BIT_WIDTH_CONFIG \
    --layer $LAYER \
    --output $OUTPUT \
    --method $METHOD \
    --max_iterations $MAX_ITERATIONS \
    --gradient_samples $GRADIENT_SAMPLES \
    --population_size $POPULATION_SIZE \
    --crossover_rate $CROSSOVER_RATE \
    --mutation_rate $MUTATION_RATE \
    --elite_size $ELITE_SIZE \
    --seed $SEED"

if [ "$USE_SENSITIVITY" = true ]; then
    CMD="$CMD --use_sensitivity"
fi

# 运行训练
echo "开始训练..."
echo ""
$CMD

echo ""
echo "=================================================================================="
echo "训练完成！"
echo "结果已保存到: $OUTPUT"
echo "=================================================================================="



