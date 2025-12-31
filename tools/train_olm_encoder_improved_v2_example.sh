#!/bin/bash
# 使用改进的OLM编码器 V2 训练示例
# 在方法2（多bit翻转）基础上的数学改进

echo "=================================================================================="
echo "改进的OLM编码器 V2 训练 - 在方法2基础上的数学改进"
echo "=================================================================================="
echo ""

# 基础参数
CONFIG="configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json"
OUTPUT="olm_encoding_improved_v2.json"

# 方法V2: 完整改进方案（值重要性 + 局部一致性）
echo "方法V2: 完整改进方案（值重要性 + 局部一致性）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_improved_v2.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0,classifier.1" \
    --output "${OUTPUT%.json}_full.json" \
    --method simulated_annealing \
    --max_iterations 500000 \
    --ber 0.1 \
    --consider_multi_bit \
    --max_hamming_dist 3 \
    --use_value_importance \
    --use_local_consistency \
    --local_consistency_weight 0.1 \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo ""

# 方法V2-1: 只使用值重要性（不使用局部一致性）
echo "方法V2-1: 只使用值重要性（不使用局部一致性）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_improved_v2.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0,classifier.1" \
    --output "${OUTPUT%.json}_value_importance_only.json" \
    --method simulated_annealing \
    --max_iterations 200000 \
    --ber 0.1 \
    --consider_multi_bit \
    --max_hamming_dist 3 \
    --use_value_importance \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo ""

# 方法V2-2: 只使用局部一致性（不使用值重要性）
echo "方法V2-2: 只使用局部一致性（不使用值重要性）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_improved_v2.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0,classifier.1" \
    --output "${OUTPUT%.json}_local_consistency_only.json" \
    --method simulated_annealing \
    --max_iterations 200000 \
    --ber 0.1 \
    --consider_multi_bit \
    --max_hamming_dist 3 \
    --use_local_consistency \
    --local_consistency_weight 0.1 \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo "训练完成！"
echo "=================================================================================="
echo ""
echo "生成的文件："
echo "  - ${OUTPUT%.json}_full.json: 完整改进方案"
echo "  - ${OUTPUT%.json}_value_importance_only.json: 只使用值重要性"
echo "  - ${OUTPUT%.json}_local_consistency_only.json: 只使用局部一致性"
echo ""
echo "建议："
echo "  1. 比较三种方法的准确率提升"
echo "  2. 选择效果最好的方案"
echo "  3. 可以调整 --local_consistency_weight 参数进一步优化"
echo ""

