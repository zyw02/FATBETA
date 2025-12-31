#!/bin/bash
# 使用改进的OLM编码器训练示例
# 改进点：
# 1. 使用梯度信息衡量权重重要性 (--use_gradient)
# 2. 考虑多bit翻转 (--consider_multi_bit --max_hamming_dist 3)
# 3. 针对高BER场景优化 (--ber 0.1)

echo "=================================================================================="
echo "改进的OLM编码器训练 - 使用梯度信息和多bit翻转"
echo "=================================================================================="
echo ""

# 基础参数
CONFIG="configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json"
OUTPUT="olm_encoding_improved.json"

# 方法1: 只使用梯度信息（不考虑多bit翻转）
echo "方法1: 只使用梯度信息（单bit翻转）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_improved.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0,classifier.1" \
    --output "${OUTPUT%.json}_gradient_only.json" \
    --method simulated_annealing \
    --max_iterations 200000 \
    --ber 0.1 \
    --use_gradient \
    --gradient_samples 100 \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo ""

# 方法2: 只考虑多bit翻转（不使用梯度信息）
echo "方法2: 只考虑多bit翻转（不使用梯度信息）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_improved.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0,classifier.1" \
    --output "${OUTPUT%.json}_multibit_only.json" \
    --method simulated_annealing \
    --max_iterations 200000 \
    --ber 0.1 \
    --consider_multi_bit \
    --max_hamming_dist 3 \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo ""

# 方法3: 同时使用梯度信息和多bit翻转（完整改进方案）
echo "方法3: 同时使用梯度信息和多bit翻转（完整改进方案）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_improved.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0,classifier.1" \
    --output "${OUTPUT%.json}_full_improved.json" \
    --method simulated_annealing \
    --max_iterations 200000 \
    --ber 0.1 \
    --consider_multi_bit \
    --max_hamming_dist 3 \
    --use_gradient \
    --gradient_samples 100 \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo "训练完成！"
echo "=================================================================================="
echo ""
echo "生成的文件："
echo "  - ${OUTPUT%.json}_gradient_only.json: 只使用梯度信息"
echo "  - ${OUTPUT%.json}_multibit_only.json: 只考虑多bit翻转"
echo "  - ${OUTPUT%.json}_full_improved.json: 完整改进方案"
echo ""
echo "建议："
echo "  1. 比较三种方法的准确率提升"
echo "  2. 选择效果最好的方案"
echo "  3. 可以调整 --max_iterations 和 --max_hamming_dist 参数进一步优化"
echo ""

