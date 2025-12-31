#!/bin/bash
# 增强的OLM编码器训练示例（基于Gemini建议）

echo "=================================================================================="
echo "增强的OLM编码器训练 - 基于Gemini建议的改进"
echo "=================================================================================="
echo ""

# 基础参数
CONFIG="configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json"
OUTPUT="olm_encoding_enhanced.json"

# 方法1: 只使用敏感度权重（Hessian感知）
echo "方法1: 只使用敏感度权重（Hessian感知）"
echo "----------------------------------------------------------------------------------"
python tools/train_olm_encoder_enhanced.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --bit_width_config "$BIT_WIDTH_CONFIG" \
    --layer "features.0" \
    --output "${OUTPUT%.json}_sensitivity_only.json" \
    --method genetic \
    --use_sensitivity \
    --gradient_samples -1 \
    --population_size 200 \
    --max_iterations 2000 \
    --test_fault_injection \
    --test_ber 0.1

echo ""
echo "=================================================================================="
echo ""

# 方法2: 只使用多对一映射（Surjective Mapping）
# echo "方法2: 只使用多对一映射（Surjective Mapping）"
# echo "----------------------------------------------------------------------------------"
# python tools/train_olm_encoder_enhanced.py \
#     --config "$CONFIG" \
#     --ckpt "$CKPT" \
#     --bit_width_config "$BIT_WIDTH_CONFIG" \
#     --layer "features.0,classifier.1" \
#     --output "${OUTPUT%.json}_surjective_only.json" \
#     --method genetic \
#     --use_surjective \
#     --population_size 200 \
#     --max_iterations 2000 \
#     --top_k_values 10 \
#     --test_fault_injection \
#     --test_ber 0.1

# echo ""
# echo "=================================================================================="
# echo ""

# # 方法3: 同时使用敏感度权重和多对一映射（完整改进方案）
# echo "方法3: 同时使用敏感度权重和多对一映射（完整改进方案）"
# echo "----------------------------------------------------------------------------------"
# python tools/train_olm_encoder_enhanced.py \
#     --config "$CONFIG" \
#     --ckpt "$CKPT" \
#     --bit_width_config "$BIT_WIDTH_CONFIG" \
#     --layer "features.0,classifier.1" \
#     --output "${OUTPUT%.json}_full_enhanced.json" \
#     --method genetic \
#     --use_sensitivity \
#     --use_surjective \
#     --top_k_values 10 \
#     --gradient_samples -1 \
#     --population_size 200 \
#     --max_iterations 2000 \
#     --test_fault_injection \
#     --test_ber 0.1

# echo ""
# echo "=================================================================================="
# echo ""



# echo ""
# echo "=================================================================================="
# echo "训练完成！"
# echo "=================================================================================="
# echo ""
# echo "生成的文件："
# echo "  - ${OUTPUT%.json}_sensitivity_only.json: 只使用敏感度权重"
# echo "  - ${OUTPUT%.json}_surjective_only.json: 只使用多对一映射"
# echo "  - ${OUTPUT%.json}_full_enhanced.json: 完整改进方案（贪婪）"
# echo "  - ${OUTPUT%.json}_sa_full.json: 完整改进方案（模拟退火）"
# echo ""
# echo "建议："
# echo "  1. 比较四种方法的准确率提升"
# echo "  2. 选择效果最好的方案"
# echo "  3. 可以调整 --top_k_values 参数（5-20）进一步优化"
# echo "  4. 可以调整 --gradient_samples 参数（50-200）平衡计算时间和精度"
# echo ""

