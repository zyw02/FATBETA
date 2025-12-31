#!/bin/bash
# 训练多个层的OLM编码示例
# 注意：classifier.0是Dropout层，不是Linear层，应该使用classifier.1

python tools/train_olm_encoder.py \
    --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
    --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
    --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
    --layer "features.0,classifier.1,features.3,features.6" \
    --output olm_encoding_features_0_classifier_1_search.json \
    --method both \
    --max_iterations 200000 \
    --test_fault_injection \
    --test_ber 0.1

# ================================================================================
# 故障注入测试完成!
# ================================================================================
# Baseline:        88.91%
# 二进制编码:      30.51% (下降 58.40%)
# 格雷码编码:      34.15% (改进 3.64%)
# OLM编码:         48.75% (改进 18.24%)
#   → OLM编码优于格雷码 14.60%    

