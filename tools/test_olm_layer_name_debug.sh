#!/bin/bash
# 测试OLM编码时layer_name的检查脚本

# 使用之前生成的OLM映射文件
python tools/test_olm_layer_name_check.py \
    --config configs/training/train_alexnet_cifar10_learnable_olm_fat.yaml \
    --ckpt training/alexnet_cifar10_learnable_olm_fat_v3/alexnet_cifar10_learnable_olm_fat_v3_checkpoint.pth.tar \
    --olm_json olm_encoding_features_0_classifier_1.json \
    --layers features.0 classifier.1 \
    --num_batches 1 \
    2>&1 | tee olm_layer_name_debug.log



