#!/bin/bash
# 测试混合保护OLM编码的故障注入效果

# python tools/test_hybrid_olm_fault_injection.py \
#     --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
#     --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar \
#     --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
#     --olm_json olm_encoding_hybrid_v2_features_0.json \
#     --layer features.0 \
#     --ber 1e-1 \
#     --device cuda \
#     --seed 42

python tools/test_hybrid_olm_fault_injection.py \
    --config configs/eval/eval_alexnet_cifar10_single_gpu_v2.yaml \
    --ckpt training/alexnet_cifar10_single_gpu_normal_srqat_ls1e_m6/alexnet_cifar10_single_gpu_normal_srqat_ls1e_m6_checkpoint.pth.tar \
    --bit_width_config search/alexnet_cifar10_single_gpu_v2_search_bit_width_config.json \
    --olm_json olm_encoding_hybrid_v2_features_0.json \
    --layer features.0 \
    --ber 1e-1 \
    --device cuda \
    --seed 42