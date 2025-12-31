#!/bin/bash
# 训练 DenoiseRestorer
# 使用 BER=1e-2 的 activation dumps，MSE loss，Adam 优化器，step decay LR

set -euo pipefail

CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
DEVICE="cuda:0"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_FAT_a92b25search_bit_width_config.json"
BIT_CONFIG_INDEX=0

# 数据收集参数
TARGET_LAYERS="features.0"
COLLECT_BER=1e-2
COLLECT_BATCHES=80
USE_TRAIN_SPLIT=true

# 训练参数
EPOCHS=100
BATCH_SIZE=8
LR=1e-3
HIDDEN_DIM=64
CLIP_MARGIN=1.25
NUM_STAGES=3  # 堆叠的 DenoiseRestorer 数量（默认 2）

# 输出路径
ACTIVATION_DIR="activation_dumps/denoise_restorer"
CLEAN_DIR="${ACTIVATION_DIR}/clean"
FAULT_DIR="${ACTIVATION_DIR}/fault"
PROFILE_DIR="layer_profiles"
CLEAN_PROFILE="${PROFILE_DIR}/clean_w6a6.pt"
OUTPUT_CKPT="checkpoints/denoise_restorer.pt"

echo "=========================================="
echo "Training DenoiseRestorer"
echo "=========================================="
echo "Target layers: ${TARGET_LAYERS}"
echo "BER for data collection: ${COLLECT_BER}"
echo "Training epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "=========================================="

# Step 1: 收集 clean activations (w6a6)
echo ""
echo "Step 1: Collecting clean activations (w6a6)..."
mkdir -p "${CLEAN_DIR}"
python tools/collect_layer_statistics.py \
    --config "${CONFIG}" \
    --stage1_ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --mode clean \
    --target_layers "${TARGET_LAYERS}" \
    --output "${CLEAN_PROFILE}" \
    --save_activations \
    --activations_dir "${ACTIVATION_DIR}" \
    --num_batches "${COLLECT_BATCHES}" \
    --force_w6a6 \
    $([[ "${USE_TRAIN_SPLIT}" == "true" ]] && echo "--use_train_split")

echo "✓ Clean activations saved to ${CLEAN_DIR}"

# Step 2: 收集 fault activations (BER=1e-2, w6a6)
echo ""
echo "Step 2: Collecting fault activations (BER=${COLLECT_BER}, w6a6)..."
mkdir -p "${FAULT_DIR}"
python tools/collect_layer_statistics.py \
    --config "${CONFIG}" \
    --stage1_ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --mode fault \
    --target_layers "${TARGET_LAYERS}" \
    --ber "${COLLECT_BER}" \
    --output "${PROFILE_DIR}/fault_${COLLECT_BER}.pt" \
    --save_activations \
    --activations_dir "${ACTIVATION_DIR}" \
    --num_batches "${COLLECT_BATCHES}" \
    --force_w6a6 \
    $([[ "${USE_TRAIN_SPLIT}" == "true" ]] && echo "--use_train_split")

echo "✓ Fault activations saved to ${FAULT_DIR}"

# Step 3: 训练 DenoiseRestorer
echo ""
echo "Step 3: Training DenoiseRestorer..."
echo "  - Loss: MSE"
echo "  - Optimizer: Adam"
echo "  - LR scheduler: Step decay (every 30% epochs, gamma=0.1)"
echo "  - Epochs: ${EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Initial LR: ${LR}"

mkdir -p "$(dirname "${OUTPUT_CKPT}")"

python tools/train_sensitive_repair_head.py \
    --clean_dir "${CLEAN_DIR}" \
    --fault_dir "${FAULT_DIR}" \
    --layers "${TARGET_LAYERS}" \
    --output "${OUTPUT_CKPT}" \
    --repair_mode restorer_v4 \
    --layer_profile "${CLEAN_PROFILE}" \
    --model_config "${CONFIG}" \
    --model_ckpt "${CKPT}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --clip_margin "${CLIP_MARGIN}" \
    --num_stages "${NUM_STAGES}" \
    --device "${DEVICE}"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoint saved to: ${OUTPUT_CKPT}"
echo "Clean activations: ${CLEAN_DIR}"
echo "Fault activations: ${FAULT_DIR}"
echo "Clean profile: ${CLEAN_PROFILE}"
echo ""
echo "To evaluate the trained restorer, use:"
echo "  python tools/eval_gradient_statistics_restorer.py \\"
echo "    --config ${CONFIG} \\"
echo "    --stage1_ckpt ${CKPT} \\"
echo "    --repair_mode denoise_restorer \\"
echo "    --repair_head_ckpt ${OUTPUT_CKPT} \\"
echo "    --layer_profile ${CLEAN_PROFILE} \\"
echo "    --force_w6a6 \\"
echo "    --device ${DEVICE}"

