#!/bin/bash
#
# Unified pipeline for the sensitive-layer restorer workflow:
#   1. Layer sensitivity analysis
#   2. Clean / fault statistics collection
#   3. Evaluation with the sensitive restorer
#
# Usage (example):
#   bash tools/run_sensitive_restorer_pipeline.sh \
#       --config configs/training/train_alexnet_cifar10_sensitive_stage1.yaml \
#       --ckpt training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"
DEVICE="cuda:0"
BIT_WIDTH_CONFIG="search/alexnet_cifar10_FAT_a92b25search_bit_width_config.json"
BIT_CONFIG_INDEX=0
TARGET_LAYERS="features.0,features.3,features.8"
SENSITIVE_LAYER_GROUPS=""
SENSITIVITY_SAMPLES=1000
SENSITIVITY_BER=1e-1
STAT_BATCHES=80
COLLECT_BER=1e-1
REST_BERS="1e-3 1e-2 5e-2 1e-1"
REST_SKIP_FIRST_LAST=false
REST_SEED=42
BER_POLICY="configs/restorer/ber_policy_default.json"

REST_Z_THRESH=3.0
REST_STD_LOW=0.5
REST_STD_HIGH=2.0
REST_CLIP_MARGIN=1.25
REPAIR_MODE="lightweight_denoiser"
RUN_TAG="$(date +'%Y%m%d-%H%M%S')"
REPAIR_HEAD_CKPT=""
MLP_HIDDEN_DIM=64
MLP_TRAIN_EPOCHS=100
MLP_TRAIN_BATCH=4
MLP_TRAIN_LR=1e-1
ENABLE_COMBO_EVAL=false
COMBO_SIZES=""
AUTO_TOP_K=2

PROFILE_DIR="layer_profiles"
LOG_DIR="logs/restorer_pipeline"
SUMMARY_PATH="${LOG_DIR}/${RUN_TAG}_sensitivity_summary.json"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --config PATH           Config file (default: ${CONFIG})
  --ckpt PATH             Stage1 checkpoint (default: ${CKPT})
  --device STR            Device string (default: ${DEVICE})
  --bit-width-config PATH Bit-width JSON (default: ${BIT_WIDTH_CONFIG})
  --bit-config-index IDX  Index inside JSON (default: ${BIT_CONFIG_INDEX})
  --target-layers STR     Comma-separated sensitive layers (default: ${TARGET_LAYERS})
  --layer-groups STR      Pipe-separated layer groups for evaluation (e.g. "f0|f0,f3")
  --sensitivity-samples N Number of samples for ranking (default: ${SENSITIVITY_SAMPLES})
  --sensitivity-ber VAL   BER used in ranking (default: ${SENSITIVITY_BER})
  --stat-batches N        Batches for statistics collection (default: ${STAT_BATCHES})
  --collect-ber VAL       BER for fault statistics (default: ${COLLECT_BER})
  --rest-ber-values STR   Quote-wrapped list for evaluation (default: "${REST_BERS}")
  --skip-first-last (true|false)  Skip FI on first/last layers (default: ${REST_SKIP_FIRST_LAST})
  --seed N                Seed for FI / scripts (default: ${REST_SEED})
  --repair-mode STR       Restorer repair mode (rule/mlp_local/mlp_poly, default: ${REPAIR_MODE})
  --ber-policy PATH       BER policy JSON (default: ${BER_POLICY})
  --repair-head-ckpt PATH Optional repair head checkpoint for learning mode
  --mlp-hidden-dim N      Hidden dim for MLP repair head (default: ${MLP_HIDDEN_DIM})
  --combo-sizes STR       Space-separated combo sizes (default: "${COMBO_SIZES}")
  --enable-combo-eval (true|false)  Run combo eval (default: ${ENABLE_COMBO_EVAL})
  --mlp-train-epochs N    Training epochs for repair head (default: ${MLP_TRAIN_EPOCHS})
  --mlp-train-batch N     Activation pairs per step (default: ${MLP_TRAIN_BATCH})
  --mlp-train-lr VAL      Learning rate for repair head (default: ${MLP_TRAIN_LR})
  --auto-top-k N          Automatically pick top-K sensitive layers (default: ${AUTO_TOP_K})
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --ckpt) CKPT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --bit-width-config) BIT_WIDTH_CONFIG="$2"; shift 2 ;;
        --bit-config-index) BIT_CONFIG_INDEX="$2"; shift 2 ;;
        --target-layers) TARGET_LAYERS="$2"; shift 2 ;;
        --layer-groups) SENSITIVE_LAYER_GROUPS="$2"; shift 2 ;;
        --sensitivity-samples) SENSITIVITY_SAMPLES="$2"; shift 2 ;;
        --sensitivity-ber) SENSITIVITY_BER="$2"; shift 2 ;;
        --stat-batches) STAT_BATCHES="$2"; shift 2 ;;
        --collect-ber) COLLECT_BER="$2"; shift 2 ;;
        --rest-ber-values) REST_BERS="$2"; shift 2 ;;
        --skip-first-last) REST_SKIP_FIRST_LAST="$2"; shift 2 ;;
        --seed) REST_SEED="$2"; shift 2 ;;
        --repair-mode) REPAIR_MODE="$2"; shift 2 ;;
        --ber-policy) BER_POLICY="$2"; shift 2 ;;
        --repair-head-ckpt) REPAIR_HEAD_CKPT="$2"; shift 2 ;;
        --mlp-hidden-dim) MLP_HIDDEN_DIM="$2"; shift 2 ;;
        --combo-sizes) COMBO_SIZES="$2"; shift 2 ;;
        --enable-combo-eval) ENABLE_COMBO_EVAL="$2"; shift 2 ;;
        --mlp-train-epochs) MLP_TRAIN_EPOCHS="$2"; shift 2 ;;
        --mlp-train-batch) MLP_TRAIN_BATCH="$2"; shift 2 ;;
        --mlp-train-lr) MLP_TRAIN_LR="$2"; shift 2 ;;
        --auto-top-k) AUTO_TOP_K="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "${LOG_DIR}" "${PROFILE_DIR}"
LOG_PREFIX="${LOG_DIR}/${RUN_TAG}"

LAYER_GROUP_ARRAY=()
COLLECT_LAYERS=""
ACTIVATION_DIR="activation_dumps/${RUN_TAG}"
ACTIVATION_ARGS=()
COLLECT_BATCHES="${STAT_BATCHES}"
if [ "${REPAIR_MODE}" != "rule" ]; then
    COLLECT_BATCHES=0
    ACTIVATION_ARGS=(
        --save_activations
        --activations_dir "${ACTIVATION_DIR}"
        --max_activation_batches 200
        --force_w6a6
    )
fi

echo ">>> Step 1: Layer sensitivity analysis"
python tools/analyze_multi_bit_sensitivity.py \
    --config "${CONFIG}" \
    --resume_path "${CKPT}" \
    --ber "${SENSITIVITY_BER}" \
    --num_samples "${SENSITIVITY_SAMPLES}" \
    --summary_path "${SUMMARY_PATH}" \
    > "${LOG_PREFIX}_sensitivity.log"
echo "    Log saved to ${LOG_PREFIX}_sensitivity.log"

if [ "${AUTO_TOP_K}" -gt 0 ]; then
    if [ ! -f "${SUMMARY_PATH}" ]; then
        echo "Auto-top-k requested but summary file not found at ${SUMMARY_PATH}"
        exit 1
    fi
    TARGET_LAYERS=$(python - <<PY
import json
from pathlib import Path
summary = Path(r"${SUMMARY_PATH}")
data = json.load(summary.open())
ranking = data.get("ranking") or []
top = [entry["layer"] for entry in ranking[:${AUTO_TOP_K}]]
print(",".join(top))
PY
)
    echo "Auto-selected top-${AUTO_TOP_K} layers: ${TARGET_LAYERS}"
fi

if [ -n "${SENSITIVE_LAYER_GROUPS}" ]; then
    IFS='|' read -r -a LAYER_GROUP_ARRAY <<< "${SENSITIVE_LAYER_GROUPS}"
elif [ -n "${TARGET_LAYERS}" ]; then
    LAYER_GROUP_ARRAY=("${TARGET_LAYERS}")
else
    echo "No target layers specified; set TARGET_LAYERS or use --auto-top-k."
    exit 1
fi

COLLECT_LAYERS=$(printf "%s\n" "${LAYER_GROUP_ARRAY[@]}" | tr ',' '\n' | tr -d ' ' | sed '/^$/d' | sort -u | tr '\n' ',')
COLLECT_LAYERS=${COLLECT_LAYERS%,}

echo ">>> Step 2: Collect per-layer statistics"
CLEAN_PROFILE="${PROFILE_DIR}/clean_${RUN_TAG}.pt"
FAULT_PROFILE="${PROFILE_DIR}/fault_${COLLECT_BER}_${RUN_TAG}.pt"

python tools/collect_layer_statistics.py \
    --config "${CONFIG}" \
    --stage1_ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --mode clean \
    --output "${CLEAN_PROFILE}" \
    --num_batches "${COLLECT_BATCHES}" \
    --target_layers "${COLLECT_LAYERS}" \
    --bit_width_config "${BIT_WIDTH_CONFIG}" \
    --config_index "${BIT_CONFIG_INDEX}" \
    --seed "${REST_SEED}" \
    --use_train_split \
    "${ACTIVATION_ARGS[@]}" \
    > "${LOG_PREFIX}_collect_clean.log"
echo "    Clean stats saved to ${CLEAN_PROFILE}"

python tools/collect_layer_statistics.py \
    --config "${CONFIG}" \
    --stage1_ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --mode fault \
    --ber "${COLLECT_BER}" \
    --output "${FAULT_PROFILE}" \
    --num_batches "${COLLECT_BATCHES}" \
    --target_layers "${COLLECT_LAYERS}" \
    --bit_width_config "${BIT_WIDTH_CONFIG}" \
    --config_index "${BIT_CONFIG_INDEX}" \
    $([[ "${REST_SKIP_FIRST_LAST}" == "true" ]] && echo "--skip_first_last") \
    --seed "${REST_SEED}" \
    --use_train_split \
    "${ACTIVATION_ARGS[@]}" \
    > "${LOG_PREFIX}_collect_fault.log"
echo "    Fault stats saved to ${FAULT_PROFILE}"

if [ "${REPAIR_MODE}" != "rule" ]; then
    DEFAULT_HEAD_PATH="learning_heads/repair_${RUN_TAG}.pt"
    if [ -z "${REPAIR_HEAD_CKPT}" ]; then
        REPAIR_HEAD_CKPT="${DEFAULT_HEAD_PATH}"
    fi
    mkdir -p "$(dirname "${REPAIR_HEAD_CKPT}")"
    echo ">>> Training learning-based repair head"
    python tools/train_sensitive_repair_head.py \
        --clean_dir "${ACTIVATION_DIR}/clean" \
        --fault_dir "${ACTIVATION_DIR}/fault" \
        --layers "${TARGET_LAYERS}" \
        --output "${REPAIR_HEAD_CKPT}" \
        --layer_profile "${CLEAN_PROFILE}" \
        --repair_mode "${REPAIR_MODE}" \
        --hidden_dim "${MLP_HIDDEN_DIM}" \
        --clip_margin "${REST_CLIP_MARGIN}" \
        --epochs "${MLP_TRAIN_EPOCHS}" \
        --batch_size "${MLP_TRAIN_BATCH}" \
        --lr "${MLP_TRAIN_LR}" \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG_PREFIX}_train_repair.log"
    echo "    Learning repair head saved to ${REPAIR_HEAD_CKPT}"
fi

echo ">>> Step 3: Evaluate with sensitive restorer"
for GROUP in "${LAYER_GROUP_ARRAY[@]}"; do
    GROUP_TRIMMED=$(echo "${GROUP}" | tr -d ' ')
    GROUP_TAG=$(echo "${GROUP_TRIMMED}" | tr ',' '_' | tr '.' '_')
    LOG_FILE="${LOG_PREFIX}_evaluation_${GROUP_TAG}.log"
    python tools/eval_gradient_statistics_restorer.py \
        --config "${CONFIG}" \
        --stage1_ckpt "${CKPT}" \
        --device "${DEVICE}" \
        --bit_width_config "${BIT_WIDTH_CONFIG}" \
        --config_index "${BIT_CONFIG_INDEX}" \
        --ber_values ${REST_BERS} \
        --seed "${REST_SEED}" \
        $([[ "${REST_SKIP_FIRST_LAST}" == "true" ]] && echo "--skip_first_last") \
        --restorer_mode sensitive \
        --layer_profile "${CLEAN_PROFILE}" \
        --fault_layer_profile "${FAULT_PROFILE}" \
        --fault_profile_ber "${COLLECT_BER}" \
        $([ -n "${BER_POLICY}" ] && echo "--ber_policy ${BER_POLICY}") \
        --sensitive_layers "${GROUP_TRIMMED}" \
        --sensitive_z_thresh "${REST_Z_THRESH}" \
        --sensitive_std_ratio_bounds "${REST_STD_LOW}" "${REST_STD_HIGH}" \
        --sensitive_clip_margin "${REST_CLIP_MARGIN}" \
        --repair_mode "${REPAIR_MODE}" \
        $([ -n "${REPAIR_HEAD_CKPT}" ] && echo "--repair_head_ckpt ${REPAIR_HEAD_CKPT}") \
        --mlp_hidden_dim "${MLP_HIDDEN_DIM}" \
        > "${LOG_FILE}"
    echo "    Evaluation (${GROUP_TRIMMED}) log saved to ${LOG_FILE}"
done

if [ "${ENABLE_COMBO_EVAL}" = "true" ] && [ -f "${SUMMARY_PATH}" ]; then
    echo ">>> Step 4: Evaluating additional sensitive-layer combinations"
    COMBO_LOG="${LOG_PREFIX}_combo_eval.log"
    python tools/eval_sensitive_layer_sets.py \
        --config "${CONFIG}" \
        --ckpt "${CKPT}" \
        --device "${DEVICE}" \
        --layer_profile "${CLEAN_PROFILE}" \
        --summary_json "${SUMMARY_PATH}" \
        --bit_width_config "${BIT_WIDTH_CONFIG}" \
        --config_index "${BIT_CONFIG_INDEX}" \
        --ber_values ${REST_BERS} \
        --combo_sizes ${COMBO_SIZES} \
        --sensitive_args "--fault_layer_profile ${FAULT_PROFILE} --fault_profile_ber ${COLLECT_BER} $([[ -n \"${BER_POLICY}\" ]] && echo --ber_policy ${BER_POLICY}) --sensitive_z_thresh ${REST_Z_THRESH} --sensitive_std_ratio_bounds ${REST_STD_LOW} ${REST_STD_HIGH} --sensitive_clip_margin ${REST_CLIP_MARGIN} --repair_mode ${REPAIR_MODE} $([[ -n \"${REPAIR_HEAD_CKPT}\" ]] && echo --repair_head_ckpt ${REPAIR_HEAD_CKPT}) --mlp_hidden_dim ${MLP_HIDDEN_DIM}" \
        > "${COMBO_LOG}" 2>&1
    echo "    Combo evaluation summary saved to ${COMBO_LOG}"
fi

echo "Pipeline finished. Profiles stored in ${PROFILE_DIR}, logs in ${LOG_DIR}."

