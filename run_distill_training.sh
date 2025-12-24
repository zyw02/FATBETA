#!/bin/bash
set -e

# Knowledge Distillation Training Pipeline
# Step 1: Train large teacher restorer
# Step 2: Distill knowledge to small student restorer

STAGE1_CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage1.yaml"
STAGE2_CONFIG="configs/training/train_alexnet_cifar10_sensitive_stage2.yaml"
STAGE1_CKPT="training/alexnet_cifar10_sensitive_stage1/alexnet_cifar10_sensitive_stage1_checkpoint.pth.tar"

# Teacher checkpoint path
TEACHER_CKPT="sensitive_training/alexnet_cifar10_sensitive_stage2_teacher/teacher_checkpoint.pth.tar"

echo "=========================================="
echo "Knowledge Distillation Training Pipeline"
echo "=========================================="

# Step 1: Train teacher restorer (large model)
echo ""
echo "==== Step 1: Training Teacher Restorer (large model) ===="
echo "Temporarily using teacher_hidden_dim for training..."

# Modify config temporarily for teacher training
python - <<'PYTHON_SCRIPT'
import yaml
import shutil

# Read config
with open('configs/training/train_alexnet_cifar10_sensitive_stage2.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set teacher hidden_dim for training
config['sensitive_restorer']['restorer_hidden_dim'] = config['sensitive_restorer'].get('teacher_hidden_dim', 256)

# Save temporary config
with open('configs/training/train_alexnet_cifar10_sensitive_stage2_teacher.yaml', 'w') as f:
    yaml.dump(config, f)

print(f"Teacher hidden_dim: {config['sensitive_restorer']['restorer_hidden_dim']}")
PYTHON_SCRIPT

python train_sensitive_restorer.py \
    --config configs/training/train_alexnet_cifar10_sensitive_stage2_teacher.yaml \
    --stage1_ckpt ${STAGE1_CKPT} \
    --output_dir sensitive_training/alexnet_cifar10_sensitive_stage2_teacher

# Rename checkpoint
if [ -f "sensitive_training/alexnet_cifar10_sensitive_stage2_teacher/sensitive_stage2_checkpoint.pth.tar" ]; then
    mv sensitive_training/alexnet_cifar10_sensitive_stage2_teacher/sensitive_stage2_checkpoint.pth.tar ${TEACHER_CKPT}
    echo "✓ Teacher checkpoint saved to ${TEACHER_CKPT}"
fi

# Step 2: Distill knowledge to student restorer (small model)
echo ""
echo "==== Step 2: Distilling Knowledge to Student Restorer (small model) ===="
echo "Using teacher checkpoint: ${TEACHER_CKPT}"

python train_sensitive_restorer_distill.py \
    --config ${STAGE2_CONFIG} \
    --stage1_ckpt ${STAGE1_CKPT} \
    --teacher_ckpt ${TEACHER_CKPT} \
    --output_dir sensitive_training/alexnet_cifar10_sensitive_stage2_distilled

echo ""
echo "=========================================="
echo "Knowledge Distillation Complete!"
echo "=========================================="
echo "Teacher checkpoint: ${TEACHER_CKPT}"
echo "Student checkpoint: sensitive_training/alexnet_cifar10_sensitive_stage2_distilled/distilled_student_checkpoint.pth.tar"

