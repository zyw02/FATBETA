# 单GPU (11GB 2080Ti) 复现指南

本文档说明如何在单张11GB显存的2080Ti上复现RFQuant项目。

## 主要修改

### 1. 配置文件调整

针对11GB显存的限制，做了以下调整：

**训练配置** (`configs/training/train_resnet18_w2to6_a2to6_single_gpu.yaml`):
- `batch_size`: 128 → 32 (降低4倍)
- `workers`: 8 → 4 (减少数据加载线程)
- `information_distortion_mitigation`: true → false (禁用以节省显存)
- `lr`: 0.04 → 0.005 (根据batch size线性缩放)

**搜索配置** (`configs/search/search_resnet18_single_gpu.yaml`):
- `batch_size`: 256 → 64 (降低4倍)
- `workers`: 24 → 4

### 2. 代码修改

修改了 `main.py` 以支持单GPU训练（不使用分布式训练）。

## 使用步骤

### 步骤1: 环境准备

```bash
cd retraining-free-quantization
conda create -n RFQuant python=3.9
conda activate RFQuant
pip install -r requirements.txt
```

### 步骤2: 准备ImageNet数据集

确保ImageNet数据集路径正确。在配置文件中修改 `dataloader.path` 为你的ImageNet路径。

### 步骤3: 训练ResNet18

```bash
bash run_training_single_gpu.sh
```

或者直接运行：

```bash
export CUDA_VISIBLE_DEVICES=0
python main.py configs/training/train_resnet18_w2to6_a2to6_single_gpu.yaml
```

**注意事项：**
- 训练大约需要150个epoch，在单卡上可能需要较长时间
- 监控显存使用，如果OOM，可以进一步降低batch_size到16或8
- 训练checkpoint会保存在 `training/resnet18_single_gpu/` 目录

### 步骤4: 搜索最优量化配置

训练完成后，运行搜索：

```bash
# 修改 run_search_single_gpu.sh 中的 resume.path 为训练得到的checkpoint路径
bash run_search_single_gpu.sh
```

或者：

```bash
export CUDA_VISIBLE_DEVICES=0
python main.py configs/search/search_resnet18_single_gpu.yaml \
    --resume.path ./training/resnet18_single_gpu/checkpoint_best.pth.tar
```

## 显存优化建议

如果遇到OOM（显存不足）错误，可以尝试：

1. **进一步降低batch size**: 
   - 训练: `batch_size: 32` → `16` 或 `8`
   - 搜索: `batch_size: 64` → `32` 或 `16`

2. **使用梯度累积**（如果需要，需要修改代码）：
   - 例如：batch_size=16，累积4步 ≈ 有效batch_size=64

3. **禁用混合精度训练的部分功能**：
   - 已经禁用了 `information_distortion_mitigation`

4. **减少量化bit候选数**：
   - 例如：只使用 `target_bits: [4, 3, 2]` 而不是 `[6, 5, 4, 3, 2]`

## 预期时间

在2080Ti上的大致时间：
- **训练阶段**: 约150 epochs，每epoch约15-20分钟 → 总计约38-50小时
- **搜索阶段**: 取决于搜索算法迭代次数，通常数小时到数十小时

## 检查点保存

训练过程中的checkpoint会保存在：
- `training/resnet18_single_gpu/checkpoint_best.pth.tar` (最佳模型)
- `training/resnet18_single_gpu/checkpoint_latest.pth.tar` (最新模型)

## 故障排除

1. **OOM错误**: 降低batch_size或禁用更多功能
2. **数据加载慢**: 减少workers数量（已设置为4）
3. **训练不稳定**: 可以尝试使用更小的学习率

## 与原配置对比

| 配置项 | 原始(4×A100) | 单GPU (2080Ti) |
|--------|-------------|----------------|
| batch_size (训练) | 128×4=512 | 32 |
| batch_size (搜索) | 256×4=1024 | 64 |
| information_distortion_mitigation | ✅ | ❌ |
| 分布式训练 | ✅ | ❌ |
| 显存使用 | ~40GB/GPU | ~9-10GB |




