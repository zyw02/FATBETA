# TRADES风格故障感知训练使用指南

## 概述

本功能实现了**TRADES风格的故障感知训练**，用于提高量化神经网络在单粒子翻转故障（Single-Event Upset, SEU）下的鲁棒性。该实现完全隔离，不影响原有训练流程。

## 功能特点

1. **代码隔离**: 通过条件判断完全隔离，不影响原有训练流程
2. **TRADES损失**: 支持两种TRADES损失函数：
   - KL散度版本：`L = L(x_normal, y) + β * KL(p(x_normal), p(x_faulted))`
   - 简单组合版本：`L = α * L(x_normal, y) + β * L(x_faulted, y)`
3. **灵活配置**: 通过YAML配置文件轻松启用/禁用
4. **梯度保持**: 故障注入不会破坏梯度计算

## 使用方法

### 1. 启用故障感知训练

在训练配置文件中添加以下配置：

```yaml
# 故障感知训练配置（TRADES风格）
fault_aware_training:
  enabled: true   # 设置为true启用
  ber: 1e-2       # Bit-Error-Rate: 单粒子翻转故障率
  
  # TRADES损失函数配置
  trades:
    use_kl: false   # 是否使用KL散度
    alpha: 0.6      # 正常样本权重（仅当use_kl=false时使用）
    beta: 1.0       # 故障样本权重
```

### 2. 配置参数说明

#### `enabled` (bool)
- **默认值**: `false`
- **说明**: 是否启用故障感知训练
- **注意**: 设置为`false`时，代码完全不会执行故障感知训练逻辑，保持原有行为

#### `ber` (float)
- **默认值**: `1e-2`
- **说明**: Bit-Error-Rate，单粒子翻转故障率
- **范围**: `0.0` 到 `1.0`
- **推荐值**: 
  - 小故障：`1e-4` 到 `1e-3`
  - 中等故障：`1e-3` 到 `1e-2`
  - 大故障：`1e-2` 到 `5e-2`

#### `trades.use_kl` (bool)
- **默认值**: `false`
- **说明**: 是否使用KL散度版本的TRADES损失
- **选择**:
  - `true`: 使用KL散度，更符合TRADES原论文
  - `false`: 使用简单组合，更稳定且计算更快

#### `trades.alpha` (float)
- **默认值**: `0.6`
- **说明**: 正常样本损失权重（仅当`use_kl=false`时使用）
- **范围**: `0.0` 到 `1.0`
- **推荐值**: `0.5` 到 `0.7`

#### `trades.beta` (float)
- **默认值**: `1.0`
- **说明**: 
  - 当`use_kl=true`时：KL散度的权重
  - 当`use_kl=false`时：故障样本损失的权重
- **推荐值**: `0.3` 到 `1.0`

### 3. 运行训练

运行训练脚本（与正常训练相同）：

```bash
python main.py configs/training/train_alexnet_cifar10_single_gpu_v2.yaml
```

如果启用了故障感知训练，日志中会显示：

```
FaultInjector initialized for fault-aware training (BER=0.01)
Fault-aware training (TRADES style) is ENABLED
  TRADES config: use_kl=False, alpha=0.6, beta=1.0
```

## 工作原理

### TRADES风格训练流程

1. **第一次Forward（正常）**:
   - 禁用故障注入器
   - 正常forward，计算`loss_normal`
   - 获取正常输出`outputs_normal`

2. **第二次Forward（故障）**:
   - 启用故障注入器
   - 在量化权重上注入bit-flip故障
   - Forward，计算`loss_faulted`
   - 获取故障输出`outputs_faulted`

3. **TRADES损失计算**:
   - **KL散度版本**: `L = loss_normal + β * KL(p_normal, p_faulted)`
   - **简单组合版本**: `L = α * loss_normal + β * loss_faulted`

4. **反向传播**:
   - 使用组合损失进行反向传播
   - 梯度计算正确（故障注入不会破坏梯度）

### 代码隔离机制

所有故障感知训练相关的代码都通过条件判断隔离：

```python
# 在process.py中
if use_fault_aware_training and fault_injector is not None:
    # TRADES训练逻辑
    ...
else:
    # 原有训练流程（完全不变）
    ...
```

当`fault_aware_training.enabled=false`时：
- `fault_injector`保持为`None`
- `use_fault_aware_training`为`False`
- 代码执行原有训练流程，**完全不受影响**

## 性能影响

### 训练时间
- **增加**: 约2倍（需要两次forward）
- **原因**: 每个batch需要执行两次forward pass

### 内存使用
- **增加**: 约1.5倍
- **原因**: 需要存储normal和faulted两个输出

### GPU使用
- **增加**: 约10-20%
- **原因**: 故障注入的计算开销

## 实验建议

### 实验1: 基线对比
1. 禁用故障感知训练训练一个模型
2. 启用故障感知训练训练一个模型
3. 在相同BER下评估两个模型的准确率

### 实验2: BER值选择
- 测试不同BER值（`1e-4`, `1e-3`, `1e-2`, `2e-2`）
- 观察模型在不同BER下的表现

### 实验3: 参数调优
- 测试不同的`alpha`和`beta`组合
- 测试KL散度 vs 简单组合

### 实验4: 渐进式训练
- 前期使用较小BER（如`1e-4`）
- 后期逐步增加BER（如`1e-2`）
- 观察是否比固定BER效果更好

## 注意事项

1. **训练时间**: 启用后训练时间会增加约2倍
2. **BER选择**: 建议从`1e-2`开始，根据实验结果调整
3. **参数调优**: 需要根据具体模型和数据集调整`alpha`和`beta`
4. **代码隔离**: 默认`enabled=false`，不影响原有代码

## 故障排除

### 问题1: 训练loss不下降
- **可能原因**: BER太大，故障过多
- **解决**: 降低BER值（如改为`1e-3`）

### 问题2: 训练时间过长
- **可能原因**: 正常现象（需要两次forward）
- **解决**: 可以减小batch_size或使用更少的epochs进行实验

### 问题3: 内存不足
- **可能原因**: 两次forward需要更多内存
- **解决**: 减小batch_size或使用梯度累积

## 示例配置

### 示例1: 简单组合（推荐开始使用）

```yaml
fault_aware_training:
  enabled: true
  ber: 1e-2
  trades:
    use_kl: false
    alpha: 0.6
    beta: 0.4
```

### 示例2: KL散度版本（理论最优）

```yaml
fault_aware_training:
  enabled: true
  ber: 1e-2
  trades:
    use_kl: true
    beta: 1.0
```

### 示例3: 小故障训练

```yaml
fault_aware_training:
  enabled: true
  ber: 1e-3
  trades:
    use_kl: false
    alpha: 0.7
    beta: 0.3
```

## 相关文档

- `TRADES_STYLE_FAULT_TRAINING.md`: TRADES风格的详细实现指南
- `SOTA_ADVERSARIAL_TRAINING_ANALOGY.md`: SOTA对抗训练方法及其类比
- `FAULT_INJECTION_MECHANISM.md`: 故障注入机制详解



