# 故障感知搜索 - 方案3实现说明

## 概述

方案3（故障感知的贪婪搜索）已在 `util/greedy_search.py` 中实现。该方案在搜索过程中同时考虑正常准确率和故障注入下的容错能力。

## 功能特点

1. **代码隔离**：通过配置开关完全隔离，不影响原有搜索流程
2. **多目标优化**：同时优化正常准确率和容错性
3. **灵活配置**：通过YAML配置文件轻松启用/禁用
4. **向后兼容**：默认禁用，不影响现有代码

## 使用方法

### 1. 启用故障感知搜索

在搜索配置文件中添加以下配置：

```yaml
# 故障感知搜索配置（方案3：故障感知的贪婪搜索）
fault_aware_search:
  enabled: true   # 设置为true启用
  # 故障注入评估配置
  fault_injection:
    ber: 2e-2     # 评估时使用的BER（建议与FAT训练的BER匹配或稍大）
  # 多目标优化权重
  weights:
    alpha: 0.5    # 正常准确率权重
    beta: 0.3     # 容错性权重（故障下的准确率）
```

### 2. 配置参数说明

#### `enabled` (bool)
- **默认值**: `false`
- **说明**: 是否启用故障感知搜索
- **注意**: 设置为`false`时，使用标准搜索（只考虑正常准确率）

#### `fault_injection.ber` (float)
- **默认值**: `2e-2`
- **说明**: 评估时使用的BER（Bit-Error-Rate）
- **推荐值**: 
  - 与FAT训练的BER匹配：`2e-2`（如果FAT训练使用2e-2）
  - 或稍大于FAT训练的BER：`3e-2`（如果FAT训练使用2e-2）

#### `weights.alpha` (float)
- **默认值**: `0.5`
- **说明**: 正常准确率权重
- **范围**: `0.0` 到 `1.0`
- **推荐值**: `0.5` 到 `0.7`

#### `weights.beta` (float)
- **默认值**: `0.3`
- **说明**: 容错性权重（故障下的准确率）
- **范围**: `0.0` 到 `1.0`
- **推荐值**: `0.3` 到 `0.5`
- **注意**: `alpha + beta` 应该 <= 1.0（实际使用中会自动归一化）

### 3. 运行搜索

运行搜索脚本（与正常搜索相同）：

```bash
python main.py configs/search/search_alexnet_cifar10_single_gpu_v2.yaml
```

如果启用了故障感知搜索，日志中会显示：

```
================================================================================
🔍 FAULT-AWARE SEARCH (Scheme 3) - ENABLED
================================================================================
  ✅ FaultInjector initialized for search
  ✅ Search BER: 0.02
  ✅ Multi-objective weights: alpha=0.5 (normal), beta=0.3 (fault)
  ✅ Optimization: score = α * acc_normal + β * acc_fault - γ * bitops
================================================================================
```

## 工作原理

### 优化目标

**原始搜索**（`enabled=false`）：
```
score = acc_scale * normalized_acc - normalized_bitops
```

**故障感知搜索**（`enabled=true`）：
```
combined_acc = α * acc_normal + β * acc_fault
score = acc_scale * normalized_combined_acc - normalized_bitops
```

### 评估流程

1. **正常评估**：禁用故障注入，评估正常准确率 `acc_normal`
2. **故障评估**：启用故障注入（BER=2e-2），评估故障准确率 `acc_fault`
3. **综合评分**：计算 `combined_acc = α * acc_normal + β * acc_fault`
4. **优化选择**：选择综合评分最高的候选配置

## 性能影响

### 搜索时间
- **增加**: 约2倍（每次评估需要两次forward pass）
- **原因**: 每个候选配置需要评估正常和故障两种情况

### 内存使用
- **增加**: 约10-20%
- **原因**: 故障注入的计算开销

### GPU使用
- **增加**: 约10-20%
- **原因**: 故障注入的计算开销

## 实验建议

### 实验1: 对比标准搜索和故障感知搜索

1. **禁用故障感知搜索**训练一个配置：
   ```yaml
   fault_aware_search:
     enabled: false
   ```

2. **启用故障感知搜索**训练另一个配置：
   ```yaml
   fault_aware_search:
     enabled: true
     fault_injection:
       ber: 2e-2
     weights:
       alpha: 0.5
       beta: 0.3
   ```

3. 对比两个配置在故障注入下的准确率

### 实验2: 权重调优

- 测试不同的`alpha`和`beta`组合：
  - `alpha=0.7, beta=0.3`：更重视正常准确率
  - `alpha=0.5, beta=0.5`：平衡正常和容错性
  - `alpha=0.3, beta=0.7`：更重视容错性

### 实验3: BER值选择

- 测试不同的BER值：
  - `ber: 1e-2`：小故障
  - `ber: 2e-2`：中等故障（推荐）
  - `ber: 3e-2`：大故障

## 注意事项

1. **搜索时间**：启用后搜索时间会增加约2倍
2. **BER选择**：建议与FAT训练的BER匹配或稍大
3. **权重调优**：需要根据具体模型和数据集调整`alpha`和`beta`
4. **代码隔离**：默认`enabled=false`，不影响原有代码

## 示例配置

### 示例1: 启用故障感知搜索（推荐）

```yaml
fault_aware_search:
  enabled: true
  fault_injection:
    ber: 2e-2
  weights:
    alpha: 0.5
    beta: 0.3
```

### 示例2: 更重视容错性

```yaml
fault_aware_search:
  enabled: true
  fault_injection:
    ber: 2e-2
  weights:
    alpha: 0.3
    beta: 0.7
```

### 示例3: 禁用（默认）

```yaml
fault_aware_search:
  enabled: false
```

## 代码修改说明

### 修改的文件

1. **`util/greedy_search.py`**:
   - 添加 `forward_loss_with_fault()` 函数：同时评估正常和故障情况
   - 修改 `search()` 函数：添加故障感知搜索逻辑
   - 完全通过配置控制，代码隔离

2. **`main.py`**:
   - 修改 `search()` 调用：传入 `fault_injector=None`（由search函数内部处理）

3. **`configs/search/search_alexnet_cifar10_single_gpu_v2.yaml`**:
   - 添加 `fault_aware_search` 配置节

### 代码隔离机制

所有故障感知搜索相关的代码都通过条件判断隔离：

```python
# 在greedy_search.py中
if use_fault_aware_search and search_fault_injector is not None:
    # 故障感知评估逻辑
    ...
else:
    # 原有评估逻辑（完全不变）
    ...
```

当`fault_aware_search.enabled=false`时：
- `use_fault_aware_search`为`False`
- 代码执行原有搜索流程，**完全不受影响**

## 故障排除

### 问题1: 搜索时间过长
- **可能原因**: 正常现象（需要两次forward）
- **解决**: 可以减小batch_size或使用更少的候选配置

### 问题2: 内存不足
- **可能原因**: 故障注入需要额外内存
- **解决**: 减小batch_size

### 问题3: 搜索结果不理想
- **可能原因**: 权重配置不当
- **解决**: 调整`alpha`和`beta`权重，或尝试不同的BER值

## 相关文档

- `FAULT_AWARE_SEARCH_STRATEGY.md`: 完整的设计方案文档
- `FAULT_AWARE_TRAINING_README.md`: FAT训练使用指南



