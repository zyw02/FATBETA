# 故障感知训练（Fault-Aware Training）实现指南

## 1. 概述

故障感知训练通过在训练时引入故障注入，让模型学会在bit-flip故障下工作，从而提高对BER > 1e-2大故障的容错性。

## 2. 实现方案

### 方案A: 简单的故障感知训练（推荐用于快速验证）

在训练脚本中启用故障注入：

```python
# 在训练循环中
from util.fault_injector import FaultInjector

# 创建故障注入器
injector = FaultInjector(
    model=model,
    mode="ber",
    ber=1e-2,  # 目标BER值
    enable_in_training=True,   # ⭐ 训练时启用
    enable_in_inference=False, # 评估时禁用（使用无故障模型）
    seed=42
)

# 在训练开始前启用
injector.enable()

# 正常训练循环
for epoch in range(epochs):
    for batch in train_loader:
        # 训练时自动注入故障
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 方案B: 渐进式故障训练（推荐用于最佳效果）

逐步增加BER值，让模型逐步适应故障：

```python
def get_ber_schedule(epoch, total_epochs, max_ber=1e-2):
    """
    渐进式BER调度
    - 前30%: BER = 1e-4 (几乎无故障)
    - 30-60%: BER = 1e-3 (小故障)
    - 60-100%: BER = 1e-2 (目标故障)
    """
    progress = epoch / total_epochs
    if progress < 0.3:
        return 1e-4
    elif progress < 0.6:
        return 1e-3
    else:
        return max_ber

# 训练循环中
for epoch in range(epochs):
    # 根据epoch调整BER
    current_ber = get_ber_schedule(epoch, total_epochs, max_ber=1e-2)
    
    # 更新故障注入器的BER
    injector.ber = current_ber
    injector.enable()  # 重新启用以应用新的BER
    
    for batch in train_loader:
        # 训练...
        pass
```

### 方案C: 混合故障训练（平衡baseline和容错性）

每个batch随机决定是否注入故障：

```python
import random

for epoch in range(epochs):
    for batch_idx, (input, target) in enumerate(train_loader):
        # 50%概率注入故障
        if random.random() < 0.5:
            # 随机BER值在[1e-3, 1e-2]之间
            current_ber = random.uniform(1e-3, 1e-2)
            injector.ber = current_ber
            injector.enable()
        else:
            injector.disable()
        
        # 训练...
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 3. 修改训练脚本

需要在`process.py`或`main.py`中集成故障注入：

```python
# 在main.py中添加
from util.fault_injector import FaultInjector

# 如果启用故障感知训练
if getattr(configs, 'fault_aware_training', False):
    fault_injector = FaultInjector(
        model=model,
        mode="ber",
        ber=getattr(configs.fault_aware_training, 'ber', 1e-2),
        enable_in_training=True,
        enable_in_inference=False,
        seed=getattr(configs, 'seed', 42)
    )
    fault_injector.enable()
    logger_info(logger, f"Fault-aware training enabled with BER={fault_injector.ber}")
else:
    fault_injector = None
```

## 4. 训练配置示例

在YAML配置文件中添加：

```yaml
# 故障感知训练配置
fault_aware_training:
  enabled: true
  ber: 1e-2  # 目标BER值
  schedule: "progressive"  # "constant", "progressive", "mixed"
  # 渐进式调度参数
  progressive:
    phase1_epochs: 0.3    # 前30%使用BER=1e-4
    phase2_epochs: 0.6    # 30-60%使用BER=1e-3
    phase3_epochs: 1.0    # 60-100%使用BER=1e-2
  # 混合训练参数
  mixed:
    probability: 0.5      # 50%概率注入故障
    ber_range: [1e-3, 1e-2]  # BER范围
```

## 5. 预期效果

### 优点
- ✅ 直接针对bit-flip故障优化
- ✅ 可以让模型学习到对故障更鲁棒的参数
- ✅ 理论上最有效的方法

### 注意事项
- ⚠️ 可能降低baseline精度（无故障时）
- ⚠️ 训练时间增加
- ⚠️ 需要仔细调整BER值和训练策略

## 6. 评估方法

训练完成后，使用故障注入评估模型：

```python
# 禁用训练时的故障注入
if fault_injector is not None:
    fault_injector.disable()

# 创建评估用的故障注入器
eval_injector = FaultInjector(
    model=model,
    mode="ber",
    ber=1e-2,  # 或更大的BER值
    enable_in_training=False,
    enable_in_inference=True,
    seed=42
)

# 评估
eval_injector.enable()
accuracy = evaluate(model, test_loader)
eval_injector.disable()
```

## 7. 实验建议

1. **先做小规模实验**：
   - 使用少量epoch（如20-50）
   - 对比有无故障感知训练的差异

2. **逐步增加BER**：
   - 从1e-3开始
   - 逐步增加到1e-2

3. **监控指标**：
   - Baseline精度（无故障）
   - 故障下的精度
   - 训练loss曲线

4. **调优参数**：
   - BER值
   - 学习率（可能需要调整）
   - 训练策略（渐进式/混合）



