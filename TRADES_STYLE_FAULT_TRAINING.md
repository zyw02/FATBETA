# TRADES风格故障感知训练实现指南

## 1. TRADES方法简介

**TRADES (Trade-off between Adversarial Robustness and Accuracy, 2019)**:
- **核心思想**: 将对抗鲁棒性分解为两个目标：
  1. 在干净样本上的准确率
  2. 干净和对抗样本预测的一致性

- **损失函数**:
  ```
  L = L(x, y) + β * KL(p(x), p(x_adv))
  ```
  其中:
  - `L(x, y)`: 干净样本的损失
  - `KL(p(x), p(x_adv))`: 干净和对抗预测的KL散度
  - `β`: 平衡权重

- **关键创新**: 使用KL散度衡量预测一致性，而不是直接使用对抗样本的损失

## 2. 类比到故障感知训练

### 2.1 TRADES风格损失函数

**方案A: 使用KL散度（推荐）**
```
L = L(x_normal, y) + β * KL(p(x_normal), p(x_faulted))
```

**方案B: 使用简单组合**
```
L = α * L(x_normal, y) + β * L(x_faulted, y)
```

### 2.2 实现伪代码

```python
# 在process.py的train()函数中
from util.fault_injector import FaultInjector
import torch.nn.functional as F

# 初始化故障注入器（在main.py或train()开始时）
fault_injector = FaultInjector(
    model=model,
    mode="ber",
    ber=getattr(configs.fault_aware_training, 'ber', 1e-2),
    enable_in_training=True,
    enable_in_inference=False,
    seed=getattr(configs, 'seed', 42)
)

# 在训练循环中
for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.cuda()
    targets = targets.cuda()
    
    optimizer.zero_grad()
    if optimizer_q is not None:
        optimizer_q.zero_grad()
    
    # === TRADES风格训练 ===
    # 第一次forward: 正常情况
    fault_injector.disable()
    outputs_normal = model(inputs)
    loss_normal = criterion(outputs_normal, targets)
    probs_normal = F.softmax(outputs_normal, dim=1)
    
    # 第二次forward: 故障注入
    fault_injector.enable()
    outputs_faulted = model(inputs)
    loss_faulted = criterion(outputs_faulted, targets)
    probs_faulted = F.softmax(outputs_faulted, dim=1)
    
    # TRADES损失: L(x_normal, y) + β * KL(p(x_normal), p(x_faulted))
    beta = getattr(configs.fault_aware_training, 'beta', 1.0)
    kl_div = F.kl_div(
        F.log_softmax(outputs_faulted, dim=1),
        probs_normal,
        reduction='batchmean'
    )
    
    loss = loss_normal + beta * kl_div
    
    # 或者使用简单组合（更稳定）
    # alpha = getattr(configs.fault_aware_training, 'alpha', 0.6)
    # beta = getattr(configs.fault_aware_training, 'beta', 0.4)
    # loss = alpha * loss_normal + beta * loss_faulted
    
    loss.backward()
    optimizer.step()
    if optimizer_q is not None:
        optimizer_q.step()
```

## 3. 配置示例

在YAML配置文件中添加:

```yaml
# 故障感知训练配置（TRADES风格）
fault_aware_training:
  enabled: true
  method: "trades"  # "trades", "mart", "simple"
  ber: 1e-2
  
  # TRADES参数
  trades:
    alpha: 0.6      # 正常样本权重（如果使用简单组合）
    beta: 1.0       # KL散度权重（如果使用KL散度）
    use_kl: true    # 是否使用KL散度（true）或简单组合（false）
  
  # 渐进式BER调度（可选）
  schedule:
    enabled: true
    type: "progressive"  # "constant", "progressive", "adaptive"
    progressive:
      phase1_epochs: 0.3    # 前30%: BER=1e-4
      phase2_epochs: 0.6    # 30-60%: BER=1e-3
      phase3_epochs: 1.0    # 60-100%: BER=1e-2
```

## 4. 与现有训练流程的集成

### 4.1 修改process.py

需要修改`train()`函数，添加TRADES风格的故障感知训练：

```python
def train(..., fault_injector=None, ...):
    # ... 现有代码 ...
    
    # 故障感知训练（TRADES风格）
    use_fault_aware_training = (
        getattr(configs, 'fault_aware_training', False) and
        getattr(configs.fault_aware_training, 'enabled', False) and
        fault_injector is not None
    )
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # ... 现有代码 ...
        
        if use_fault_aware_training:
            # TRADES风格训练
            # ... 实现TRADES损失 ...
        else:
            # 原有训练流程
            # ... 现有代码 ...
```

### 4.2 修改main.py

在`main.py`中初始化故障注入器：

```python
# 在main.py中
from util.fault_injector import FaultInjector

# 如果启用故障感知训练
fault_injector = None
if getattr(configs, 'fault_aware_training', False) and \
   getattr(configs.fault_aware_training, 'enabled', False):
    fault_injector = FaultInjector(
        model=model,
        mode="ber",
        ber=getattr(configs.fault_aware_training, 'ber', 1e-2),
        enable_in_training=True,
        enable_in_inference=False,
        seed=getattr(configs, 'seed', 42)
    )
    logger_info(logger, f"Fault-aware training enabled (TRADES style)")

# 在训练循环中传递fault_injector
t_top1, t_top5, t_loss = train(
    train_loader, model, criterion, optimizer, epoch, monitors, configs,
    model_ema=target_model, ...,
    fault_injector=fault_injector  # 传递故障注入器
)
```

## 5. 实验对比

### 5.1 实验设置

1. **Baseline**: 标准QAT训练（无故障注入）
2. **实验1**: 简单故障感知训练（只在故障下训练）
3. **实验2**: TRADES风格（α=0.6, β=0.4，简单组合）
4. **实验3**: TRADES风格（β=1.0，使用KL散度）
5. **实验4**: 渐进式BER + TRADES

### 5.2 评估指标

- Baseline准确率（无故障）
- 故障下准确率（BER=1e-2, 2e-2, 3e-2等）
- 训练稳定性
- 训练时间（相对baseline）

## 6. 预期效果

### 优点
- ✅ 同时保持baseline和容错性
- ✅ 有理论保证（类似TRADES）
- ✅ 可以调整α和β控制平衡点

### 注意事项
- ⚠️ 需要两次forward，计算成本增加
- ⚠️ 需要调整α和β权重
- ⚠️ 可能略微降低baseline精度



