# SOTA对抗训练方法及其在故障感知训练中的应用

## 1. SOTA对抗训练方法概览

### 1.1 经典方法

#### PGD-AT (Madry et al., 2018)
- **核心思想**: 在对抗样本上直接训练
- **损失函数**: `L = L(x_adv, y)`
- **优点**: 简单直接
- **缺点**: 可能过拟合对抗样本，降低干净准确率

#### TRADES (Zhang et al., 2019) ⭐⭐⭐⭐⭐
- **核心思想**: 平衡干净样本和对抗样本的性能
- **损失函数**: `L = L(x, y) + β * KL(p(x), p(x_adv))`
- **关键创新**: 使用KL散度衡量干净和对抗预测的差异
- **优点**: 
  - 同时保持干净和对抗准确率
  - 避免过拟合对抗样本
  - 有理论保证

#### MART (Wang et al., 2020) ⭐⭐⭐⭐
- **核心思想**: Misclassified-Aware Regularization
- **损失函数**: `L = BCE(x, y) + λ * BCE(x_adv, y) * (1 - p_y(x_adv))`
- **关键创新**: 对分类错误的样本给予更多权重
- **优点**: 更好地处理困难样本

#### AWP (Wu et al., 2020) ⭐⭐⭐⭐⭐
- **核心思想**: Adversarial Weight Perturbation（参数空间扰动）
- **损失函数**: `L = max_{||δ||≤ρ} L(x, y; θ+δ)`
- **关键创新**: 在参数空间而不是输入空间添加扰动
- **优点**: 
  - 更直接优化模型鲁棒性
  - 与SAQ的思想非常相似

### 1.2 最新SOTA方法

#### LAS-AT (Learnable Attack Strategy, 2022)
- **核心思想**: 使用策略网络自动生成攻击策略
- **关键创新**: 根据模型状态动态调整攻击参数
- **优点**: 自适应，更灵活

#### HFAT (Hider-Focused AT, 2023)
- **核心思想**: 关注"隐藏者"（反复出现问题的样本）
- **关键创新**: 识别训练中的高风险区域
- **优点**: 更有效地处理困难情况

#### FGSM-LASS (Learnable Attack Step Size)
- **核心思想**: 可学习的攻击步长
- **关键创新**: 为每个样本预测最优攻击步长
- **优点**: 更精细的控制

## 2. 类比到故障感知训练

### 2.1 TRADES风格（最推荐）⭐⭐⭐⭐⭐

**核心思想**: 平衡正常和故障下的性能

**损失函数**:
```
L = α * L(x_normal, y) + β * KL(p(x_normal), p(x_faulted))
```

或更简单的版本:
```
L = α * L(x_normal, y) + β * L(x_faulted, y)
```

**实现方式**:
```python
# 在训练循环中
for batch in train_loader:
    inputs, targets = batch
    
    # 第一次forward: 正常情况
    fault_injector.disable()
    outputs_normal = model(inputs)
    loss_normal = criterion(outputs_normal, targets)
    
    # 第二次forward: 故障注入
    fault_injector.enable()
    outputs_faulted = model(inputs)
    loss_faulted = criterion(outputs_faulted, targets)
    
    # 组合损失（TRADES风格）
    alpha = 0.6  # 正常样本权重
    beta = 0.4   # 故障样本权重
    loss = alpha * loss_normal + beta * loss_faulted
    
    loss.backward()
    optimizer.step()
```

**优点**:
- ✅ 明确控制baseline和容错性的平衡
- ✅ 有理论保证（类似TRADES）
- ✅ 可以调整α和β控制平衡点

**缺点**:
- ❌ 需要两次forward，计算成本增加
- ❌ 需要调整α和β权重

### 2.2 MART风格（困难故障挖掘）⭐⭐⭐⭐

**核心思想**: 对故障下分类错误的样本给予更多关注

**损失函数**:
```
L = L(x_normal, y) + λ * L(x_faulted, y) * (1 - p_y(x_faulted))
```

其中:
- `p_y(x_faulted)`: 故障下正确类别的概率
- `(1 - p_y(x_faulted))`: 错误权重，分类越错误权重越大

**实现方式**:
```python
# 在训练循环中
for batch in train_loader:
    inputs, targets = batch
    
    # 正常情况
    fault_injector.disable()
    outputs_normal = model(inputs)
    loss_normal = criterion(outputs_normal, targets)
    
    # 故障注入
    fault_injector.enable()
    outputs_faulted = model(inputs)
    probs_faulted = F.softmax(outputs_faulted, dim=1)
    p_correct = probs_faulted.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # MART风格损失
    loss_faulted = criterion(outputs_faulted, targets)
    error_weight = (1 - p_correct).detach()  # 错误权重
    loss_faulted_weighted = (loss_faulted * error_weight).mean()
    
    lambda_weight = 1.0  # MART权重
    loss = loss_normal + lambda_weight * loss_faulted_weighted
    
    loss.backward()
    optimizer.step()
```

**优点**:
- ✅ 关注困难故障情况
- ✅ 自动调整样本权重
- ✅ 可能提高对BER > 1e-2的鲁棒性

**缺点**:
- ❌ 需要计算预测概率
- ❌ 可能不稳定（如果p_y很小）

### 2.3 AWP风格（参数空间扰动）⭐⭐⭐⭐⭐

**核心思想**: 结合SAQ和故障感知训练

**方法**: 在量化权重空间添加扰动（类似SAQ），然后在扰动后的权重上注入故障

**实现方式**:
```python
# 结合SAQ和故障注入
# 1. SAQ的ascent_step: 在量化权重上添加扰动
sam.ascent_step()  # 添加epsilon扰动到量化权重

# 2. 在扰动后的权重上注入故障
fault_injector.enable()
outputs = model(inputs)  # 使用扰动+故障的权重
loss = criterion(outputs, targets)
loss.backward()

# 3. SAQ的descent_step: 恢复参数并更新
sam.descent_step()
```

**优点**:
- ✅ 结合SAQ和故障感知训练的优势
- ✅ 在参数空间优化，更直接
- ✅ 理论上最优雅

**缺点**:
- ❌ 实现复杂
- ❌ 需要修改SAQ实现
- ❌ 计算成本高

### 2.4 自适应BER策略（类似LAS-AT）⭐⭐⭐⭐

**核心思想**: 根据模型状态和训练阶段动态调整BER

**实现方式**:
```python
class AdaptiveBERStrategy:
    def __init__(self, min_ber=1e-4, max_ber=1e-2):
        self.min_ber = min_ber
        self.max_ber = max_ber
        self.current_ber = min_ber
        
    def update(self, model_performance, target_acc=0.80):
        """
        根据模型性能自适应调整BER
        """
        if model_performance > target_acc:
            # 性能好，增加BER
            self.current_ber = min(self.current_ber * 1.1, self.max_ber)
        else:
            # 性能差，减少BER
            self.current_ber = max(self.current_ber * 0.9, self.min_ber)
        return self.current_ber

# 在训练循环中
adaptive_ber = AdaptiveBERStrategy()
for epoch in range(epochs):
    for batch in train_loader:
        # 使用当前BER
        fault_injector.ber = adaptive_ber.current_ber
        fault_injector.enable()
        
        # 训练...
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 定期评估并调整BER
        if batch_idx % 100 == 0:
            acc = evaluate_fault_accuracy(model, val_loader, fault_injector)
            adaptive_ber.update(acc)
```

**优点**:
- ✅ 自适应调整
- ✅ 灵活
- ✅ 类似LAS-AT的思想

**缺点**:
- ❌ 需要额外的评估步骤
- ❌ 可能不稳定

### 2.5 层自适应BER（类似HFAT）⭐⭐⭐⭐

**核心思想**: 根据层的敏感性自适应调整BER

**实现方式**:
```python
# 分析层的敏感性
def analyze_layer_sensitivity(model, fault_injector, val_loader):
    """
    分析各层对故障的敏感性
    """
    sensitivities = {}
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            # 测试该层在不同BER下的影响
            # ... 敏感性分析 ...
            sensitivities[name] = sensitivity_score
    return sensitivities

# 根据敏感性设置层级BER
layer_bers = analyze_layer_sensitivity(model, fault_injector, val_loader)
for name, module in model.named_modules():
    if name in layer_bers:
        # 对敏感层使用较小BER，对不敏感层使用较大BER
        module.ber = layer_bers[name]
```

**优点**:
- ✅ 更精细的控制
- ✅ 针对性强
- ✅ 可能提高效率

**缺点**:
- ❌ 需要敏感性分析
- ❌ 实现复杂

## 3. 推荐实现方案

### 方案A: TRADES风格（快速实现）⭐⭐⭐⭐⭐

**优点**: 实现简单，效果明确，易于调优

**实现步骤**:
1. 修改`process.py`的`train()`函数
2. 添加两次forward（正常和故障）
3. 组合损失函数

### 方案B: MART风格（困难故障挖掘）⭐⭐⭐⭐

**优点**: 自动关注困难样本，可能对BER > 1e-2更有效

**实现步骤**:
1. 在故障注入下forward
2. 计算预测概率和错误权重
3. 加权损失函数

### 方案C: AWP + 故障注入（理论最优）⭐⭐⭐⭐⭐

**优点**: 结合SAQ和故障感知训练，理论上最优

**实现步骤**:
1. 先实现SAQ
2. 在SAQ的扰动权重上注入故障
3. 联合优化

## 4. 实验建议

### 4.1 优先级排序

1. **第一优先级**: TRADES风格（快速验证）
   - 实现简单
   - 效果明确
   - 易于调优

2. **第二优先级**: MART风格（困难故障挖掘）
   - 可能对BER > 1e-2更有效
   - 自动关注困难样本

3. **第三优先级**: AWP + 故障注入（长期优化）
   - 需要先实现SAQ
   - 理论最优

### 4.2 实验设计

1. **Baseline**: 标准QAT训练（无故障注入）
2. **实验1**: TRADES风格（α=0.6, β=0.4）
3. **实验2**: MART风格（λ=1.0）
4. **实验3**: 渐进式BER + TRADES
5. **实验4**: 自适应BER + TRADES

### 4.3 评估指标

- Baseline准确率（无故障）
- 故障下准确率（BER=1e-2, 2e-2, 3e-2等）
- 训练稳定性（loss曲线）
- 训练时间（相对baseline）



