# Fault Injector Tool 使用说明

## 概述

故障注入工具（Fault Injector）专门为 `retraining-free-quantization` 框架设计，用于研究量化模型的单粒子翻转（SEU）鲁棒性。

## 主要特性

1. **代码隔离**：使用 hook 机制包装量化器，可随时启用/禁用，不影响原有功能
2. **自动读取位宽配置**：自动从量化层的 `bits` 或 `fixed_bits` 属性读取位宽配置
3. **支持训练和推理模式**：可分别控制训练时和推理时的故障注入
4. **GPU 加速**：使用 GPU 并行加速位翻转操作，提升效率
5. **保持梯度流**：故障注入不会破坏反向传播的计算图
6. **BER 模式**：支持 Bit-Error-Rate（位错误率）模式的故障注入

## 使用方法

### 混合精度模型使用（推荐）

对于混合精度模型，需要先从搜索生成的JSON配置文件中加载位宽配置：

```python
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
import torch

# 1. 创建或加载量化模型
model = ...  # 你的量化模型（包含 QuanConv2d/QuanLinear 层）

# 2. 从JSON文件加载位宽配置并设置到模型
# 这一步会读取搜索生成的配置文件，并根据配置设置每层的位宽
setup_model_with_bit_width_config(
    model,
    json_path="search/resnet18_cifar10_single_gpu_search_bit_width_config.json",
    config_index=0,  # 使用第一个配置（如果有多个）
    verbose=True
)

# 3. 创建故障注入器（会自动读取每层的位宽配置）
injector = FaultInjector(
    model=model,
    mode="ber",
    ber=1e-6,  # 位错误率：每比特翻转概率为 1e-6
    enable_in_training=False,   # 训练时是否注入
    enable_in_inference=True,   # 推理时是否注入
    seed=42,  # 随机种子，用于可重复性
)

# 4. 启用故障注入
injector.enable()

# 5. 进行推理（故障会被注入，每层根据其配置的位宽进行翻转）
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# 6. 禁用故障注入
injector.disable()
```

### 基本使用（单一精度模型）

对于单一精度模型或已设置好位宽的模型，可以直接使用：

```python
from util.fault_injector import FaultInjector
import torch

# 创建或加载量化模型（位宽已配置）
model = ...  # 你的量化模型（包含 QuanConv2d/QuanLinear 层）

# 创建故障注入器
injector = FaultInjector(
    model=model,
    mode="ber",
    ber=1e-6,  # 位错误率：每比特翻转概率为 1e-6
    enable_in_training=False,   # 训练时是否注入
    enable_in_inference=True,   # 推理时是否注入
    seed=42,  # 随机种子，用于可重复性
)

# 启用故障注入
injector.enable()

# 进行推理（故障会被注入）
model.eval()
with torch.no_grad():
    output = model(input_tensor)

# 禁用故障注入
injector.disable()
```

### 在训练循环中使用

```python
injector = FaultInjector(
    model=model,
    mode="ber",
    ber=1e-6,
    enable_in_training=True,   # 训练时也注入故障
    enable_in_inference=True,
)

# 训练时启用
injector.enable()

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

# 评估时保持启用
model.eval()
with torch.no_grad():
    for batch in val_loader:
        output = model(inputs)
        # ... 计算准确率等指标

injector.disable()  # 训练结束后禁用
```

### 评估不同 BER 值（混合精度模型）

```python
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config

# 加载模型和位宽配置
model = load_model(...)
setup_model_with_bit_width_config(
    model,
    "search/resnet18_cifar10_single_gpu_search_bit_width_config.json"
)

ber_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

# 基准准确率（无故障）
baseline_acc = evaluate_model(model, test_loader)
print(f"Baseline Accuracy: {baseline_acc:.2f}%")

for ber in ber_values:
    injector = FaultInjector(
        model=model,
        mode="ber",
        ber=ber,
        enable_in_inference=True,
        seed=42,  # 使用相同种子确保可重复性
    )
    
    injector.enable()
    acc_with_fault = evaluate_model(model, test_loader)
    injector.disable()
    
    acc_drop = baseline_acc - acc_with_fault
    print(f"BER={ber:.1e}, Accuracy: {acc_with_fault:.2f}%, Drop: {acc_drop:.2f}%")
```

### 手动加载位宽配置

如果需要更精细的控制，可以手动加载和设置：

```python
from util.fault_injector import load_bit_width_config_from_json
from util.qat import set_bit_width

# 加载配置
weight_bits, act_bits = load_bit_width_config_from_json(
    "search/resnet18_cifar10_single_gpu_search_bit_width_config.json",
    config_index=0
)

# 设置到模型
set_bit_width(model, weight_bits, act_bits)

# 现在可以启用故障注入
injector = FaultInjector(model, mode="ber", ber=1e-6)
injector.enable()
```

## 工作原理

1. **包装量化器**：故障注入器在 `enable()` 时包装所有 `QuanConv2d` 和 `QuanLinear` 层的 `quan_w_fn.forward` 方法

2. **位宽自动检测**：从量化层的 `bits` 或 `fixed_bits` 属性自动读取位宽配置
   - 首层和末层通常配置为 8-bit（通过 `fixed_bits=(8, 8)`）
   - 其他层根据搜索/训练配置使用不同的位宽（如 3-bit, 4-bit 等）

3. **故障注入流程**：
   - 量化器首先执行正常的权重量化
   - 将量化后的浮点值转换回整数码（0 到 2^k-1）
   - 使用 GPU 加速生成随机翻转掩码
   - 对整数码的每一位进行 XOR 操作（翻转）
   - 将翻转后的整数码转换回浮点值
   - 使用 `detach()` 技巧保持梯度流

4. **梯度保持**：使用 `x_faulted.detach() + (x_q - x_q.detach())` 确保：
   - 前向传播使用故障后的值
   - 反向传播使用原始值，保持梯度正确

## 参数说明

### FaultInjector 参数

- `model` (torch.nn.Module): 量化模型（必须包含 QuanConv2d/QuanLinear 层）
- `mode` (str): 故障注入模式，目前仅支持 `"ber"`
- `ber` (float): 位错误率，范围 [0, 1]，每比特翻转的概率
- `device` (torch.device, optional): 故障注入使用的设备，默认使用模型所在设备
- `enable_in_training` (bool): 是否在训练模式下注入故障，默认 `True`
- `enable_in_inference` (bool): 是否在推理模式下注入故障，默认 `True`
- `seed` (int, optional): 随机种子，用于可重复性

### 方法

- `enable()`: 启用故障注入
- `disable()`: 禁用故障注入并恢复原始行为

## 注意事项

1. **模型必须已量化**：确保模型包含 `QuanConv2d` 或 `QuanLinear` 层，并且这些层已配置量化位宽

2. **位宽配置**：故障注入工具会自动读取每层的位宽配置：
   - 如果层有 `fixed_bits` 属性，使用该值（通常首层末层为 8-bit）
   - 否则使用 `bits` 属性（动态位宽）

3. **性能考虑**：
   - GPU 加速的位翻转操作通常很快
   - 对于大模型，故障注入的开销很小（< 5%）
   - 如果遇到性能问题，可以只启用推理时注入

4. **可重复性**：使用相同的 `seed` 可以确保每次运行产生相同的故障模式

5. **代码隔离**：故障注入工具完全独立，不会修改原始模型代码，可以随时启用/禁用

## 测试

使用提供的测试脚本进行测试：

```bash
python tools/test_fault_injector.py \
    --config configs/eval/eval_resnet18_cifar10_single_gpu.yaml \
    --ber 1e-6 \
    --enable_in_inference
```

## 示例：评估模型鲁棒性

```python
from util.fault_injector import FaultInjector
from process import validate
from util.data_loader import init_dataloader

# 加载模型和测试集
model = load_model(...)
test_loader = init_dataloader(configs, configs.arch)

# 基准准确率（无故障）
baseline_acc = validate(test_loader, model, criterion, -1, monitors, configs)

# 测试不同 BER 值
ber_results = []
for ber in [1e-8, 1e-7, 1e-6, 1e-5]:
    injector = FaultInjector(model, mode="ber", ber=ber, seed=42)
    injector.enable()
    acc = validate(test_loader, model, criterion, -1, monitors, configs)
    injector.disable()
    ber_results.append((ber, acc))
    print(f"BER={ber:.1e}: Accuracy={acc:.2f}%")

# 计算鲁棒性指标（例如：准确率下降 1% 时的 BER 阈值）
```

