# 故障注入机制详解：如何在 forward 过程中插入故障注入

## 1. 核心原理：Python 函数替换

故障注入工具使用了 **Python 函数替换机制**，通过**替换量化器的 forward 方法**来实现故障注入。

### 1.1 关键概念

在 Python 中，**函数是对象**，可以像变量一样被赋值和替换：

```python
class MyClass:
    def original_method(self, x):
        return x * 2

obj = MyClass()
original = obj.original_method  # 保存原始函数
obj.original_method = lambda self, x: x * 3  # 替换函数
obj.original_method = original  # 恢复原始函数
```

## 2. 故障注入的执行流程

### 2.1 正常的 forward 流程（无故障注入）

```
模型 forward 调用链：
  model.forward(x)
    ↓
  QuanConv2d.forward(x)
    ↓
    调用 self.quan_w_fn(self.weight, wbits)
    ↓
    quan_w_fn.forward(weight, wbits)  ← 原始量化函数
    ↓
    返回量化后的权重 quantized_weight
    ↓
    使用量化权重进行卷积
```

### 2.2 故障注入的流程

```
1. 启用故障注入：
   injector.enable()
   ↓
   遍历所有 QuanConv2d/QuanLinear 层
   ↓
   对每个层的 quan_w_fn.forward 进行包装：
   - 保存原始函数：orig_fn = module.quan_w_fn.forward
   - 创建包装函数：wrapped_fn
   - 替换函数：module.quan_w_fn.forward = wrapped_fn

2. 模型 forward 调用链（故障注入启用后）：
   model.forward(x)
     ↓
   QuanConv2d.forward(x)
     ↓
     调用 self.quan_w_fn(self.weight, wbits)
     ↓
     quan_w_fn.forward(weight, wbits)  ← ⚠️ 现在是包装后的函数！
     ↓
     包装函数内部：
       1. 调用原始量化函数：x_q = orig_fn(weight, wbits)
       2. 对量化权重注入故障：x_faulted = inject_faults(x_q)
       3. 返回故障注入后的权重：return x_faulted
     ↓
     返回故障注入后的权重给 QuanConv2d.forward
     ↓
     使用故障注入后的权重进行卷积

3. 禁用故障注入：
   injector.disable()
   ↓
   恢复原始函数：module.quan_w_fn.forward = orig_fn
```

## 3. 代码实现详解

### 3.1 包装函数的关键代码

```python
def _wrap_modules(self) -> None:
    """包装量化器的 forward 方法"""
    for name, module in self.model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'quan_w_fn'):
                # ⭐ 步骤1：保存原始函数
                orig_quan_forward = module.quan_w_fn.forward
                
                # ⭐ 步骤2：创建包装函数
                def make_quan_wrapper(quantizer_instance, module_instance, orig_fn):
                    def wrapped_quan_forward(x, bits, is_activation=False, **kwargs):
                        # 如果不是权重，直接返回原始结果
                        if is_activation or bits is None or bits >= 32:
                            return orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        # ⭐ 步骤3：调用原始量化函数
                        x_q = orig_fn(x, bits, is_activation=is_activation, **kwargs)
                        
                        # ⭐ 步骤4：注入故障
                        scale = quantizer_instance.get_scale(bits, detach=True)
                        x_faulted = self._inject_on_quantized_tensor(x_q, int(bits), scale)
                        
                        # ⭐ 步骤5：保持梯度（重要！）
                        return x_faulted.detach() + (x_q - x_q.detach())
                    
                    return wrapped_quan_forward
                
                # ⭐ 步骤6：替换原始函数
                module.quan_w_fn.forward = make_quan_wrapper(
                    module.quan_w_fn, module, orig_quan_forward
                )
                
                # ⭐ 步骤7：保存原始函数引用（用于恢复）
                self._wrapped[id(module.quan_w_fn)] = orig_quan_forward
```

### 3.2 恢复原始函数

```python
def _restore_modules(self) -> None:
    """恢复原始 forward 方法"""
    for module in self.model.modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if hasattr(module, 'quan_w_fn'):
                key = id(module.quan_w_fn)
                if key in self._wrapped:
                    # ⭐ 恢复原始函数
                    module.quan_w_fn.forward = self._wrapped[key]
    self._wrapped.clear()
```

## 4. 关键点解释

### 4.1 为什么替换 `quan_w_fn.forward` 而不是 `QuanConv2d.forward`？

**原因**：
- `QuanConv2d.forward` 包含很多逻辑（量化权重、量化激活、卷积等）
- `quan_w_fn.forward` 只负责权重量化，更精确
- 只需要对**量化后的权重**注入故障，不需要修改其他逻辑

**执行位置**：
```python
# QuanConv2d.forward 中
weight = self.quan_w_fn(weight, wbits, is_activation=False)
#                        ↑
#                    这里被包装了！
#                    返回的 weight 已经是故障注入后的
```

### 4.2 如何保存原始函数？

**方法 1：使用字典保存**
```python
self._wrapped = {}  # 字典：{量化器ID: 原始函数}
orig_fn = module.quan_w_fn.forward
self._wrapped[id(module.quan_w_fn)] = orig_fn
```

**为什么使用 `id()`？**
- `id()` 返回对象的内存地址，作为唯一标识符
- 同一个量化器对象总是有相同的 ID
- 可以确保恢复时找到正确的原始函数

### 4.3 包装函数的闭包（Closure）

**问题**：为什么需要 `make_quan_wrapper` 函数？

**原因**：Python 闭包机制，确保每个包装函数都能访问：
- 对应的量化器实例（`quantizer_instance`）
- 对应的模块实例（`module_instance`）
- 对应的原始函数（`orig_fn`）

```python
def make_quan_wrapper(quantizer_instance, module_instance, orig_fn):
    # 这个函数创建了一个闭包，捕获了三个变量
    def wrapped_quan_forward(x, bits, ...):
        # 可以访问：
        # - quantizer_instance: 用于获取 scale
        # - module_instance: 用于检查训练模式
        # - orig_fn: 用于调用原始量化函数
        # - self: 用于调用 _inject_on_quantized_tensor
        pass
    return wrapped_quan_forward
```

**如果不使用闭包会怎样？**
- 所有包装函数可能共享同一个 `orig_fn`（最后一个）
- 导致所有层都使用同一个量化器的原始函数（错误！）

### 4.4 梯度保持机制

**关键代码**：
```python
return x_faulted.detach() + (x_q - x_q.detach())
```

**解释**：
- `x_faulted.detach()`: 故障注入后的值，但**不参与梯度计算**
- `(x_q - x_q.detach())`: 原始量化权重的梯度部分
- 结果：Forward 使用故障注入后的值，Backward 使用原始值的梯度

**为什么需要这样？**
- 如果训练时启用故障注入，需要保持梯度流
- 故障是随机引入的，不应该影响梯度计算
- 这样可以在训练时测试故障鲁棒性（虽然通常不推荐）

## 5. 完整示例

### 5.1 模拟代码执行

```python
# === 1. 创建模型和故障注入器 ===
model = create_model(...)
injector = FaultInjector(model, mode="ber", ber=1e-6)

# === 2. 启用故障注入 ===
injector.enable()

# 此时，所有 QuanConv2d 层的 quan_w_fn.forward 都被替换了
# 原始函数保存在 injector._wrapped 中

# === 3. 模型 forward ===
output = model(input)

# 内部执行：
#   QuanConv2d.forward(x)
#     → self.quan_w_fn(weight, wbits)
#     → quan_w_fn.forward(weight, wbits)  ← 包装后的函数
#       → 调用原始函数：x_q = orig_fn(weight, wbits)
#       → 注入故障：x_faulted = inject_faults(x_q)
#       → 返回：x_faulted.detach() + (x_q - x_q.detach())
#     → 使用故障注入后的权重进行卷积

# === 4. 禁用故障注入 ===
injector.disable()

# 此时，所有 quan_w_fn.forward 都恢复为原始函数
# injector._wrapped 被清空

# === 5. 再次 forward（无故障） ===
output = model(input)

# 内部执行：
#   QuanConv2d.forward(x)
#     → self.quan_w_fn(weight, wbits)
#     → quan_w_fn.forward(weight, wbits)  ← 原始函数
#       → 直接返回量化后的权重
#     → 使用原始量化权重进行卷积
```

## 6. 与其他方法的对比

### 6.1 方法对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| **函数替换**（当前方法） | 1. 简单直接<br>2. 不需要修改原始代码<br>3. 可以随时启用/禁用 | 1. 需要保存原始函数<br>2. 闭包可能复杂 |
| **Hook 机制** | 1. PyTorch 原生支持<br>2. 更灵活 | 1. 需要注册多个 hook<br>2. 管理复杂 |
| **继承和重写** | 1. 类型安全<br>2. 代码清晰 | 1. 需要修改原始代码<br>2. 不够灵活 |

### 6.2 为什么选择函数替换？

1. **代码隔离**：不需要修改 `QuanConv2d` 或量化器的代码
2. **灵活性**：可以随时启用/禁用，不影响原有功能
3. **精确性**：只替换量化器，不影响其他逻辑
4. **简单性**：实现相对简单，容易理解和维护

## 7. 注意事项

### 7.1 多次启用/禁用

```python
injector.enable()   # 第一次启用
injector.disable()  # 禁用
injector.enable()   # 第二次启用 - 需要重新包装
```

**实现**：`enable()` 方法会检查 `self._enabled`，避免重复包装。

### 7.2 模型状态保存

**问题**：如果保存模型 checkpoint，包装后的函数会被保存吗？

**答案**：
- `state_dict()` 只保存参数，不保存函数
- 但如果在 `enable()` 状态下保存，原始函数引用可能丢失
- **建议**：在 `disable()` 状态下保存模型

### 7.3 多线程/多进程

**问题**：函数替换是否线程安全？

**答案**：
- 函数替换是 Python 对象属性的修改
- 如果在多线程/多进程环境中使用，需要加锁
- **建议**：在单线程环境中使用，或使用 `torch.multiprocessing`

## 8. 总结

故障注入工具通过以下机制插入到 forward 过程：

1. **保存原始函数**：在 `enable()` 时保存 `quan_w_fn.forward`
2. **创建包装函数**：包装函数内部调用原始函数，然后注入故障
3. **替换函数**：将 `quan_w_fn.forward` 替换为包装函数
4. **恢复原始函数**：在 `disable()` 时恢复原始函数

**关键优势**：
- ✅ 代码隔离：不需要修改原始代码
- ✅ 灵活控制：可以随时启用/禁用
- ✅ 精确插入：只影响权重量化，不影响其他逻辑
- ✅ 梯度保持：支持训练时的故障注入（如果需要）

**这种机制在 Python 中非常常见**，类似于装饰器模式，但直接在运行时修改函数，更加灵活。



