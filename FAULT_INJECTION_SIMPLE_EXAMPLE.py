"""
故障注入机制简单示例
演示如何通过函数替换实现故障注入
"""

import torch
import torch.nn as nn

# === 1. 模拟量化器 ===
class SimpleQuantizer:
    def forward(self, x, bits):
        """原始量化函数"""
        # 简单量化：将浮点数转换为整数
        scale = 2 ** (bits - 1)
        quantized = torch.round(x * scale) / scale
        return quantized

# === 2. 模拟量化层 ===
class QuantizedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.quan_w_fn = SimpleQuantizer()
    
    def forward(self, x):
        # 调用量化器
        quantized_weight = self.quan_w_fn.forward(self.weight, bits=4)
        # 使用量化权重进行计算
        output = torch.matmul(x, quantized_weight)
        return output

# === 3. 故障注入器 ===
class SimpleFaultInjector:
    def __init__(self, model, ber=0.1):
        self.model = model
        self.ber = ber
        self._wrapped = {}  # 保存原始函数
    
    def enable(self):
        """启用故障注入"""
        print("=== 启用故障注入 ===")
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'quan_w_fn'):
                # 保存原始函数
                orig_forward = module.quan_w_fn.forward
                
                # 创建包装函数
                def make_wrapper(orig_fn):
                    def wrapped_forward(x, bits):
                        # 1. 调用原始量化函数
                        x_q = orig_fn(x, bits)
                        
                        # 2. 注入故障（随机翻转）
                        fault_mask = torch.rand_like(x_q) < self.ber
                        x_faulted = x_q.clone()
                        x_faulted[fault_mask] = -x_faulted[fault_mask]  # 翻转
                        
                        print(f"  故障注入：{fault_mask.sum().item()} 个值被翻转")
                        return x_faulted
                    
                    return wrapped_forward
                
                # 替换函数
                module.quan_w_fn.forward = make_wrapper(orig_forward)
                self._wrapped[id(module.quan_w_fn)] = orig_forward
                print(f"  已包装 {name} 的量化器")
    
    def disable(self):
        """禁用故障注入"""
        print("=== 禁用故障注入 ===")
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'quan_w_fn'):
                key = id(module.quan_w_fn)
                if key in self._wrapped:
                    # 恢复原始函数
                    module.quan_w_fn.forward = self._wrapped[key]
                    print(f"  已恢复 {name} 的量化器")
        
        self._wrapped.clear()

# === 4. 测试 ===
if __name__ == "__main__":
    # 创建模型
    model = QuantizedLayer()
    input_tensor = torch.randn(2, 3)
    
    print("=== 测试1：无故障注入 ===")
    output1 = model(input_tensor)
    print(f"输出: {output1}\n")
    
    # 创建故障注入器
    injector = SimpleFaultInjector(model, ber=0.2)
    
    # 启用故障注入
    injector.enable()
    
    print("\n=== 测试2：有故障注入 ===")
    output2 = model(input_tensor)
    print(f"输出: {output2}\n")
    
    # 再次测试
    print("=== 测试3：再次有故障注入（不同故障） ===")
    output3 = model(input_tensor)
    print(f"输出: {output3}\n")
    
    # 禁用故障注入
    injector.disable()
    
    print("=== 测试4：禁用后无故障注入 ===")
    output4 = model(input_tensor)
    print(f"输出: {output4}\n")
    
    print("=== 验证：测试1和测试4的输出应该相同 ===")
    print(f"输出1和输出4是否相同: {torch.allclose(output1, output4)}")

