"""
测试故障感知训练中的梯度传播是否正常
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("梯度传播验证测试")
print("=" * 60)

# 模拟故障注入器的梯度保持机制
def simulate_fault_injection(x_q, fault_mask):
    """
    模拟故障注入，使用梯度保持技巧
    x_q: 原始量化值（需要梯度）
    fault_mask: 故障掩码（随机bit-flip）
    """
    # 模拟bit-flip故障（简化版）
    x_faulted = x_q * fault_mask  # 简化的故障注入
    
    # 关键：梯度保持技巧
    # forward使用故障值，backward使用原始值的梯度
    return x_faulted.detach() + (x_q - x_q.detach())

# 测试1: 梯度是否正常传播
print("\n【测试1: 梯度传播验证】")
print("-" * 60)

# 创建简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3, requires_grad=True))
    
    def forward(self, x, fault_mask=None):
        # 模拟量化
        x_q = x * self.weight
        
        if fault_mask is not None:
            # 应用故障注入（梯度保持）
            x_q = simulate_fault_injection(x_q, fault_mask)
        
        return x_q.sum()

model = SimpleModel()
x = torch.randn(3, 3, requires_grad=False)
fault_mask = torch.rand(3, 3) > 0.5  # 随机故障掩码

# 正常forward
output_normal = model(x, fault_mask=None)
output_normal.backward()
grad_normal = model.weight.grad.clone()
model.weight.grad.zero_()

# 故障注入forward
output_faulted = model(x, fault_mask=fault_mask)
output_faulted.backward()
grad_faulted = model.weight.grad.clone()

print(f"正常forward梯度: {grad_normal.norm().item():.6f}")
print(f"故障注入梯度: {grad_faulted.norm().item():.6f}")
print(f"梯度差异: {(grad_normal - grad_faulted).norm().item():.6f}")

if grad_normal.norm() > 0 and grad_faulted.norm() > 0:
    print("✅ 梯度正常传播！")
    print("   说明：故障注入不会破坏梯度流")
else:
    print("❌ 梯度传播异常！")

# 测试2: TRADES训练的梯度传播
print("\n【测试2: TRADES训练梯度传播验证】")
print("-" * 60)

model2 = SimpleModel()
x2 = torch.randn(3, 3, requires_grad=False)
target = torch.randn(3, 3)
fault_mask2 = torch.rand(3, 3) > 0.5

# 模拟TRADES训练
optimizer = torch.optim.SGD(model2.parameters(), lr=0.1)

# 第一次forward: 正常
output_normal2 = model2(x2, fault_mask=None)
loss_normal = F.mse_loss(output_normal2, target)

# 第二次forward: 故障注入
output_faulted2 = model2(x2, fault_mask=fault_mask2)
loss_faulted = F.mse_loss(output_faulted2, target)

# TRADES损失（简单组合）
alpha, beta = 0.6, 0.4
loss = alpha * loss_normal + beta * loss_faulted

# 反向传播
optimizer.zero_grad()
loss.backward()
grad_trades = model2.weight.grad.clone()

print(f"TRADES损失: {loss.item():.6f}")
print(f"TRADES梯度: {grad_trades.norm().item():.6f}")

if grad_trades.norm() > 0:
    print("✅ TRADES训练梯度正常传播！")
    print("   说明：两次forward的梯度都能正确组合")
else:
    print("❌ TRADES训练梯度传播异常！")

# 测试3: 验证梯度保持技巧
print("\n【测试3: 梯度保持技巧验证】")
print("-" * 60)

x3 = torch.randn(2, 2, requires_grad=True)
fault_mask3 = torch.rand(2, 2) > 0.5

# 原始值
x_q = x3 * 2  # 模拟量化

# 故障值
x_faulted = x_q * fault_mask3.float()

# 梯度保持技巧
result = x_faulted.detach() + (x_q - x_q.detach())

# 计算梯度
loss = result.sum()
loss.backward()

print(f"原始x3梯度: {x3.grad.norm().item():.6f}")
print(f"x_q值: {x_q.detach().norm().item():.6f}")
print(f"x_faulted值: {x_faulted.detach().norm().item():.6f}")
print(f"result值: {result.detach().norm().item():.6f}")

if x3.grad is not None and x3.grad.norm() > 0:
    print("✅ 梯度保持技巧正常工作！")
    print("   说明：前向使用故障值，反向使用原始值梯度")
else:
    print("❌ 梯度保持技巧异常！")

# 测试4: 验证计算图完整性
print("\n【测试4: 计算图完整性验证】")
print("-" * 60)

x4 = torch.randn(2, 2, requires_grad=True)
model4 = nn.Linear(2, 2)

# 正常forward
output4 = model4(x4)
loss4 = output4.sum()

# 检查计算图
print(f"output4.requires_grad: {output4.requires_grad}")
print(f"loss4.requires_grad: {loss4.requires_grad}")

# 故障注入forward
x_q4 = model4(x4)
fault_mask4 = torch.rand(2, 2) > 0.5
x_faulted4 = x_q4 * fault_mask4.float()
result4 = x_faulted4.detach() + (x_q4 - x_q4.detach())
loss4_fault = result4.sum()

print(f"result4.requires_grad: {result4.requires_grad}")
print(f"loss4_fault.requires_grad: {loss4_fault.requires_grad}")

if result4.requires_grad and loss4_fault.requires_grad:
    print("✅ 计算图完整！")
    print("   说明：故障注入后计算图仍然完整")
    
    # 尝试反向传播
    loss4_fault.backward()
    if model4.weight.grad is not None:
        print(f"✅ 梯度计算成功！梯度: {model4.weight.grad.norm().item():.6f}")
    else:
        print("❌ 梯度计算失败！")
else:
    print("❌ 计算图不完整！")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)



