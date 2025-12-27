import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import create_model
from util import load_checkpoint, get_config, init_dataloader
from quan import find_modules_to_quantize, replace_module_by_names
from util.mpq import switch_bit_width
from util.utils import accuracy

@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    # 仅测试 5 个 batch 以保证分析速度
    for i, (inputs, targets) in enumerate(loader):
        if i >= 5: break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

def analyze_model(checkpoint_path, configs, device, loader):
    print(f"\nAnalyzing Bit Sensitivity for: {checkpoint_path}")
    model = create_model(configs.arch, dataset=configs.dataloader.dataset)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model, _, _ = load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    switch_bit_width(model, configs.quan, 8, 8) # 使用 8-bit 进行敏感度分析
    
    # 找到所有量化层
    quant_layers = []
    for name, m in model.named_modules():
        from quan.func import QuanConv2d, QuanLinear
        if isinstance(m, (QuanConv2d, QuanLinear)):
            quant_layers.append((name, m))
    
    base_acc = test_accuracy(model, loader, device)
    print(f"Base Accuracy: {base_acc:.2f}%")
    
    # [Bits, Layers] 敏感度矩阵
    sensitivity_matrix = np.zeros((8, len(quant_layers)))
    
    for layer_idx, (name, layer) in enumerate(quant_layers):
        print(f"Testing Layer {layer_idx+1}/{len(quant_layers)}: {name}")
        orig_weight = layer.weight.data.clone()
        # 获取该层的 scale
        scale = layer.quan_w_fn.get_scale(8, detach=True)
        
        for bit_idx in range(8):
            # 模拟比特翻转：在量化域反转第 bit_idx 位
            # 翻转量 delta = 2^bit_idx * scale
            delta = (2**bit_idx) * scale
            
            # 简单的注入：我们直接给权重加上这个偏移（模拟 0->1 翻转的平均效应）
            layer.weight.data = orig_weight + delta
            
            fault_acc = test_accuracy(model, loader, device)
            # 记录准确率下降值
            sensitivity_matrix[bit_idx, layer_idx] = max(0, base_acc - fault_acc)
            
            # 恢复权重
            layer.weight.data.copy_(orig_weight)
            
    return sensitivity_matrix, [n for n, _ in quant_layers]

def main():
    device = torch.device("cuda")
    configs = get_config(default_file='template.yaml')
    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)

    models = [
        "plot_model/resnet18_cifar10_nude_srqat_ls0_checkpoint.pth.tar",
        "plot_model/resnet18_cifar10_baseline_checkpoint.pth.tar"
    ]
    
    results = {}
    for m_path in models:
        if os.path.exists(m_path):
            mat, layer_names = analyze_model(m_path, configs, device, train_loader)
            results[os.path.basename(m_path)] = mat

    # 绘制对比热力图
    num_models = len(results)
    fig, axes = plt.subplots(num_models, 1, figsize=(15, 5 * num_models))
    if num_models == 1: axes = [axes]
    
    for i, (name, mat) in enumerate(results.items()):
        sns.heatmap(mat, ax=axes[i], cmap="YlOrRd", annot=False)
        axes[i].set_title(f"Bit Sensitivity (Acc Drop): {name}")
        axes[i].set_xlabel("Layer Index")
        axes[i].set_ylabel("Bit Position (0=LSB, 7=MSB)")
        axes[i].invert_yaxis() # 让 MSB 在上面

    plt.tight_layout()
    plt.savefig("bit_sensitivity_comparison.png")
    print("\nHeatmap saved as bit_sensitivity_comparison.png")

if __name__ == "__main__":
    main()

