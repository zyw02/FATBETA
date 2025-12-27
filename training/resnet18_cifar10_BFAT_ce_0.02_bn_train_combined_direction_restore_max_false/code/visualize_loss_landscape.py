import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import os
from pathlib import Path
from model import create_model
from util import get_config, load_checkpoint, init_dataloader
from util.utils import accuracy
from quan import find_modules_to_quantize, replace_module_by_names

def get_directions(model):
    """创建随机方向向量并进行 Filter-wise Normalization"""
    directions = []
    for param in model.parameters():
        if param.dim() <= 1:
            directions.append(torch.zeros_like(param))
            continue
        
        # 生成随机方向
        d = torch.randn_like(param)
        
        # Filter-wise Normalization: 确保扰动的比例与权重本身一致
        if param.dim() == 4: # Conv
            for i in range(d.size(0)):
                d[i].mul_(param[i].norm() / (d[i].norm() + 1e-10))
        elif param.dim() == 2: # Linear
            for i in range(d.size(0)):
                d[i].mul_(param[i].norm() / (d[i].norm() + 1e-10))
        
        directions.append(d)
    return directions

def apply_perturbation(model, base_params, directions_x, directions_y, alpha, beta):
    """应用扰动: W = W0 + alpha * dx + beta * dy"""
    for param, base, dx, dy in zip(model.parameters(), base_params, directions_x, directions_y):
        param.data.copy_(base + alpha * dx + beta * dy)

@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """同时返回 Loss 和 Accuracy"""
    model.eval()
    total_loss = 0
    total_acc1 = 0
    num_batches = 5
    for i, (inputs, targets) in enumerate(loader):
        if i >= num_batches: break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        
        total_loss += loss.item()
        total_acc1 += acc1.item()
        
    return total_loss / num_batches, total_acc1 / num_batches

def calibrate_bn(model, loader, device):
    """简单的 BN 校准逻辑 - 针对混合精度模型切换位宽后的必要步骤"""
    print("Recalibrating BatchNorm statistics for the current bit-width...")
    model.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if i >= 50: break # 使用 50 个 batch 进行校准
            model(inputs.to(device))
    model.eval()

def main():
    # 1. 解析命令行参数
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Loss Landscape Visualization', add_help=False)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--bit', type=int, default=None, help='Specific bit-width to test')
    temp_args, remaining_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_args
    
    if len(remaining_args) < 1:
        possible_config = "configs/training/train_resnet18_cifar10_single_gpu.yaml"
        if os.path.exists(possible_config):
            sys.argv.append(possible_config)
        else:
            print("Error: No config file found.")
            return

    # 2. 环境准备
    checkpoint_path = temp_args.checkpoint
    if not checkpoint_path:
        checkpoint_path = "training/resnet18_cifar10_BFAT_ber_0.02_best_res/resnet18_cifar10_BFAT_ber_0.02_checkpoint.pth.tar"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_config(default_file='template.yaml')
    
    # 3. 创建模型并加载权重
    model = create_model(configs.arch, dataset=configs.dataloader.dataset)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model, _, _ = load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    # 4. 显式设置位宽
    from util.mpq import switch_bit_width
    target_bits = configs.target_bits
    run_bit = temp_args.bit if temp_args.bit is not None else max(target_bits)
    
    print(f"Switching model to: {run_bit}-bit")
    switch_bit_width(model, quan_scheduler=configs.quan, wbit=run_bit, abits=run_bit)
    
    # 5. 数据准备
    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    
    # --- 重要：进行 BN 校准 ---
    # 对于混合精度模型，切换位宽后必须校准 BN 统计量，否则准确率会极低
    calibrate_bn(model, train_loader, device)
    
    # 验证当前点性能
    test_loss, test_acc = evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"Current Model Performance at {run_bit}-bit: Loss={test_loss:.4f}, Top1-Acc={test_acc:.2f}%")
    
    # 6. 绘图准备
    model_name = Path(checkpoint_path).stem
    plot_name = f"loss_landscape_{model_name}_{run_bit}bit.png"
    base_params = [p.data.clone() for p in model.parameters()]
    dir_x = get_directions(model)
    dir_y = get_directions(model)

    res = 21 
    range_val = 0.5 
    alpha_vals = np.linspace(-range_val, range_val, res)
    beta_vals = np.linspace(-range_val, range_val, res)
    loss_grid = np.zeros((res, res))

    print(f"Starting landscape sampling ({res}x{res} grid)...")
    criterion = torch.nn.CrossEntropyLoss()
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            apply_perturbation(model, base_params, dir_x, dir_y, alpha, beta)
            # 在绘图点采样时，我们只需要 Loss
            loss, _ = evaluate_model(model, train_loader, criterion, device)
            loss_grid[i, j] = loss
            if (i * res + j + 1) % 50 == 0:
                print(f"Progress: {((i * res + j + 1) / (res*res)) * 100:.1f}%")

    apply_perturbation(model, base_params, dir_x, dir_y, 0, 0)

    # 7. 绘图与保存
    X, Y = np.meshgrid(alpha_vals, beta_vals)
    Z = loss_grid
    
    # 确保目标文件夹存在
    os.makedirs("exp/2d", exist_ok=True)
    os.makedirs("exp/3d", exist_ok=True)
    
    model_name = Path(checkpoint_path).stem
    
    # --- 绘制 2D 等高线图 ---
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)
    plt.xlabel('Direction X')
    plt.ylabel('Direction Y')
    plt.colorbar(cp, label='Loss')
    
    plot_path_2d = f"exp/2d/loss_2d_{model_name}_{run_bit}bit.png"
    plt.savefig(plot_path_2d, bbox_inches='tight')
    plt.close()
    print(f"2D Plot saved as {plot_path_2d}")

    # --- 绘制 3D 地形图 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)
    ax.set_xlabel('Direction X')
    ax.set_ylabel('Direction Y')
    ax.set_zlabel('Loss')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    # 调整视角以获得更好的视觉效果
    ax.view_init(elev=30, azim=45)
    
    plot_path_3d = f"exp/3d/loss_3d_{model_name}_{run_bit}bit.png"
    plt.savefig(plot_path_3d, bbox_inches='tight')
    plt.close()
    print(f"3D Plot saved as {plot_path_3d}")

if __name__ == "__main__":
    main()
