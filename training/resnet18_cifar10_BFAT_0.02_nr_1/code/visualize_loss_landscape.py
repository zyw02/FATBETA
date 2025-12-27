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

def get_directions(model, use_norm=True):
    """创建随机方向向量"""
    directions = []
    for param in model.parameters():
        if param.dim() <= 1:
            directions.append(torch.zeros_like(param))
            continue
        
        # 生成随机方向
        d = torch.randn_like(param)
        
        if use_norm:
            # Filter-wise Normalization: 确保扰动的比例与权重本身一致
            if param.dim() == 4: # Conv
                for i in range(d.size(0)):
                    d[i].mul_(param[i].norm() / (d[i].norm() + 1e-10))
            elif param.dim() == 2: # Linear
                for i in range(d.size(0)):
                    d[i].mul_(param[i].norm() / (d[i].norm() + 1e-10))
        else:
            # 不使用归一化：所有模型承受相同绝对模长的扰动
            # 将扰动缩放到模型总参数量的万分之一级别作为基础（这是一个经验值）
            d.div_(d.norm() + 1e-10)
        
        directions.append(d)
    return directions

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

def apply_perturbation(model, base_params, directions_x, directions_y, alpha, beta):
    """应用扰动: W = W0 + alpha * dx + beta * dy"""
    for param, base, dx, dy in zip(model.parameters(), base_params, directions_x, directions_y):
        param.data.copy_(base + alpha * dx + beta * dy)

def calibrate_bn(model, loader, device):
    """简单的 BN 校准逻辑"""
    print("Recalibrating BatchNorm statistics...")
    model.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if i >= 50: break
            model(inputs.to(device))
    model.eval()

def main():
    # 1. 解析命令行参数
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Loss Landscape Visualization', add_help=False)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--bit', type=int, default=None, help='Specific bit-width to test')
    parser.add_argument('--no_norm', action='store_true', help='Disable filter-wise normalization (to show absolute sharpness)')
    parser.add_argument('--z_max', type=float, default=None, help='Fix maximum Z-axis (Loss) value for comparison')
    parser.add_argument('--range', type=float, default=0.5, help='Perturbation range')
    temp_args, remaining_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining_args
    
    if len(remaining_args) < 1:
        possible_config = "configs/training/train_resnet18_cifar10_single_gpu.yaml"
        if os.path.exists(possible_config): sys.argv.append(possible_config)
        else: return

    # 2. 环境准备
    checkpoint_path = temp_args.checkpoint
    if not checkpoint_path:
        checkpoint_path = "training/resnet18_cifar10_BFAT_ber_0.02_best_res/resnet18_cifar10_BFAT_ber_0.02_checkpoint.pth.tar"
    
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
    switch_bit_width(model, quan_scheduler=configs.quan, wbit=run_bit, abits=run_bit)
    
    # 5. 数据与校准
    train_loader, _, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    calibrate_bn(model, train_loader, device)
    
    test_loss, test_acc = evaluate_model(model, train_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"Initial Performance: Loss={test_loss:.4f}, Top1-Acc={test_acc:.2f}%")
    
    # 6. 绘图准备
    model_name = Path(checkpoint_path).stem
    norm_status = "nonorm" if temp_args.no_norm else "filternorm"
    base_params = [p.data.clone() for p in model.parameters()]
    dir_x = get_directions(model, use_norm=not temp_args.no_norm)
    dir_y = get_directions(model, use_norm=not temp_args.no_norm)

    res = 21 
    range_val = temp_args.range
    alpha_vals = np.linspace(-range_val, range_val, res)
    beta_vals = np.linspace(-range_val, range_val, res)
    loss_grid = np.zeros((res, res))

    print(f"Sampling ({norm_status}, range={range_val})...")
    criterion = torch.nn.CrossEntropyLoss()
    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            apply_perturbation(model, base_params, dir_x, dir_y, alpha, beta)
            loss, _ = evaluate_model(model, train_loader, criterion, device)
            loss_grid[i, j] = loss

    apply_perturbation(model, base_params, dir_x, dir_y, 0, 0)

    # 7. 绘图与保存
    X, Y = np.meshgrid(alpha_vals, beta_vals)
    Z = loss_grid
    os.makedirs("exp/2d", exist_ok=True)
    os.makedirs("exp/3d", exist_ok=True)
    
    # --- 2D ---
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.colorbar(cp, label='Loss')
    plot_path_2d = f"exp/2d/loss_2d_{model_name}_{run_bit}bit_{norm_status}.png"
    plt.savefig(plot_path_2d, bbox_inches='tight')
    plt.close()

    # --- 3D ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)
    if temp_args.z_max: ax.set_zlim(Z.min(), temp_args.z_max)
    ax.view_init(elev=30, azim=45)
    plot_path_3d = f"exp/3d/loss_3d_{model_name}_{run_bit}bit_{norm_status}.png"
    plt.savefig(plot_path_3d, bbox_inches='tight')
    plt.close()
    print(f"Results saved to exp/ with suffix {norm_status}")

if __name__ == "__main__":
    main()
