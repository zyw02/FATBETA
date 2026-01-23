#!/usr/bin/env python3
"""
训练 ResNet18 在 CIFAR-10 数据集上
支持使用 torchrun 进行分布式训练
单卡训练: python train_r18_224.py
多卡训练: torchrun --nproc_per_node=NUM_GPUS train_r18_224.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import os

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun 会自动设置这些环境变量
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化进程组
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        # 单卡训练模式
        return False, 0, 1, 0

def cleanup():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train():
    # 初始化分布式训练
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)  # 只有主进程进行打印和保存
    
    # 超参数设置
    EPOCHS = 30
    BATCH_SIZE = 32  # 每个GPU的batch size
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # 设备设置
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print(f"分布式训练: {is_distributed}")
        print(f"World Size: {world_size}")
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(local_rank)}")
            print(f"显存容量: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.2f} GB")
    
    # 数据预处理和增强
    if is_main_process:
        print("\n准备数据集...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=transform_train
    )
    
    # 使用 DistributedSampler 进行分布式采样
    train_sampler = DistributedSampler(
        trainset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    ) if is_distributed else None
    
    trainloader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=(train_sampler is None),  # 使用sampler时不能shuffle
        num_workers=4, 
        pin_memory=True,
        sampler=train_sampler
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform_test
    )
    
    # 测试集也使用 DistributedSampler 避免重复评估
    test_sampler = DistributedSampler(
        testset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # 测试集不需要shuffle
    ) if is_distributed else None
    
    testloader = DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        sampler=test_sampler
    )
    
    if is_main_process:
        print(f"训练集大小: {len(trainset)}")
        print(f"测试集大小: {len(testset)}")
    
    # 创建模型
    print("\n创建 ResNet18 模型...")
    model = models.resnet18(pretrained=False)
    
    # 修改第一层（CIFAR-10图像较小，不需要那么大的kernel）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除maxpool
    
    # 修改最后一层为10类
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    model = model.to(device)
    
    # 使用 DDP 包装模型（分布式训练）
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度器（cosine annealing）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练循环
    if is_main_process:
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
    
    best_acc = 0.0
    training_start_time = time.time()  # 记录训练开始时间
    
    for epoch in range(EPOCHS):
        # 设置 epoch 以确保每个 epoch 的数据打乱不同
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0 and is_main_process:
                print(f'Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(trainloader)}] '
                      f'Loss: {loss.item():.3f} Acc: {100.*correct/total:.2f}%')
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(trainloader)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 在分布式训练中同步测试结果
        if is_distributed:
            # 将指标转换为tensor以进行all_reduce
            metrics = torch.tensor([correct, total, test_loss], dtype=torch.float32, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            correct, total, test_loss = metrics[0].item(), metrics[1].item(), metrics[2].item()
        
        test_acc = 100. * correct / total
        avg_test_loss = test_loss / len(testloader) / world_size if is_distributed else test_loss / len(testloader)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # 计算预计剩余时间
        total_elapsed = time.time() - training_start_time
        avg_epoch_time = total_elapsed / (epoch + 1)
        remaining_epochs = EPOCHS - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # 只在主进程打印
        if is_main_process:
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{EPOCHS} 完成 (耗时: {epoch_time:.1f}s)')
            print(f'训练 - Loss: {avg_train_loss:.3f} | Acc: {train_acc:.2f}%')
            print(f'测试 - Loss: {avg_test_loss:.3f} | Acc: {test_acc:.2f}%')
            print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 显示预计剩余时间
            if remaining_epochs > 0:
                if estimated_remaining >= 3600:
                    eta_str = f"{estimated_remaining/3600:.1f}小时"
                elif estimated_remaining >= 60:
                    eta_str = f"{estimated_remaining/60:.1f}分钟"
                else:
                    eta_str = f"{estimated_remaining:.0f}秒"
                print(f'预计剩余时间: {eta_str} (平均每epoch: {avg_epoch_time:.1f}s)')
            else:
                print(f'训练即将完成！')
            
            print(f'{"="*60}\n')
        
        # 只在主进程保存模型
        if test_acc > best_acc:
            if is_main_process:
                print(f'✓ 保存最佳模型 (准确率: {test_acc:.2f}%)')
            best_acc = test_acc
            if is_main_process:
                # 保存DDP模型时需要访问module属性
                model_state = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc,
                    'loss': avg_test_loss,
                }, 'resnet18_cifar10_best.pth')
    
    # 保存最终模型（只在主进程）
    if is_main_process:
        model_state = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': test_acc,
            'loss': avg_test_loss,
        }, 'resnet18_cifar10_final.pth')
    
    if is_main_process:
        print("\n" + "="*60)
        print("训练完成！")
        print(f"最佳测试准确率: {best_acc:.2f}%")
        print(f"最终测试准确率: {test_acc:.2f}%")
        print("模型已保存:")
        print("  - resnet18_cifar10_best.pth (最佳)")
        print("  - resnet18_cifar10_final.pth (最终)")
        print("="*60)
    
    # 清理分布式训练环境
    cleanup()

if __name__ == '__main__':
    train()

