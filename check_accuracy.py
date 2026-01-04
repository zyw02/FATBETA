import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.mobilenetv2 import mobilenet_v2  # 导入我们修改后的模型

def test():
    # 1. 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备数据增强（必须与训练时的测试配置一致）
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # 3. 创建模型（确保参数与你训练时一致：32x32输入，10类）
    # 注意：我们之前修改了 mobilenet_v2，它现在能自动处理 stride=1
    model = mobilenet_v2(pretrained=False, num_classes=10, input_size=32, dropout=0.2)
    model = model.to(device)

    # 4. 加载权重
    checkpoint_path = '/root/autodl-tmp/FATBETA/training/mobilenetv2_cifar10_fp32/mobilenetv2_cifar10_fp32_checkpoint.pth.tar'
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    # 优先尝试加载 EMA 权重（通常精度更高），如果没有则加载普通权重
    if 'state_dict_ema' in checkpoint and checkpoint['state_dict_ema'] is not None:
        print("Loading EMA weights...")
        state_dict = checkpoint['state_dict_ema']
    else:
        print("Loading normal weights...")
        state_dict = checkpoint['state_dict']

    # 移除 DDP 包装产生的 'module.' 前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # 5. 测试循环
    correct_t1 = 0
    correct_t5 = 0
    total = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 计算 Top-1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_t1 += predicted.eq(targets).sum().item()

            # 计算 Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_t5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

    print(f"\nTest Results on CIFAR-10 Test Set:")
    print(f"Total Samples: {total}")
    print(f"Top-1 Accuracy: {100. * correct_t1 / total:.2f}%")
    print(f"Top-5 Accuracy: {100. * correct_t5 / total:.2f}%")

if __name__ == '__main__':
    test()


