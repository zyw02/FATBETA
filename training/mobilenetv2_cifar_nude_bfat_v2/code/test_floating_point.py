import torch
import torch.nn as nn
import os
import sys
from model import create_model
from util import get_config, init_dataloader, load_checkpoint
from util.utils import accuracy

def test():
    # 1. 加载配置
    # 使用项目标准的 get_config，它会自动从 sys.argv 读取配置文件
    if len(sys.argv) < 2:
        print("Usage: python test_floating_point.py <config_path>")
        return
        
    configs = get_config(default_file='template.yaml')
    
    # 强制覆盖一些测试用的参数
    configs.eval = True
    configs.distributed = False
    configs.local_rank = 0
    
    # 2. 创建模型
    print(f"==> Creating model: {configs.arch}")
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=False)
    model = model.cuda()
    
    # 3. 确定权重路径
    # 优先使用 best 模型，如果不存在则使用最新的 checkpoint
    best_path = f'training/{configs.name}/{configs.name}_best.pth.tar'
    last_path = f'training/{configs.name}/{configs.name}_checkpoint.pth.tar'
    
    checkpoint_path = best_path if os.path.exists(best_path) else last_path
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Cannot find checkpoint at {best_path} or {last_path}")
        return
    
    print(f"==> Loading checkpoint: {checkpoint_path}")
    model, _, _ = load_checkpoint(model, checkpoint_path, 'cuda', lean=True)
    model.eval()
    
    # 4. 初始化数据加载器
    print(f"==> Initializing dataloader for {configs.dataloader.dataset}")
    _, _, test_loader, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    
    # 5. 纯净的测试循环
    top1_sum = 0
    top5_sum = 0
    total_samples = 0
    
    print(f"==> Starting Inference on {len(test_loader)} batches...")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = model(inputs)
            
            # 计算准确率
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            batch_size = inputs.size(0)
            top1_sum += acc1.item() * batch_size
            top5_sum += acc5.item() * batch_size
            total_samples += batch_size
            
            if (i + 1) % 20 == 0:
                print(f"Batch [{i+1:3d}/{len(test_loader)}] | Current Top-1: {top1_sum/total_samples:6.2f}%")
                
    final_top1 = top1_sum / total_samples
    final_top5 = top5_sum / total_samples
    
    print("\n" + "="*40)
    print(f"Final Test Results (Floating Point)")
    print(f"Model: {configs.arch}")
    print(f"Dataset: {configs.dataloader.dataset}")
    print(f"Total samples: {total_samples}")
    print("-" * 40)
    print(f"Top-1 Accuracy: {final_top1:6.2f}%")
    print(f"Top-5 Accuracy: {final_top5:6.2f}%")
    print("="*40)

if __name__ == '__main__':
    test()
