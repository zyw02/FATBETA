import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# 添加项目根目录
sys.path.append(str(Path.cwd()))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.mpq import switch_bit_width
from util.utils import accuracy

def main():
    # 1. 模拟命令行参数并加载配置 (如果没有通过命令行指定，则使用默认的)
    if len(sys.argv) < 2:
        sys.argv = ['eval_standalone.py', 'configs/training/r20.yaml']
    
    config = get_config(default_file='template.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 创建原始模型
    print(f"==> Creating model: {config.arch}")
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    model = model.to(device)
    
    # 3. 检查是否执行量化
    # 逻辑：如果在命令行指定的配置文件内容里没有 'quan:' 关键字，则视为浮点模型
    is_quantized = False
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        raw_config_text = f.read()
        if 'quan:' in raw_config_text:
            is_quantized = True

    if is_quantized:
        print(f"==> [Quantized Mode] Detected 'quan' in {config_path}, applying quantization structure...")
        modules_to_replace = find_modules_to_quantize(model, config)
        replace_module_by_names(model, modules_to_replace)
    else:
        print(f"==> [FP Mode] No 'quan' detected in {config_path}, treating as a floating-point model.")
    
    # 4. 加载 Checkpoint
    # 如果是测试 resnet56_fp，请手动修改此处的 checkpoint_path 或通过环境变量传入
    checkpoint_path = '/workspace/FATBETA/training/resnet20_c10_fp/resnet20_c10_fp_checkpoint.pth.tar'
    if not is_quantized and 'resnet20' in config.arch:
        checkpoint_path = '/workspace/FATBETA/training/resnet20_c10_fp/resnet20_c10_fp_checkpoint.pth.tar'
        
    print(f"==> Loading checkpoint: {checkpoint_path}")
    # load_checkpoint 内部会处理 EMA 的选择，并打印加载详情
    load_checkpoint(model, checkpoint_path, model_device=device, strict=False)
    
    # 5. 初始化数据加载器
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 6. 如果是量化模型，进行初始化和位宽切换
    if is_quantized:
        print("==> Initializing quantizers with a warm-up forward pass...")
        model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(test_loader))
            inputs = inputs.to(device)
            _ = model(inputs)
        
        print("==> Locking bit-width to 8-bit...")
        from quan.func import QuanConv2d, QuanLinear
        for name, module in model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                target_b = 8
                if hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    target_b = module.fixed_bits
                
                module.bits = (target_b, target_b)
                if hasattr(module, 'current_bit_cands_w'):
                    module.current_bit_cands_w = [torch.tensor(target_b).to(device)]
                if hasattr(module, 'current_bit_cands_a'):
                    module.current_bit_cands_a = [torch.tensor(target_b).to(device)]

        switch_bit_width(model, quan_scheduler=config.quan, wbit=8, abits=8)
    
    # 7. 开始直接遍历测试
    top1_sum = 0
    total_count = 0
    print("==> Starting direct evaluation over test_loader...")
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            acc1, _ = accuracy(outputs.data, targets.data, topk=(1, 5))
            
            bs = inputs.size(0)
            top1_sum += acc1.item() * bs
            total_count += bs
            
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(test_loader)} | Current Acc: {top1_sum/total_count:.2f}%")

    final_acc = top1_sum / total_count
    print("-" * 30)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print(f"Total samples tested: {total_count}")

if __name__ == "__main__":
    main()
