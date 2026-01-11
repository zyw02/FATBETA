import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径中
sys.path.append(str(Path.cwd()))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.config import get_config
from util.utils import preprocess_model

def main():
    # 1. 模拟命令行参数并加载 r20.yaml 配置
    config_path = 'configs/training/r20.yaml'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found!")
        return

    sys.argv = ['inspect_resnet20.py', config_path]
    config = get_config(default_file='template.yaml')
    
    print(f"==> Loading Configuration from: {config_path}")
    print(f"==> Target Arch: {config.arch}")
    
    # 2. 创建原始模型
    print("==> Creating raw model...")
    model = create_model(config.arch, dataset=config.dataloader.dataset, pre_trained=config.pre_trained)
    
    # 3. 应用量化预处理 (将普通 BN 替换为支持多位宽切换的 SwithableBatchNorm)
    print("==> Applying preprocess_model (Switchable BN)...")
    model = preprocess_model(model, config)
    
    # 4. 识别并替换量化层 (根据 config 中的 quan 设置)
    print("==> Replacing modules with quantization layers (LSQ)...")
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 5. 打印模型结构
    print("\n" + "="*50)
    print("FINAL QUANTIZED MODEL STRUCTURE")
    print("="*50)
    print(model)
    print("="*50)

if __name__ == "__main__":
    main()