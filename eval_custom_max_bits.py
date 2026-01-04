import torch
import yaml
import munch
import os
from pathlib import Path
from model import create_model
from util import preprocess_model, init_dataloader, load_checkpoint
from quan import find_modules_to_quantize, replace_module_by_names
from util.mpq import switch_bit_width

def main():
    # 1. Load configs
    config_path = '/root/autodl-tmp/FATBETA/training/mobilenetv2_cifar10_w2to6_a2to6/configs.yaml'
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    configs = munch.munchify(cfg_dict)
    
    # Update path to be relative or absolute correctly
    configs.dataloader.path = './data/cifar10'

    # 2. Create model
    print(f"Creating model: {configs.arch}")
    model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    
    # 3. Insert quantizers
    print("Inserting quantizers...")
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model = model.cuda()

    # 4. Load checkpoint
    checkpoint_path = '/root/autodl-tmp/FATBETA/training/mobilenetv2_cifar10_w2to6_a2to6/mobilenetv2_cifar10_w2to6_a2to6_checkpoint.pth.tar'
    print(f"Loading checkpoint from {checkpoint_path}")
    # load_checkpoint expects (model, path)
    load_checkpoint(model, checkpoint_path)

    # 5. Set to max bits
    max_bits = max(configs.target_bits)
    print(f"Switching to max bits: {max_bits}")
    switch_bit_width(model, configs.quan, max_bits, max_bits)
    model.eval()

    # 6. Init dataloader
    print("Initializing dataloader...")
    # configs.dataloader is a munch object, init_dataloader expects it
    _, _, test_loader, _, _ = init_dataloader(configs.dataloader, configs.arch)

    # 7. Evaluate
    print("Starting evaluation...")
    correct = 0
    total = 0
    
    # Using the same evaluation logic as typical PyTorch loops
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda()
            target = target.cuda()
            
            output = model(images)
            
            # Get accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (i + 1) % 20 == 0:
                print(f"Batch {i+1}/{len(test_loader)}, Current Acc: {100 * correct / total:.2f}%")

    final_acc = 100 * correct / total
    print("-" * 30)
    print(f'Final Accuracy of {configs.arch} on CIFAR-10 test set (bits={max_bits}): {final_acc:.2f}%')
    print("-" * 30)

if __name__ == '__main__':
    main()

