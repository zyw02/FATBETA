
import sys
import os
sys.path.insert(0, '/root/autodl-tmp/retraining-free-quantization')

import torch
import yaml
import logging
from pathlib import Path
from util import get_config, init_logger, set_global_seed, load_checkpoint
from process_normal import validate, PerformanceScoreboard
from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.dist import logger_info

# Load config
script_dir = Path.cwd()
with open('configs/eval/eval_resnet18_cifar10_fault_tolerance_test.yaml.temp_ber_0.001', 'r') as f:
    config_dict = yaml.safe_load(f)

# Create a simple config object
class Config:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

configs = Config(config_dict)

# Setup device
configs.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = create_model(configs.arch, dataset=configs.dataloader['dataset'], pre_trained=configs.pre_trained)
model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
model = model.to(configs.device)

# Load checkpoint
model, _, _ = load_checkpoint(model, configs.resume['path'], 'cuda', lean=configs.resume['lean'])

# Create dataloader
from util import init_dataloader
_, _, test_loader, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)

# Create fault injector
from util.fault_injector import FaultInjector
fault_injector = FaultInjector(
    model=model,
    mode="ber",
    ber=0.001,
    enable_in_training=False,
    enable_in_inference=True,
    protect_highest_bit=configs.fault_aware_training.get('protect_highest_bit', False)
)

# Enable fault injection
fault_injector.enable()

# Run validation
criterion = torch.nn.CrossEntropyLoss().cuda()
monitors = []  # Empty monitors for simple evaluation

top1, top5, loss = validate(test_loader, model, criterion, 0, monitors, configs)

print(f"BER_0.001_Top1_{top1:.3f}_Top5_{top5:.3f}_Loss_{loss:.4f}")
