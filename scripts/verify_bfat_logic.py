
import torch
import torch.nn as nn
import sys
import os
import argparse
from types import SimpleNamespace

# Add workspace to path
sys.path.append(os.getcwd())

# Mock modules
from quan.func import QuanConv2d, QuanLinear
# We need to import process_gs to test it
import process_gs
from util import AverageMeter

class MockFaultInjector:
    def __init__(self):
        self.enabled = False
        self.all_bits = False
        self.only_msb = False
        self.ber = 0.0
        self.seed = 0
        self.whitelist_layer = None
        
    def enable(self):
        self.enabled = True
        print("[MockInjector] Enabled")
        
    def disable(self):
        self.enabled = False
        print("[MockInjector] Disabled")
        
    def reset_forward_seed(self):
        print(f"[MockInjector] Reset Seed: {self.seed}")

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 3, padding=1)
        self.fc = nn.Linear(10 * 32 * 32, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def verify_bfat():
    print("=== Verifying BFAT Integration in process_gs.py ===")
    
    # 1. Setup Mock Environment
    model = MockModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_q = None
    criterion = nn.CrossEntropyLoss()
    
    # Mock Data
    inputs = torch.randn(2, 3, 32, 32).cuda()
    targets = torch.tensor([0, 1]).cuda()
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Mock Fault Injector
    fault_injector = MockFaultInjector()
    
    # Mock Monitors (simple list)
    monitors = []
    
    # Mock Configs
    configs = SimpleNamespace()
    configs.epochs = 1
    configs.dataloader = SimpleNamespace(batch_size=2)
    configs.world_size = 1
    configs.log = SimpleNamespace(print_freq=1)
    configs.target_bits = [4, 8]
    configs.sandwich_training = False
    configs.adaptive_region_weight_decay = 0.0
    configs.weight_decay = 0.0
    configs.gno_alpha = 0.0
    configs.enable_dynamic_bit_training = False
    configs.quan = None # or SimpleNamespace() if needed
    configs.post_training_batchnorm_calibration = False
    configs.distributed = False
    
    # BFAT Config
    configs.bfat = SimpleNamespace()
    configs.bfat.enabled = True
    configs.bfat.start_epoch = 0
    configs.bfat.all_bits = True
    configs.bfat.ber = 0.01
    configs.bfat.projection_mode = "direction" # Test direction first
    
    # Run Train for 1 epoch (1 batch)
    try:
        print("\n--- Running process_gs.train with BFAT Enabled ---")
        process_gs.train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=0,
            monitors=monitors,
            configs=configs,
            nr_random_sample=1,
            fault_injector=fault_injector,
            annealing_schedule=lambda x: 0.5,
            freezing_annealing_schedule=lambda x: 0.0,
        )
        print("✅ process_gs.train completed without error")
        
        # Check if gradients exist
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                print(f"Parameter {p.shape} has gradient norm: {p.grad.norm().item():.4f}")
        
        if not has_grad:
            print("❌ No gradients found!")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = verify_bfat()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
