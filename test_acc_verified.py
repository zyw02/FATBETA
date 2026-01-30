import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.mpq import switch_bit_width
from util.qat import get_quantized_layers, set_bit_width

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100. * correct / total

def check_sd_match(model, ckpt_sd):
    model_sd = model.state_dict()
    matched = 0
    mismatched = 0
    missing_in_ckpt = 0
    
    for k, v in model_sd.items():
        if k in ckpt_sd:
            if v.shape == ckpt_sd[k].shape:
                matched += 1
            else:
                print(f"Shape mismatch: {k} (Model: {v.shape}, Ckpt: {ckpt_sd[k].shape})")
                mismatched += 1
        else:
            # print(f"Missing in ckpt: {k}")
            missing_in_ckpt += 1
            
    print(f"Match Stats: Matched={matched}, Mismatched={mismatched}, MissingInCkpt={missing_in_ckpt}")

def main():
    try:
        config_file = "configs/nights/2.yaml"
        ckpt_path = "/workspace/FATBETA/training/gs/r20_c10_gs_path1_sen/r20_c10_gs_path1_sen_checkpoint.pth.tar"
        device = torch.device('cuda:0')
        
        print(f"Using config: {config_file}")
        print(f"Using ckpt: {ckpt_path}")

        # 1. Load Config
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0], config_file]
        config = get_config(default_file=config_file)
        sys.argv = original_argv
        print("Config loaded.")
        
        # 2. Create Model
        print("Creating model...")
        model = create_model(config.arch, dataset=config.dataloader.dataset)
        print("Model created.")
        
        print("Replacing modules for quantization...")
        modules_to_replace = find_modules_to_quantize(model, config)
        replace_module_by_names(model, modules_to_replace)
        model = model.to(device)
        print("Modules replaced and model moved to device.")
        
        # 3. Load Data
        print("Initializing dataloader...")
        train_loader, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
        print("Dataloader initialized.")
        
        # 3.5 Warmup
        print("Warm-up forward pass...")
        model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(test_loader))
            model(inputs.to(device))
        print("Warm-up completed.")
        
        # Load checkpoint raw to inspect
        print("Loading checkpoint for inspection...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("Checkpoint loaded into CPU.")
        
        # 4. EMA Test
        print("\n--- Testing EMA ---")
        load_checkpoint(model, ckpt_path, model_device=device, use_ema=True)
        print("EMA weights loaded.")
        check_sd_match(model, checkpoint['state_dict_ema'])
        
        # 5. Set bit-width to 8
        bits = 8
        q_layers, _ = get_quantized_layers(model)
        set_bit_width(model, [bits]*len(q_layers), [bits]*len(q_layers))
        switch_bit_width(model, quan_scheduler=config.quan, wbit=bits, abits=bits)
        print(f"Bit-width set to {bits}.")
        
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy with EMA: {test_acc:.2f}%")
        
        # Evaluate a subset of training data for speed
        print("Evaluating training subset for EMA...")
        train_subset_loader = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=config.dataloader.batch_size, 
            sampler=torch.utils.data.SubsetRandomSampler(range(min(5000, len(train_loader.dataset)))),
            num_workers=0 # Use 0 workers for debugging stability
        )
        train_acc = evaluate_model(model, train_subset_loader, device)
        print(f"Train Accuracy (Subset) with EMA: {train_acc:.2f}%")
        
        # 6. Standard Test
        print("\n--- Testing Standard ---")
        load_checkpoint(model, ckpt_path, model_device=device, use_ema=False)
        print("Standard weights loaded.")
        check_sd_match(model, checkpoint['state_dict'])
        
        # Re-apply bit-width
        q_layers_std, _ = get_quantized_layers(model)
        set_bit_width(model, [bits]*len(q_layers_std), [bits]*len(q_layers_std))
        switch_bit_width(model, quan_scheduler=config.quan, wbit=bits, abits=bits)
        print(f"Bit-width reset to {bits} for standard weights.")
        
        # --- DEBUG: Inspect Model State ---
        print("\n[DEBUG] Inspecting Standard Model State:")
        layer = model.layer1[0]
        print(f"Layer1.0.conv1.weight mean: {layer.conv1.weight.mean().item():.4f}")
        print(f"Layer1.0.conv1 bits: {layer.conv1.bits}")
        print(f"Layer1.0.conv1 scale: {layer.conv1.quan_w_fn.s}")
        
        bn = layer.bn1
        print(f"Layer1.0.bn1 type: {type(bn)}")
        if hasattr(bn, '_active_bn'):
            print(f"Layer1.0.bn1._active_bn type: {type(bn._active_bn)}")
            print(f"Layer1.0.bn1._active_bn.running_mean mean: {bn._active_bn.running_mean.mean().item():.4f}")
            print(f"Layer1.0.bn1._active_bn.running_var mean: {bn._active_bn.running_var.mean().item():.4f}")
            
            # Check against bn_list[0][0]
            if hasattr(bn, 'bn_list'):
                bn00 = bn.bn_list[0][0]
                print(f"Layer1.0.bn1.bn_list[0][0].running_mean mean: {bn00.running_mean.mean().item():.4f}")
                print(f"Layer1.0.bn1.bn_list[0][0].running_var mean: {bn00.running_var.mean().item():.4f}")
                
                is_same = (bn._active_bn is bn00)
                print(f"Is _active_bn same object as bn_list[0][0]? {is_same}")
        # ----------------------------------
        
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy without EMA: {test_acc:.2f}%")
        
        print("Evaluating training subset for Standard...")
        train_acc = evaluate_model(model, train_subset_loader, device)
        print(f"Train Accuracy (Subset) without EMA: {train_acc:.2f}%")

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
