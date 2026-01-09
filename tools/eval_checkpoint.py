import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader

def evaluate(model, dataloader, device):
    model.eval()
    correct_t1 = 0
    correct_t5 = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Top-1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_t1 += predicted.eq(targets).sum().item()
            
            # Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_t5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            if (i + 1) % 20 == 0:
                print(f"Batch {i+1}/{len(dataloader)}: Acc1 {100. * correct_t1 / total:.2f}%")

    acc1 = 100. * correct_t1 / total
    acc5 = 100. * correct_t5 / total
    return acc1, acc5

def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint accuracy')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/training/train_mc_cifar10_fp32.yaml', help='Config file')
    parser.add_argument('--arch', type=str, default='mobilenetv2_mc', help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--use-ema', action='store_true', help='Use EMA weights if available')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config
    # Spoof sys.argv for get_config
    import sys
    orig_argv = sys.argv
    sys.argv = [sys.argv[0], args.config]
    config = get_config(default_file=args.config)
    sys.argv = orig_argv

    # 2. Create Model
    # Note: Use pre_trained=False because we are loading a specific checkpoint
    model = create_model(args.arch, dataset=args.dataset, pre_trained=False)
    model = model.to(device)

    # 3. Load Checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    # load_checkpoint already handles EMA and filtering
    load_checkpoint(model, args.ckpt, model_device=device, use_ema=args.use_ema)

    # 4. Init Dataloader
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, args.arch)

    # 5. Run Evaluation
    acc1, acc5 = evaluate(model, test_loader, device)
    
    print("\n" + "="*30)
    print(f"Final Results for: {Path(args.ckpt).name}")
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print("="*30)

if __name__ == '__main__':
    main()





