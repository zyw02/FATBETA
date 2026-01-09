import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quan.func import QuanConv2d, QuanLinear, SwithableBatchNorm
from quan.quantizer.lsq import compute_thd
from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.config import get_config
from util.data_loader import init_dataloader

def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total if total > 0 else 0.0

def bfa_attack(model, test_loader, device, criterion, target_acc=11.0, n_candidates=100):
    """
    完整的 PBS (Progressive Bit Search) 实现 (ICCV'19)
    1. 使用梯度敏感度初步筛选全局 Top-N 候选池
    2. 对候选池中的位进行真实翻转并评估 Loss，选出最优 Winner
    """
    model.eval()
    quantized_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            quantized_layers.append((name, module))
    
    flipped_bits = set()
    # 使用 256 的 Batch Size 减少梯度噪声
    search_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=256, shuffle=True)
    search_iter = iter(search_loader)

    flip_idx = 0
    while True:
        flip_idx += 1
        try:
            inputs, targets = next(search_iter)
        except StopIteration:
            search_iter = iter(search_loader)
            inputs, targets = next(search_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        # --- Step 1: 获取当前状态下的权重梯度 ---
        model.zero_grad()
        for _, m in quantized_layers: 
            m.weight.requires_grad = True
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # --- Step 2: 梯度敏感度预筛选候选位 (Taylor Expansion) ---
        candidate_pool = []
        for name, module in quantized_layers:
            w = module.weight
            if w.grad is None: continue
            grad = w.grad.data
            
            k = module.bits[0] if (hasattr(module, 'bits') and module.bits) else 8
            thd_neg, thd_pos = compute_thd(module.quan_w_fn, k)
            s = module.quan_w_fn.get_scale(k, detach=True)
            
            code = torch.round(w.data / s).clamp(thd_neg, thd_pos)
            code_unsigned = (code - thd_neg).to(torch.int64)
            
            for bit_i in range(k):
                bit_val = (code_unsigned >> bit_i) & 1
                delta_w = (1 - 2 * bit_val) * (2 ** bit_i) * s
                
                # 预估 Loss 变化量: Delta_L = grad * delta_w
                sens_all = grad * delta_w
                
                # 每层提取最敏感的 Top-20 候选
                v_top, i_top = torch.topk(sens_all.view(-1), min(20, sens_all.numel()))
                
                for v, idx in zip(v_top, i_top):
                    idx_val = idx.item()
                    if (name, idx_val, bit_i) not in flipped_bits:
                        candidate_pool.append({
                            'name': name, 'module': module, 'idx': idx_val, 
                            'bit_i': bit_i, 'delta_w': delta_w.view(-1)[idx_val].item(),
                            'sens': v.item()
                        })
        
        # 全局筛选前 n_candidates 个最强候选 (基于梯度预估)
        candidate_pool = sorted(candidate_pool, key=lambda x: x['sens'], reverse=True)[:n_candidates]

        if not candidate_pool:
            print("No more candidates."); break

        # --- Step 3: 真实 Loss 验证 (找出真正能让 Loss 最大的 Winner) ---
        best_winner = None
        max_loss = -1e9
        
        with torch.no_grad():
            for cand in candidate_pool:
                # 尝试翻转
                cand['module'].weight.data.view(-1)[cand['idx']] += cand['delta_w']
                # 计算在该 Batch 上的真实 Loss
                try_loss = criterion(model(inputs), targets).item()
                if try_loss > max_loss:
                    max_loss = try_loss
                    best_winner = cand
                # 还原权重
                cand['module'].weight.data.view(-1)[cand['idx']] -= cand['delta_w']

        # --- Step 4: 执行永久翻转 ---
        if best_winner:
            best_winner['module'].weight.data.view(-1)[best_winner['idx']] += best_winner['delta_w']
            flipped_bits.add((best_winner['name'], best_winner['idx'], best_winner['bit_i']))
            
            # 评估全量测试集 Acc
            acc = evaluate_model(model, test_loader, device)
            print(f"Flip {flip_idx:03d}: Layer {best_winner['name']:20s}, Bit {best_winner['bit_i']}, Acc {acc:6.2f}%, Batch_Loss {max_loss:.4f}")
            
            if acc <= target_acc:
                print(f"\n[Success] Accuracy reached random guess level ({acc:.2f}%) after {flip_idx} flips.")
                break
            if flip_idx >= 1000: break
        else:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--bits', type=int, default=8)
    bfa_args, remaining_args = parser.parse_known_args()
    
    sys.argv = [sys.argv[0], bfa_args.config] + remaining_args
    config = get_config(default_file=bfa_args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(config.arch, dataset=config.dataloader.dataset).to(device)
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    checkpoint = torch.load(bfa_args.ckpt, map_location=device)
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(filtered_dict, strict=False)
    
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # 初始化位宽设置
    next_bn_bits = None
    for name, module in model.named_modules():
        if isinstance(module, (QuanConv2d, QuanLinear)):
            if not (hasattr(config.quan, 'excepts') and name in config.quan.excepts):
                b_list = module.quan_w_fn.bit_list
                t_bit = bfa_args.bits if bfa_args.bits in b_list else max(b_list)
                module.bits = (t_bit, t_bit)
                if hasattr(module, 'current_bit_cands_w'):
                    module.current_bit_cands_w.data = torch.tensor([t_bit], device=device).to(module.current_bit_cands_w.dtype)
                if hasattr(module, 'current_bit_cands_a'):
                    module.current_bit_cands_a.data = torch.tensor([t_bit], device=device).to(module.current_bit_cands_a.dtype)
                if hasattr(module.quan_w_fn, 'bits'):
                    module.quan_w_fn.bits = t_bit
            else:
                module.bits = (8, 8)
            next_bn_bits = module.bits
        elif isinstance(module, SwithableBatchNorm):
            if next_bn_bits is not None:
                module.switch_bn(next_bn_bits)

    print(f"Initial Accuracy: {evaluate_model(model, test_loader, device):.2f}%")
    bfa_attack(model, test_loader, device, nn.CrossEntropyLoss(), target_acc=11.0)

if __name__ == '__main__':
    main()
