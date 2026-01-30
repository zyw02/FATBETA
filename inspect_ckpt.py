import torch
import sys

ckpt_path = "/workspace/FATBETA/training/gs/r20_c10_gs_path1_sen/r20_c10_gs_path1_sen_checkpoint.pth.tar"
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("Keys in checkpoint:", checkpoint.keys())
print("\nSome state_dict keys and shapes:")
sd = checkpoint['state_dict']
count = 0
for k, v in sd.items():
    if 'quan_w_fn.s' in k or 'quan_a_fn.s' in k or 'init_state' in k:
        print(f"{k}: {v.shape} | {v.flatten()[:5]}")
        count += 1
    if count > 10:
        break

ema_sd = checkpoint.get('state_dict_ema', None)
if ema_sd:
    print("\nSome state_dict_ema keys and shapes:")
    count = 0
    for k, v in ema_sd.items():
        if 'quan_w_fn.s' in k or 'quan_a_fn.s' in k:
            print(f"{k}: {v.shape}")
            count += 1
        if count > 10:
            break
else:
    print("\nNo state_dict_ema found.")
