import torch

ckpt_path = "/workspace/FATBETA/training/gs/r20_c10_gs_path1_sen/r20_c10_gs_path1_sen_checkpoint.pth.tar"
checkpoint = torch.load(ckpt_path, map_location='cpu')

sd = checkpoint['state_dict']
ema_sd = checkpoint['state_dict_ema']

print(f"{'Key':<40} | {'SD Mean':<10} | {'EMA Mean':<10} | {'Diff Norm':<10}")
print("-" * 80)

diffs = []
for k in sd.keys():
    if k in ema_sd:
        if 'weight' not in k or 'bn' in k:
            continue
        v1 = sd[k].float()
        v2 = ema_sd[k].float()
        diff_norm = torch.norm(v1 - v2).item()
        mean_diff = (v1.mean() - v2.mean()).item()
        diffs.append({
            'key': k,
            'std_mean': v1.mean().item(),
            'ema_mean': v2.mean().item(),
            'diff_norm': diff_norm,
            'mean_diff': mean_diff
        })

# Sort by diff_norm descending
diffs.sort(key=lambda x: x['diff_norm'], reverse=True)

print(f"{'Key':<60} | {'SD Mean':<10} | {'EMA Mean':<10} | {'Diff Norm':<10}")
print("-" * 110)
for d in diffs[:100]:
    print(f"{d['key']:<60} | {d['std_mean']:<10.4f} | {d['ema_mean']:<10.4f} | {d['diff_norm']:<10.4f}")
