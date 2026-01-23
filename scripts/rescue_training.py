import os
import subprocess
import yaml
import time
from pathlib import Path

# 基础配置
template_cfg_path = "configs/training/r20.yaml"
output_dir = "training_rescue"  # 分开存放，避免污染 grid_search

# 待跑的“救援”配置列表
# 重点验证：200 Epochs + High LR (0.6/0.8) + EMA Fix
experiments = [
    # 1. 之前的【鲁棒性冠军】复刻 (BER=0.02, MSB=0.02)
    {
        "name": "rescue_champion_lr0.6",
        "lr": 0.6,
        "ema_decay": 0.998,
        "bfat": {
            "ber": 0.02,
            "ber_base_restricted": 0.005,
            "ber_msb": 0.02
        }
    },
    # 2. 之前的【Clean Acc 冠军】复刻 (BER=0.005, MSB=0.001) -> 冲刺 93.8%
    {
        "name": "rescue_clean_king_lr0.6",
        "lr": 0.6,
        "ema_decay": 0.998,
        "bfat": {
            "ber": 0.005,
            "ber_base_restricted": 0.001,
            "ber_msb": 0.001
        }
    },
    # 3. 更激进的 LR=0.8 测试 (理论值)
    {
        "name": "rescue_champion_lr0.8",
        "lr": 0.8,
        "ema_decay": 0.998,
        "bfat": {
            "ber": 0.02,
            "ber_base_restricted": 0.005,
            "ber_msb": 0.02
        }
    },
    # 4. Clean Acc 冠军的激进版 (LR=0.8) -> 冲击极限 Clean Acc
    {
        "name": "rescue_clean_king_lr0.8",
        "lr": 0.8,
        "ema_decay": 0.998,
        "bfat": {
            "ber": 0.005,
            "ber_base_restricted": 0.001,
            "ber_msb": 0.001
        }
    },
    # 5. 全面扩倍实验：LR, Min_LR, Warmup_LR 全部 x8 (对比单卡基础 1e-5 -> 8e-5/1e-4)
    {
        "name": "rescue_fully_scaled_lr0.6",
        "lr": 0.6,
        "min_lr": 0.00006,
        "warmup_lr": 0.00006,
        "ema_decay": 0.998,
        "bfat": {
            "ber": 0.02,
            "ber_base_restricted": 0.005,
            "ber_msb": 0.02
        }
    }
]

def run_rescue():
    # 1. 读取模板
    with open(template_cfg_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # 强制修正基础参数 (Double Check)
    base_cfg['epochs'] = 200     # 关键修正
    base_cfg['output_dir'] = output_dir

    tmp_dir = Path("scripts/rescue_configs")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚑 Starting Rescue Training: {len(experiments)} missions queued.")
    print(f"   Common Settings: Epochs=200, Output={output_dir}, WorldSize=8")

    for i, exp_cfg in enumerate(experiments):
        exp_name = exp_cfg['name']
        print(f"\n--- Mission {i+1}/{len(experiments)}: {exp_name} ---")

        # 检查是否已存在
        log_file = Path(output_dir) / exp_name / f"{exp_name}.log"
        if log_file.exists():
            print(f"⏩ Experiment {exp_name} already exists. Skipping...")
            continue

        # 2. 覆盖参数
        current_cfg = base_cfg.copy()
        current_cfg['name'] = exp_name
        current_cfg['lr'] = exp_cfg['lr']
        current_cfg['ema_decay'] = exp_cfg['ema_decay']
        current_cfg['sync_bn'] = False # 强制关闭 SyncBN
        if 'min_lr' in exp_cfg: current_cfg['min_lr'] = exp_cfg['min_lr']
        if 'warmup_lr' in exp_cfg: current_cfg['warmup_lr'] = exp_cfg['warmup_lr']
        
        # Deep update BFAT
        current_cfg['bfat'].update(exp_cfg['bfat'])

        # 3. 保存临时配置
        tmp_cfg_path = tmp_dir / f"{exp_name}.yaml"
        with open(tmp_cfg_path, 'w') as f:
            yaml.dump(current_cfg, f)

        # 4. 运行训练
        cmd = [
            "torchrun", 
            "--nproc_per_node", "8", 
            "--master_port", str(29600 + i), # 防止端口冲突
            "main_nude.py",
            str(tmp_cfg_path)
        ]

        try:
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = (time.time() - start_time) / 3600
            print(f"✅ Mission {exp_name} COMPLETE in {duration:.2f} hours.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Mission {exp_name} FAILED (Code {e.returncode}). Moving to next...")

if __name__ == "__main__":
    run_rescue()
