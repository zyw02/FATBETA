import os
import subprocess
import yaml
import copy
from pathlib import Path
import time

def run_experiment(base_cfg_path, ber, ber_base, ber_msb):
    # 1. 读取基础配置
    with open(base_cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 2. 修改参数
    cfg['bfat']['ber'] = ber
    cfg['bfat']['ber_base_restricted'] = ber_base
    cfg['bfat']['ber_msb'] = ber_msb
    
    # 3. 构造唯一的实验名称
    # 格式：resnet20_b{ber}_base{ber_base}_msb{ber_msb}
    exp_name = f"r56_b{ber:.4f}_base{ber_base:.4f}_msb{ber_msb:.4f}".replace('0.', '.')
    cfg['name'] = exp_name
    
    # 4. 创建临时配置文件
    tmp_config_dir = Path("scripts/texp/tmp_configs")
    tmp_config_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = tmp_config_dir / f"{exp_name}.yaml"
    
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(cfg, f)
    
    # 5. 检查是否已经跑过 (通过检查日志文件是否存在)
    log_file = Path(cfg.get('output_dir', 'training')) / exp_name / f"{exp_name}.log"
    skip_training = False
    if log_file.exists():
        print(f"⏩ [SKIP TRAINING] Experiment {exp_name} already exists. Skipping training, but will check sweep...")
        skip_training = True
    
    # 6. 执行训练命令
    # 注意：main_nude.py 内部已默认加载 template.yaml，此处只需传入临时的 override 配置
    # 修正 torchrun 参数拆分，并增加 master_port 避免端口冲突
    if not skip_training:
        cmd = [
            "torchrun", 
            "--nproc_per_node", "8", 
            "--master_port", "29505",
            "main_nude.py",
            str(tmp_cfg_path)
        ]
        
        print(f"\n🚀 [1/2 TRAINING] Running experiment: {exp_name}")
        print(f"   Config: ber={ber}, ber_base={ber_base}, ber_msb={ber_msb}")
        
        try:
            # 使用 subprocess.run 等待完成
            start_time = time.time()
            subprocess.run(cmd, check=True)
            duration = time.time() - start_time
            print(f"✅ [FINISHED] {exp_name} in {duration/3600:.2f} hours")
        except subprocess.CalledProcessError as e:
            print(f"❌ [ERROR] {exp_name} failed with exit code {e.returncode}")
            return # Training failed, so we stop here
        
    # --- NEW: 自动执行故障注入 Sweep 测试 ---
    # 1. 寻找训练好的 checkpoint
    # 根据 main_nude.py 的逻辑，它会保存在 output_dir/exp_name/ 目录下
    checkpoint_name = f"{cfg['name']}_checkpoint.pth.tar"
    checkpoint_path = (Path.cwd() / cfg.get('output_dir', 'training') / exp_name / checkpoint_name).resolve()
    
    if checkpoint_path.exists():
        print(f"🔍 [2/2 SWEEPING] Found checkpoint, starting fault injection sweep (No EMA)...")
        sweep_log_path = (Path.cwd() / cfg.get('output_dir', 'training') / exp_name / "fast_sweep_results.log").resolve()
        
        # 2. 构造 sweep 命令 (务必传入 --config 指向本次实验的临时配置)
        # Added --no_ema as requested
        sweep_cmd = [
            "python", "scripts/fast_sweep.py",
            "--ckpt", str(checkpoint_path),
            "--config", str(tmp_cfg_path),
            "--bers", "0.0,1e-6,1e-5,1e-4,1e-3,1e-2,0.02,0.03,0.04,0.05",
            "--no_ema"
        ]
        
        # 3. 清理环境变量，确保单进程脚本不会尝试启动 DDP 逻辑
        sweep_env = os.environ.copy()
        for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            sweep_env.pop(var, None)

        # 4. 执行 sweep 并将结果记录到专门的 log 文件
        with open(sweep_log_path, "w") as sweep_f:
            # 同时写入一些基础信息
            sweep_f.write(f"Experiment: {exp_name}\n")
            sweep_f.write(f"Checkpoint: {checkpoint_path}\n")
            sweep_f.write("-" * 50 + "\n")
            sweep_f.flush()
            
            # 执行脚本，传入清理后的环境
            try:
                subprocess.run(sweep_cmd, stdout=sweep_f, stderr=subprocess.STDOUT, check=True, env=sweep_env)
                print(f"📊 [SWEEP DONE] Results saved to: {sweep_log_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ [SWEEP ERROR] Sweep failed with exit code {e.returncode}")
    else:
        print(f"⚠️ [SWEEP SKIP] Checkpoint not found at {checkpoint_path}")

def main():
    base_config = "configs/training/r20.yaml"
    
    # 定义搜索空间 (你可以根据需要调整这里的列表)
    # 区间 [0.001, 0.02]
    # 如果做全网格搜索，点数不宜过多，否则耗时极长
    bers = [0.005, 0.01, 0.02]
    ber_bases = [0.001, 0.005, 0.01, 0.02]
    ber_msbs = [0.005, 0.01, 0.02]
    
    # 如果你希望这三个参数保持一致同步变化，可以这样：
    # for b in [0.001, 0.002, 0.005, 0.01, 0.015, 0.02]:
    #     run_experiment(base_config, b, b, b)
    
    # 如果是全网格搜索：
    total_runs = len(bers) * len(ber_bases) * len(ber_msbs)
    current_run = 0
    
    print(f"📊 Starting Grid Search: Total {total_runs} combinations")
    
    for b in bers:
        for bb in ber_bases:
            for bm in ber_msbs:
                current_run += 1
                print(f"\n--- Progress: {current_run}/{total_runs} ---")
                run_experiment(base_config, b, bb, bm)

if __name__ == "__main__":
    main()

