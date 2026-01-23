
import os
import subprocess
import time
from pathlib import Path

models = [
    "rescue_champion_lr0.6",
    "rescue_clean_king_lr0.6",
    "rescue_champion_lr0.8",
    "rescue_clean_king_lr0.8",
    "rescue_fully_scaled_lr0.6"
]

def main():
    base_dir = Path("training_rescue")
    config_dir = Path("scripts/rescue_configs")
    
    print(f"🚀 Starting BER Sweep for {len(models)} models...")

    for model_name in models:
        print(f"\n--- {model_name} ---")
        
        checkpoint_path = base_dir / model_name / f"{model_name}_checkpoint.pth.tar"
        config_path = config_dir / f"{model_name}.yaml"
        log_path = base_dir / model_name / "sweep_results.log"
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            continue
            
        if not config_path.exists():
            print(f"❌ Config not found: {config_path}")
            continue
            
        # Command from grid_search.py (Using the same BER list and --no_ema)
        cmd = [
            "python", "scripts/fast_sweep.py",
            "--ckpt", str(checkpoint_path),
            "--config", str(config_path),
            "--bers", "0.0,1e-6,1e-5,1e-4,1e-3,1e-2,0.02,0.03,0.04,0.05",
            "--no_ema" 
        ]
        
        # Clean env to prevent DDP interference
        env = os.environ.copy()
        for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            env.pop(var, None)
            
        print(f"   Writing results to: {log_path}")
        with open(log_path, "w") as f:
            f.write(f"Experiment: {model_name}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write("-" * 50 + "\n")
            f.flush()
            
            try:
                start_time = time.time()
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
                duration = time.time() - start_time
                print(f"✅ Finished in {duration:.1f}s")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed with exit code {e.returncode}")

    print("\n🎉 All sweeps completed.")

if __name__ == "__main__":
    main()
