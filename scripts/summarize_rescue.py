
import re
from pathlib import Path
import pandas as pd

def parse_sweep_log(log_path):
    results = {}
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        if "BER        |  All Bits" in line:
            start_parsing = True
            continue
        if "----------------" in line:
            continue
            
        if start_parsing:
            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    ber = float(parts[0].strip())
                    acc_all = float(parts[1].strip())  # Remove color codes if needed, but python float() handles strict numbers. 
                    # Actually valid data lines have color codes. Regex is safer.
                    pass
                except:
                    continue
                    
    # Re-parsing with regex to handle ANSI codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    results = []
    for line in lines:
        # Strip ANSI codes first
        clean_line = ansi_escape.sub('', line)
        
        # Match lines starting with a number (integer or float)
        # 0.0        | 93.19        | 93.19        | 93.19   
        parts = clean_line.split('|')
        if len(parts) == 4:
            try:
                ber_str = parts[0].strip()
                if not ber_str: continue
                
                # Check if it looks like a number
                if not re.match(r'^\d', ber_str): continue
                
                ber = float(ber_str)
                acc_all = float(parts[1].strip())
                acc_skip = float(parts[2].strip())
                acc_only = float(parts[3].strip())
                
                results.append({
                    "BER": ber,
                    "All Bits": acc_all,
                    "Skip MSB": acc_skip,
                    "Only MSB": acc_only
                })
            except ValueError:
                continue
    return results

def main():
    root = Path("training_rescue")
    models = sorted([d.name for d in root.iterdir() if d.is_dir()])
    
    print(f"{'Model':<30} | {'Clean Acc':<10} | {'BER=0.01':<10} | {'BER=0.02':<10} | {'BER=0.05':<10}")
    print("-" * 85)
    
    for model in models:
        log_path = root / model / "sweep_results.log"
        if not log_path.exists():
            continue
            
        data = parse_sweep_log(log_path)
        if not data:
            continue
            
        # Extract key metrics
        clean_acc = next((d['All Bits'] for d in data if d['BER'] == 0.0), 0.0)
        ber01_acc = next((d['All Bits'] for d in data if d['BER'] == 0.01), 0.0)
        ber02_acc = next((d['All Bits'] for d in data if d['BER'] == 0.02), 0.0)
        ber05_acc = next((d['All Bits'] for d in data if d['BER'] == 0.05), 0.0)
        
        print(f"{model:<30} | {clean_acc:<10.2f} | {ber01_acc:<10.2f} | {ber02_acc:<10.2f} | {ber05_acc:<10.2f}")

if __name__ == "__main__":
    main()
