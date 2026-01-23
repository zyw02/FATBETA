import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 用兼容性好的字体
plt.rcParams['axes.unicode_minus'] = False

def parse_fast_sweep_log(log_path):
    """解析 fast_sweep_results.log 文件"""
    data_lines = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    # 找到表格的开始
    start_idx = -1
    for i, line in enumerate(lines):
        if "BER" in line and "All Bits" in line:
            start_idx = i + 2  # 跳过表头和分割线
            break
            
    if start_idx == -1:
        return None

    results = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
            
        # 移除 ANSI 颜色代码
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        parts = line.split('|')
        if len(parts) >= 4:
            try:
                ber = float(parts[0].strip())
                all_bits = float(parts[1].strip())
                skip_msb = float(parts[2].strip())
                only_msb = float(parts[3].strip())
                results.append({
                    'BER': ber,
                    'All Bits': all_bits,
                    'Skip MSB': skip_msb,
                    'Only MSB': only_msb
                })
            except ValueError:
                continue
                
    return results

def main():
    training_dir = Path("training")
    all_experiments = []
    
    # 1. 遍历所有实验目录
    print(f"Scanning {training_dir} for results...")
    for exp_dir in training_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("r56_b"):
            continue
            
        # 解析实验名称获取参数
        # 格式: r56_b0.0050_base0.0050_msb0.0100
        # 注意: 之前的 grid_search.py 里面的 replace('0.', '.') 可能把 b0.0050 变成了 b.0050
        name = exp_dir.name
        
        # 尝试提取参数
        try:
            # 使用简单的字符串分割提取
            parts = name.split('_')
            # r56, b.0050, base.0050, msb.0100
            
            # Helper to restore "0." if needed
            def norm_val(s):
                if s.startswith('.'):
                    return float("0" + s)
                return float(s)

            train_ber = norm_val(parts[1].replace('b', ''))
            train_base = norm_val(parts[2].replace('base', ''))
            train_msb = norm_val(parts[3].replace('msb', ''))
            
        except Exception as e:
            print(f"Skipping {name}: could not parse name ({e})")
            continue
            
        log_path = exp_dir / "fast_sweep_results.log"
        if not log_path.exists():
            continue
            
        sweep_data = parse_fast_sweep_log(log_path)
        if not sweep_data:
            print(f"Skipping {name}: empty or invalid log")
            continue
            
        # 将每个 BER 点的数据铺平
        for row in sweep_data:
            all_experiments.append({
                'Experiment': name,
                'Train_BER': train_ber,
                'Train_Base': train_base,
                'Train_MSB': train_msb,
                'Eval_BER': row['BER'],
                'Acc_AllBits': row['All Bits'],
                'Acc_SkipMSB': row['Skip MSB'],
                'Acc_OnlyMSB': row['Only MSB']
            })

    if not all_experiments:
        print("No valid data found.")
        return

    df = pd.DataFrame(all_experiments)
    
    # 2. 导出完整数据
    output_csv = "grid_search_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")
    
    # 3. 计算综合指标 (Score) 来寻找"最佳"配置
    # 定义 Best: 在高 BER (>= 0.01) 下 'All Bits' 的平均准确率最高
    high_ber_df = df[df['Eval_BER'] >= 0.01]
    
    score_df = high_ber_df.groupby(['Experiment', 'Train_BER', 'Train_Base', 'Train_MSB'])['Acc_AllBits'].mean().reset_index()
    score_df.rename(columns={'Acc_AllBits': 'Avg_HighBER_Acc'}, inplace=True)
    score_df = score_df.sort_values(by='Avg_HighBER_Acc', ascending=False)
    
    best_config = score_df.iloc[0]
    print("\n" + "="*50)
    print(f"🌟 Best Configuration (based on Avg Acc @ BER>=0.01):")
    print(f"Experiment: {best_config['Experiment']}")
    print(f"Params: BER={best_config['Train_BER']}, Base={best_config['Train_Base']}, MSB={best_config['Train_MSB']}")
    print(f"Score (Avg Acc @ High BER): {best_config['Avg_HighBER_Acc']:.2f}%")
    print("="*50 + "\n")
    
    print("Top 5 Configurations:")
    print(score_df.head(5).to_string(index=False))
    
    # 4. 可视化
    print("\nGenerating plots...")
    
    # 4.1 核心对比图：选出 Top 3 和 Baseline (如果存在)
    top_exps = score_df.head(3)['Experiment'].tolist()
    plot_df = df[df['Experiment'].isin(top_exps)]
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=plot_df, x='Eval_BER', y='Acc_AllBits', hue='Experiment', style='Experiment', markers=True, dashes=False)
    plt.xscale('log')
    plt.title('Fault Tolerance Comparison (Top 3 Configs)')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Bit Error Rate (BER)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.ylim(10, 95) # Zoom in on reasonable range
    
    plot_path = "grid_search_top3_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # 4.2 热力图 (如果参数维度合适)
    # 固定一个维度，看另外两个维度对 Score 的影响
    # 例如固定 Train_MSB 为其中的众数，或者最佳配置的值
    target_msb = best_config['Train_MSB']
    heatmap_data = score_df[score_df['Train_MSB'] == target_msb]
    
    if not heatmap_data.empty and len(heatmap_data['Train_BER'].unique()) > 1:
        plt.figure(figsize=(10, 8))
        pivot_table = heatmap_data.pivot(index='Train_BER', columns='Train_Base', values='Avg_HighBER_Acc')
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title(f'Performance Heatmap (Fixed MSB={target_msb})')
        plt.tight_layout()
        heatmap_path = "grid_search_heatmap.png"
        plt.savefig(heatmap_path)
        print(f"Heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    main()
