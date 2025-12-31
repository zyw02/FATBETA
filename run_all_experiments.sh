#!/bin/bash

# 自动化实验运行脚本
# 按顺序运行多个 BFAT 实验配置

# 1. 运行当前的综合实验（假设目前已经在运行中，或者你想再跑一遍确保万无一失）
# 如果你目前的实验已经在运行，你可以注释掉这一行，或者等待它结束

python main_nude.py configs/training/r18_c10_nude_standard_bfat_allbits_1.yaml
# 2. 运行 CAGrad 投影模式实验

python main_nude.py configs/training/r18_c10_nude_standard_bfat_allbits_2.yaml




echo "All experiments completed!"