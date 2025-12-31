#!/usr/bin/env python3
"""
诊断Corrector训练/测试gap的根本原因
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from util.output_corrector import OutputCorrector, compute_probability_features, compute_energy_features

def analyze_corrector_behavior():
    """分析corrector在训练和测试时的行为差异"""
    
    print("=" * 80)
    print("Corrector训练/测试Gap诊断分析")
    print("=" * 80)
    
    print("\n【问题1：Baseline分布差异】")
    print("-" * 80)
    print("训练时：")
    print("  - 每个batch都用normal样本更新baseline（EMA）")
    print("  - baseline基于训练集的normal样本分布")
    print("  - 训练集normal样本的logits/activations分布")
    print("\n测试时：")
    print("  - 用测试集的500个normal样本初始化baseline（一次性）")
    print("  - baseline基于测试集的normal样本分布")
    print("  - 测试集normal样本的logits/activations分布可能不同")
    print("\n⚠️  问题：训练集和测试集的normal样本分布可能不同，导致WM-ID计算不准确")
    
    print("\n【问题2：WM-ID门控行为不一致】")
    print("-" * 80)
    print("训练时（从日志看到）：")
    print("  - WM-ID_e: 12.22, scale: 1.0000 (门控全开)")
    print("  - WM-ID_e: 1.73, scale: 0.3693 (门控部分开启)")
    print("  - tau=2.0, beta=2.0")
    print("  - scale = sigmoid(beta * (WM-ID_e - tau))")
    print("\n测试时：")
    print("  - WM-ID_e可能更大或更小（取决于测试集分布）")
    print("  - 如果WM-ID_e < tau，scale会很小，修正量被抑制")
    print("  - 如果WM-ID_e >> tau，scale接近1.0，但修正方向可能不对")
    print("\n⚠️  问题：训练时和测试时的WM-ID分布可能不同，导致门控行为不一致")
    
    print("\n【问题3：自监督目标的不一致性】")
    print("-" * 80)
    print("V8的训练目标：")
    print("  - 学习预测：delta_logits = normal_logits - faulted_logits")
    print("  - 训练时：normal和faulted的差异可能较小（模型还在学习）")
    print("  - 测试时：normal和faulted的差异可能很大（模型已收敛）")
    print("\n训练时的场景：")
    print("  - 模型还在学习，normal logits可能不够准确")
    print("  - faulted logits和normal logits的差异可能较小")
    print("  - corrector学习到的修正模式可能不够'激进'")
    print("\n测试时的场景：")
    print("  - 模型已收敛，normal logits很准确")
    print("  - faulted logits和normal logits的差异可能很大")
    print("  - corrector学习到的修正模式可能不够'激进'，无法应对大差异")
    print("\n⚠️  问题：训练时和测试时的normal-faulted差异分布不同，导致corrector修正不足")
    
    print("\n【问题4：特征提取的分布偏移】")
    print("-" * 80)
    print("训练时：")
    print("  - logits统计特征（top-k, gap, std, entropy等）基于训练集分布")
    print("  - 流形特征（z_e, z_p）基于训练集的baseline计算")
    print("\n测试时：")
    print("  - logits统计特征基于测试集分布（可能不同）")
    print("  - 流形特征基于测试集的baseline计算（可能不同）")
    print("\n⚠️  问题：特征分布偏移导致corrector的预测不准确")
    
    print("\n【问题5：过拟合到训练集的故障模式】")
    print("-" * 80)
    print("训练时：")
    print("  - 使用10个不同的seed进行故障注入")
    print("  - 但每个epoch的故障模式是固定的（seed固定）")
    print("  - corrector可能学习到训练集特有的故障模式")
    print("\n测试时：")
    print("  - 使用不同的seed进行故障注入")
    print("  - 故障模式可能与训练时不同")
    print("\n⚠️  问题：corrector过拟合到训练集的故障模式，无法泛化到新的故障模式")
    
    print("\n【问题6：修正方向的学习问题】")
    print("-" * 80)
    print("V8的设计：")
    print("  - 学习预测delta_logits = normal_logits - faulted_logits")
    print("  - 这是一个回归任务，需要准确预测每个维度的修正量")
    print("\n训练时：")
    print("  - 有target label，可以计算CE loss")
    print("  - 但主要目标是MSE loss（预测delta_logits）")
    print("  - 如果MSE loss小，但CE loss大，说明修正方向不对")
    print("\n测试时：")
    print("  - 没有target label，只能依赖预测的delta_logits")
    print("  - 如果修正方向不对，即使修正量很大，也不会改善准确率")
    print("\n⚠️  问题：corrector可能学习到了错误的修正方向（之前V7的问题）")
    
    print("\n" + "=" * 80)
    print("【根本原因总结】")
    print("=" * 80)
    print("1. **分布偏移**：训练集和测试集的normal样本分布不同，导致baseline和WM-ID计算不准确")
    print("2. **门控不一致**：训练时和测试时的WM-ID分布不同，导致门控行为不一致")
    print("3. **目标不一致**：训练时和测试时的normal-faulted差异分布不同，导致修正不足")
    print("4. **特征偏移**：训练时和测试时的特征分布不同，导致预测不准确")
    print("5. **过拟合**：corrector过拟合到训练集的故障模式，无法泛化")
    print("6. **修正方向**：corrector可能学习到了错误的修正方向")
    
    print("\n" + "=" * 80)
    print("【可能的解决方案】")
    print("=" * 80)
    print("1. **统一baseline**：训练时和测试时使用相同的baseline校准方法")
    print("2. **自适应门控**：根据WM-ID的分布动态调整tau和beta")
    print("3. **更强的正则化**：增加sparse loss和stability loss的权重，防止过拟合")
    print("4. **数据增强**：增加更多样化的故障模式（更多seed，更多BER）")
    print("5. **修正方向约束**：增加方向损失，确保修正方向正确")
    print("6. **测试时校准**：在测试时用少量样本重新校准baseline（已实现，但可能不够）")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_corrector_behavior()

