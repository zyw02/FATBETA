"""
改进的OLM编码器 - 遗传算法版本

使用遗传算法替代模拟退火，提高搜索效率

优点：
1. 多点并行搜索，探索效率高
2. 通过交叉操作，可以组合好的编码片段
3. 收敛速度通常比模拟退火快
4. 适合组合优化问题
"""

import torch
import torch.nn as nn
import math
import random
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from util.olm_encoder import compute_lrobust
from util.improved_olm_encoder_v2 import (
    compute_lrobust_improved_v2,
    compute_value_importance_by_distribution,
    compute_hamming_distance_weights,
    compute_local_consistency_penalty,
    _get_codes_with_hamming_dist,
    _comb
)


class GeneticAlgorithmOLM:
    """
    遗传算法优化OLM编码
    """
    
    def __init__(
        self,
        distribution: Dict[int, int],
        k: int,
        population_size: int = 50,
        max_generations: int = 1000,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 5,
        ber: float = 1e-2,
        consider_multi_bit: bool = True,
        max_hamming_dist: int = 3,
        use_value_importance: bool = True,
        use_local_consistency: bool = True,
        local_consistency_weight: float = 0.1
    ):
        self.distribution = distribution
        self.k = k
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.ber = ber
        self.consider_multi_bit = consider_multi_bit
        self.max_hamming_dist = max_hamming_dist
        self.use_value_importance = use_value_importance
        self.use_local_consistency = use_local_consistency
        self.local_consistency_weight = local_consistency_weight
        
        # 获取所有量化值
        self.values = sorted(distribution.keys())
        self.n_values = len(self.values)
        self.n_codes = 1 << k
        
        # 如果量化值数量超过编码数量，只使用频率最高的n_codes个
        if self.n_values > self.n_codes:
            sorted_values = sorted(self.values, key=lambda v: distribution[v], reverse=True)
            self.values = sorted_values[:self.n_codes]
            self.n_values = len(self.values)
    
    def create_individual(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        创建一个个体（一个完整的映射）
        
        Returns:
            (value_to_code, code_to_value) 映射
        """
        # 随机分配编码
        codes = list(range(self.n_codes))
        random.shuffle(codes)
        
        value_to_code = {}
        code_to_value = {}
        
        # 为每个量化值分配编码
        for i, value in enumerate(self.values):
            if i < len(codes):
                code = codes[i]
                value_to_code[value] = code
                code_to_value[code] = value
        
        # 填充剩余的编码（使用最近的值）
        for code in range(self.n_codes):
            if code not in code_to_value:
                # 找到最近的已分配值
                closest_value = min(self.values, key=lambda v: abs(v - (code - (1 << (self.k-1)))))
                code_to_value[code] = closest_value
        
        return value_to_code, code_to_value
    
    def evaluate_fitness(
        self,
        value_to_code: Dict[int, int],
        code_to_value: Dict[int, int]
    ) -> float:
        """
        评估个体的适应度（使用改进的LRobust）
        
        Returns:
            适应度值（越小越好）
        """
        return compute_lrobust_improved_v2(
            value_to_code, code_to_value, self.distribution, self.k,
            self.ber, self.consider_multi_bit, self.max_hamming_dist,
            self.use_value_importance, self.use_local_consistency,
            self.local_consistency_weight
        )
    
    def crossover(
        self,
        parent1: Tuple[Dict[int, int], Dict[int, int]],
        parent2: Tuple[Dict[int, int], Dict[int, int]]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        交叉操作：创建两个子代
        
        Args:
            parent1, parent2: 父代个体
        
        Returns:
            子代个体
        """
        value_to_code1, code_to_value1 = parent1
        value_to_code2, code_to_value2 = parent2
        
        # 单点交叉：随机选择一部分值，交换它们的编码
        crossover_point = random.randint(1, len(self.values) - 1)
        selected_values = random.sample(self.values, crossover_point)
        
        # 创建子代
        child_value_to_code = dict(value_to_code1)
        child_code_to_value = dict(code_to_value1)
        
        # 交换选中的值的编码
        for value in selected_values:
            if value in value_to_code1 and value in value_to_code2:
                code1 = value_to_code1[value]
                code2 = value_to_code2[value]
                
                # 交换编码
                if code1 in child_code_to_value and code2 in child_code_to_value:
                    # 先清除旧的映射
                    old_value1 = child_code_to_value[code1]
                    old_value2 = child_code_to_value[code2]
                    
                    # 更新映射
                    child_value_to_code[value] = code2
                    child_value_to_code[old_value2] = code1
                    child_code_to_value[code1] = old_value2
                    child_code_to_value[code2] = value
        
        return child_value_to_code, child_code_to_value
    
    def mutate(
        self,
        individual: Tuple[Dict[int, int], Dict[int, int]]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        变异操作：随机交换两个值的编码
        
        Args:
            individual: 个体
        
        Returns:
            变异后的个体
        """
        value_to_code, code_to_value = individual
        
        # 随机选择两个值，交换它们的编码
        if len(self.values) >= 2:
            v1, v2 = random.sample(self.values, 2)
            
            if v1 in value_to_code and v2 in value_to_code:
                code1 = value_to_code[v1]
                code2 = value_to_code[v2]
                
                # 交换编码
                new_value_to_code = dict(value_to_code)
                new_code_to_value = dict(code_to_value)
                
                new_value_to_code[v1] = code2
                new_value_to_code[v2] = code1
                new_code_to_value[code1] = v2
                new_code_to_value[code2] = v1
                
                return new_value_to_code, new_code_to_value
        
        return value_to_code, code_to_value
    
    def select_parents(
        self,
        population: List[Tuple[Dict[int, int], Dict[int, int]]],
        fitness: List[float]
    ) -> Tuple[Tuple[Dict[int, int], Dict[int, int]], Tuple[Dict[int, int], Dict[int, int]]]:
        """
        选择父代（锦标赛选择）
        
        Args:
            population: 种群
            fitness: 适应度列表
        
        Returns:
            两个父代个体
        """
        tournament_size = 3
        
        def tournament_select():
            # 随机选择tournament_size个个体
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness[i] for i in tournament_indices]
            
            # 选择适应度最好的（适应度越小越好）
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            return population[winner_idx]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2
    
    def optimize(self) -> Tuple[Dict[int, int], Dict[int, int], float]:
        """
        运行遗传算法优化
        
        Returns:
            (value_to_code, code_to_value, best_fitness)
        """
        # 初始化种群
        print(f"  初始化种群（大小: {self.population_size}）...")
        population = [self.create_individual() for _ in range(self.population_size)]
        
        # 评估初始种群
        fitness = [self.evaluate_fitness(ind[0], ind[1]) for ind in population]
        
        # 记录最优解
        best_idx = fitness.index(min(fitness))
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        print(f"  初始最优适应度: {best_fitness:.4f}")
        
        # 进化过程
        for generation in range(self.max_generations):
            # 创建新种群
            new_population = []
            
            # 精英保留
            elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代
                parent1, parent2 = self.select_parents(population, fitness)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1 if random.random() < 0.5 else parent2
                
                # 变异
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            # 更新种群
            population = new_population[:self.population_size]
            
            # 评估新种群
            fitness = [self.evaluate_fitness(ind[0], ind[1]) for ind in population]
            
            # 更新最优解
            current_best_idx = fitness.index(min(fitness))
            current_best_fitness = fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx]
            
            # 每100代打印一次进度
            if (generation + 1) % 100 == 0:
                avg_fitness = sum(fitness) / len(fitness)
                print(f"  第 {generation + 1}/{self.max_generations} 代: "
                      f"最优={best_fitness:.4f}, 平均={avg_fitness:.4f}")
        
        print(f"  最终最优适应度: {best_fitness:.4f}")
        
        return best_individual[0], best_individual[1], best_fitness


def optimize_olm_mapping_genetic(
    distribution: Dict[int, int],
    k: int,
    max_generations: int = 1000,
    population_size: int = 50,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elite_size: int = 5,
    ber: float = 1e-2,
    consider_multi_bit: bool = True,
    max_hamming_dist: int = 3,
    use_value_importance: bool = True,
    use_local_consistency: bool = True,
    local_consistency_weight: float = 0.1
) -> Tuple[Dict[int, int], Dict[int, int], float]:
    """
    使用遗传算法优化OLM映射
    
    Args:
        distribution: 量化值分布
        k: 位宽
        max_generations: 最大代数
        population_size: 种群大小
        crossover_rate: 交叉率
        mutation_rate: 变异率
        elite_size: 精英数量
        ber: Bit-error-rate
        consider_multi_bit: 是否考虑多bit翻转
        max_hamming_dist: 最大Hamming距离
        use_value_importance: 是否使用值重要性
        use_local_consistency: 是否使用局部一致性
        local_consistency_weight: 局部一致性权重
    
    Returns:
        (value_to_code, code_to_value, best_fitness)
    """
    ga = GeneticAlgorithmOLM(
        distribution=distribution,
        k=k,
        population_size=population_size,
        max_generations=max_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        ber=ber,
        consider_multi_bit=consider_multi_bit,
        max_hamming_dist=max_hamming_dist,
        use_value_importance=use_value_importance,
        use_local_consistency=use_local_consistency,
        local_consistency_weight=local_consistency_weight
    )
    
    return ga.optimize()



