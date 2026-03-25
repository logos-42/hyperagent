/// 进化选择策略
/// 实现多种选择机制：轮盘赌、锦标赛、截断选择等

use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use crate::agent::Agent;

/// 带分数的个体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub agent: Agent,
    pub fitness: f32,
}

/// 选择策略类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionType {
    /// 轮盘赌选择（适应度比例）
    RouletteWheel,
    /// 锦标赛选择
    Tournament { tournament_size: usize },
    /// 截断选择（只选 top-k）
    Truncation { top_k: usize },
    /// Boltzmann 选择（基于温度）
    Boltzmann { temperature: f32 },
    /// 排名选择
    RankBased,
    /// 多样性保护选择
    DiversityPreserving { diversity_weight: f32 },
}

impl Default for SelectionType {
    fn default() -> Self {
        Self::Tournament { tournament_size: 3 }
    }
}

/// 选择器
pub struct Selector {
    strategy: SelectionType,
}

impl Selector {
    pub fn new(strategy: SelectionType) -> Self {
        Self { strategy }
    }

    /// 选择一个个体
    pub fn select<'a>(&self, population: &'a [Individual]) -> Option<&'a Individual> {
        if population.is_empty() {
            return None;
        }

        match &self.strategy {
            SelectionType::RouletteWheel => self.roulette_wheel_select(population),
            SelectionType::Tournament { tournament_size } => {
                self.tournament_select(population, *tournament_size)
            }
            SelectionType::Truncation { top_k } => self.truncation_select(population, *top_k),
            SelectionType::Boltzmann { temperature } => {
                self.boltzmann_select(population, *temperature)
            }
            SelectionType::RankBased => self.rank_select(population),
            SelectionType::DiversityPreserving { diversity_weight } => {
                self.diversity_select(population, *diversity_weight)
            }
        }
    }

    /// 选择多个个体（用于生成下一代）
    pub fn select_many<'a>(&self, population: &'a [Individual], count: usize) -> Vec<&'a Individual> {
        let mut selected = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(ind) = self.select(population) {
                selected.push(ind);
            }
        }
        selected
    }

    // === 选择策略实现 ===

    /// 轮盘赌选择：P(i) = f_i / Σf
    fn roulette_wheel_select<'a>(&self, population: &'a [Individual]) -> Option<&'a Individual> {
        let total_fitness: f32 = population.iter().map(|i| i.fitness).sum();
        if total_fitness <= 0.0 {
            return population.first();
        }

        let mut rng = thread_rng();
        let mut rand_point = rng.gen_range(0.0..total_fitness);

        for individual in population {
            rand_point -= individual.fitness;
            if rand_point <= 0.0 {
                return Some(individual);
            }
        }

        population.last()
    }

    /// 锦标赛选择：随机选 k 个，取最优
    fn tournament_select<'a>(
        &self,
        population: &'a [Individual],
        tournament_size: usize,
    ) -> Option<&'a Individual> {
        if population.is_empty() {
            return None;
        }

        let actual_size = tournament_size.min(population.len());
        let mut rng = thread_rng();

        let mut best: Option<&Individual> = None;
        for _ in 0..actual_size {
            let idx = rng.gen_range(0..population.len());
            if best.is_none() || population[idx].fitness > best.unwrap().fitness {
                best = Some(&population[idx]);
            }
        }

        best
    }

    /// 截断选择：只选前 k 个
    fn truncation_select<'a>(&self, population: &'a [Individual], top_k: usize) -> Option<&'a Individual> {
        if population.is_empty() {
            return None;
        }

        let mut sorted: Vec<&Individual> = population.iter().collect();
        sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let idx = thread_rng().gen_range(0..top_k.min(sorted.len()));
        Some(sorted[idx])
    }

    /// Boltzmann 选择：P(i) = exp(f_i/T) / Σexp(f_j/T)
    fn boltzmann_select<'a>(&self, population: &'a [Individual], temperature: f32) -> Option<&'a Individual> {
        if population.is_empty() || temperature <= 0.0 {
            return self.truncation_select(population, 1);
        }

        let mut rng = thread_rng();

        // 高温时接近随机选择，低温时接近截断选择
        if temperature > 10.0 {
            let idx = rng.gen_range(0..population.len());
            return Some(&population[idx]);
        }

        // 计算 Boltzmann 权重
        let max_fitness = population.iter().map(|i| i.fitness).fold(
            f32::NEG_INFINITY,
            f32::max,
        );

        let weights: Vec<f32> = population
            .iter()
            .map(|i| ((i.fitness - max_fitness) / temperature).exp())
            .collect();

        let total_weight: f32 = weights.iter().sum();
        let mut rand_point = rng.gen_range(0.0..total_weight);

        for (i, weight) in weights.iter().enumerate() {
            rand_point -= weight;
            if rand_point <= 0.0 {
                return Some(&population[i]);
            }
        }

        population.last()
    }

    /// 排名选择：按排名而非绝对适应度
    fn rank_select<'a>(&self, population: &'a [Individual]) -> Option<&'a Individual> {
        if population.is_empty() {
            return None;
        }

        let mut sorted: Vec<(usize, &Individual)> = population.iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.fitness.partial_cmp(&a.1.fitness).unwrap());

        // 排名权重：P(i) ∝ (N - rank_i)
        let n = sorted.len();
        let total_weight: f32 = (1..=n).sum::<usize>() as f32;

        let mut rng = thread_rng();
        let mut rand_point = rng.gen_range(0.0..total_weight);

        for (i, (_, individual)) in sorted.iter().enumerate() {
            let weight = (n - i) as f32;
            rand_point -= weight;
            if rand_point <= 0.0 {
                return Some(individual);
            }
        }

        sorted.last().map(|(_, ind)| *ind)
    }

    /// 多样性保护选择：结合适应度和新颖性
    fn diversity_select<'a>(
        &self,
        population: &'a [Individual],
        diversity_weight: f32,
    ) -> Option<&'a Individual> {
        if population.is_empty() {
            return None;
        }

        // 计算每个个体的新颖性（与其他个体的平均距离）
        let novelties: Vec<f32> = population
            .iter()
            .map(|ind| {
                let avg_distance: f32 = population
                    .iter()
                    .filter(|other| other.agent.id != ind.agent.id)
                    .map(|other| self.genotype_distance(&ind.agent, &other.agent))
                    .sum::<f32>()
                    / (population.len() - 1) as f32;
                avg_distance
            })
            .collect();

        let max_novelty = novelties.iter().cloned().fold(0.0_f32, f32::max);
        let max_fitness = population.iter().map(|i| i.fitness).fold(0.0_f32, f32::max);

        // 综合分数 = (1-w)*fitness + w*novelty
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, ind) in population.iter().enumerate() {
            let norm_fitness = if max_fitness > 0.0 {
                ind.fitness / max_fitness
            } else {
                0.0
            };
            let norm_novelty = if max_novelty > 0.0 {
                novelties[i] / max_novelty
            } else {
                0.0
            };

            let score = (1.0 - diversity_weight) * norm_fitness
                + diversity_weight * norm_novelty;

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        Some(&population[best_idx])
    }

    /// 计算基因型距离（简化版：基于代码相似度）
    fn genotype_distance(&self, a: &Agent, b: &Agent) -> f32 {
        // 简化的编辑距离比例
        let len_a = a.code.len() as f32;
        let len_b = b.code.len() as f32;

        if len_a == 0.0 && len_b == 0.0 {
            return 0.0;
        }

        // 使用 Jaccard 距离近似
        let words_a: std::collections::HashSet<_> = a
            .code
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();
        let words_b: std::collections::HashSet<_> = b
            .code
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            1.0 - (intersection as f32 / union as f32)
        }
    }
}

/// 种群统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStats {
    pub size: usize,
    pub mean_fitness: f32,
    pub std_fitness: f32,
    pub max_fitness: f32,
    pub min_fitness: f32,
    pub diversity: f32,
}

impl PopulationStats {
    pub fn calculate(population: &[Individual]) -> Self {
        if population.is_empty() {
            return Self {
                size: 0,
                mean_fitness: 0.0,
                std_fitness: 0.0,
                max_fitness: 0.0,
                min_fitness: 0.0,
                diversity: 0.0,
            };
        }

        let fitnesses: Vec<f32> = population.iter().map(|i| i.fitness).collect();
        let n = fitnesses.len() as f32;

        let mean = fitnesses.iter().sum::<f32>() / n;
        let variance = fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let max = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = fitnesses.iter().cloned().fold(f32::INFINITY, f32::min);

        // 多样性：适应度标准差 / 均值
        let diversity = if mean.abs() > 1e-6 { std / mean.abs() } else { 0.0 };

        Self {
            size: population.len(),
            mean_fitness: mean,
            std_fitness: std,
            max_fitness: max,
            min_fitness: min,
            diversity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;

    fn create_test_population() -> Vec<Individual> {
        vec![
            Individual {
                agent: Agent::new("code1".to_string(), "prompt".to_string()),
                fitness: 5.0,
            },
            Individual {
                agent: Agent::new("code2".to_string(), "prompt".to_string()),
                fitness: 8.0,
            },
            Individual {
                agent: Agent::new("code3".to_string(), "prompt".to_string()),
                fitness: 3.0,
            },
            Individual {
                agent: Agent::new("code4".to_string(), "prompt".to_string()),
                fitness: 10.0,
            },
        ]
    }

    #[test]
    fn test_tournament_selection() {
        let selector = Selector::new(SelectionType::Tournament { tournament_size: 2 });
        let population = create_test_population();
        let selected = selector.select(&population);
        assert!(selected.is_some());
        assert!(selected.unwrap().fitness >= 3.0); // 至少不会选最差的
    }

    #[test]
    fn test_truncation_selection() {
        let selector = Selector::new(SelectionType::Truncation { top_k: 2 });
        let population = create_test_population();
        let selected = selector.select(&population);
        assert!(selected.is_some());
        assert!(selected.unwrap().fitness >= 8.0); // 应该在前 2 个中
    }

    #[test]
    fn test_population_stats() {
        let population = create_test_population();
        let stats = PopulationStats::calculate(&population);
        assert_eq!(stats.size, 4);
        assert!((stats.mean_fitness - 6.5).abs() < 0.01);
        assert!((stats.max_fitness - 10.0).abs() < 0.01);
        assert!((stats.min_fitness - 3.0).abs() < 0.01);
    }
}
