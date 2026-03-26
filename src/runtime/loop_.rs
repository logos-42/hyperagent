use anyhow::Result;

use crate::agent::{Executor, MetaMutator, Mutator};
use crate::eval::Evaluator;
use crate::llm::LLMClient;

use super::state::{RuntimePhase, RuntimeState, ThermodynamicSnapshot};
use super::thermodynamics::{
    compute_fitness, compute_novelty, jaccard_similarity,
    DissipationScale, EnergyState, FitnessLandscape, InfoEnergyCoupling,
};

/// 单个探索分支的结果
#[derive(Debug, Clone)]
struct BranchResult {
    agent: crate::agent::Agent,
    score: f32,
    novelty: f32,
    fitness: f32,
    _output: String,
}

pub struct EvolutionLoop<C: LLMClient> {
    executor: Executor<C>,
    evaluator: Evaluator<C>,
    mutator: Mutator<C>,
    meta_mutator: MetaMutator<C>,
    state: RuntimeState,
    // 热力学状态
    energy: EnergyState,
    landscape: FitnessLandscape,
    info_coupling: InfoEnergyCoupling,
    dissipation: DissipationScale,
    fitness_history: Vec<f32>,
    current_temperature: f32,
    // 多分支探索
    branches: Vec<BranchResult>,
    recent_codes: Vec<String>,
    // 前一代的熵值（用于计算熵产生率）
    prev_entropy: f32,
}

impl<C: LLMClient + Clone> EvolutionLoop<C> {
    pub fn new(client: C, state: RuntimeState) -> Self {
        let executor = Executor::new(client.clone());
        let evaluator = Evaluator::new(client.clone());
        let mutator = Mutator::new(client.clone());
        let meta_mutator = MetaMutator::new(client);

        let initial_temperature = state.config.initial_temperature;
        let free_energy = (state.config.max_generations as f32)
            * (state.config.num_branches as f32)
            * 2.0;

        let dissipation = DissipationScale::new(
            state.config.population_size.max(state.config.num_branches),
            state.config.mutation_rate,
            state.config.selection_pressure,
        );

        Self {
            executor,
            evaluator,
            mutator,
            meta_mutator,
            state,
            energy: EnergyState::new(free_energy, initial_temperature),
            landscape: FitnessLandscape::new(),
            info_coupling: InfoEnergyCoupling::new(initial_temperature),
            dissipation,
            fitness_history: Vec::new(),
            current_temperature: initial_temperature,
            branches: Vec::new(),
            recent_codes: Vec::new(),
            prev_entropy: 0.0,
        }
    }

    /// 使用确定性拥挤算法选择多样性父代
    fn select_diverse_parents(&self) -> Vec<crate::agent::Agent> {
        let num_branches = self.state.config.num_branches;
        let threshold = self.state.config.diversity_threshold;

        // 候选池：当前分支 + archive 中的高分方案
        let mut candidates: Vec<(crate::agent::Agent, f32)> = self
            .branches
            .iter()
            .map(|b| (b.agent.clone(), b.fitness))
            .collect();

        for record in self.state.archive.top_k(num_branches * 3) {
            let novelty = compute_novelty(&record.agent.code, &self.recent_codes);
            let fitness = compute_fitness(record.score.value, novelty, self.state.config.novelty_weight);
            candidates.push((record.agent.clone(), fitness));
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // 按 fitness 降序排列
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 去重（同 id 只保留最高 fitness）
        candidates.dedup_by(|a, b| a.0.id == b.0.id);

        // 预计算候选者之间的相似度矩阵，避免重复计算
        let candidate_codes: Vec<&String> = candidates.iter().map(|(a, _)| &a.code).collect();
        let candidate_count = candidates.len();
        
        // 构建相似度矩阵：sim_matrix[i][j] = jaccard_similarity(code_i, code_j)
        // 这是对称矩阵，只需计算上三角
        let mut sim_matrix: Vec<Vec<f32>> = vec![vec![0.0; candidate_count]; candidate_count];
        for i in 0..candidate_count {
            sim_matrix[i][i] = 1.0; // 自相似度为1
            for j in (i + 1)..candidate_count {
                let sim = jaccard_similarity(candidate_codes[i], candidate_codes[j]);
                sim_matrix[i][j] = sim;
                sim_matrix[j][i] = sim; // 对称性
            }
        }

        // 确定性拥挤选择
        let mut selected: Vec<usize> = Vec::new(); // 存储选中的候选索引
        let mut selected_fitness: Vec<f32> = Vec::new();
        
        for (idx, (_, fitness)) in candidates.iter().enumerate() {
            if selected.len() >= num_branches {
                break;
            }

            if selected.is_empty() {
                // 第一个候选直接入选
                selected.push(idx);
                selected_fitness.push(*fitness);
                continue;
            }

            // 计算与已选个体的最大相似度（使用预计算矩阵）
            let max_similarity = selected
                .iter()
                .map(|&s_idx| sim_matrix[idx][s_idx])
                .fold(0.0_f32, f32::max);

            // 确定性拥挤接受条件
            if max_similarity < threshold {
                selected.push(idx);
                selected_fitness.push(*fitness);
            } else {
                // 收集所有可能的替换选项
                let mut best_replacement: Option<(usize, f32)> = None;
                
                for (pos, &sel_idx) in selected.iter().enumerate() {
                    // 只有当新候选与已选个体的相似度超过阈值时才考虑替换
                    if sim_matrix[idx][sel_idx] >= threshold {
                        // 只有当新候选的 fitness 更高时才考虑替换
                        if *fitness > selected_fitness[pos] {
                            // 计算替换后的总相似度变化
                            // 旧相似度和：被替换个体与所有其他已选个体的相似度之和
                            let old_sim_sum: f32 = selected
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != pos)
                                .map(|(_, &other_idx)| sim_matrix[sel_idx][other_idx])
                                .sum();
                            
                            // 新相似度和：新候选与所有其他已选个体的相似度之和
                            let new_sim_sum: f32 = selected
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != pos)
                                .map(|(_, &other_idx)| sim_matrix[idx][other_idx])
                                .sum();
                            
                            // 选择能最大化降低总相似度的替换
                            if new_sim_sum < old_sim_sum {
                                let new_total_sim = new_sim_sum;
                                match &best_replacement {
                                    None => best_replacement = Some((pos, new_total_sim)),
                                    Some((_, prev_sim)) if new_total_sim < *prev_sim => {
                                        best_replacement = Some((pos, new_total_sim));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                
                // 应用最佳替换
                if let Some((pos, _)) = best_replacement {
                    selected[pos] = idx;
                    selected_fitness[pos] = *fitness;
                }
            }
        }

        // 如果数量不足，放宽条件补齐
        if selected.len() < num_branches {
            for (idx, _) in candidates.iter().enumerate() {
                if selected.len() >= num_branches {
                    break;
                }
                if !selected.contains(&idx) {
                    selected.push(idx);
                }
            }
        }

        // 转换索引为实际的 agent
        selected
            .into_iter()
            .map(|idx| candidates[idx].0.clone())
            .collect()
    }

    /// 计算无偏样本方差（Bessel校正）
    fn compute_unbiased_variance(&self, values: &[f32]) -> Option<f32> {
        let n = values.len();
        if n < 2 {
            return None;
        }
        let mean: f32 = values.iter().sum::<f32>() / n as f32;
        let sum_sq_diff: f32 = values.iter().map(|x| (x - mean).powi(2)).sum();
        // Bessel's correction: divide by (n-1) for unbiased estimate
        Some(sum_sq_diff / (n - 1) as f32)
    }

    /// 计算熵产生率（改进版本：正确的时间导数 + 数值稳定性）
    /// 熵产生率 = dS/dt，表示系统熵随时间的变化率
    fn compute_entropy_production_rate(&self, current_entropy: f32) -> f32 {
        // 第一代没有前一代数据，熵产生率为0
        if self.state.current_generation <= 1 {
            return 0.0;
        }
        
        // 熵的变化量
        let delta_s = current_entropy - self.prev_entropy;
        
        // 熵产生率 = 熵变化 / 时间步（每代为一个时间单位）
        // 正值表示熵增（系统变得更无序），负值表示熵减（系统变得更有序）
        let rate = delta_s; // 因为时间步 = 1 代
        
        // 数值稳定性：使用自适应钳位，基于当前熵值缩放
        // 允许的熵产生率范围应与当前熵值成比例
        let max_rate = (current_entropy + 0.1).max(0.5);
        rate.clamp(-max_rate, max_rate)
    }

    /// 更新热力学状态
    fn update_thermodynamics(&mut self, results: &[BranchResult]) {
        let config = &self.state.config;

        // 保存当前熵值作为前一代熵值（用于下一次计算熵产生率）
        self.prev_entropy = self.energy.entropy;

        // 追踪最佳 fitness 历史
        if let Some(best) = results.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            self.fitness_history.push(best.fitness);
        }

        // 温度退火
        let gen = self.state.current_generation as f32;
        let base_temp = config.initial_temperature * config.annealing_rate.powf(gen);
        self.current_temperature = base_temp.max(0.01);

        // 适应度景观
        self.landscape.update(&self.fitness_history);
        if self.landscape.escape_probability < 1.0 {
            self.current_temperature =
                (self.current_temperature * 2.5).min(config.initial_temperature);
        }

        // 能量状态
        self.energy.temperature = self.current_temperature;
        self.energy.free_energy = (self.energy.free_energy
            - (config.num_branches as f32) * 2.0)
            .max(0.0);

        // 熵 = 使用无偏估计的各分支 fitness 标准差
        let fitness_values: Vec<f32> = results.iter().map(|r| r.fitness).collect();
        if results.len() >= 2 {
            let new_entropy = self.compute_unbiased_variance(&fitness_values)
                .map(|v| v.sqrt())
                .unwrap_or(0.0);
            
            // 使用改进的熵产生率计算
            self.energy.entropy_production_rate = self.compute_entropy_production_rate(new_entropy);
            self.energy.entropy = new_entropy;
        }

        // 信息-能量耦合（使用无偏协方差估计）
        if self.fitness_history.len() >= 3 {
            let n = self.fitness_history.len() as f32;
            let mean: f32 = self.fitness_history.iter().sum::<f32>() / n;
            
            // 无偏方差估计
            let fitness_var: f32 = if self.fitness_history.len() > 1 {
                self.compute_unbiased_variance(&self.fitness_history)
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            let gen_mean = (self.fitness_history.len() - 1) as f32 / 2.0;
            
            // 无偏基因型方差估计
            let gen_indices: Vec<f32> = (0..self.fitness_history.len())
                .map(|i| i as f32)
                .collect();
            let genotype_var: f32 = if self.fitness_history.len() > 1 {
                self.compute_unbiased_variance(&gen_indices)
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            // 无偏协方差估计
            let covariance: f32 = if self.fitness_history.len() > 1 {
                let sum_xy: f32 = (0..self.fitness_history.len())
                    .map(|i| ((i as f32) - gen_mean) * (self.fitness_history[i] - mean))
                    .sum();
                // Bessel's correction for covariance
                sum_xy / (n - 1.0)
            } else {
                0.0
            };

            self.info_coupling
                .update_mutual_information(fitness_var, genotype_var, covariance);
        }

        // 动态耗散尺度
        self.dissipation = DissipationScale::new(
            config.population_size.max(config.num_branches),
            config.mutation_rate,
            config.selection_pressure
                + (1.0 - self.current_temperature / config.initial_temperature) * 0.5,
        );

        if self.dissipation.near_critical(0.5) {
            tracing::info!(
                "  ⚡ Near critical point! De={:.2} (phase transition)",
                self.dissipation.deborah_number,
            );
        }

        self.state.thermo = ThermodynamicSnapshot {
            temperature: self.current_temperature,
            entropy: self.energy.entropy,
            entropy_production_rate: self.energy.entropy_production_rate,
            free_energy: self.energy.free_energy,
            landscape_gradient: self.landscape.gradient,
            landscape_curvature: self.landscape.curvature,
            escape_probability: self.landscape.escape_probability,
            info_energy: self.info_coupling.info_energy,
            deborah_number: self.dissipation.deborah_number,
            metropolis_accept_prob: 0.0,
        };
    }

    /// 执行一个完整的进化代
    async fn run_generation(&mut self, task: &str) -> Result<Vec<BranchResult>> {
        let novelty_weight = self.state.config.novelty_weight;

        // 0. Gen 1 初始化：直接执行初始 agent，不依赖父代
        if self.branches.is_empty() && self.state.archive.size() == 0 {
            self.state.set_phase(RuntimePhase::Executing);
            let initial_agent = crate::agent::Agent::new(String::new(), String::new());
            let mut results = Vec::new();

            for i in 0..self.state.config.num_branches {
                let mut agent = initial_agent.clone();
                agent.generation = self.state.current_generation;
                agent.id = format!("gen{}_init{}", self.state.current_generation, i);

                match self.executor.run(&agent, task).await {
                    Ok(exec_result) => {
                        match self.evaluator.score(task, &exec_result).await {
                            Ok(eval_result) => {
                                let score = eval_result.score.value;
                                let novelty = 1.0; // 首次执行，最大新颖度
                                let fitness = compute_fitness(score, novelty, novelty_weight);

                                self.state.archive.store(
                                    agent.clone(),
                                    eval_result.score,
                                    task.to_string(),
                                    exec_result.output.clone(),
                                );
                                self.state.lineage.add(&agent, None, score);
                                self.state.update_best(agent.clone(), score, fitness);

                                self.recent_codes.push(agent.code.clone());

                                results.push(BranchResult {
                                    agent,
                                    score,
                                    novelty,
                                    fitness,
                                    _output: exec_result.output,
                                });

                                tracing::info!(
                                    "Generation {}: init branch {}/{} | score={:.2}, novelty=1.0, fitness={:.2}",
                                    self.state.current_generation, i + 1,
                                    self.state.config.num_branches, score, fitness,
                                );
                            }
                            Err(e) => {
                                tracing::warn!("Initial eval failed (branch {}): {}", i, e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Initial execution failed (branch {}): {}", i, e);
                    }
                }
            }

            // 热力学初始化
            self.update_thermodynamics(&results);
            self.branches = results;
            return Ok(self.branches.clone());
        }

        // 1. 从 archive + 当前分支中选择多样性父代（使用确定性拥挤）
        let parents = self.select_diverse_parents();

        if parents.is_empty() {
            tracing::warn!(
                "Generation {}: no diverse parents available, generating fresh agents",
                self.state.current_generation
            );
            // 回退：生成全新的 agent
            let mut agent = crate::agent::Agent::new(String::new(), String::new());
            agent.generation = self.state.current_generation;
            let mut results = Vec::new();
            match self.executor.run(&agent, task).await {
                Ok(exec_result) => match self.evaluator.score(task, &exec_result).await {
                    Ok(eval_result) => {
                        let score = eval_result.score.value;
                        let novelty = compute_novelty(&agent.code, &self.recent_codes);
                        let fitness = compute_fitness(score, novelty, novelty_weight);
                        self.state.archive.store(
                            agent.clone(),
                            eval_result.score,
                            task.to_string(),
                            exec_result.output.clone(),
                        );
                        self.state.lineage.add(&agent, None, score);
                        self.state.update_best(agent.clone(), score, fitness);
                        self.recent_codes.push(agent.code.clone());
                        results.push(BranchResult {
                            agent,
                            score,
                            novelty,
                            fitness,
                            _output: exec_result.output,
                        });
                    }
                    Err(e) => tracing::warn!("Fallback eval failed: {}", e),
                },
                Err(e) => tracing::warn!("Fallback exec failed: {}", e),
            }
            self.update_thermodynamics(&results);
            self.branches = results;
            return Ok(self.branches.clone());
        }

        // 2. Meta-mutation
        if self.state.should_meta_mutate() {
            self.state.set_phase(RuntimePhase::MetaMutating);
            let history = format!(
                "Generations: {}, Branches: {}, Best Fitness: {:.2}, T: {:.3}, S: {:.3}",
                self.state.current_generation,
                self.branches.len(),
                self.branches
                    .iter()
                    .map(|b| b.fitness)
                    .fold(0.0_f32, f32::max),
                self.current_temperature,
                self.energy.entropy,
            );
            if let Ok(new_strategy) = self.meta_mutator.evolve(&history).await {
                self.state.mutation_strategy = new_strategy.clone();
                self.mutator.update_strategy(new_strategy.prompt.clone());
            }
        }

        // 3. 对每个父代进行变异
        self.state.set_phase(RuntimePhase::Mutating);
        let failures = self.state.archive.get_failures_text();
        let failures_vec: Vec<String> = if failures.is_empty() {
            vec![]
        } else {
            failures.lines().map(|s| s.to_string()).collect()
        };

        let mut mutated_agents = Vec::new();
        for parent in &parents {
            match self.mutator.mutate(parent, &failures_vec).await {
                Ok(agent) => mutated_agents.push(agent),
                Err(e) => {
                    tracing::warn!("Mutation failed: {}, using parent", e);
                    mutated_agents.push(parent.clone());
                }
            }
        }

        // 4. 执行 + 评估所有分支
        self.state.set_phase(RuntimePhase::Evaluating);
        let mut results = Vec::new();

        for agent in &mutated_agents {
            let exec_result = match self.executor.run(agent, task).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Execution failed: {}", e);
                    continue;
                }
            };

            let eval_result = match self.evaluator.score(task, &exec_result).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("Evaluation failed: {}", e);
                    continue;
                }
            };

            let score = eval_result.score.value;
            let novelty = compute_novelty(&agent.code, &self.recent_codes);
            let fitness = compute_fitness(score, novelty, novelty_weight);

            // 存档所有结果（不管分数高低）
            self.state.archive.store(
                agent.clone(),
                eval_result.score,
                task.to_string(),
                exec_result.output.clone(),
            );
            self.state.lineage.add(agent, None, score);
            self.state.update_best(agent.clone(), score, fitness);

            results.push(BranchResult {
                agent: agent.clone(),
                score,
                novelty,
                fitness,
                _output: exec_result.output,
            });

            // 记录代码用于后续新颖度计算
            self.recent_codes.push(agent.code.clone());
            if self.recent_codes.len() > 100 {
                self.recent_codes.remove(0);
            }
        }

        // 5. 更新热力学
        self.state.set_phase(RuntimePhase::Selecting);
        self.update_thermodynamics(&results);

        // 6. 日志
        if let Some(best_by_fitness) = results.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let _avg_score: f32 =
                results.iter().map(|b| b.score).sum::<f32>() / results.len().max(1) as f32;
            let avg_novelty: f32 =
                results.iter().map(|b| b.novelty).sum::<f32>() / results.len().max(1) as f32;
            let best_by_score = results
                .iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                .map(|b| b.score)
                .unwrap_or(0.0);

            tracing::info!(
                "Generation {}: {} branches | scores=[{:.2}..{:.2}] avg_novelty={:.2} | best_fit={:.2} (score={:.2}, nov={:.2}) | best_score={:.2} [T={:.3}, S={:.3}]",
                self.state.current_generation,
                results.len(),
                results.iter().map(|b| b.score).fold(f32::INFINITY, f32::min),
                best_by_score,
                avg_novelty,
                best_by_fitness.fitness, best_by_fitness.score, best_by_fitness.novelty,
                best_by_score,
                self.current_temperature, self.energy.entropy,
            );
        }

        Ok(results)
    }

    pub async fn run(&mut self, task: &str) -> Result<RuntimeState> {
        self.state.set_phase(RuntimePhase::Executing);
        self.state.current_task = Some(task.to_string());

        while !self.state.is_finished() {
            self.state.increment_generation();
            self.branches = self.run_generation(task).await?;
            self.state.save();
        }

        self.state.set_phase(RuntimePhase::Finished);
        self.state.save();
        Ok(self.state.clone())
    }

    pub async fn run_with_iterations(&mut self, task: &str, iterations: usize) -> Result<RuntimeState> {
        self.state.set_phase(RuntimePhase::Executing);
        self.state.current_task = Some(task.to_string());

        for _ in 0..iterations {
            if self.state.is_finished() {
                break;
            }
            self.state.increment_generation();
            self.branches = self.run_generation(task).await?;
            self.state.save();
        }

        self.state.set_phase(RuntimePhase::Finished);
        self.state.save();
        Ok(self.state.clone())
    }

    pub fn get_state(&self) -> &RuntimeState {
        &self.state
    }

    pub fn get_best_agent(&self) -> Option<&crate::agent::Agent> {
        self.state.best_agent.as_ref()
    }
}