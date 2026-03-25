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
    #[allow(dead_code)]
    output: String,
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
        }
    }

    /// 从 archive + 当前分支中选择多样性父代（核心：非只取最优）
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

        // 按 fitness 降序排列
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 去重（同 id 只保留最高 fitness）
        candidates.dedup_by(|a, b| a.0.id == b.0.id);

        // 贪心选择多样性父代
        let mut selected: Vec<crate::agent::Agent> = Vec::new();
        for (agent, _) in &candidates {
            if selected.len() >= num_branches {
                break;
            }
            let is_diverse = selected.is_empty()
                || selected.iter().all(|s| {
                    jaccard_similarity(&s.code, &agent.code) < threshold
                });
            if is_diverse {
                selected.push(agent.clone());
            }
        }

        // 如果多样性过滤太严格导致不足，放宽阈值补齐
        if selected.len() < num_branches && !candidates.is_empty() {
            for (agent, _) in &candidates {
                if selected.len() >= num_branches {
                    break;
                }
                if !selected.iter().any(|s| s.id == agent.id) {
                    selected.push(agent.clone());
                }
            }
        }

        selected
    }

    /// 更新热力学状态
    fn update_thermodynamics(&mut self, results: &[BranchResult]) {
        let config = &self.state.config;

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

        // 熵 = 各分支 fitness 的标准差
        if results.len() >= 2 {
            let mean: f32 = results.iter().map(|r| r.fitness).sum::<f32>() / results.len() as f32;
            let variance: f32 = results
                .iter()
                .map(|r| (r.fitness - mean).powi(2))
                .sum::<f32>()
                / results.len() as f32;
            let new_entropy = variance.sqrt();
            self.energy.entropy_production_rate = new_entropy - self.energy.entropy;
            self.energy.entropy = new_entropy;
        }

        // 信息-能量耦合
        if self.fitness_history.len() >= 3 {
            let len = self.fitness_history.len() as f32;
            let mean: f32 = self.fitness_history.iter().sum::<f32>() / len;
            let fitness_var: f32 = self
                .fitness_history
                .iter()
                .map(|f| (f - mean).powi(2))
                .sum::<f32>()
                / len;

            let gen_mean = (self.fitness_history.len() - 1) as f32 / 2.0;
            let genotype_var: f32 = (0..self.fitness_history.len())
                .map(|i| ((i as f32) - gen_mean).powi(2))
                .sum::<f32>()
                / len;

            let covariance: f32 = (0..self.fitness_history.len())
                .map(|i| ((i as f32) - gen_mean) * (self.fitness_history[i] - mean))
                .sum::<f32>()
                / len;

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
                                    output: exec_result.output,
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

        // 1. 从 archive + 当前分支中选择多样性父代
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
                            output: exec_result.output,
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
                output: exec_result.output,
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
