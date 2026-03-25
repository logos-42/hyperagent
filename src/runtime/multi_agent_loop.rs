use anyhow::Result;

use crate::agent::population::{AgentRole, MultiAgentSystem, PopulationConfig, PopulationAgent};
use crate::llm::LLMClient;

use super::state::{RuntimeConfig, RuntimePhase, RuntimeState, ThermodynamicSnapshot};
use super::thermodynamics::{
    DissipationScale, EnergyState, FitnessLandscape, InfoEnergyCoupling,
};

pub struct MultiAgentEvolutionLoop<C: LLMClient> {
    client: C,
    state: RuntimeState,
    config: PopulationConfig,
    // 热力学状态
    energy: EnergyState,
    landscape: FitnessLandscape,
    info_coupling: InfoEnergyCoupling,
    dissipation: DissipationScale,
    score_history: Vec<f32>,
    current_temperature: f32,
}

impl<C: LLMClient + Clone> MultiAgentEvolutionLoop<C> {
    pub fn new(client: C, config: RuntimeConfig, pop_config: PopulationConfig) -> Self {
        let initial_temperature = config.initial_temperature;
        let free_energy = (config.max_generations as f32)
            * (pop_config.population_size as f32)
            * 2.0;

        let dissipation = DissipationScale::new(
            pop_config.population_size.max(1),
            config.mutation_rate,
            config.selection_pressure,
        );

        Self {
            client,
            state: RuntimeState::new(config),
            config: pop_config,
            energy: EnergyState::new(free_energy, initial_temperature),
            landscape: FitnessLandscape::new(),
            info_coupling: InfoEnergyCoupling::new(initial_temperature),
            dissipation,
            score_history: Vec::new(),
            current_temperature: initial_temperature,
        }
    }

    fn update_thermodynamics(&mut self, best_score: f32, _mean_fitness: f64, std_fitness: f64) {
        let config = &self.state.config;
        self.score_history.push(best_score);

        // 1. 温度退火
        let gen = self.state.current_generation as f32;
        let base_temp = config.initial_temperature * config.annealing_rate.powf(gen);
        self.current_temperature = base_temp.max(0.01);

        // 2. 适应度景观
        self.landscape.update(&self.score_history);
        if self.landscape.escape_probability < 1.0 {
            self.current_temperature = (self.current_temperature * 2.0)
                .min(config.initial_temperature);
        }

        // 3. 能量状态
        self.energy.temperature = self.current_temperature;
        self.energy.free_energy = (self.energy.free_energy
            - (self.config.population_size as f32) * 2.0)
            .max(0.0);

        // 4. 熵 = 种群适应度标准差
        self.energy.entropy = std_fitness as f32;
        if self.score_history.len() >= 2 {
            let prev = self.score_history[self.score_history.len() - 2];
            self.energy.entropy_production_rate = best_score - prev;
        }

        // 5. 信息-能量耦合
        let fitness_var = (std_fitness * std_fitness) as f32;
        let genotype_var = (self.score_history.len() as f32).powi(2) / 12.0; // uniform variance
        if self.score_history.len() >= 2 && genotype_var > 1e-6 {
            let covariance = self.energy.entropy_production_rate; // trend as covariance proxy
            self.info_coupling.update_mutual_information(fitness_var, genotype_var, covariance);
        }

        // 6. 动态耗散尺度
        self.dissipation = DissipationScale::new(
            self.config.population_size.max(1),
            config.mutation_rate,
            config.selection_pressure + (1.0 - self.current_temperature / config.initial_temperature) * 0.5,
        );

        // 7. 临界点检测
        if self.dissipation.near_critical(0.5) {
            tracing::info!(
                "Generation {}: Near critical point! De={:.2} (phase transition likely)",
                self.state.current_generation, self.dissipation.deborah_number,
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

    pub async fn run(&mut self, task: &str) -> Result<RuntimeState> {
        let mut system = MultiAgentSystem::new(self.config.clone());

        self.state.set_phase(RuntimePhase::Executing);
        self.state.current_task = Some(task.to_string());

        while !self.state.is_finished() {
            self.state.increment_generation();
            system.increment_generation();

            tracing::info!(
                "Generation {} - Phase: {:?}",
                self.state.current_generation,
                self.state.phase
            );

            self.state.set_phase(RuntimePhase::Executing);
            self.execute_round(&mut system, task).await?;

            self.state.set_phase(RuntimePhase::Evaluating);
            self.evaluate_round(&mut system, task).await?;

            // 更新热力学状态
            let (mean, std, best) = system.population_fitness_stats();
            let best_fitness = best as f32;
            self.update_thermodynamics(best_fitness, mean, std);

            tracing::info!(
                "Generation {}: Pop fitness - mean={:.2}, std={:.2}, best={:.2} [T={:.3}, S={:.3}, De={:.1}]",
                self.state.current_generation, mean, std, best,
                self.current_temperature, self.energy.entropy, self.dissipation.deborah_number,
            );

            self.state.set_phase(RuntimePhase::Mutating);
            self.mutate_round(&mut system).await?;

            self.state.set_phase(RuntimePhase::Selecting);
            self.select_round(&mut system);

            let best = system.get_best();
            if let Some(b) = best {
                let fitness = b.fitness as f32;
                self.state.update_best(b.agent.clone(), fitness, fitness);
            }
        }

        self.state.set_phase(RuntimePhase::Finished);
        Ok(self.state.clone())
    }

    async fn execute_round(&mut self, system: &mut MultiAgentSystem, task: &str) -> Result<()> {
        let population: Vec<(String, AgentRole, String)> = system
            .population
            .iter()
            .map(|a| (a.id.clone(), a.role.clone(), a.agent.prompt.clone()))
            .collect();

        let mut tasks = Vec::new();

        for (_, _role, prompt) in &population {
            let task_clone = task.to_string();
            let prompt_clone = prompt.clone();
            let client_clone = self.client.clone();

            tasks.push(async move {
                execute_agent_task(&client_clone, prompt_clone, task_clone).await
            });
        }

        let results = futures::future::join_all(tasks).await;

        for (i, result) in results.iter().enumerate() {
            if let Ok(content) = result {
                let agent_id = &population[i].0;
                let role_name = role_debug_name(&population[i].1);
                system.broadcast_message(agent_id, content.clone());
                tracing::debug!(
                    "Agent {} executed: {}",
                    role_name,
                    &content[..content.len().min(50)]
                );
            }
        }

        Ok(())
    }

    async fn evaluate_round(&mut self, system: &mut MultiAgentSystem, task: &str) -> Result<()> {
        let population: Vec<(String, AgentRole, Vec<String>)> = system
            .population
            .iter()
            .map(|a| {
                (
                    a.id.clone(),
                    a.role.clone(),
                    a.messages
                        .iter()
                        .map(|m| format!("[{}]: {}", role_debug_name(&m.role), m.content))
                        .collect(),
                )
            })
            .collect();

        let mut fitness_updates = Vec::new();

        for (id, role, messages) in &population {
            let messages_text = messages.join("\n---\n");

            let eval_prompt = format!(
                "Task: {}\n\nAgent Outputs:\n{}\n\nEvaluate the quality (0-100):",
                task, messages_text
            );

            let response = self.client.complete(&eval_prompt).await?;
            let fitness = parse_score(&response.content);

            fitness_updates.push((id.clone(), fitness));
            tracing::info!("Agent {} fitness: {:.2}", role_debug_name(role), fitness);
        }

        for (id, fitness) in fitness_updates {
            system.update_fitness(&id, fitness);
        }

        Ok(())
    }

    async fn mutate_round(&mut self, system: &mut MultiAgentSystem) -> Result<()> {
        let (mean, std, _) = system.population_fitness_stats();

        // 低多样性时触发交叉变异
        if std < 0.1 && system.population.len() > 2 {
            let parents = system.select_parents(2);
            self.create_diversity_mutants(system, &parents).await?;
        }

        let mut code_updates = Vec::new();

        for pop_agent in &system.population {
            // Metropolis-based mutation: use temperature to decide whether to mutate even above-average agents
            let should_mutate = if pop_agent.fitness < mean {
                true // below mean: always try to improve
            } else {
                // above mean: mutate with probability based on temperature
                pseudo_random() < self.current_temperature
            };

            if should_mutate {
                let mutation_prompt = format!(
                    "Current agent ({}):\n{}\n\nTask: improve this agent's code.\n\nRules:\n- Keep improvements minimal but effective\n- Focus on the weak aspects\n\nReturn improved code:",
                    role_debug_name(&pop_agent.role),
                    pop_agent.agent.code
                );

                let response = self.client.complete(&mutation_prompt).await?;
                code_updates.push((pop_agent.id.clone(), response.content));
            }
        }

        for (id, new_code) in code_updates {
            if let Some(agent) = system.population.iter_mut().find(|a| a.id == id) {
                agent.agent.code = new_code;
            }
        }

        system.clear_round();
        Ok(())
    }

    async fn create_diversity_mutants(
        &mut self,
        system: &mut MultiAgentSystem,
        parents: &[PopulationAgent],
    ) -> Result<()> {
        if parents.len() < 2 || !self.config.crossover_enabled {
            return Ok(());
        }

        let crossover_prompt = format!(
            "Agent A ({}):\n{}\n\nAgent B ({}):\n{}\n\nCreate a hybrid agent that combines strengths from both:\n\nReturn hybrid code:",
            role_debug_name(&parents[0].role),
            parents[0].agent.code,
            role_debug_name(&parents[1].role),
            parents[1].agent.code
        );

        if let Ok(response) = self.client.complete(&crossover_prompt).await {
            let hybrid_code = response.content;

            let weakest_id = system
                .population
                .iter()
                .min_by(|a, b| {
                    a.fitness
                        .partial_cmp(&b.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|a| a.id.clone());

            if let Some(id) = weakest_id {
                if let Some(agent) = system.population.iter_mut().find(|a| a.id == id) {
                    agent.agent.code = hybrid_code;
                    agent.agent.generation = system.generation;
                    tracing::info!("Created diversity mutant via crossover (T={:.3})", self.current_temperature);
                }
            }
        }

        Ok(())
    }

    /// Metropolis-based selection: replace weak agents with best, but accept
    /// some sub-optimal replacements at high temperature for exploration
    fn select_round(&mut self, system: &mut MultiAgentSystem) {
        let (mean, std, _) = system.population_fitness_stats();

        let best_code = system.get_best().map(|a| a.agent.code.clone());

        for agent in &mut system.population {
            if agent.fitness < mean - std {
                if let Some(ref best) = best_code {
                    // Metropolis: accept replacement with probability based on temperature
                    let delta = agent.fitness - mean; // negative
                    let accept_prob = if delta < 0.0 {
                        let energy_diff = -delta as f32;
                        (-energy_diff / self.current_temperature).exp()
                    } else {
                        1.0
                    };

                    if pseudo_random() < accept_prob {
                        agent.agent.code = best.clone();
                        agent.agent.generation = system.generation;
                        tracing::info!(
                            "Replaced weak agent (fitness={:.2}) via Metropolis (P={:.3}, T={:.3})",
                            agent.fitness, accept_prob, self.current_temperature,
                        );
                    }
                    // else: keep weak agent for diversity (exploration)
                }
            }
        }
    }

    pub fn get_state(&self) -> &RuntimeState {
        &self.state
    }

    pub fn get_best_agent(&self) -> Option<PopulationAgent> {
        self.state.best_agent.as_ref().map(|a| {
            let role = AgentRole::Executor;
            let mut pa = PopulationAgent::new(a.clone(), role);
            pa.fitness = self.state.best_score as f64;
            pa
        })
    }
}

async fn execute_agent_task<C: LLMClient>(client: &C, prompt: String, task: String) -> Result<String> {
    let full_prompt = format!("{}\n\nTask: {}", prompt, task);
    let response = client.complete(&full_prompt).await?;
    Ok(response.content)
}

fn parse_score(text: &str) -> f64 {
    let numbers: Vec<f64> = text
        .split(|c: char| !c.is_numeric() && c != '.')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();

    numbers.first().copied().unwrap_or(50.0).min(100.0).max(0.0)
}

fn pseudo_random() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    ((x >> 33) as f32) / (u32::MAX as f32)
}

fn role_debug_name(role: &AgentRole) -> String {
    match role {
        AgentRole::Planner => "Planner".to_string(),
        AgentRole::Executor => "Executor".to_string(),
        AgentRole::Evaluator => "Evaluator".to_string(),
        AgentRole::Reflector => "Reflector".to_string(),
        AgentRole::Synthesizer => "Synthesizer".to_string(),
    }
}
