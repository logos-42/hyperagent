use anyhow::Result;

use crate::agent::population::{AgentRole, MultiAgentSystem, PopulationConfig, PopulationAgent};
use crate::llm::LLMClient;

use super::state::{RuntimeConfig, RuntimePhase, RuntimeState};

pub struct MultiAgentEvolutionLoop<C: LLMClient> {
    client: C,
    state: RuntimeState,
    config: PopulationConfig,
}

impl<C: LLMClient + Clone> MultiAgentEvolutionLoop<C> {
    pub fn new(client: C, config: RuntimeConfig, pop_config: PopulationConfig) -> Self {
        Self {
            client,
            state: RuntimeState::new(config),
            config: pop_config,
        }
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

            self.state.set_phase(RuntimePhase::Mutating);
            self.mutate_round(&mut system).await?;

            self.state.set_phase(RuntimePhase::Selecting);
            self.select_round(&mut system);

            let best = system.get_best();
            if let Some(b) = best {
                self.state.update_best(b.agent.clone(), b.fitness as f32);
            }
        }

        self.state.set_phase(RuntimePhase::Finished);
        Ok(self.state.clone())
    }

    async fn execute_round(&mut self, system: &mut MultiAgentSystem, task: &str) -> Result<()> {
        let population: Vec<(String, AgentRole, String)> = system.population.iter()
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
                tracing::debug!("Agent {} executed: {}", role_name, &content[..content.len().min(50)]);
            }
        }

        Ok(())
    }

    async fn evaluate_round(&mut self, system: &mut MultiAgentSystem, task: &str) -> Result<()> {
        let population: Vec<(String, AgentRole, Vec<String>)> = system.population.iter()
            .map(|a| (a.id.clone(), a.role.clone(), 
                a.messages.iter().map(|m| format!("[{}]: {}", role_debug_name(&m.role), m.content)).collect()))
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
        
        if std < 0.1 && system.population.len() > 2 {
            let parents = system.select_parents(2);
            self.create_diversity_mutants(system, &parents).await?;
        }

        let mut code_updates = Vec::new();

        for pop_agent in &system.population {
            if pop_agent.fitness < mean {
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

    async fn create_diversity_mutants(&mut self, system: &mut MultiAgentSystem, parents: &[PopulationAgent]) -> Result<()> {
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
            
            let weakest_id = system.population.iter()
                .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
                .map(|a| a.id.clone());

            if let Some(id) = weakest_id {
                if let Some(agent) = system.population.iter_mut().find(|a| a.id == id) {
                    agent.agent.code = hybrid_code;
                    agent.agent.generation = system.generation;
                    tracing::info!("Created diversity mutant via crossover");
                }
            }
        }

        Ok(())
    }

    fn select_round(&mut self, system: &mut MultiAgentSystem) {
        let (mean, std, _) = system.population_fitness_stats();
        
        let to_replace: Vec<_> = system.population.iter()
            .filter(|a| a.fitness < mean - std)
            .map(|a| a.id.clone())
            .collect();

        let best_code = system.get_best().map(|a| a.agent.code.clone());

        for weak_id in to_replace {
            if let Some(best) = best_code.clone() {
                if let Some(agent) = system.population.iter_mut().find(|a| a.id == weak_id) {
                    agent.agent.code = best.clone();
                    agent.agent.generation = system.generation;
                    tracing::info!("Replaced weak agent with best performer");
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

fn role_debug_name(role: &AgentRole) -> String {
    match role {
        AgentRole::Planner => "Planner".to_string(),
        AgentRole::Executor => "Executor".to_string(),
        AgentRole::Evaluator => "Evaluator".to_string(),
        AgentRole::Reflector => "Reflector".to_string(),
        AgentRole::Synthesizer => "Synthesizer".to_string(),
    }
}
