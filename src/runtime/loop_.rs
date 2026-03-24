use anyhow::Result;

use crate::agent::{Executor, MetaMutator, Mutator};
use crate::eval::Evaluator;
use crate::llm::LLMClient;

use super::state::{RuntimeConfig, RuntimePhase, RuntimeState};

pub struct EvolutionLoop<C: LLMClient> {
    executor: Executor<C>,
    evaluator: Evaluator<C>,
    mutator: Mutator<C>,
    meta_mutator: MetaMutator<C>,
    state: RuntimeState,
}

impl<C: LLMClient + Clone> EvolutionLoop<C> {
    pub fn new(
        client: C,
        config: RuntimeConfig,
    ) -> Self {
        let executor = Executor::new(client.clone());
        let evaluator = Evaluator::new(client.clone());
        let mutator = Mutator::new(client.clone());
        let meta_mutator = MetaMutator::new(client);

        Self {
            executor,
            evaluator,
            mutator,
            meta_mutator,
            state: RuntimeState::new(config),
        }
    }

    pub async fn run(&mut self, task: &str) -> Result<RuntimeState> {
        self.state.set_phase(RuntimePhase::Executing);
        self.state.current_task = Some(task.to_string());

        let initial_agent = crate::agent::Agent::from_prompt(
            "You are an autonomous task-solving agent. Be concise and correct.".to_string()
        );

        let mut current_agent = initial_agent;

        while !self.state.is_finished() {
            self.state.increment_generation();
            tracing::info!(
                "Generation {}: {:?}",
                self.state.current_generation,
                self.state.phase
            );

            self.state.set_phase(RuntimePhase::Executing);
            let result = match self.executor.run(&current_agent, task).await {
                Ok(r) => r,
                Err(e) => {
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            self.state.set_phase(RuntimePhase::Evaluating);
            let eval_result = match self.evaluator.score(task, &result).await {
                Ok(e) => e,
                Err(e) => {
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            self.state.archive.store(
                current_agent.clone(),
                eval_result.score,
                task.to_string(),
                result.output.clone(),
            );

            self.state.lineage.add(
                &current_agent,
                Some(&current_agent.id),
                eval_result.score.value,
            );

            self.state.update_best(current_agent.clone(), eval_result.score.value);

            self.state.set_phase(RuntimePhase::Mutating);
            let failures = self.state.archive.get_failures_text();
            let failures_vec: Vec<String> = if failures.is_empty() {
                vec![]
            } else {
                failures.lines().map(|s| s.to_string()).collect()
            };
            
            current_agent = match self.mutator.mutate(&current_agent, &failures_vec).await {
                Ok(agent) => agent,
                Err(e) => {
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            if self.state.should_meta_mutate() {
                self.state.set_phase(RuntimePhase::MetaMutating);
                let history = format!(
                    "Generations: {}, Best Score: {:.2}",
                    self.state.current_generation,
                    self.state.best_score
                );
                
                if let Ok(new_strategy) = self.meta_mutator.evolve(&history).await {
                    self.state.mutation_strategy = new_strategy.clone();
                    self.mutator.update_strategy(new_strategy.prompt.clone());
                }
            }

            self.state.set_phase(RuntimePhase::Selecting);
            if let Some(best) = self.state.archive.get_best() {
                if best.score.value > current_agent.generation as f32 {
                    current_agent = best.agent.clone();
                }
            }
        }

        self.state.set_phase(RuntimePhase::Finished);
        Ok(self.state.clone())
    }

    pub async fn run_with_iterations(&mut self, task: &str, iterations: usize) -> Result<RuntimeState> {
        self.state.set_phase(RuntimePhase::Executing);
        self.state.current_task = Some(task.to_string());

        let initial_agent = crate::agent::Agent::from_prompt(
            "You are an autonomous task-solving agent. Be concise and correct.".to_string()
        );

        let mut current_agent = initial_agent;

        for _ in 0..iterations {
            if self.state.is_finished() {
                break;
            }

            self.state.increment_generation();
            tracing::info!(
                "Generation {}: {:?}",
                self.state.current_generation,
                self.state.phase
            );

            self.state.set_phase(RuntimePhase::Executing);
            let result = match self.executor.run(&current_agent, task).await {
                Ok(r) => r,
                Err(e) => {
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            self.state.set_phase(RuntimePhase::Evaluating);
            let eval_result = match self.evaluator.score(task, &result).await {
                Ok(e) => e,
                Err(e) => {
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            self.state.archive.store(
                current_agent.clone(),
                eval_result.score,
                task.to_string(),
                result.output.clone(),
            );

            self.state.lineage.add(
                &current_agent,
                Some(&current_agent.id),
                eval_result.score.value,
            );

            self.state.update_best(current_agent.clone(), eval_result.score.value);

            self.state.set_phase(RuntimePhase::Mutating);
            let failures = self.state.archive.get_failures_text();
            let failures_vec: Vec<String> = if failures.is_empty() {
                vec![]
            } else {
                failures.lines().map(|s| s.to_string()).collect()
            };

            current_agent = match self.mutator.mutate(&current_agent, &failures_vec).await {
                Ok(agent) => agent,
                Err(e) => {
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            if self.state.should_meta_mutate() {
                self.state.set_phase(RuntimePhase::MetaMutating);
                let history = format!(
                    "Generations: {}, Best Score: {:.2}",
                    self.state.current_generation,
                    self.state.best_score
                );
                
                if let Ok(new_strategy) = self.meta_mutator.evolve(&history).await {
                    self.state.mutation_strategy = new_strategy.clone();
                    self.mutator.update_strategy(new_strategy.prompt.clone());
                }
            }

            self.state.set_phase(RuntimePhase::Selecting);
            if let Some(best) = self.state.archive.get_best() {
                if best.score.value > current_agent.generation as f32 {
                    current_agent = best.agent.clone();
                }
            }
        }

        self.state.set_phase(RuntimePhase::Finished);
        Ok(self.state.clone())
    }

    pub fn get_state(&self) -> &RuntimeState {
        &self.state
    }

    pub fn get_best_agent(&self) -> Option<&crate::agent::Agent> {
        self.state.best_agent.as_ref()
    }
}
