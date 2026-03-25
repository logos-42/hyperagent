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

            let gen_score = eval_result.score.value;

            self.state.archive.store(
                current_agent.clone(),
                eval_result.score,
                task.to_string(),
                result.output.clone(),
            );

            self.state.lineage.add(
                &current_agent,
                Some(&current_agent.id),
                gen_score,
            );

            self.state.update_best(current_agent.clone(), gen_score);

            self.state.set_phase(RuntimePhase::Mutating);
            let failures = self.state.archive.get_failures_text();
            let failures_vec: Vec<String> = if failures.is_empty() {
                vec![]
            } else {
                failures.lines().map(|s| s.to_string()).collect()
            };
            
            let pre_mutation_agent = current_agent.clone();
            current_agent = match self.mutator.mutate(&current_agent, &failures_vec).await {
                Ok(agent) => agent,
                Err(e) => {
                    tracing::error!("Generation {}: Mutation failed: {}", self.state.current_generation, e);
                    self.state.add_error(e.to_string());
                    current_agent = pre_mutation_agent;
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

            // Evaluate the mutated agent for informed selection
            self.state.set_phase(RuntimePhase::Selecting);
            let mutant_result = match self.executor.run(&current_agent, task).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Generation {}: Mutant evaluation failed, keeping previous agent: {}", self.state.current_generation, e);
                    current_agent = pre_mutation_agent;
                    continue;
                }
            };

            let mutant_score = match self.evaluator.score(task, &mutant_result).await {
                Ok(e) => {
                    tracing::info!("Generation {}: Mutant score = {:.2}", self.state.current_generation, e.score.value);
                    e.score.value
                }
                Err(e) => {
                    tracing::warn!("Generation {}: Mutant scoring failed, keeping previous agent: {}", self.state.current_generation, e);
                    current_agent = pre_mutation_agent;
                    continue;
                }
            };

            if mutant_score < gen_score {
                tracing::info!(
                    "Generation {}: Reverting to previous agent ({:.2} > {:.2})",
                    self.state.current_generation, gen_score, mutant_score
                );
                current_agent = pre_mutation_agent;
            } else {
                tracing::info!(
                    "Generation {}: Keeping mutant ({:.2} >= {:.2})",
                    self.state.current_generation, mutant_score, gen_score
                );
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
                    tracing::error!("Generation {}: Execution failed: {}", self.state.current_generation, e);
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            self.state.set_phase(RuntimePhase::Evaluating);
            let eval_result = match self.evaluator.score(task, &result).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!("Generation {}: Evaluation failed: {}", self.state.current_generation, e);
                    self.state.add_error(e.to_string());
                    continue;
                }
            };

            tracing::info!("Generation {}: Score = {:.2}", self.state.current_generation, eval_result.score.value);

            let gen_score = eval_result.score.value;

            self.state.archive.store(
                current_agent.clone(),
                eval_result.score,
                task.to_string(),
                result.output.clone(),
            );

            self.state.lineage.add(
                &current_agent,
                Some(&current_agent.id),
                gen_score,
            );

            self.state.update_best(current_agent.clone(), gen_score);

            self.state.set_phase(RuntimePhase::Mutating);
            let failures = self.state.archive.get_failures_text();
            let failures_vec: Vec<String> = if failures.is_empty() {
                vec![]
            } else {
                failures.lines().map(|s| s.to_string()).collect()
            };

            let pre_mutation_agent = current_agent.clone();
            current_agent = match self.mutator.mutate(&current_agent, &failures_vec).await {
                Ok(agent) => agent,
                Err(e) => {
                    tracing::error!("Generation {}: Mutation failed: {}", self.state.current_generation, e);
                    self.state.add_error(e.to_string());
                    current_agent = pre_mutation_agent;
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

            // Evaluate the mutated agent to make an informed selection
            self.state.set_phase(RuntimePhase::Selecting);
            let mutant_result = match self.executor.run(&current_agent, task).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Generation {}: Mutant evaluation failed, keeping previous agent: {}", self.state.current_generation, e);
                    current_agent = pre_mutation_agent;
                    continue;
                }
            };

            let mutant_score = match self.evaluator.score(task, &mutant_result).await {
                Ok(e) => {
                    tracing::info!("Generation {}: Mutant score = {:.2}", self.state.current_generation, e.score.value);
                    e.score.value
                }
                Err(e) => {
                    tracing::warn!("Generation {}: Mutant scoring failed, keeping previous agent: {}", self.state.current_generation, e);
                    current_agent = pre_mutation_agent;
                    continue;
                }
            };

            // Elitist selection: keep the better agent between pre-mutation and mutant
            if mutant_score < gen_score {
                tracing::info!(
                    "Generation {}: Reverting to previous agent ({:.2} > {:.2})",
                    self.state.current_generation, gen_score, mutant_score
                );
                current_agent = pre_mutation_agent;
            } else {
                tracing::info!(
                    "Generation {}: Keeping mutant ({:.2} >= {:.2})",
                    self.state.current_generation, mutant_score, gen_score
                );
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
