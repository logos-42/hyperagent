use serde::{Deserialize, Serialize};

use crate::agent::{Agent, MutationStrategy};
use crate::memory::{Archive, Lineage};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RuntimePhase {
    Initializing,
    Executing,
    Evaluating,
    Mutating,
    MetaMutating,
    Selecting,
    Checkpointing,
    Finished,
    Error,
}

impl Default for RuntimePhase {
    fn default() -> Self {
        Self::Initializing
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub max_generations: u32,
    pub population_size: usize,
    pub top_k_selection: usize,
    pub checkpoint_interval: u32,
    pub meta_mutation_interval: u32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_generations: 100,
            population_size: 5,
            top_k_selection: 3,
            checkpoint_interval: 10,
            meta_mutation_interval: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    pub phase: RuntimePhase,
    pub current_generation: u32,
    pub current_task: Option<String>,
    pub best_agent: Option<Agent>,
    pub best_score: f32,
    pub config: RuntimeConfig,
    pub archive: Archive,
    pub lineage: Lineage,
    pub mutation_strategy: MutationStrategy,
    pub errors: Vec<String>,
}

impl RuntimeState {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            phase: RuntimePhase::Initializing,
            current_generation: 0,
            current_task: None,
            best_agent: None,
            best_score: 0.0,
            config,
            archive: Archive::new(),
            lineage: Lineage::new(),
            mutation_strategy: MutationStrategy::default(),
            errors: Vec::new(),
        }
    }

    pub fn set_phase(&mut self, phase: RuntimePhase) {
        self.phase = phase;
    }

    pub fn increment_generation(&mut self) {
        self.current_generation += 1;
    }

    pub fn update_best(&mut self, agent: Agent, score: f32) {
        if score > self.best_score {
            self.best_score = score;
            self.best_agent = Some(agent);
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn is_finished(&self) -> bool {
        self.phase == RuntimePhase::Finished
            || self.current_generation >= self.config.max_generations
    }

    pub fn should_meta_mutate(&self) -> bool {
        self.current_generation > 0
            && self.current_generation % self.config.meta_mutation_interval == 0
    }

    pub fn should_checkpoint(&self) -> bool {
        self.current_generation > 0
            && self.current_generation % self.config.checkpoint_interval == 0
    }

    pub fn summary(&self) -> String {
        format!(
            "Generation: {}/{}, Best Score: {:.2}, Phase: {:?}, Errors: {}",
            self.current_generation,
            self.config.max_generations,
            self.best_score,
            self.phase,
            self.errors.len()
        )
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new(RuntimeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_state_creation() {
        let state = RuntimeState::default();
        assert_eq!(state.phase, RuntimePhase::Initializing);
        assert_eq!(state.current_generation, 0);
    }

    #[test]
    fn test_update_best() {
        let mut state = RuntimeState::default();
        let agent = Agent::new("code".to_string(), "prompt".to_string());

        state.update_best(agent.clone(), 8.5);

        assert_eq!(state.best_score, 8.5);
        assert!(state.best_agent.is_some());
    }

    #[test]
    fn test_should_meta_mutate() {
        let config = RuntimeConfig {
            meta_mutation_interval: 10,
            ..Default::default()
        };
        let mut state = RuntimeState::new(config);

        state.current_generation = 10;
        assert!(state.should_meta_mutate());

        state.current_generation = 5;
        assert!(!state.should_meta_mutate());
    }
}
