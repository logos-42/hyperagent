use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::llm::LLMClient;

use super::MutationStrategy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionHistory {
    pub generation: u32,
    pub strategy_version: u32,
    pub mutation_prompt: String,
    pub improvement_rate: f32,
    pub notes: String,
}

pub struct MetaMutator<C: LLMClient> {
    client: C,
    current_strategy: MutationStrategy,
    history: Vec<EvolutionHistory>,
    generation: u32,
}

impl<C: LLMClient> MetaMutator<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            current_strategy: MutationStrategy::default(),
            history: Vec::new(),
            generation: 0,
        }
    }

    pub fn with_strategy(client: C, strategy: MutationStrategy) -> Self {
        Self {
            client,
            current_strategy: strategy,
            history: Vec::new(),
            generation: 0,
        }
    }

    pub async fn evolve(&mut self, history: &str) -> Result<MutationStrategy> {
        self.generation += 1;

        let prompt = format!(
            r#"You are a meta-learning system.

Current mutation strategy:
{}

History of improvements:
{}

Problem:
The current mutation strategy is not improving fast enough.

Goal:
Modify the mutation strategy itself.

Think:
- Are we exploring enough?
- Are we exploiting too early?
- Are we missing structural changes?

Return ONLY the new mutation strategy prompt, nothing else."#,
            self.current_strategy.prompt,
            history
        );

        let response = self.client.complete(&prompt).await?;

        let new_strategy = MutationStrategy::new(response.content.clone());
        let old_version = self.current_strategy.version;

        self.history.push(EvolutionHistory {
            generation: self.generation,
            strategy_version: old_version,
            mutation_prompt: self.current_strategy.prompt.clone(),
            improvement_rate: 0.0,
            notes: String::new(),
        });

        self.current_strategy = new_strategy;

        Ok(self.current_strategy.clone())
    }

    pub fn get_strategy(&self) -> &MutationStrategy {
        &self.current_strategy
    }

    pub fn get_history(&self) -> &[EvolutionHistory] {
        &self.history
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn set_strategy(&mut self, strategy: MutationStrategy) {
        self.current_strategy = strategy;
    }
}

#[async_trait]
pub trait MetaEvolutionary: Send + Sync {
    async fn evolve(&mut self, history: &str) -> Result<MutationStrategy>;
    fn get_strategy(&self) -> &MutationStrategy;
    fn get_history(&self) -> &[EvolutionHistory];
    fn generation(&self) -> u32;
}

impl<C: LLMClient> MetaEvolutionary for MetaMutator<C> {
    async fn evolve(&mut self, history: &str) -> Result<MutationStrategy> {
        self.evolve(history).await
    }

    fn get_strategy(&self) -> &MutationStrategy {
        &self.current_strategy
    }

    fn get_history(&self) -> &[EvolutionHistory] {
        &self.history
    }

    fn generation(&self) -> u32 {
        self.generation
    }
}

pub struct EnsembleMetaMutator<C: LLMClient> {
    mutators: Vec<MetaMutator<C>>,
    strategy: MutationStrategy,
}

impl<C: LLMClient> EnsembleMetaMutator<C> {
    pub fn new(clients: Vec<C>) -> Self {
        let mutators = clients
            .into_iter()
            .map(MetaMutator::new)
            .collect();

        Self {
            mutators,
            strategy: MutationStrategy::default(),
        }
    }

    pub async fn evolve(&mut self, history: &str) -> Result<MutationStrategy> {
        let mut new_strategies = Vec::new();

        for mutator in &mut self.mutators {
            match mutator.evolve(history).await {
                Ok(strategy) => new_strategies.push(strategy),
                Err(e) => {
                    tracing::warn!("Mutator evolution failed: {}", e);
                }
            }
        }

        if new_strategies.is_empty() {
            return Ok(self.strategy.clone());
        }

        let best = new_strategies.into_iter().next().unwrap();
        self.strategy = best;

        Ok(self.strategy.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_mutator_creation() {
        // Would need mock client
    }
}
