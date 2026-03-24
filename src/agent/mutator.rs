use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Agent, MutationStrategy};
use crate::llm::{LLMClient, PromptManager};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationHistory {
    pub agent_id: String,
    pub parent_id: String,
    pub strategy_version: u32,
    pub score_before: f32,
    pub score_after: f32,
    pub success: bool,
}

pub struct Mutator<C: LLMClient> {
    client: C,
    strategy: MutationStrategy,
}

impl<C: LLMClient> Mutator<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            strategy: MutationStrategy::default(),
        }
    }

    pub fn with_strategy(client: C, strategy: MutationStrategy) -> Self {
        Self { client, strategy }
    }

    pub fn get_strategy(&self) -> &MutationStrategy {
        &self.strategy
    }

    pub async fn mutate(
        &self,
        agent: &Agent,
        failures: &[String],
    ) -> Result<Agent> {
        let failures_str = if failures.is_empty() {
            "No failures recorded".to_string()
        } else {
            failures.join("\n")
        };

        let prompt = self.strategy.prompt
            .replace("{agent_code}", &agent.code)
            .replace("{failures}", &failures_str);

        let response = self.client.complete(&prompt).await?;

        let mut new_agent = Agent::new(
            response.content.clone(),
            agent.prompt.clone(),
        );
        new_agent.generation = agent.generation + 1;

        Ok(new_agent)
    }

    pub fn update_strategy(&mut self, new_prompt: String) {
        self.strategy.evolve(new_prompt);
    }
}

#[async_trait]
pub trait AgentMutator: Send + Sync {
    async fn mutate(&self, agent: &Agent, failures: &[String]) -> Result<Agent>;
    fn get_strategy(&self) -> &MutationStrategy;
    fn update_strategy(&mut self, new_prompt: String);
}

impl<C: LLMClient> AgentMutator for Mutator<C> {
    async fn mutate(&self, agent: &Agent, failures: &[String]) -> Result<Agent> {
        self.mutate(agent, failures).await
    }

    fn get_strategy(&self) -> &MutationStrategy {
        &self.strategy
    }

    fn update_strategy(&mut self, new_prompt: String) {
        self.strategy.evolve(new_prompt);
    }
}

pub struct PopulationMutator<C: LLMClient> {
    client: C,
    population_size: usize,
}

impl<C: LLMClient> PopulationMutator<C> {
    pub fn new(client: C, population_size: usize) -> Self {
        Self {
            client,
            population_size,
        }
    }

    pub async fn mutate_population(
        &self,
        agents: &[Agent],
        failures: &[String],
    ) -> Result<Vec<Agent>> {
        let mut new_agents = Vec::new();

        for agent in agents {
            let mutator = Mutator::new(self.client.clone());
            match mutator.mutate(agent, failures).await {
                Ok(new_agent) => new_agents.push(new_agent),
                Err(e) => {
                    tracing::warn!("Failed to mutate agent {}: {}", agent.id, e);
                }
            }
        }

        Ok(new_agents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_strategy() {
        let strategy = MutationStrategy::default();
        assert!(!strategy.prompt.is_empty());
    }
}
