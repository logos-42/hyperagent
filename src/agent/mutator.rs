use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{Agent, MutationStrategy};
use crate::llm::LLMClient;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_strategy() {
        let strategy = MutationStrategy::default();
        assert!(!strategy.prompt.is_empty());
    }
}
