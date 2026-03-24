use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::Agent;
use crate::llm::LLMClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub agent_id: String,
    pub task: String,
    pub output: String,
    pub reasoning: Option<String>,
    pub success: bool,
    pub error: Option<String>,
}

pub struct Executor<C: LLMClient> {
    client: C,
}

impl<C: LLMClient> Executor<C> {
    pub fn new(client: C) -> Self {
        Self { client }
    }

    pub async fn run(&self, agent: &Agent, task: &str) -> Result<ExecutionResult> {
        let prompt = format!(
            "{}\n\nTask: {}",
            agent.prompt, task
        );

        let response = self.client.complete(&prompt).await?;

        Ok(ExecutionResult {
            agent_id: agent.id.clone(),
            task: task.to_string(),
            output: response.content,
            reasoning: None,
            success: true,
            error: None,
        })
    }

    pub async fn run_with_context(
        &self,
        agent: &Agent,
        task: &str,
        context: &str,
    ) -> Result<ExecutionResult> {
        let prompt = format!(
            "{}\n\nContext:\n{}\n\nTask: {}",
            agent.prompt, context, task
        );

        let response = self.client.complete(&prompt).await?;

        Ok(ExecutionResult {
            agent_id: agent.id.clone(),
            task: task.to_string(),
            output: response.content,
            reasoning: None,
            success: true,
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_creation() {
        // This would need a mock client
    }
}
