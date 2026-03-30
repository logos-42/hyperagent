pub mod executor;
pub mod mutator;
pub mod meta_mutator;
pub mod population;

pub use executor::Executor;
pub use mutator::Mutator;
pub use meta_mutator::MetaMutator;
pub use population::{MultiAgentSystem, PopulationConfig, PopulationAgent, AgentRole, AgentMessage};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub code: String,
    pub prompt: String,
    pub generation: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Agent {
    pub fn new(code: String, prompt: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            code,
            prompt,
            generation: 0,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn with_generation(mut self, generation: u32) -> Self {
        self.generation = generation;
        self
    }

    pub fn with_prompt(mut self, prompt: String) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_code(mut self, code: String) -> Self {
        self.code = code;
        self
    }

    pub fn from_prompt(prompt: String) -> Self {
        Self::new(String::new(), prompt)
    }

    pub fn evolve(&mut self, new_code: String) -> &mut Self {
        self.code = new_code;
        self.generation += 1;
        self
    }

    /// Creates a new Agent with updated code and incremented generation.
    /// This is a functional alternative to `evolve` that returns a new instance.
    pub fn evolve_with(&self, new_code: String) -> Self {
        Self {
            id: self.id.clone(),
            code: new_code,
            prompt: self.prompt.clone(),
            generation: self.generation + 1,
            created_at: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationStrategy {
    pub id: String,
    pub prompt: String,
    pub version: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Default for MutationStrategy {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: r#"You are an AI system that improves agents.

Current agent:
{agent_code}

Past failures:
{failures}

Goal:
Generate a modified version of the agent that performs better.

Rules:
- You may change structure, tools, or reasoning strategy
- Keep it minimal but effective

Return:
NEW_AGENT_CODE"#.to_string(),
            version: 1,
            created_at: chrono::Utc::now(),
        }
    }
}

impl MutationStrategy {
    pub fn new(prompt: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            prompt,
            version: 1,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn evolve(&mut self, new_prompt: String) {
        self.version += 1;
        self.prompt = new_prompt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let agent = Agent::new("code".to_string(), "prompt".to_string());
        assert!(!agent.id.is_empty());
        assert_eq!(agent.code, "code");
    }

    #[test]
    fn test_mutation_strategy_default() {
        let strategy = MutationStrategy::default();
        assert!(!strategy.prompt.is_empty());
        assert_eq!(strategy.version, 1);
    }

    #[test]
    fn test_agent_with_prompt() {
        let agent = Agent::new("code".to_string(), "old".to_string())
            .with_prompt("new prompt".to_string());
        assert_eq!(agent.prompt, "new prompt");
    }

    #[test]
    fn test_agent_evolve() {
        let mut agent = Agent::new("old code".to_string(), "prompt".to_string());
        agent.evolve("new code".to_string());
        assert_eq!(agent.code, "new code");
        assert_eq!(agent.generation, 1);
        
        agent.evolve("even newer code".to_string());
        assert_eq!(agent.code, "even newer code");
        assert_eq!(agent.generation, 2);
    }

    #[test]
    fn test_agent_with_code() {
        let agent = Agent::new("old".to_string(), "prompt".to_string())
            .with_code("new code".to_string());
        assert_eq!(agent.code, "new code");
        assert_eq!(agent.generation, 0);
    }

    #[test]
    fn test_agent_evolve_with() {
        let original = Agent::new("old code".to_string(), "prompt".to_string());
        let evolved = original.evolve_with("new code".to_string());
        
        // Original unchanged
        assert_eq!(original.code, "old code");
        assert_eq!(original.generation, 0);
        
        // Evolved has new code and incremented generation
        assert_eq!(evolved.code, "new code");
        assert_eq!(evolved.generation, 1);
        assert_eq!(evolved.id, original.id);
        assert_eq!(evolved.prompt, original.prompt);
    }

    #[test]
    fn test_agent_evolve_with_chain() {
        let agent = Agent::new("v1".to_string(), "prompt".to_string());
        let v2 = agent.evolve_with("v2".to_string());
        let v3 = v2.evolve_with("v3".to_string());
        
        assert_eq!(agent.generation, 0);
        assert_eq!(v2.generation, 1);
        assert_eq!(v3.generation, 2);
    }
}