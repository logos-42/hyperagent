use anyhow::Result;
use async_trait::async_trait;
use rig_core::providers::openai::OpenAi;
use rig_core::completion::Prompt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string()),
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
}

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;
    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse>;
}

pub struct RigClient {
    provider: OpenAi,
    model: String,
}

impl RigClient {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let provider = OpenAi::from_env();
        Ok(Self {
            provider,
            model: config.model,
        })
    }
}

#[async_trait]
impl LLMClient for RigClient {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let completion = self.provider
            .completion(&self.model)
            .prompt(prompt)
            .await?;

        Ok(LLMResponse {
            content: completion,
            model: self.model.clone(),
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let completion = self.provider
            .completion(&self.model)
            .system(system_prompt)
            .user(user_prompt)
            .await?;

        Ok(LLMResponse {
            content: completion,
            model: self.model.clone(),
        })
    }
}

pub type DynLLMClient = Box<dyn LLMClient>;

pub fn create_llm_client(config: LLMConfig) -> Result<DynLLMClient> {
    Ok(Box::new(RigClient::new(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LLMConfig::default();
        assert_eq!(config.model, "gpt-4o");
    }
}
