use anyhow::Result;
use async_trait::async_trait;
use rig::client::Client;
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
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;
    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse>;
}

pub struct RigClient {
    client: Client,
    model: String,
}

impl RigClient {
    pub fn new(config: LLMConfig) -> Result<Self> {
        let client = if let Some(base_url) = config.base_url {
            Client::new()
                .with_api_key(&config.api_key)
                .with_base_url(&base_url)?
                .build()
        } else {
            Client::new()
                .with_api_key(&config.api_key)
                .build()
        };

        Ok(Self {
            client,
            model: config.model,
        })
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[async_trait]
impl LLMClient for RigClient {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let completion = self.client
            .model(&self.model)
            .prompt(prompt)
            .await?;

        Ok(LLMResponse {
            content: completion,
            model: self.model.clone(),
            usage: None,
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let completion = self.client
            .model(&self.model)
            .system(system_prompt)
            .user(user_prompt)
            .await?;

        Ok(LLMResponse {
            content: completion,
            model: self.model.clone(),
            usage: None,
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
