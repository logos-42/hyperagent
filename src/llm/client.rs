use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::completion::message::{AssistantContent, Text};
use rig::providers::openai::Client as OpenAiClient;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
    pub max_concurrent: usize,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string()),
            base_url: None,
            max_concurrent: 8,
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

/// Shared LLM client pool. Uses `Arc` to share a single HTTP client (and its
/// connection pool) across all clones, plus a `Semaphore` to cap concurrent
/// requests so we don't overwhelm the API.
#[derive(Clone)]
pub struct RigClient {
    client: Arc<OpenAiClient>,
    model: String,
    semaphore: Arc<Semaphore>,
}

impl RigClient {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        let mut builder = OpenAiClient::builder().api_key(&config.api_key);

        if let Some(base_url) = &config.base_url {
            builder = builder.base_url(base_url);
        }

        let client = Arc::new(builder.build()?);
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

        Ok(Self {
            client,
            model: config.model.clone(),
            semaphore,
        })
    }

    /// Convenience: create from env vars with defaults.
    pub fn from_env() -> Result<Self> {
        Self::new(&LLMConfig::default())
    }
}

fn extract_text(content: &rig::OneOrMany<AssistantContent>) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(Text { text }) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[async_trait]
impl LLMClient for RigClient {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;
        let model = self.client.completion_model(&self.model);
        let request = model
            .completion_request(prompt)
            .send()
            .await?;

        Ok(LLMResponse {
            content: extract_text(&request.choice),
            model: self.model.clone(),
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;
        let model = self.client.completion_model(&self.model);
        let request = model
            .completion_request(user_prompt)
            .preamble(system_prompt.to_string())
            .send()
            .await?;

        Ok(LLMResponse {
            content: extract_text(&request.choice),
            model: self.model.clone(),
        })
    }
}

pub type DynLLMClient = Box<dyn LLMClient>;

pub fn create_llm_client(config: LLMConfig) -> Result<DynLLMClient> {
    Ok(Box::new(RigClient::new(&config)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LLMConfig::default();
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.max_concurrent, 8);
    }

    #[test]
    fn test_pool_shares_client() {
        let config = LLMConfig::default();
        let a = RigClient::new(&config).unwrap();
        let b = a.clone();

        // Both clones point to the same underlying client.
        assert!(Arc::ptr_eq(&a.client, &b.client));
        // Semaphore is also shared.
        assert!(Arc::ptr_eq(&a.semaphore, &b.semaphore));
    }
}
