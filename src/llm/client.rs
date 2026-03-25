use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use http::{HeaderMap, HeaderValue};
use rig::{
    client::CompletionClient,
    completion::Prompt,
    providers::openai::Client as OpenAiClient,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

/// LLM Provider enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LLMProvider {
    OpenAI,
    Ollama,
    GLM,
    MiniMax,
    Qwen,
}

impl Default for LLMProvider {
    fn default() -> Self {
        LLMProvider::OpenAI
    }
}

impl std::fmt::Display for LLMProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMProvider::OpenAI => write!(f, "openai"),
            LLMProvider::Ollama => write!(f, "ollama"),
            LLMProvider::GLM => write!(f, "glm"),
            LLMProvider::MiniMax => write!(f, "minimax"),
            LLMProvider::Qwen => write!(f, "qwen"),
        }
    }
}

/// LLM Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub model: String,
    pub api_key: String,
    pub base_url: Option<String>,
    pub max_concurrent: usize,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::OpenAI,
            model: "gpt-4o".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".to_string()),
            base_url: None,
            max_concurrent: 8,
            temperature: Some(0.7),
            max_tokens: Some(2000),
        }
    }
}

impl LLMConfig {
    pub fn ollama(model: &str, base_url: Option<&str>) -> Self {
        Self {
            provider: LLMProvider::Ollama,
            model: model.to_string(),
            api_key: String::new(),
            base_url: base_url.map(|s| s.to_string()),
            max_concurrent: 4,
            temperature: Some(0.7),
            max_tokens: Some(2000),
        }
    }

    pub fn openai(model: &str, api_key: &str) -> Self {
        Self {
            provider: LLMProvider::OpenAI,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: None,
            max_concurrent: 8,
            temperature: Some(0.7),
            max_tokens: Some(2000),
        }
    }

    pub fn qwen(model: &str, api_key: &str) -> Self {
        Self {
            provider: LLMProvider::Qwen,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: None,
            max_concurrent: 8,
            temperature: Some(0.7),
            max_tokens: Some(2000),
        }
    }

    pub fn glm(model: &str, api_key: &str) -> Self {
        Self {
            provider: LLMProvider::GLM,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: None,
            max_concurrent: 8,
            temperature: Some(0.7),
            max_tokens: Some(2000),
        }
    }

    pub fn minimax(model: &str, api_key: &str) -> Self {
        Self {
            provider: LLMProvider::MiniMax,
            model: model.to_string(),
            api_key: api_key.to_string(),
            base_url: None,
            max_concurrent: 8,
            temperature: Some(0.7),
            max_tokens: Some(2000),
        }
    }
}

/// LLM Response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub provider: String,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// LLM Client trait
#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse>;
    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse>;
    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse>;
}

/// Message structure for multi-turn conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

// ---------------------------------------------------------------------------
// rig-powered backends (enum dispatch over non-object-safe agent types)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum InnerBackend {
    OpenAI(Arc<OpenAiClient>, String),
    Ollama(Arc<rig::providers::ollama::Client>, String),
}

/// Unified backend powered by rig providers.
///
/// - OpenAI-compatible providers (OpenAI, GLM, MiniMax, Qwen): rig's OpenAI client + base_url
/// - Ollama (local + cloud): rig's native Ollama client + optional http_headers
#[derive(Clone)]
struct RigBackend {
    inner: InnerBackend,
    model_name: String,
    provider_name: String,
    semaphore: Arc<Semaphore>,
    #[allow(dead_code)]
    temperature: Option<f32>,
    #[allow(dead_code)]
    max_tokens: Option<i32>,
}

impl RigBackend {
    async fn complete_prompt(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;
        let content = match &self.inner {
            InnerBackend::OpenAI(client, model) => {
                client.agent(model.as_str()).build().prompt(prompt).await?
            }
            InnerBackend::Ollama(client, model) => {
                client.agent(model.as_str()).build().prompt(prompt).await?
            }
        };
        Ok(self.make_response(content))
    }

    async fn complete_with_preamble(&self, system: &str, user: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;
        let content = match &self.inner {
            InnerBackend::OpenAI(client, model) => {
                client.agent(model.as_str())
                    .preamble(system)
                    .build()
                    .prompt(user)
                    .await?
            }
            InnerBackend::Ollama(client, model) => {
                client.agent(model.as_str())
                    .preamble(system)
                    .build()
                    .prompt(user)
                    .await?
            }
        };
        Ok(self.make_response(content))
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut preamble: Option<String> = None;
        let mut user_content = String::new();

        for msg in &messages {
            match msg.role {
                MessageRole::System => {
                    preamble = Some(msg.content.clone());
                }
                MessageRole::User => {
                    user_content.push_str(&msg.content);
                }
                MessageRole::Assistant => {
                    user_content.push_str(&format!("\nAssistant: {}", msg.content));
                }
            }
        }

        let content = match &self.inner {
            InnerBackend::OpenAI(client, model) => {
                let mut builder = client.agent(model.as_str());
                if let Some(p) = &preamble {
                    builder = builder.preamble(p);
                }
                builder.build().prompt(&user_content).await?
            }
            InnerBackend::Ollama(client, model) => {
                let mut builder = client.agent(model.as_str());
                if let Some(p) = &preamble {
                    builder = builder.preamble(p);
                }
                builder.build().prompt(&user_content).await?
            }
        };
        Ok(self.make_response(content))
    }

    fn make_response(&self, content: String) -> LLMResponse {
        LLMResponse {
            content,
            model: self.model_name.clone(),
            provider: self.provider_name.clone(),
            usage: None,
        }
    }
}

/// Main LLM Client that wraps rig backends
#[derive(Clone)]
pub struct LLMClientImpl {
    backend: RigBackend,
    provider: LLMProvider,
}

impl LLMClientImpl {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        let (inner, provider_name) = match config.provider {
            LLMProvider::OpenAI => {
                let client = build_openai_client(config.api_key.as_str(), config.base_url.as_deref())?;
                (InnerBackend::OpenAI(client, config.model.clone()), "openai".to_string())
            }
            LLMProvider::Ollama => {
                let client = build_ollama_client(config.base_url.as_deref(), &config.api_key)?;
                (InnerBackend::Ollama(client, config.model.clone()), "ollama".to_string())
            }
            LLMProvider::GLM => {
                let base_url = config.base_url.as_deref()
                    .unwrap_or("https://open.bigmodel.cn/api/paas/v4");
                let client = build_openai_client(config.api_key.as_str(), Some(base_url))?;
                (InnerBackend::OpenAI(client, config.model.clone()), "glm".to_string())
            }
            LLMProvider::MiniMax => {
                let base_url = config.base_url.as_deref()
                    .unwrap_or("https://api.minimax.chat/v1");
                let client = build_openai_client(config.api_key.as_str(), Some(base_url))?;
                (InnerBackend::OpenAI(client, config.model.clone()), "minimax".to_string())
            }
            LLMProvider::Qwen => {
                let base_url = config.base_url.as_deref()
                    .unwrap_or("https://dashscope.aliyuncs.com/compatible-mode/v1");
                let client = build_openai_client(config.api_key.as_str(), Some(base_url))?;
                (InnerBackend::OpenAI(client, config.model.clone()), "qwen".to_string())
            }
        };

        Ok(Self {
            backend: RigBackend {
                inner,
                model_name: config.model.clone(),
                provider_name,
                semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                temperature: config.temperature,
                max_tokens: config.max_tokens,
            },
            provider: config.provider.clone(),
        })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self> {
        let provider_str = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());
        let config = match provider_str.to_lowercase().as_str() {
            "ollama" => {
                let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama2".to_string());
                let base_url = std::env::var("OLLAMA_BASE_URL").ok();
                let api_key = std::env::var("OLLAMA_API_KEY").unwrap_or_default();
                let mut cfg = LLMConfig::ollama(&model, base_url.as_deref());
                cfg.api_key = api_key;
                cfg
            }
            "openai" => {
                let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
                let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
                LLMConfig::openai(&model, &api_key)
            }
            "glm" => {
                let model = std::env::var("GLM_MODEL").unwrap_or_else(|_| "glm-4".to_string());
                let api_key = std::env::var("GLM_API_KEY").unwrap_or_default();
                let base_url = std::env::var("GLM_BASE_URL").ok();
                let mut cfg = LLMConfig::glm(&model, &api_key);
                cfg.base_url = base_url;
                cfg
            }
            "minimax" => {
                let model = std::env::var("MINIMAX_MODEL").unwrap_or_else(|_| "abab6.5s-chat".to_string());
                let api_key = std::env::var("MINIMAX_API_KEY").unwrap_or_default();
                let base_url = std::env::var("MINIMAX_BASE_URL").ok();
                let mut cfg = LLMConfig::minimax(&model, &api_key);
                cfg.base_url = base_url;
                cfg
            }
            "qwen" => {
                let model = std::env::var("QWEN_MODEL").unwrap_or_else(|_| "qwen-turbo".to_string());
                let api_key = std::env::var("QWEN_API_KEY").unwrap_or_default();
                let base_url = std::env::var("QWEN_BASE_URL").ok();
                let mut cfg = LLMConfig::qwen(&model, &api_key);
                cfg.base_url = base_url;
                cfg
            }
            other => anyhow::bail!("Unknown LLM provider: {}", other),
        };
        Self::new(&config)
    }

    pub fn provider(&self) -> &LLMProvider {
        &self.provider
    }

    pub fn model(&self) -> &str {
        &self.backend.model_name
    }
}

#[async_trait]
impl LLMClient for LLMClientImpl {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        self.backend.complete_prompt(prompt).await
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        self.backend.complete_with_preamble(system_prompt, user_prompt).await
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        self.backend.complete_with_messages(messages).await
    }
}

// ---------------------------------------------------------------------------
// rig client builders
// ---------------------------------------------------------------------------

/// Build an OpenAI-compatible client (covers OpenAI, GLM, MiniMax, Qwen).
fn build_openai_client(
    api_key: &str,
    base_url: Option<&str>,
) -> Result<Arc<OpenAiClient>> {
    let mut builder = OpenAiClient::builder().api_key(api_key);

    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }

    Ok(Arc::new(builder.build()?))
}

/// Build an Ollama client (local or cloud).
///
/// Note: We set `NO_PROXY=localhost,127.0.0.1` so that rig's reqwest 0.13
/// (which auto-detects system proxy) doesn't route local Ollama requests
/// through a proxy and get 502 errors.
fn build_ollama_client(
    base_url: Option<&str>,
    api_key: &str,
) -> Result<Arc<rig::providers::ollama::Client>> {
    use rig::client::Nothing;

    // Ensure local Ollama traffic bypasses any system proxy
    if std::env::var("NO_PROXY").is_err() {
        std::env::set_var("NO_PROXY", "localhost,127.0.0.1");
    }

    let mut builder = rig::providers::ollama::Client::builder().api_key(Nothing);

    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }

    if !api_key.is_empty() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| anyhow::anyhow!("Invalid API key: {}", e))?,
        );
        builder = builder.http_headers(headers);
    }

    Ok(Arc::new(builder.build()?))
}

// Type aliases for convenience
pub type DynLLMClient = Box<dyn LLMClient>;

/// Create an LLM client from config
pub fn create_llm_client(config: LLMConfig) -> Result<DynLLMClient> {
    Ok(Box::new(LLMClientImpl::new(&config)?))
}

/// Switch provider at runtime
pub fn switch_provider(_client: &LLMClientImpl, new_config: LLMConfig) -> Result<LLMClientImpl> {
    LLMClientImpl::new(&new_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LLMConfig::default();
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_ollama_config() {
        let config = LLMConfig::ollama("llama2", Some("http://localhost:11434"));
        assert_eq!(config.model, "llama2");
        assert_eq!(config.provider, LLMProvider::Ollama);
        assert_eq!(config.base_url, Some("http://localhost:11434".to_string()));
    }

    #[test]
    fn test_provider_display() {
        assert_eq!(LLMProvider::OpenAI.to_string(), "openai");
        assert_eq!(LLMProvider::Ollama.to_string(), "ollama");
        assert_eq!(LLMProvider::Qwen.to_string(), "qwen");
    }
}
