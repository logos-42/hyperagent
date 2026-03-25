use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use reqwest::Client as ReqwestClient;
use rig::{
    client::CompletionClient,
    providers::openai::Client as OpenAiClient,
    completion::Prompt,
};

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
    /// Create config for Ollama (local)
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

    /// Create config for OpenAI
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

    /// Create config for Qwen
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

    /// Create config for GLM
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

    /// Create config for MiniMax
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

/// Internal client implementations
#[derive(Clone)]
enum ClientBackend {
    OpenAI(OpenAIBackend),
    Ollama(OllamaBackend),
    GLM(GLMBackend),
    MiniMax(MiniMaxBackend),
    Qwen(QwenBackend),
}

#[derive(Clone)]
#[allow(dead_code)]
struct OpenAIBackend {
    client: Arc<OpenAiClient>,
    model: String,
    semaphore: Arc<Semaphore>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct OllamaBackend {
    http_client: ReqwestClient,
    base_url: String,
    model: String,
    semaphore: Arc<Semaphore>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Clone)]
struct GLMBackend {
    http_client: ReqwestClient,
    base_url: String,
    model: String,
    api_key: String,
    semaphore: Arc<Semaphore>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Clone)]
struct MiniMaxBackend {
    http_client: ReqwestClient,
    base_url: String,
    model: String,
    api_key: String,
    semaphore: Arc<Semaphore>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Clone)]
struct QwenBackend {
    http_client: ReqwestClient,
    base_url: String,
    model: String,
    api_key: String,
    semaphore: Arc<Semaphore>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

/// Main LLM Client that wraps different backends
#[derive(Clone)]
pub struct LLMClientImpl {
    backend: ClientBackend,
    provider: LLMProvider,
}

impl LLMClientImpl {
    pub fn new(config: &LLMConfig) -> Result<Self> {
        let backend = match config.provider {
            LLMProvider::OpenAI => {
                let mut builder = OpenAiClient::builder().api_key(&config.api_key);
                if let Some(base_url) = &config.base_url {
                    builder = builder.base_url(base_url);
                }
                ClientBackend::OpenAI(OpenAIBackend {
                    client: Arc::new(builder.build()?),
                    model: config.model.clone(),
                    semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                })
            }
            LLMProvider::Ollama => {
                let base_url = config.base_url.as_deref().unwrap_or("http://localhost:11434").to_string();
                ClientBackend::Ollama(OllamaBackend {
                    http_client: ReqwestClient::builder()
                        .no_proxy()
                        .build()?,
                    base_url,
                    model: config.model.clone(),
                    semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                })
            }
            LLMProvider::GLM => {
                let base_url = config.base_url.as_deref().unwrap_or("https://open.bigmodel.cn/api/paas/v4").to_string();
                ClientBackend::GLM(GLMBackend {
                    http_client: ReqwestClient::new(),
                    base_url,
                    model: config.model.clone(),
                    api_key: config.api_key.clone(),
                    semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                })
            }
            LLMProvider::MiniMax => {
                let base_url = config.base_url.as_deref().unwrap_or("https://api.minimax.chat/v1").to_string();
                ClientBackend::MiniMax(MiniMaxBackend {
                    http_client: ReqwestClient::new(),
                    base_url,
                    model: config.model.clone(),
                    api_key: config.api_key.clone(),
                    semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                })
            }
            LLMProvider::Qwen => {
                let base_url = config.base_url.as_deref().unwrap_or("https://dashscope.aliyuncs.com/compatible-mode/v1").to_string();
                ClientBackend::Qwen(QwenBackend {
                    http_client: ReqwestClient::new(),
                    base_url,
                    model: config.model.clone(),
                    api_key: config.api_key.clone(),
                    semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                })
            }
        };

        Ok(Self {
            backend,
            provider: config.provider.clone(),
        })
    }

    /// Create from environment variables (reads LLM_PROVIDER, provider-specific vars)
    pub fn from_env() -> Result<Self> {
        let provider_str = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());
        let config = match provider_str.to_lowercase().as_str() {
            "ollama" => {
                let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama2".to_string());
                let base_url = std::env::var("OLLAMA_BASE_URL").ok();
                LLMConfig::ollama(&model, base_url.as_deref())
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

    /// Get the current provider
    pub fn provider(&self) -> &LLMProvider {
        &self.provider
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        match &self.backend {
            ClientBackend::OpenAI(b) => &b.model,
            ClientBackend::Ollama(b) => &b.model,
            ClientBackend::GLM(b) => &b.model,
            ClientBackend::MiniMax(b) => &b.model,
            ClientBackend::Qwen(b) => &b.model,
        }
    }
}

#[async_trait]
impl LLMClient for LLMClientImpl {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        match &self.backend {
            ClientBackend::OpenAI(client) => client.complete(prompt).await,
            ClientBackend::Ollama(client) => client.complete(prompt).await,
            ClientBackend::GLM(client) => client.complete(prompt).await,
            ClientBackend::MiniMax(client) => client.complete(prompt).await,
            ClientBackend::Qwen(client) => client.complete(prompt).await,
        }
        .map(|mut resp| {
            resp.provider = self.provider.to_string();
            resp
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        match &self.backend {
            ClientBackend::OpenAI(client) => client.complete_with_system(system_prompt, user_prompt).await,
            ClientBackend::Ollama(client) => client.complete_with_system(system_prompt, user_prompt).await,
            ClientBackend::GLM(client) => client.complete_with_system(system_prompt, user_prompt).await,
            ClientBackend::MiniMax(client) => client.complete_with_system(system_prompt, user_prompt).await,
            ClientBackend::Qwen(client) => client.complete_with_system(system_prompt, user_prompt).await,
        }
        .map(|mut resp| {
            resp.provider = self.provider.to_string();
            resp
        })
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        match &self.backend {
            ClientBackend::OpenAI(client) => client.complete_with_messages(messages).await,
            ClientBackend::Ollama(client) => client.complete_with_messages(messages).await,
            ClientBackend::GLM(client) => client.complete_with_messages(messages).await,
            ClientBackend::MiniMax(client) => client.complete_with_messages(messages).await,
            ClientBackend::Qwen(client) => client.complete_with_messages(messages).await,
        }
        .map(|mut resp| {
            resp.provider = self.provider.to_string();
            resp
        })
    }
}

// OpenAI Implementation
impl OpenAIBackend {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let agent = self.client.agent(&self.model).build();
        let response = agent.prompt(prompt).await?;

        Ok(LLMResponse {
            content: response,
            model: self.model.clone(),
            provider: "openai".to_string(),
            usage: None,
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let agent = self.client.agent(&self.model)
            .preamble(system_prompt)
            .build();
        let response = agent.prompt(user_prompt).await?;

        Ok(LLMResponse {
            content: response,
            model: self.model.clone(),
            provider: "openai".to_string(),
            usage: None,
        })
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut preamble = String::new();
        let mut user_prompt = String::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    preamble = msg.content;
                }
                MessageRole::User => {
                    user_prompt = msg.content;
                }
                MessageRole::Assistant => {
                    // For simplicity, append to user prompt
                    user_prompt.push_str(&format!("\nAssistant: {}", msg.content));
                }
            }
        }

        let agent = if !preamble.is_empty() {
            self.client.agent(&self.model)
                .preamble(&preamble)
                .build()
        } else {
            self.client.agent(&self.model).build()
        };

        let response = agent.prompt(&user_prompt).await?;

        Ok(LLMResponse {
            content: response,
            model: self.model.clone(),
            provider: "openai".to_string(),
            usage: None,
        })
    }
}

// Ollama Implementation
impl OllamaBackend {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": false
        });

        if let Some(temp) = self.temperature {
            request_body["options"] = serde_json::json!({
                "temperature": temp
            });
        }

        let resp = self.http_client
            .post(&format!("{}/api/chat", self.base_url))
            .json(&request_body)
            .send()
            .await?;

        let status = resp.status();
        let body_text = resp.text().await?;

        if !status.is_success() {
            anyhow::bail!("Ollama API error {}: {}", status, body_text);
        }

        let response: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse Ollama response: {} | body: {}", e, &body_text[..body_text.len().min(200)]))?;

        let content = response.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "ollama".to_string(),
            usage: None,
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "stream": false
        });

        if let Some(temp) = self.temperature {
            request_body["options"] = serde_json::json!({
                "temperature": temp
            });
        }

        let resp = self.http_client
            .post(&format!("{}/api/chat", self.base_url))
            .json(&request_body)
            .send()
            .await?;

        let status = resp.status();
        let body_text = resp.text().await?;

        if !status.is_success() {
            anyhow::bail!("Ollama API error {}: {}", status, body_text);
        }

        let response: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse Ollama response: {} | body: {}", e, &body_text[..body_text.len().min(200)]))?;

        let content = response.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "ollama".to_string(),
            usage: None,
        })
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let ollama_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                };
                serde_json::json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect();

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": ollama_messages,
            "stream": false
        });

        if let Some(temp) = self.temperature {
            request_body["options"] = serde_json::json!({
                "temperature": temp
            });
        }

        let response = self.http_client
            .post(&format!("{}/api/chat", self.base_url))
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "ollama".to_string(),
            usage: None,
        })
    }
}

// GLM Implementation
impl GLMBackend {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        if let Some(temp) = self.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            request_body["max_tokens"] = serde_json::json!(tokens);
        }

        let response = self.http_client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = response.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        });

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "glm".to_string(),
            usage,
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        });

        if let Some(temp) = self.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            request_body["max_tokens"] = serde_json::json!(tokens);
        }

        let response = self.http_client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = response.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        });

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "glm".to_string(),
            usage,
        })
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let api_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                };
                serde_json::json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect();

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": api_messages
        });

        if let Some(temp) = self.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            request_body["max_tokens"] = serde_json::json!(tokens);
        }

        let response = self.http_client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = response.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        });

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "glm".to_string(),
            usage,
        })
    }
}

// MiniMax Implementation
impl MiniMaxBackend {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let messages = serde_json::json!([
            {
                "role": "user",
                "content": prompt
            }
        ]);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        if let Some(temp) = self.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            request_body["max_tokens"] = serde_json::json!(tokens);
        }

        let response = self.http_client
            .post(&format!("{}/text/chatcompletion_v2", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "minimax".to_string(),
            usage: None,
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let messages = serde_json::json!([
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]);

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        if let Some(temp) = self.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            request_body["max_tokens"] = serde_json::json!(tokens);
        }

        let response = self.http_client
            .post(&format!("{}/text/chatcompletion_v2", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "minimax".to_string(),
            usage: None,
        })
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let api_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                };
                serde_json::json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect();

        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": api_messages,
        });

        if let Some(temp) = self.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            request_body["max_tokens"] = serde_json::json!(tokens);
        }

        let response = self.http_client
            .post(&format!("{}/text/chatcompletion_v2", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "minimax".to_string(),
            usage: None,
        })
    }
}

// Qwen Implementation
impl QwenBackend {
    async fn complete(&self, prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut request_body = serde_json::json!({
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        });

        let mut parameters = serde_json::json!({});
        if let Some(temp) = self.temperature {
            parameters["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            parameters["max_tokens"] = serde_json::json!(tokens);
        }
        request_body["parameters"] = parameters;

        let response = self.http_client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = response.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        });

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "qwen".to_string(),
            usage,
        })
    }

    async fn complete_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let mut request_body = serde_json::json!({
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
        });

        let mut parameters = serde_json::json!({});
        if let Some(temp) = self.temperature {
            parameters["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            parameters["max_tokens"] = serde_json::json!(tokens);
        }
        request_body["parameters"] = parameters;

        let response = self.http_client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = response.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        });

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "qwen".to_string(),
            usage,
        })
    }

    async fn complete_with_messages(&self, messages: Vec<Message>) -> Result<LLMResponse> {
        let _permit = self.semaphore.acquire().await?;

        let api_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                };
                serde_json::json!({
                    "role": role,
                    "content": msg.content
                })
            })
            .collect();

        let mut request_body = serde_json::json!({
            "model": self.model,
            "input": {
                "messages": api_messages
            }
        });

        let mut parameters = serde_json::json!({});
        if let Some(temp) = self.temperature {
            parameters["temperature"] = serde_json::json!(temp);
        }
        if let Some(tokens) = self.max_tokens {
            parameters["max_tokens"] = serde_json::json!(tokens);
        }
        request_body["parameters"] = parameters;

        let response = self.http_client
            .post(&format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let content = response.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = response.get("usage").map(|u| TokenUsage {
            prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            completion_tokens: u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            total_tokens: u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        });

        Ok(LLMResponse {
            content,
            model: self.model.clone(),
            provider: "qwen".to_string(),
            usage,
        })
    }
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
