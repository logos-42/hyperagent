//! Local Execution Runtime
//!
//! This module provides a local execution runtime that supports switching between
//! different LLM API providers (OpenAI, Ollama, GLM, MiniMax, Qwen) at runtime.

use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};

use crate::llm::{
    LLMClient, LLMClientImpl, LLMConfig, LLMProvider, LLMResponse, Message,
};
use crate::runtime::environment::{Environment, EnvironmentConfig, IterationMetrics};

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Environment configuration
    pub environment: EnvironmentConfig,
    /// Default LLM configuration
    pub default_llm: LLMConfig,
    /// Enable provider auto-fallback on errors
    pub auto_fallback: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            environment: EnvironmentConfig::default(),
            default_llm: LLMConfig::default(),
            auto_fallback: false,
            max_retries: 3,
            timeout_secs: 60,
        }
    }
}

impl RuntimeConfig {
    /// Create a new runtime config for Ollama (local execution)
    pub fn ollama(model: &str, base_url: Option<&str>) -> Self {
        Self {
            environment: EnvironmentConfig::default(),
            default_llm: LLMConfig::ollama(model, base_url),
            auto_fallback: false,
            max_retries: 3,
            timeout_secs: 120,
        }
    }

    /// Create a new runtime config for OpenAI
    pub fn openai(model: &str, api_key: &str) -> Self {
        Self {
            environment: EnvironmentConfig::default(),
            default_llm: LLMConfig::openai(model, api_key),
            auto_fallback: false,
            max_retries: 3,
            timeout_secs: 60,
        }
    }

    /// Create a new runtime config for Qwen
    pub fn qwen(model: &str, api_key: &str) -> Self {
        Self {
            environment: EnvironmentConfig::default(),
            default_llm: LLMConfig::qwen(model, api_key),
            auto_fallback: false,
            max_retries: 3,
            timeout_secs: 60,
        }
    }
}

/// Execution context for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub session_id: String,
    pub iteration_id: String,
    pub provider: String,
    pub model: String,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub response: LLMResponse,
    pub execution_time_ms: u64,
    pub retries: usize,
    pub context: ExecutionContext,
}

/// Provider statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens: i64,
    pub avg_response_time_ms: f64,
}

impl ProviderStats {
    fn record_success(&mut self, tokens: i32, response_time_ms: u64) {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.total_tokens += tokens as i64;
        
        // Update moving average
        let n = self.successful_requests as f64;
        self.avg_response_time_ms = 
            ((self.avg_response_time_ms * (n - 1.0)) + response_time_ms as f64) / n;
    }

    fn record_failure(&mut self) {
        self.total_requests += 1;
        self.failed_requests += 1;
    }
}

/// Local Execution Runtime
/// 
/// Provides a unified interface for executing LLM requests with support for:
/// - Multiple API providers
/// - Runtime provider switching
/// - Automatic fallback on errors
/// - Request retry with exponential backoff
/// - Execution tracking and statistics
pub struct LocalRuntime {
    config: RuntimeConfig,
    environment: Arc<RwLock<Environment>>,
    current_client: Arc<RwLock<LLMClientImpl>>,
    provider_stats: Arc<RwLock<std::collections::HashMap<String, ProviderStats>>>,
}

impl LocalRuntime {
    /// Create a new local runtime
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        let environment = Environment::new(config.environment.clone())?;
        let client = LLMClientImpl::new(&config.default_llm)?;
        
        let mut stats = std::collections::HashMap::new();
        stats.insert(
            config.default_llm.provider.to_string(),
            ProviderStats::default(),
        );

        Ok(Self {
            config,
            environment: Arc::new(RwLock::new(environment)),
            current_client: Arc::new(RwLock::new(client)),
            provider_stats: Arc::new(RwLock::new(stats)),
        })
    }

    /// Get the current provider
    pub async fn current_provider(&self) -> LLMProvider {
        let client = self.current_client.read().await;
        client.provider().clone()
    }

    /// Switch to a different provider
    pub async fn switch_provider(&self, config: LLMConfig) -> Result<LLMProvider> {
        let new_client = LLMClientImpl::new(&config)?;
        let old_provider = {
            let client = self.current_client.read().await;
            client.provider().clone()
        };

        {
            let mut client = self.current_client.write().await;
            *client = new_client;
        }

        // Initialize stats for new provider if not exists
        {
            let mut stats = self.provider_stats.write().await;
            stats.entry(config.provider.to_string())
                .or_insert_with(ProviderStats::default);
        }

        info!("Switched provider from {} to {}", old_provider, config.provider);
        
        Ok(config.provider)
    }

    /// Switch to Ollama (local)
    pub async fn switch_to_ollama(&self, model: &str, base_url: Option<&str>) -> Result<LLMProvider> {
        let config = LLMConfig::ollama(model, base_url);
        self.switch_provider(config).await
    }

    /// Switch to OpenAI
    pub async fn switch_to_openai(&self, model: &str, api_key: &str) -> Result<LLMProvider> {
        let config = LLMConfig::openai(model, api_key);
        self.switch_provider(config).await
    }

    /// Switch to Qwen
    pub async fn switch_to_qwen(&self, model: &str, api_key: &str) -> Result<LLMProvider> {
        let config = LLMConfig::qwen(model, api_key);
        self.switch_provider(config).await
    }

    /// Switch to GLM
    pub async fn switch_to_glm(&self, model: &str, api_key: &str) -> Result<LLMProvider> {
        let config = LLMConfig::glm(model, api_key);
        self.switch_provider(config).await
    }

    /// Switch to MiniMax
    pub async fn switch_to_minimax(&self, model: &str, api_key: &str) -> Result<LLMProvider> {
        let config = LLMConfig::minimax(model, api_key);
        self.switch_provider(config).await
    }

    /// Create a new session
    pub async fn create_session(&self, name: &str) -> Result<String> {
        let client = self.current_client.read().await;
        let provider = client.provider().to_string();
        let model = client.model().to_string();
        drop(client);

        let mut env = self.environment.write().await;
        env.create_session(name, &provider, &model)
    }

    /// Load an existing session
    pub async fn load_session(&self, session_id: &str) -> Result<()> {
        let mut env = self.environment.write().await;
        env.load_session(session_id)?;
        Ok(())
    }

    /// Execute a simple completion request
    #[instrument(skip(self, prompt), fields(session_id, iteration_id))]
    pub async fn execute(&self, prompt: &str) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let client = self.current_client.read().await;
        let provider = client.provider().clone();
        let model = client.model().to_string();
        
        // Start iteration
        let iteration_id = {
            let mut env = self.environment.write().await;
            env.start_iteration(prompt)?
        };

        let session_id = {
            let env = self.environment.read().await;
            env.info().session_id.unwrap_or_default()
        };

        let context = ExecutionContext {
            session_id: session_id.clone(),
            iteration_id: iteration_id.clone(),
            provider: provider.to_string(),
            model: model.clone(),
        };

        tracing::Span::current().record("session_id", &session_id);
        tracing::Span::current().record("iteration_id", &iteration_id);

        info!("Executing request with provider: {}, model: {}", provider, model);

        // Execute with retry
        let mut retries = 0;
        let response = loop {
            match client.complete(prompt).await {
                Ok(resp) => break resp,
                Err(e) => {
                    retries += 1;
                    warn!("Request failed (attempt {}/{}): {}", retries, self.config.max_retries, e);
                    
                    if retries >= self.config.max_retries {
                        error!("All retry attempts exhausted");
                        
                        // Record failure
                        {
                            let mut stats = self.provider_stats.write().await;
                            if let Some(s) = stats.get_mut(&provider.to_string()) {
                                s.record_failure();
                            }
                        }

                        // Fail iteration
                        {
                            let mut env = self.environment.write().await;
                            let _ = env.fail_iteration(&e.to_string());
                        }

                        return Err(e);
                    }

                    // Exponential backoff
                    let delay = std::time::Duration::from_millis(100 * (1 << retries));
                    tokio::time::sleep(delay).await;
                }
            }
        };

        drop(client);

        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        let tokens_used = response.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0);

        info!(
            "Request completed in {}ms, tokens used: {}",
            execution_time_ms, tokens_used
        );

        // Record success
        {
            let mut stats = self.provider_stats.write().await;
            if let Some(s) = stats.get_mut(&provider.to_string()) {
                s.record_success(tokens_used, execution_time_ms);
            }
        }

        // Complete iteration
        {
            let mut env = self.environment.write().await;
            let metrics = IterationMetrics {
                tokens_used,
                execution_time_ms,
                success: true,
                retry_count: retries,
            };
            let _ = env.complete_iteration(&response.content, metrics);
        }

        Ok(ExecutionResult {
            success: true,
            response,
            execution_time_ms,
            retries,
            context,
        })
    }

    /// Execute with system prompt
    pub async fn execute_with_system(&self, system_prompt: &str, user_prompt: &str) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let client = self.current_client.read().await;
        let provider = client.provider().clone();
        let model = client.model().to_string();
        
        // Start iteration
        let iteration_id = {
            let mut env = self.environment.write().await;
            env.start_iteration(&format!("{}\n\n{}", system_prompt, user_prompt))?
        };

        let session_id = {
            let env = self.environment.read().await;
            env.info().session_id.unwrap_or_default()
        };

        let context = ExecutionContext {
            session_id: session_id.clone(),
            iteration_id: iteration_id.clone(),
            provider: provider.to_string(),
            model: model.clone(),
        };

        info!("Executing with system prompt, provider: {}, model: {}", provider, model);

        // Execute with retry
        let mut retries = 0;
        let response = loop {
            match client.complete_with_system(system_prompt, user_prompt).await {
                Ok(resp) => break resp,
                Err(e) => {
                    retries += 1;
                    warn!("Request failed (attempt {}/{}): {}", retries, self.config.max_retries, e);
                    
                    if retries >= self.config.max_retries {
                        error!("All retry attempts exhausted");
                        
                        {
                            let mut stats = self.provider_stats.write().await;
                            if let Some(s) = stats.get_mut(&provider.to_string()) {
                                s.record_failure();
                            }
                        }

                        {
                            let mut env = self.environment.write().await;
                            let _ = env.fail_iteration(&e.to_string());
                        }

                        return Err(e);
                    }

                    let delay = std::time::Duration::from_millis(100 * (1 << retries));
                    tokio::time::sleep(delay).await;
                }
            }
        };

        drop(client);

        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        let tokens_used = response.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0);

        {
            let mut stats = self.provider_stats.write().await;
            if let Some(s) = stats.get_mut(&provider.to_string()) {
                s.record_success(tokens_used, execution_time_ms);
            }
        }

        {
            let mut env = self.environment.write().await;
            let metrics = IterationMetrics {
                tokens_used,
                execution_time_ms,
                success: true,
                retry_count: retries,
            };
            let _ = env.complete_iteration(&response.content, metrics);
        }

        Ok(ExecutionResult {
            success: true,
            response,
            execution_time_ms,
            retries,
            context,
        })
    }

    /// Execute with message history
    pub async fn execute_with_messages(&self, messages: Vec<Message>) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let client = self.current_client.read().await;
        let provider = client.provider().clone();
        let model = client.model().to_string();
        
        let prompt_preview = messages.first()
            .map(|m| m.content.chars().take(100).collect::<String>())
            .unwrap_or_default();

        // Start iteration
        let iteration_id = {
            let mut env = self.environment.write().await;
            env.start_iteration(&prompt_preview)?
        };

        let session_id = {
            let env = self.environment.read().await;
            env.info().session_id.unwrap_or_default()
        };

        let context = ExecutionContext {
            session_id: session_id.clone(),
            iteration_id: iteration_id.clone(),
            provider: provider.to_string(),
            model: model.clone(),
        };

        info!("Executing with message history, provider: {}, model: {}", provider, model);

        // Execute with retry
        let mut retries = 0;
        let response = loop {
            match client.complete_with_messages(messages.clone()).await {
                Ok(resp) => break resp,
                Err(e) => {
                    retries += 1;
                    warn!("Request failed (attempt {}/{}): {}", retries, self.config.max_retries, e);
                    
                    if retries >= self.config.max_retries {
                        error!("All retry attempts exhausted");
                        
                        {
                            let mut stats = self.provider_stats.write().await;
                            if let Some(s) = stats.get_mut(&provider.to_string()) {
                                s.record_failure();
                            }
                        }

                        {
                            let mut env = self.environment.write().await;
                            let _ = env.fail_iteration(&e.to_string());
                        }

                        return Err(e);
                    }

                    let delay = std::time::Duration::from_millis(100 * (1 << retries));
                    tokio::time::sleep(delay).await;
                }
            }
        };

        drop(client);

        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        let tokens_used = response.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0);

        {
            let mut stats = self.provider_stats.write().await;
            if let Some(s) = stats.get_mut(&provider.to_string()) {
                s.record_success(tokens_used, execution_time_ms);
            }
        }

        {
            let mut env = self.environment.write().await;
            let metrics = IterationMetrics {
                tokens_used,
                execution_time_ms,
                success: true,
                retry_count: retries,
            };
            let _ = env.complete_iteration(&response.content, metrics);
        }

        Ok(ExecutionResult {
            success: true,
            response,
            execution_time_ms,
            retries,
            context,
        })
    }

    /// Get provider statistics
    pub async fn get_provider_stats(&self) -> std::collections::HashMap<String, ProviderStats> {
        let stats = self.provider_stats.read().await;
        stats.clone()
    }

    /// Get environment info
    pub async fn environment_info(&self) -> crate::runtime::environment::EnvironmentInfo {
        let env = self.environment.read().await;
        env.info()
    }

    /// Save an artifact
    pub async fn save_artifact(&self, name: &str, content: &[u8]) -> Result<std::path::PathBuf> {
        let env = self.environment.read().await;
        env.save_artifact(name, content)
    }

    /// Save a text artifact
    pub async fn save_artifact_text(&self, name: &str, content: &str) -> Result<std::path::PathBuf> {
        let env = self.environment.read().await;
        env.save_artifact_text(name, content)
    }

    /// Write to iteration log
    pub async fn write_log(&self, message: &str) -> Result<()> {
        let env = self.environment.read().await;
        env.write_iteration_log(message)
    }

    /// Save to short-term memory
    pub async fn save_memory(&self, key: &str, value: &str) -> Result<()> {
        let env = self.environment.read().await;
        env.save_short_term_memory(key, value)
    }

    /// Load from short-term memory
    pub async fn load_memory(&self, key: &str) -> Result<Option<String>> {
        let env = self.environment.read().await;
        env.load_short_term_memory(key)
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Result<Vec<crate::runtime::environment::SessionMeta>> {
        let env = self.environment.read().await;
        env.list_sessions()
    }

    /// Get session history
    pub async fn get_session_history(&self, session_id: &str) -> Result<Vec<crate::runtime::environment::IterationState>> {
        let env = self.environment.read().await;
        env.get_session_history(session_id)
    }
}

/// Builder for LocalRuntime
pub struct LocalRuntimeBuilder {
    config: RuntimeConfig,
}

impl LocalRuntimeBuilder {
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
        }
    }

    pub fn environment(mut self, config: EnvironmentConfig) -> Self {
        self.config.environment = config;
        self
    }

    pub fn llm_config(mut self, config: LLMConfig) -> Self {
        self.config.default_llm = config;
        self
    }

    pub fn auto_fallback(mut self, enabled: bool) -> Self {
        self.config.auto_fallback = enabled;
        self
    }

    pub fn max_retries(mut self, retries: usize) -> Self {
        self.config.max_retries = retries;
        self
    }

    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.config.timeout_secs = secs;
        self
    }

    pub fn build(self) -> Result<LocalRuntime> {
        LocalRuntime::new(self.config)
    }
}

impl Default for LocalRuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for quick runtime creation
impl LocalRuntime {
    /// Create a runtime with Ollama (local)
    pub fn ollama(model: &str) -> Result<Self> {
        let config = RuntimeConfig::ollama(model, None);
        Self::new(config)
    }

    /// Create a runtime with Ollama and custom base URL
    pub fn ollama_with_url(model: &str, base_url: &str) -> Result<Self> {
        let config = RuntimeConfig::ollama(model, Some(base_url));
        Self::new(config)
    }

    /// Create a runtime with OpenAI
    pub fn openai(model: &str, api_key: &str) -> Result<Self> {
        let config = RuntimeConfig::openai(model, api_key);
        Self::new(config)
    }

    /// Create a runtime with Qwen
    pub fn qwen(model: &str, api_key: &str) -> Result<Self> {
        let config = RuntimeConfig::qwen(model, api_key);
        Self::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = LocalRuntime::ollama("llama2").unwrap();
        assert_eq!(runtime.current_provider().await, LLMProvider::Ollama);
    }

    #[tokio::test]
    async fn test_provider_switch() {
        let _runtime = LocalRuntime::ollama("llama2").unwrap();
        
        // Note: This test would require actual API keys
        // runtime.switch_to_openai("gpt-4o", "test-key").await.unwrap();
        // assert_eq!(runtime.current_provider().await, LLMProvider::OpenAI);
    }

    #[tokio::test]
    async fn test_session_management() {
        let runtime = LocalRuntime::ollama("llama2").unwrap();

        let count_before = runtime.list_sessions().await.unwrap().len();

        let session_id = runtime.create_session("Test").await.unwrap();
        assert!(!session_id.is_empty());

        let sessions = runtime.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), count_before + 1);
    }
}
