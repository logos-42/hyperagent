//! LLM Abstraction Layer
//!
//! This module provides a unified interface for interacting with multiple Large Language Model
//! providers, enabling seamless provider switching at runtime without code changes.
//!
//! # Supported Providers
//!
//! - **OpenAI** — GPT-4, GPT-3.5-turbo, and other OpenAI models
//! - **Ollama** — Local LLM inference (Llama, Mistral, etc.)
//! - **GLM** — Zhipu AI's ChatGLM models
//! - **MiniMax** — MiniMax API models
//! - **Qwen** — Alibaba's Qwen series
//!
//! # Architecture
//!
//! The module is organized into two main components:
//!
//! - [`client`] — Core client implementation with provider abstraction, message handling,
//!   token tracking, and configuration management
//! - [`prompts`] — Prompt template system for structured prompt engineering
//!
//! # Quick Start
//!
//! ```ignore
//! use hyperagent::llm::{LLMClient, LLMConfig, LLMProvider, Message};
//!
//! // Create a client with OpenAI
//! let config = LLMConfig::openai("gpt-4").with_temperature(0.7);
//! let client = create_llm_client(config);
//!
//! // Or switch to Ollama for local inference
//! let config = LLMConfig::ollama("llama2");
//! let client = create_llm_client(config);
//! ```
//!
//! # Provider Selection
//!
//! Use [`LLMProvider`] enum to specify the backend at runtime:
//!
//! ```ignore
//! let provider = LLMProvider::OpenAI;
//! let config = LLMConfig::new(provider, "gpt-4");
//! ```
//!
//! # Prompt Templates
//!
//! Use [`PromptManager`] for structured prompt generation across different agent operations:
//!
//! ```ignore
//! use hyperagent::llm::PromptManager;
//!
//! let manager = PromptManager::new();
//! let template = manager.render(PromptType::Execute, context);
//! ```

pub mod client;
pub mod prompts;

pub use client::{
    LLMClient, LLMConfig, LLMClientImpl, LLMProvider, LLMResponse,
    Message, MessageRole, TokenUsage,
    create_llm_client, DynLLMClient,
};
pub use prompts::{PromptTemplate, PromptManager, PromptType};
