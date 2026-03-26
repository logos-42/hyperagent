//! # Hyperagent: Self-Evolving AI Research System
//!
//! A recursive self-improvement framework implementing Andrej Karpathy's research methodology:
//! **Hypothesis → Experiment → Observation → Reflection → New Hypothesis**
//!
//! ## Architecture Overview
//!
//! The system is organized into interconnected modules:
//!
//! - **[`agent`]**: Core agent primitives (`Agent`, `Executor`, `Mutator`, `MetaMutator`)
//!   for evolutionary code transformation
//! - **[`llm`]**: Multi-provider LLM client abstraction (OpenAI, Ollama, Qwen, GLM, MiniMax)
//! - **[`runtime`]**: Evolution loops, thermodynamic state management, and constraint systems
//! - **[`memory`]**: Evolutionary lineage tracking and solution archival
//! - **[`eval`]**: Multi-dimensional evaluation metrics and benchmarking
//! - **[`strategy`]**: Meta-evolutionary strategy parameter adaptation
//! - **[`auto_research`]**: Karpathy-style autonomous research loop
//! - **[`self_evolution`]**: Recursive self-improvement engine
//! - **[`codebase`]**: Codebase context and architectural awareness
//! - **[`web`]**: Web search and fetch tools for research augmentation
//! - **[`tools`]**: Local codebase introspection tools (grep, search, read, tree)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use hyperagent::{Agent, LLMClient, LLMConfig, EvolutionLoop};
//!
//! // Create an agent with an LLM client
//! let config = LLMConfig::default();
//! let agent = Agent::new(config);
//!
//! // Run evolution
//! let result = agent.evolve("Improve performance").await?;
//! ```
//!
//! ## Feature Flags
//!
//! No feature flags are currently used; all modules are always available.
//!
//! ## Error Handling
//!
//! All public APIs return a unified [`Error`] type that wraps errors from
//! different subsystems (LLM calls, I/O, evaluation, etc.) into a single
//! consistent error interface.

pub mod agent;
pub mod codebase;
pub mod eval;
pub mod llm;
pub mod memory;
pub mod runtime;
pub mod self_evolution;
pub mod strategy;
pub mod auto_research;
pub mod web;
pub mod tools;

pub use agent::{Agent, Executor, MetaMutator, Mutator};
pub use codebase::CodebaseContext;
pub use eval::{Evaluator, Benchmark, EvaluationResult};
pub use llm::{
    LLMClient, LLMClientImpl, PromptTemplate, LLMConfig, LLMProvider, LLMResponse,
    create_llm_client, DynLLMClient,
};
pub use memory::{Archive, Lineage, Record};
pub use runtime::{
    // Evolution runtime
    EvolutionLoop, RuntimeState, EvolutionRuntimeConfig as RuntimeConfig,
    EnergyState, DissipationScale, InfoEnergyCoupling, FitnessLandscape,
    ConstraintSystem, HardConstraints, SoftConstraints, CodeMetrics,
    EvolutionDirection, TopologicalConstraints,
    Selector, SelectionType, Individual, PopulationStats,
    // Multi-agent population evolution
    PopulationEvolution, PopulationConfig, PopulationEvolutionResult,
    PopulationMember, AgentRole, AgentMessage, MessageType, GenerationStats,
    // Environment
    Environment, EnvironmentConfig, EnvironmentInfo,
    SessionMeta, SessionStatus,
    IterationState, IterationStatus, IterationMetrics,
    // Local runtime
    LocalRuntime, LocalRuntimeBuilder,
    LocalRuntimeConfig, ExecutionContext, ExecutionResult, ProviderStats,
};
pub use self_evolution::{SelfEvolutionEngine, SelfEvolutionConfig, SelfEvolutionStatus};
pub use auto_research::{AutoResearch, ResearchConfig, Experiment, ExperimentOutcome, FileChange};
pub use eval::metrics::{MultiEvalResult, MetricDirection};
pub use web::{WebSearchTool, WebFetchTool, WebSearchResult, FetchOutput, SearchOutput, build_web_context_prompt};
pub use tools::{
    CodebaseGrepTool, CodebaseSearchTool, CodebaseReadTool, CodebaseTreeTool,
    GrepMatch, GrepOutput, FileEntry, SearchFilesOutput, ReadFileOutput, TreeOutput,
};

use std::fmt;
use std::error::Error as StdError;

/// Unified error type for all Hyperagent operations.
///
/// This error type wraps errors from all subsystems into a single type,
/// enabling consistent error handling across the entire crate.
#[derive(Debug)]
pub enum Error {
    /// LLM client errors (API failures, rate limits, timeouts)
    LLM(String),
    /// I/O errors (file read/write, network requests)
    Io(String),
    /// Evaluation errors (test failures, metric computation)
    Evaluation(String),
    /// Evolution errors (constraint violations, population collapse)
    Evolution(String),
    /// Memory/archive errors (persistence failures)
    Memory(String),
    /// Codebase scanning errors
    Codebase(String),
    /// Web search/fetch errors
    Web(String),
    /// Configuration errors
    Config(String),
    /// Generic error with message
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LLM(msg) => write!(f, "LLM error: {msg}"),
            Error::Io(msg) => write!(f, "I/O error: {msg}"),
            Error::Evaluation(msg) => write!(f, "Evaluation error: {msg}"),
            Error::Evolution(msg) => write!(f, "Evolution error: {msg}"),
            Error::Memory(msg) => write!(f, "Memory error: {msg}"),
            Error::Codebase(msg) => write!(f, "Codebase error: {msg}"),
            Error::Web(msg) => write!(f, "Web error: {msg}"),
            Error::Config(msg) => write!(f, "Config error: {msg}"),
            Error::Other(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl StdError for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Web(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Config(err.to_string())
    }
}

/// A specialized Result type for Hyperagent operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_llm() {
        let err = Error::LLM("rate limited".to_string());
        assert_eq!(err.to_string(), "LLM error: rate limited");
    }

    #[test]
    fn test_error_display_io() {
        let err = Error::Io("file not found".to_string());
        assert_eq!(err.to_string(), "I/O error: file not found");
    }

    #[test]
    fn test_error_display_evaluation() {
        let err = Error::Evaluation("test failed".to_string());
        assert_eq!(err.to_string(), "Evaluation error: test failed");
    }

    #[test]
    fn test_error_display_evolution() {
        let err = Error::Evolution("population collapsed".to_string());
        assert_eq!(err.to_string(), "Evolution error: population collapsed");
    }

    #[test]
    fn test_error_display_memory() {
        let err = Error::Memory("archive corrupted".to_string());
        assert_eq!(err.to_string(), "Memory error: archive corrupted");
    }

    #[test]
    fn test_error_display_codebase() {
        let err = Error::Codebase("scan failed".to_string());
        assert_eq!(err.to_string(), "Codebase error: scan failed");
    }

    #[test]
    fn test_error_display_web() {
        let err = Error::Web("timeout".to_string());
        assert_eq!(err.to_string(), "Web error: timeout");
    }

    #[test]
    fn test_error_display_config() {
        let err = Error::Config("invalid settings".to_string());
        assert_eq!(err.to_string(), "Config error: invalid settings");
    }

    #[test]
    fn test_error_display_other() {
        let err = Error::Other("unknown".to_string());
        assert_eq!(err.to_string(), "Error: unknown");
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_result() -> Result<String> {
            Ok("success".to_string())
        }
        assert!(returns_result().is_ok());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<i32>("invalid json");
        if let Err(e) = json_err {
            let err: Error = e.into();
            assert!(matches!(err, Error::Config(_)));
        }
    }
}
