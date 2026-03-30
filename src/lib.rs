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
///
/// This enum is marked as `#[non_exhaustive]` to allow adding new variants
/// without breaking changes to downstream code. Users should not exhaustively
/// match on this enum; instead, use wildcard patterns or the provided
/// accessor methods.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// LLM client errors (API failures, rate limits, timeouts)
    LLM(String),
    /// I/O errors (file read/write, network requests)
    Io(std::io::Error),
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
    /// Configuration/deserialization errors
    Config(String),
    /// Generic error with message
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LLM(msg) => write!(f, "LLM error: {msg}"),
            Error::Io(err) => write!(f, "I/O error: {err}"),
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

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            // String-based variants have no underlying source
            Error::LLM(_)
            | Error::Evaluation(_)
            | Error::Evolution(_)
            | Error::Memory(_)
            | Error::Codebase(_)
            | Error::Web(_)
            | Error::Config(_)
            | Error::Other(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    #[cold]
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<reqwest::Error> for Error {
    #[cold]
    fn from(err: reqwest::Error) -> Self {
        Error::Web(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    #[cold]
    fn from(err: serde_json::Error) -> Self {
        Error::Config(err.to_string())
    }
}

impl From<tokio::task::JoinError> for Error {
    #[cold]
    fn from(err: tokio::task::JoinError) -> Self {
        Error::Evolution(err.to_string())
    }
}

impl From<String> for Error {
    #[cold]
    fn from(msg: String) -> Self {
        Error::Other(msg)
    }
}

impl From<&str> for Error {
    #[cold]
    fn from(msg: &str) -> Self {
        Error::Other(msg.to_string())
    }
}

impl From<Box<dyn StdError + Send + Sync>> for Error {
    #[cold]
    fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<Box<dyn StdError>> for Error {
    #[cold]
    fn from(err: Box<dyn StdError>) -> Self {
        Error::Other(err.to_string())
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
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = Error::Io(io_err);
        assert!(err.to_string().contains("file not found"));
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

    #[test]
    fn test_from_tokio_join_error() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        
        // Create a scenario that can produce a JoinError
        // We test that the From impl exists and compiles correctly
        // by relying on type inference in a Result context
        fn convert_join_error(e: tokio::task::JoinError) -> Error {
            e.into()
        }
        
        // Verify the error message contains task-related info
        let handle = tokio::runtime::Handle::current();
        let result = handle.block_on(async {
            let task = tokio::spawn(async {
                panic!("test panic");
            });
            task.await
        });
        
        if let Err(e) = result {
            let err: Error = e.into();
            assert!(matches!(err, Error::Evolution(_)));
            assert!(err.to_string().contains("task") || err.to_string().contains("panic") || err.to_string().contains("Evolution"));
        }
    }

    #[test]
    fn test_from_string() {
        let msg = "something went wrong".to_string();
        let err: Error = msg.clone().into();
        assert!(matches!(err, Error::Other(m) if m == msg));
        assert_eq!(err.to_string(), "Error: something went wrong");
    }

    #[test]
    fn test_from_str_ref() {
        let err: Error = "simple error message".into();
        assert!(matches!(err, Error::Other(m) if m == "simple error message"));
        assert_eq!(err.to_string(), "Error: simple error message");
    }

    #[test]
    fn test_from_string_in_result_context() {
        fn returns_error() -> Result<String> {
            Err("operation failed".into())
        }
        
        fn returns_error_owned() -> Result<String> {
            let msg = "owned error".to_string();
            Err(msg.into())
        }
        
        assert!(returns_error().is_err());
        assert!(returns_error_owned().is_err());
        
        let err1 = returns_error().unwrap_err();
        let err2 = returns_error_owned().unwrap_err();
        
        assert!(matches!(err1, Error::Other(_)));
        assert!(matches!(err2, Error::Other(_)));
    }

    #[test]
    fn test_error_source_returns_none_for_string_variants() {
        // String-based Error variants have no source
        let errors = vec![
            Error::LLM("test".to_string()),
            Error::Evaluation("test".to_string()),
            Error::Evolution("test".to_string()),
            Error::Memory("test".to_string()),
            Error::Codebase("test".to_string()),
            Error::Web("test".to_string()),
            Error::Config("test".to_string()),
            Error::Other("test".to_string()),
        ];
        
        for err in errors {
            assert!(err.source().is_none(), "Expected source() to return None for {:?}", err);
        }
    }

    #[test]
    fn test_error_source_returns_underlying_io_error() {
        // Io variant wraps the actual std::io::Error, so source() should return it
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = Error::Io(io_err);
        
        let source = err.source();
        assert!(source.is_some(), "Expected source() to return Some for Io variant");
        
        // Verify we can downcast back to the original error type
        let source = source.unwrap();
        let io_source = source.downcast_ref::<std::io::Error>();
        assert!(io_source.is_some(), "Expected to downcast source to std::io::Error");
        assert_eq!(io_source.unwrap().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn test_error_chain_debug_format() {
        let err = Error::LLM("rate limited after 3 retries".to_string());
        
        // Verify Debug format includes variant name
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("LLM"));
        assert!(debug_str.contains("rate limited"));
        
        // Verify Display format matches expected output
        let display_str = format!("{}", err);
        assert_eq!(display_str, "LLM error: rate limited after 3 retries");
    }

    #[test]
    fn test_io_error_preserves_error_kind() {
        // Test that different std::io::ErrorKind variants are preserved
        use std::io::{Error, ErrorKind};
        
        let kinds = vec![
            ErrorKind::NotFound,
            ErrorKind::PermissionDenied,
            ErrorKind::ConnectionRefused,
            ErrorKind::TimedOut,
        ];
        
        for kind in kinds {
            let io_err = Error::new(kind, "test");
            let err: Error = io_err.into();
            
            match err {
                Error::Io(ref inner) => {
                    assert_eq!(inner.kind(), kind, "Error kind should be preserved");
                }
                _ => panic!("Expected Io variant"),
            }
        }
    }

    #[test]
    fn test_io_error_chain_uses_anyhow_style() {
        // Test that error chaining works similar to anyhow::Error
        let io_err = std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "unexpected end of file"
        );
        let err = Error::Io(io_err);
        
        // The Display should show a readable message
        let display = format!("{}", err);
        assert!(display.contains("unexpected end of file") || display.contains("I/O error"));
        
        // The source should be accessible for programmatic inspection
        let source = StdError::source(&err);
        assert!(source.is_some());
    }

    #[test]
    fn test_error_is_send_sync() {
        // Verify Error implements Send and Sync for thread safety
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn test_from_box_dyn_std_error_send_sync() {
        // Test conversion from Box<dyn StdError + Send + Sync>
        use std::io::{self, ErrorKind};
        
        let io_err = io::Error::new(ErrorKind::Other, "boxed error");
        let boxed: Box<dyn StdError + Send + Sync> = Box::new(io_err);
        let err: Error = boxed.into();
        
        assert!(matches!(err, Error::Other(_)));
        assert!(err.to_string().contains("boxed error"));
    }

    #[test]
    fn test_from_box_dyn_std_error() {
        // Test conversion from Box<dyn StdError> (without Send + Sync)
        use std::io::{self, ErrorKind};
        
        let io_err = io::Error::new(ErrorKind::Other, "non-send boxed error");
        let boxed: Box<dyn StdError> = Box::new(io_err);
        let err: Error = boxed.into();
        
        assert!(matches!(err, Error::Other(_)));
        assert!(err.to_string().contains("non-send boxed error"));
    }

    #[test]
    fn test_error_chain_with_source() {
        // Test that error chaining works correctly with nested errors
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "inner error");
        let err = Error::Io(inner);
        
        // Verify source chain
        let source = err.source();
        assert!(source.is_some());
        
        // Verify we can get the Display of both levels
        let full_msg = format!("{}", err);
        assert!(full_msg.contains("I/O error"));
        assert!(full_msg.contains("inner error"));
    }

    #[test]
    fn test_error_variants_have_distinct_prefixes() {
        // Ensure each error variant has a distinct prefix for easier parsing
        let variants = vec![
            (Error::LLM("x".into()), "LLM error:"),
            (Error::Io(std::io::Error::new(std::io::ErrorKind::Other, "x")), "I/O error:"),
            (Error::Evaluation("x".into()), "Evaluation error:"),
            (Error::Evolution("x".into()), "Evolution error:"),
            (Error::Memory("x".into()), "Memory error:"),
            (Error::Codebase("x".into()), "Codebase error:"),
            (Error::Web("x".into()), "Web error:"),
            (Error::Config("x".into()), "Config error:"),
            (Error::Other("x".into()), "Error:"),
        ];
        
        for (err, expected_prefix) in variants {
            let display = format!("{}", err);
            assert!(
                display.starts_with(expected_prefix),
                "Expected '{}' to start with '{}'",
                display,
                expected_prefix
            );
        }
    }
}
