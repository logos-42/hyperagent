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

impl Error {
    /// Returns `true` if this is an `LLM` error.
    pub fn is_llm(&self) -> bool {
        matches!(self, Error::LLM(_))
    }

    /// Returns `true` if this is an `Io` error.
    pub fn is_io(&self) -> bool {
        matches!(self, Error::Io(_))
    }

    /// Returns `true` if this is an `Evaluation` error.
    pub fn is_evaluation(&self) -> bool {
        matches!(self, Error::Evaluation(_))
    }

    /// Returns `true` if this is an `Evolution` error.
    pub fn is_evolution(&self) -> bool {
        matches!(self, Error::Evolution(_))
    }

    /// Returns `true` if this is a `Memory` error.
    pub fn is_memory(&self) -> bool {
        matches!(self, Error::Memory(_))
    }

    /// Returns `true` if this is a `Codebase` error.
    pub fn is_codebase(&self) -> bool {
        matches!(self, Error::Codebase(_))
    }

    /// Returns `true` if this is a `Web` error.
    pub fn is_web(&self) -> bool {
        matches!(self, Error::Web(_))
    }

    /// Returns `true` if this is a `Config` error.
    pub fn is_config(&self) -> bool {
        matches!(self, Error::Config(_))
    }

    /// Returns `true` if this is an `Other` error.
    pub fn is_other(&self) -> bool {
        matches!(self, Error::Other(_))
    }

    /// Returns `true` if this error is likely transient and may succeed on retry.
    ///
    /// This is useful for implementing retry logic with exponential backoff.
    /// Returns `true` for:
    /// - `LLM` errors (rate limits, API timeouts)
    /// - `Web` errors (network failures, timeouts)
    /// - `Io` errors with transient kinds (`TimedOut`, `Interrupted`, `WouldBlock`,
    ///   `ConnectionRefused`, `ConnectionReset`, `ConnectionAborted`)
    ///
    /// Returns `false` for:
    /// - `Evaluation` errors (test failures are deterministic)
    /// - `Evolution` errors (constraint violations are structural)
    /// - `Memory` errors (corruption is persistent)
    /// - `Codebase` errors (scan failures are structural)
    /// - `Config` errors (invalid config won't fix itself)
    /// - `Other` errors (unknown, assume non-transient)
    /// - `Io` errors with non-transient kinds (`NotFound`, `PermissionDenied`, etc.)
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::LLM(_) | Error::Web(_) => true,
            Error::Io(err) => matches!(
                err.kind(),
                std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
            ),
            Error::Evaluation(_)
            | Error::Evolution(_)
            | Error::Memory(_)
            | Error::Codebase(_)
            | Error::Config(_)
            | Error::Other(_) => false,
        }
    }

    /// Returns the inner `std::io::Error` if this is an `Io` error.
    ///
    /// This is useful for inspecting the specific I/O error kind
    /// (e.g., `NotFound`, `PermissionDenied`, `TimedOut`).
    pub fn as_io(&self) -> Option<&std::io::Error> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }

    /// Returns the inner `std::io::Error` if this is an `Io` error.
    ///
    /// This returns a cloned `std::io::Error` without consuming the `Error`.
    /// Use this when you need an owned error but also need to retain ownership
    /// of the original `Error`.
    pub fn to_io(&self) -> Option<std::io::Error> {
        match self {
            Error::Io(err) => Some(std::io::Error::new(err.kind(), err.to_string())),
            _ => None,
        }
    }

    /// Returns the error message string if this is a string-based variant.
    ///
    /// Returns `None` for `Io` errors which have their own display format.
    pub fn as_message(&self) -> Option<&str> {
        match self {
            Error::LLM(msg) => Some(msg),
            Error::Evaluation(msg) => Some(msg),
            Error::Evolution(msg) => Some(msg),
            Error::Memory(msg) => Some(msg),
            Error::Codebase(msg) => Some(msg),
            Error::Web(msg) => Some(msg),
            Error::Config(msg) => Some(msg),
            Error::Other(msg) => Some(msg),
            Error::Io(_) => None,
        }
    }

    /// Returns a cloned error message string if this is a string-based variant.
    ///
    /// This is the non-consuming equivalent of [`into_message`](Self::into_message).
    /// Use this when you need an owned `String` but also need to retain ownership
    /// of the original `Error`.
    ///
    /// Returns `None` for `Io` errors which cannot be meaningfully converted
    /// Consumes the error and returns the inner `std::io::Error` if this is an `Io` error.
    ///
    /// This is useful when you need to work with the underlying I/O error
    /// directly without keeping the outer `Error` wrapper.
    pub fn into_io(self) -> Option<std::io::Error> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }

    /// Returns a cloned error message string if this is a string-based variant.
    ///
    /// This is the non-consuming equivalent of [`into_message`](Self::into_message).
    /// Use this when you need an owned `String` but also need to retain ownership
    /// of the original `Error`.
    ///
    /// Returns `None` for `Io` errors which cannot be meaningfully converted
    /// to a simple string without losing the error kind information.
    pub fn to_message(&self) -> Option<String> {
        match self {
            Error::LLM(msg) => Some(msg.clone()),
            Error::Evaluation(msg) => Some(msg.clone()),
            Error::Evolution(msg) => Some(msg.clone()),
            Error::Memory(msg) => Some(msg.clone()),
            Error::Codebase(msg) => Some(msg.clone()),
            Error::Web(msg) => Some(msg.clone()),
            Error::Config(msg) => Some(msg.clone()),
            Error::Other(msg) => Some(msg.clone()),
            Error::Io(_) => None,
        }
    }

    /// Consumes the error and returns the message string.
    ///
    /// This is useful for error transformation patterns where you want to
    /// extract the message without cloning.
    ///
    /// Returns `None` for `Io` errors which cannot be meaningfully converted
    /// to a string without losing the error kind information.
    pub fn into_message(self) -> Option<String> {
        match self {
            Error::LLM(msg) => Some(msg),
            Error::Evaluation(msg) => Some(msg),
            Error::Evolution(msg) => Some(msg),
            Error::Memory(msg) => Some(msg),
            Error::Codebase(msg) => Some(msg),
            Error::Web(msg) => Some(msg),
            Error::Config(msg) => Some(msg),
            Error::Other(msg) => Some(msg),
            Error::Io(_) => None,
        }
    }

    /// Returns a structured error context for programmatic handling.
    ///
    /// The context provides categorized error information useful for
    /// programmatic error handling, logging, and telemetry.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperagent::Error;
    ///
    /// let err = Error::LLM("rate limited".into());
    /// let ctx = err.context();
    /// assert_eq!(ctx.category, "llm");
    /// assert!(ctx.retryable);
    /// ```
    pub fn context(&self) -> ErrorContext {
        match self {
            Error::LLM(msg) => ErrorContext {
                category: "llm",
                message: msg.clone(),
                retryable: true,
                action: "retry_with_backoff",
            },
            Error::Io(err) => ErrorContext {
                category: "io",
                message: err.to_string(),
                retryable: self.is_retryable(),
                action: if self.is_retryable() { "retry_with_backoff" } else { "fail_fast" },
            },
            Error::Evaluation(msg) => ErrorContext {
                category: "evaluation",
                message: msg.clone(),
                retryable: false,
                action: "log_and_continue",
            },
            Error::Evolution(msg) => ErrorContext {
                category: "evolution",
                message: msg.clone(),
                retryable: false,
                action: "fail_fast",
            },
            Error::Memory(msg) => ErrorContext {
                category: "memory",
                message: msg.clone(),
                retryable: false,
                action: "restore_from_backup",
            },
            Error::Codebase(msg) => ErrorContext {
                category: "codebase",
                message: msg.clone(),
                retryable: false,
                action: "rescan",
            },
            Error::Web(msg) => ErrorContext {
                category: "web",
                message: msg.clone(),
                retryable: true,
                action: "retry_with_backoff",
            },
            Error::Config(msg) => ErrorContext {
                category: "config",
                message: msg.clone(),
                retryable: false,
                action: "fix_configuration",
            },
            Error::Other(msg) => ErrorContext {
                category: "unknown",
                message: msg.clone(),
                retryable: false,
                action: "investigate",
            },
        }
    }

    /// Returns a human-readable recovery suggestion.
    ///
    /// This provides actionable guidance for users or operators on how
    /// to resolve or mitigate the error.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperagent::Error;
    /// use std::io::ErrorKind;
    ///
    /// let err = Error::LLM("rate limited".into());
    /// assert!(err.suggestion().contains("retry"));
    ///
    /// let err = Error::Io(std::io::Error::new(ErrorKind::NotFound, "file"));
    /// assert!(err.suggestion().contains("missing"));
    /// ```
    pub fn suggestion(&self) -> &'static str {
        match self {
            Error::LLM(_) => "Retry the request with exponential backoff. Check API rate limits and quotas.",
            Error::Io(err) => match err.kind() {
                std::io::ErrorKind::NotFound => "The requested file or resource was not found. Verify the path exists.",
                std::io::ErrorKind::PermissionDenied => "Permission denied. Check file permissions and ownership.",
                std::io::ErrorKind::TimedOut => "Operation timed out. Consider increasing timeout duration or retrying.",
                std::io::ErrorKind::ConnectionRefused => "Connection refused. Verify the service is running and accessible.",
                std::io::ErrorKind::ConnectionReset => "Connection reset by peer. This may be transient—retry the operation.",
                std::io::ErrorKind::ConnectionAborted => "Connection aborted. Check network stability and retry.",
                std::io::ErrorKind::Interrupted => "Operation was interrupted. Retry the operation.",
                _ => "An I/O error occurred. Check logs for details.",
            },
            Error::Evaluation(_) => "Evaluation failed. Review test results and fix the failing assertions.",
            Error::Evolution(_) => "Evolution process encountered an error. Check constraints and population health.",
            Error::Memory(_) => "Memory/archive error. Check disk space and archive integrity.",
            Error::Codebase(_) => "Codebase scanning failed. Verify source files are accessible and valid.",
            Error::Web(_) => "Web request failed. Check network connectivity and retry with backoff.",
            Error::Config(_) => "Configuration error. Verify settings file syntax and required fields.",
            Error::Other(_) => "An unknown error occurred. Check logs for details and investigate.",
        }
    }
}

/// Structured error context for programmatic handling.
///
/// Provides categorized error information useful for logging, telemetry,
/// and programmatic error handling decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorContext {
    /// Error category (e.g., "llm", "io", "evaluation")
    pub category: &'static str,
    /// Human-readable error message
    pub message: String,
    /// Whether this error is likely transient and may succeed on retry
    pub retryable: bool,
    /// Suggested action for handling this error
    pub action: &'static str,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} (retryable={}, action={})", self.category, self.message, self.retryable, self.action)
    }
}

impl ErrorContext {
    /// Returns a compact one-line summary of the error context.
    ///
    /// This provides a human-readable single-line representation suitable
    /// for logging and quick diagnostics.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperagent::Error;
    ///
    /// let err = Error::LLM("rate limited".into());
    /// let ctx = err.context();
    /// assert!(ctx.summary().contains("llm"));
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "{}: {} [{}]",
            self.category,
            if self.message.len() > 60 {
                let chars: String = self.message.chars().take(57).collect();
                format!("{}...", chars)
            } else {
                self.message.clone()
            },
            if self.retryable { "retry" } else { "fail" }
        )
    }

    /// Returns a structured action hint for programmatic error handling.
    ///
    /// This converts the string-based `action` field into a structured
    /// [`ErrorAction`] enum with concrete parameters (e.g., retry delays,
    /// max attempts) for automated recovery logic.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperagent::Error;
    ///
    /// let err = Error::LLM("rate limited".into());
    /// let action = err.context().action_hint();
    /// assert!(action.is_retryable());
    /// assert_eq!(action.max_attempts(), Some(3));
    /// ```
    pub fn action_hint(&self) -> ErrorAction {
        match self.action {
            "retry_with_backoff" => ErrorAction::RetryWithBackoff {
                initial_delay_ms: 100,
                max_attempts: 3,
            },
            "fail_fast" => ErrorAction::FailFast,
            "log_and_continue" => ErrorAction::LogAndContinue,
            "restore_from_backup" => ErrorAction::RestoreFromBackup,
            "rescan" => ErrorAction::Rescan,
            "fix_configuration" => ErrorAction::FixConfiguration,
            "investigate" => ErrorAction::Investigate,
            // Fallback for unknown actions
            _ => ErrorAction::Investigate,
        }
    }
}

/// Recommended action for error recovery.
///
/// This enum provides structured, actionable guidance for programmatic
/// error handling, enabling automated retry logic and recovery strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorAction {
    /// Retry with exponential backoff.
    /// Contains recommended initial delay in milliseconds.
    RetryWithBackoff {
        /// Initial delay before first retry (milliseconds)
        initial_delay_ms: u64,
        /// Maximum number of retry attempts
        max_attempts: u8,
    },
    /// Fail immediately, no retry.
    FailFast,
    /// Log and continue (non-critical error).
    LogAndContinue,
    /// Restore from backup.
    RestoreFromBackup,
    /// Rescan or refresh state.
    Rescan,
    /// Fix configuration and retry.
    FixConfiguration,
    /// Manual investigation required.
    Investigate,
}

impl ErrorAction {
    /// Returns the recommended initial delay in milliseconds for retryable actions.
    ///
    /// Returns `None` for non-retryable actions.
    pub fn initial_delay_ms(&self) -> Option<u64> {
        match self {
            ErrorAction::RetryWithBackoff { initial_delay_ms, .. } => Some(*initial_delay_ms),
            _ => None,
        }
    }

    /// Returns the maximum number of retry attempts for retryable actions.
    ///
    /// Returns `None` for non-retryable actions.
    pub fn max_attempts(&self) -> Option<u8> {
        match self {
            ErrorAction::RetryWithBackoff { max_attempts, .. } => Some(*max_attempts),
            _ => None,
        }
    }

    /// Returns `true` if this action indicates the error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(self, ErrorAction::RetryWithBackoff { .. })
    }

    /// Returns a new `RetryWithBackoff` action with customized retry parameters.
    ///
    /// This allows callers to adjust retry behavior based on specific error
    /// contexts (e.g., longer delays for rate limits vs. shorter for timeouts).
    ///
    /// Returns `None` for non-retryable actions.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperagent::ErrorAction;
    ///
    /// let action = ErrorAction::RetryWithBackoff {
    ///     initial_delay_ms: 100,
    ///     max_attempts: 3,
    /// };
    /// let custom = action.with_retry_params(500, 5).unwrap();
    /// assert_eq!(custom.initial_delay_ms(), Some(500));
    /// assert_eq!(custom.max_attempts(), Some(5));
    /// ```
    pub fn with_retry_params(self, initial_delay_ms: u64, max_attempts: u8) -> Option<Self> {
        match self {
            ErrorAction::RetryWithBackoff { .. } => Some(ErrorAction::RetryWithBackoff {
                initial_delay_ms,
                max_attempts,
            }),
            _ => None,
        }
    }

    /// Returns a new `RetryWithBackoff` action with exponential backoff scaling.
    ///
    /// Scales the initial delay by 2^attempt, useful for computing the delay
    /// at a specific retry attempt.
    ///
    /// Returns `None` for non-retryable actions or if attempt >= max_attempts.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperagent::ErrorAction;
    ///
    /// let action = ErrorAction::RetryWithBackoff {
    ///     initial_delay_ms: 100,
    ///     max_attempts: 3,
    /// };
    /// assert_eq!(action.delay_for_attempt(0), Some(100)); // 100 * 2^0
    /// assert_eq!(action.delay_for_attempt(1), Some(200)); // 100 * 2^1
    /// assert_eq!(action.delay_for_attempt(2), Some(400)); // 100 * 2^2
    /// assert_eq!(action.delay_for_attempt(3), None);      // at max
    /// ```
    pub fn delay_for_attempt(&self, attempt: u8) -> Option<u64> {
        match self {
            ErrorAction::RetryWithBackoff {
                initial_delay_ms,
                max_attempts,
            } => {
                if attempt < *max_attempts {
                    // Exponential backoff: delay = initial * 2^attempt
                    Some(initial_delay_ms.saturating_mul(1 << attempt))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl fmt::Display for ErrorAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorAction::RetryWithBackoff { initial_delay_ms, max_attempts } => {
                write!(f, "retry_with_backoff (delay={}ms, max={})", initial_delay_ms, max_attempts)
            }
            ErrorAction::FailFast => write!(f, "fail_fast"),
            ErrorAction::LogAndContinue => write!(f, "log_and_continue"),
            ErrorAction::RestoreFromBackup => write!(f, "restore_from_backup"),
            ErrorAction::Rescan => write!(f, "rescan"),
            ErrorAction::FixConfiguration => write!(f, "fix_configuration"),
            ErrorAction::Investigate => write!(f, "investigate"),
        }
    }
}

/// A specialized Result type for Hyperagent operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_action_retryable() {
        let action = ErrorAction::RetryWithBackoff {
            initial_delay_ms: 100,
            max_attempts: 3,
        };
        assert!(action.is_retryable());
        assert_eq!(action.initial_delay_ms(), Some(100));
        assert_eq!(action.max_attempts(), Some(3));
    }

    #[test]
    fn test_error_action_non_retryable() {
        let actions = [
            ErrorAction::FailFast,
            ErrorAction::LogAndContinue,
            ErrorAction::RestoreFromBackup,
            ErrorAction::Rescan,
            ErrorAction::FixConfiguration,
            ErrorAction::Investigate,
        ];
        for action in &actions {
            assert!(!action.is_retryable());
            assert!(action.initial_delay_ms().is_none());
            assert!(action.max_attempts().is_none());
        }
    }

    #[test]
    fn test_error_action_display() {
        let action = ErrorAction::RetryWithBackoff {
            initial_delay_ms: 500,
            max_attempts: 5,
        };
        let display = format!("{}", action);
        assert!(display.contains("500ms"));
        assert!(display.contains("max=5"));

        assert!(format!("{}", ErrorAction::FailFast).contains("fail_fast"));
        assert!(format!("{}", ErrorAction::LogAndContinue).contains("log_and_continue"));
    }

    #[test]
    fn test_error_context_summary_truncation() {
        let long_msg = "a".repeat(100);
        let ctx = ErrorContext {
            category: "test",
            message: long_msg.clone(),
            retryable: true,
            action: "test_action",
        };
        let summary = ctx.summary();
        assert!(summary.len() < 100);
        assert!(summary.ends_with("...]"));
    }

    #[test]
    fn test_error_context_summary_short() {
        let ctx = ErrorContext {
            category: "test",
            message: "short".into(),
            retryable: false,
            action: "test_action",
        };
        let summary = ctx.summary();
        assert!(summary.contains("short"));
        assert!(summary.contains("[fail]"));
    }

    #[test]
    fn test_error_context_action_hint_retryable() {
        let ctx = ErrorContext {
            category: "llm",
            message: "rate limited".into(),
            retryable: true,
            action: "retry_with_backoff",
        };
        let hint = ctx.action_hint();
        assert!(hint.is_retryable());
        assert_eq!(hint.initial_delay_ms(), Some(100));
        assert_eq!(hint.max_attempts(), Some(3));
    }

    #[test]
    fn test_error_context_action_hint_fail_fast() {
        let ctx = ErrorContext {
            category: "evolution",
            message: "constraint violated".into(),
            retryable: false,
            action: "fail_fast",
        };
        let hint = ctx.action_hint();
        assert!(!hint.is_retryable());
        assert!(hint.initial_delay_ms().is_none());
    }

    #[test]
    fn test_error_context_action_hint_unknown() {
        let ctx = ErrorContext {
            category: "test",
            message: "test".into(),
            retryable: false,
            action: "unknown_action",
        };
        let hint = ctx.action_hint();
        assert!(!hint.is_retryable());
    }

    #[test]
    fn test_error_action_hint_from_error() {
        let err = Error::LLM("timeout".into());
        let action = err.context().action_hint();
        assert!(matches!(action, ErrorAction::RetryWithBackoff { .. }));

        let err = Error::Config("invalid".into());
        let action = err.context().action_hint();
        assert!(matches!(action, ErrorAction::FixConfiguration));
    }

    #[test]
    fn test_error_action_with_retry_params() {
        let action = ErrorAction::RetryWithBackoff {
            initial_delay_ms: 100,
            max_attempts: 3,
        };

        // Test customization
        let custom = action.with_retry_params(500, 5).unwrap();
        assert_eq!(custom.initial_delay_ms(), Some(500));
        assert_eq!(custom.max_attempts(), Some(5));

        // Original should be unchanged (Copy trait)
        assert_eq!(action.initial_delay_ms(), Some(100));
        assert_eq!(action.max_attempts(), Some(3));
    }

    #[test]
    fn test_error_action_with_retry_params_non_retryable() {
        let actions = [
            ErrorAction::FailFast,
            ErrorAction::LogAndContinue,
            ErrorAction::RestoreFromBackup,
            ErrorAction::Rescan,
            ErrorAction::FixConfiguration,
            ErrorAction::Investigate,
        ];

        for action in actions {
            assert!(action.with_retry_params(100, 3).is_none());
        }
    }

    #[test]
    fn test_error_action_delay_for_attempt() {
        let action = ErrorAction::RetryWithBackoff {
            initial_delay_ms: 100,
            max_attempts: 3,
        };

        // Exponential backoff: 100 * 2^attempt
        assert_eq!(action.delay_for_attempt(0), Some(100));
        assert_eq!(action.delay_for_attempt(1), Some(200));
        assert_eq!(action.delay_for_attempt(2), Some(400));

        // At max attempts, should return None
        assert_eq!(action.delay_for_attempt(3), None);
        assert_eq!(action.delay_for_attempt(4), None);
    }

    #[test]
    fn test_error_action_delay_for_attempt_non_retryable() {
        let action = ErrorAction::FailFast;
        assert_eq!(action.delay_for_attempt(0), None);
        assert_eq!(action.delay_for_attempt(1), None);
    }

    #[test]
    fn test_error_action_delay_for_attempt_overflow_protection() {
        // Test that saturating_mul prevents overflow
        let action = ErrorAction::RetryWithBackoff {
            initial_delay_ms: u64::MAX / 2,
            max_attempts: 64,
        };

        // Should saturate rather than overflow
        let delay = action.delay_for_attempt(0);
        assert!(delay.is_some());
    }
}
