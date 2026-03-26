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
