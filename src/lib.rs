pub mod agent;
pub mod codebase;
pub mod eval;
pub mod llm;
pub mod memory;
pub mod runtime;
pub mod self_evolution;
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
pub use auto_research::{AutoResearch, ResearchConfig, Experiment, ExperimentOutcome};
pub use web::{WebSearchTool, WebFetchTool, WebSearchResult, FetchOutput, SearchOutput, build_web_context_prompt};
pub use tools::{
    CodebaseGrepTool, CodebaseSearchTool, CodebaseReadTool, CodebaseTreeTool,
    GrepMatch, GrepOutput, FileEntry, SearchFilesOutput, ReadFileOutput, TreeOutput,
};
