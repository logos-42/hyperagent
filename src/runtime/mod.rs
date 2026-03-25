pub mod loop_;
pub mod state;
pub mod thermodynamics;
pub mod constraints;
pub mod selection;
pub mod environment;
pub mod local_runtime;

pub use loop_::EvolutionLoop;
pub use state::{RuntimeState, RuntimeConfig as EvolutionRuntimeConfig};
pub use thermodynamics::{EnergyState, DissipationScale, InfoEnergyCoupling, FitnessLandscape};
pub use constraints::{
    ConstraintSystem, HardConstraints, SoftConstraints, CodeMetrics,
    EvolutionDirection, TopologicalConstraints,
};
pub use selection::{Selector, SelectionType, Individual, PopulationStats};
pub use environment::{
    Environment, EnvironmentConfig, EnvironmentInfo,
    SessionMeta, SessionStatus,
    IterationState, IterationStatus, IterationMetrics,
};
pub use local_runtime::{
    LocalRuntime, LocalRuntimeBuilder,
    RuntimeConfig as LocalRuntimeConfig,
    ExecutionContext, ExecutionResult, ProviderStats,
};
