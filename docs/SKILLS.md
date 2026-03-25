# Hyperagent System Skills

## System Identity

**Name**: Hyperagent - Self-Evolving AI Agent System

**GitHub**: https://github.com/logos-42/hyperagent

**Core Capability**: Execute → Evaluate → Mutate → Meta-Mutate → Select (Evolution Loop)

---

## Knowledge Base

### Required Documents

Read in this order:

1. `README.md` - System architecture and usage
2. `docs/AR.md` - Original design specification
3. `docs/arc2.md` - Theoretical framework (thermodynamics mapping)
4. `docs/arc2_part2.md` - Phase transitions and diagnostics
5. `docs/QUICKSTART.md` - Practical implementation guide
6. `docs/SUMMARY.md` - System completion summary

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EvolutionLoop                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Executor │ → │ Evaluator│ → │ Mutator  │ → │ Selector │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ↑              │                              │       │
│       │              ↓                              │       │
│       │        ┌──────────┐                         │       │
│       │        │ Archive  │ ←────────────────────────┘       │
│       │        └──────────┘                                 │
│       │                                                      │
│       └──────────────────────────────────────────────────────┘
│                        (feedback loop)
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `EvolutionLoop` | `src/runtime/loop_.rs` | Main evolution cycle |
| `Executor` | `src/agent/executor.rs` | Task execution via LLM |
| `Evaluator` | `src/eval/evaluator.rs` | Score calculation (0-10) |
| `Mutator` | `src/agent/mutator.rs` | Generate agent variants |
| `MetaMutator` | `src/agent/meta_mutator.rs` | Evolve mutation strategy |
| `Selector` | `src/runtime/selection.rs` | Selection mechanisms |
| `Archive` | `src/memory/archive.rs` | Bounded memory storage |

---

## Thermodynamic Framework

### Key Concepts (Prigogine's Dissipative Structure Theory)

| Physical Concept | Code Implementation |
|-----------------|---------------------|
| Far from equilibrium | `initial_temperature = 0.8` |
| Energy flow | LLM token consumption |
| Entropy production | `entropy_production_rate` |
| Dissipation | `Archive::compress()` |
| Deborah number | `De = τ_response / τ_drive` |
| Critical point | `near_critical(0.2)` |
| Boltzmann selection | `exp(f_i/T) / Σexp(f_j/T)` |

### Core Structures

```rust
// src/runtime/thermodynamics.rs
pub struct EnergyState {
    pub free_energy: f32,
    pub entropy: f32,
    pub entropy_production_rate: f32,
    pub temperature: f32,
}

pub struct DissipationScale {
    pub relaxation_time: f32,
    pub diffusion_length: f32,
    pub boundary_layer: f32,
    pub deborah_number: f32,
}

pub struct FitnessLandscape {
    pub current_fitness: f32,
    pub gradient: f32,
    pub curvature: f32,
    pub escape_probability: f32,
}
```

---

## Evolution Loop Algorithm

```
FOR generation = 1 TO max_generations:
    1. EXECUTE: result = executor.run(agent, task)
    2. EVALUATE: score = evaluator.score(task, result)
    3. ARCHIVE: archive.store(agent, score, task, result)
    4. MUTATE: new_agent = mutator.mutate(agent, failures)
    5. META-MUTATE: IF generation % meta_mutation_interval == 0:
                       strategy = meta_mutator.evolve(history)
    6. SELECT: agent = selector.select(archive)
    7. CHECK: IF stagnation_detected:
                 inject_diversity()
```

---

## Configuration Parameters

### Recommended Starting Values

```rust
SelfEvolvingConfig {
    // Thermodynamic parameters
    initial_temperature: 0.8,    // High exploration
    annealing_rate: 0.95,        // 5% cooling per generation
    min_temperature: 0.1,        // Minimum exploration
    
    // Evolution parameters
    population_size: 20,
    elite_ratio: 0.2,            // Keep top 20%
    mutation_rate: 0.7,          // 70% mutation
    selection_pressure: 0.3,     // Medium pressure
    
    // Dissipation parameters
    entropy_threshold: 0.5,
    stagnation_threshold: 5,     // Inject diversity after 5 stagnant generations
    
    // Constraints
    constraints: ConstraintSystem::default(),
}
```

### Parameter Tuning Guide

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| `diversity < 0.1` | Low diversity | ↑ temperature, ↑ mutation_rate |
| `stagnation_counter > 5` | Stagnation | Inject diversity, change direction |
| `De ≈ 1` | Critical point | Prepare for phase transition |
| Early convergence | Temperature too low | ↑ initial_temperature |
| Never converges | Temperature too high | ↓ annealing_rate |

---

## Selection Strategies

Implement these 6 selection mechanisms:

```rust
pub enum SelectionType {
    /// P(i) = f_i / Σf
    RouletteWheel,
    
    /// Random k, select best
    Tournament { tournament_size: usize },
    
    /// Select top-k
    Truncation { top_k: usize },
    
    /// P(i) = exp(f_i/T) / Σexp(f_j/T)
    Boltzmann { temperature: f32 },
    
    /// Rank-based selection
    RankBased,
    
    /// (1-w)*fitness + w*novelty
    DiversityPreserving { diversity_weight: f32 },
}
```

---

## Constraint System

### Hard Constraints (Non-negotiable)

```rust
pub struct HardConstraints {
    pub max_code_length: usize,           // Default: 10000
    pub max_cyclomatic_complexity: usize, // Default: 15
    pub forbidden_patterns: Vec<String>,  // ["eval(", "exec("]
    pub max_nesting_depth: usize,         // Default: 5
}
```

### Soft Constraints (Fitness Penalties)

```rust
pub struct SoftConstraints {
    pub complexity_penalty: f32,      // Default: 0.1
    pub redundancy_penalty: f32,      // Default: 0.05
    pub deviation_penalty: f32,       // Default: 0.2
    pub direction_weights: Vec<(EvolutionDirection, f32)>,
}
```

### Evolution Directions

```rust
pub enum EvolutionDirection {
    Efficiency,       // Optimize execution speed
    Robustness,       // Enhance error handling
    Generalization,   // Improve generalization
    Minimalism,       // Reduce code size
    Exploration,      // Increase diversity
}
```

---

## Code Metrics

Calculate these metrics for each agent:

```rust
pub struct CodeMetrics {
    pub cyclomatic_complexity: usize,  // Decision points
    pub lines_of_code: usize,
    pub redundancy_ratio: f32,         // Duplicate code ratio
    pub code_entropy: f32,             // Information density
    pub vocabulary_diversity: f32,     // Unique tokens / total tokens
}
```

---

## Diagnostic Tools

### Population Statistics

```rust
pub struct PopulationStats {
    pub size: usize,
    pub mean_fitness: f32,
    pub std_fitness: f32,
    pub max_fitness: f32,
    pub min_fitness: f32,
    pub diversity: f32,  // std / mean
}
```

### Critical Point Detection

```rust
impl DissipationScale {
    pub fn near_critical(&self, threshold: f32) -> bool {
        (self.deborah_number - 1.0).abs() < threshold
    }
}
```

### Stagnation Detection

```rust
fn check_stagnation(&self, fitness_history: &[f32]) -> bool {
    if fitness_history.len() < 10 {
        return false;
    }
    
    let recent = fitness_history.iter().rev().take(5).sum::<f32>();
    let previous = fitness_history.iter().rev().skip(5).take(5).sum::<f32>();
    
    (recent - previous).abs() < 0.01  // No improvement
}
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure

- [ ] `Agent` struct with id, code, prompt, generation
- [ ] `Executor` with LLM integration
- [ ] `Evaluator` with scoring (0-10)
- [ ] `Archive` with bounded storage

### Phase 2: Evolution System

- [ ] `Mutator` with strategy-based mutation
- [ ] `MetaMutator` for strategy evolution
- [ ] `Selector` with 6 selection types
- [ ] `EvolutionLoop` main cycle

### Phase 3: Thermodynamic Framework

- [ ] `EnergyState` with Boltzmann factor
- [ ] `DissipationScale` with critical detection
- [ ] `FitnessLandscape` with gradient/curvature
- [ ] `InfoEnergyCoupling` with mutual information

### Phase 4: Constraints

- [ ] `HardConstraints` validation
- [ ] `SoftConstraints` penalty calculation
- [ ] `CodeMetrics` computation
- [ ] `EvolutionDirection` switching

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_boltzmann_factor() {
    let state = EnergyState::new(100.0, 0.5);
    assert!(state.boltzmann_factor(-1.0) > 0.5);
    assert!(state.boltzmann_factor(1.0) < 0.5);
}

#[test]
fn test_tournament_selection() {
    let selector = Selector::new(SelectionType::Tournament { tournament_size: 2 });
    let population = create_test_population();
    let selected = selector.select(&population);
    assert!(selected.unwrap().fitness >= 3.0);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_evolution_loop() {
    let config = SelfEvolvingConfig::default();
    let mut system = SelfEvolvingSystem::new(config);
    let result = system.run("test task").await;
    assert!(result.is_ok());
}
```

---

## Example Usage

### Basic Execution

```rust
use hyperagent::{EvolutionLoop, RuntimeConfig, LLMConfig, RigClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm_config = LLMConfig::default();
    let runtime_config = RuntimeConfig {
        max_generations: 100,
        population_size: 10,
        top_k_selection: 5,
        checkpoint_interval: 10,
        meta_mutation_interval: 20,
    };

    let client = RigClient::new(&llm_config)?;
    let mut loop_ = EvolutionLoop::new(client, runtime_config);
    let state = loop_.run("Write a sorting function").await?;

    println!("{}", state.summary());
    Ok(())
}
```

### Self-Evolving Mode

```rust
use hyperagent::{SelfEvolvingConfig, SelfEvolvingSystem};

let config = SelfEvolvingConfig::high_exploration();
let mut system = SelfEvolvingSystem::new(config);
system.run("Write a function to calculate Fibonacci numbers").await?;
```

---

## File Structure Reference

```
src/
├── lib.rs                  # Public API exports
├── main.rs                 # Binary entry point
├── agent/
│   ├── mod.rs              # Agent, MutationStrategy structs
│   ├── executor.rs         # Task execution
│   ├── mutator.rs          # Agent mutation
│   └── meta_mutator.rs     # Strategy evolution
├── eval/
│   ├── mod.rs              # Re-exports
│   ├── evaluator.rs        # LLM/Rule/Ensemble evaluators
│   └── benchmark.rs        # Benchmark suite
├── llm/
│   ├── mod.rs              # Re-exports
│   ├── client.rs           # LLMClient trait + RigClient
│   └── prompts.rs          # Prompt templates
├── memory/
│   ├── mod.rs              # Re-exports, Record struct
│   ├── archive.rs          # Bounded archive with compression
│   └── lineage.rs          # Evolution lineage tree
└── runtime/
    ├── mod.rs              # Re-exports
    ├── state.rs            # RuntimeState, RuntimeConfig
    ├── loop_.rs            # EvolutionLoop implementation
    ├── thermodynamics.rs   # Energy, entropy, dissipation
    ├── constraints.rs      # Hard/soft constraints
    └── selection.rs        # 6 selection strategies

examples/
└── self_evolving.rs        # Self-evolving system demo

docs/
├── AR.md                   # Original design
├── arc2.md                 # Theoretical framework
├── arc2_part2.md           # Phase transitions
├── QUICKSTART.md           # Quick start guide
├── SUMMARY.md              # Completion summary
└── SKILLS.md               # This document
```

---

## Key Formulas

### Boltzmann Factor
```
P(accept) = exp(-ΔE / kT)
where k = 0.1 (normalized)
```

### Mutual Information
```
I(X;Y) ≈ 0.5 * log(1 + cov²/(var_x * var_y))
```

### Deborah Number
```
De = τ_response / τ_drive
where τ_response = L²/D (L=population, D=mutation_rate)
```

### Annealing Schedule
```
T(generation) = T₀ * rate^generation
```

### Fitness Penalty
```
penalized_score = base_score - (complexity * 0.1) - (redundancy * 0.05)
```

---

## Debugging Guide

### Common Issues

**Issue**: Compilation error `unresolved import reqwest`
**Solution**: Add `reqwest = { version = "0.11", features = ["rustls-tls"] }` to Cargo.toml

**Issue**: Evolution stagnation
**Solution**: 
```rust
config.mutation_rate = 0.8;
config.initial_temperature = 0.9;
```

**Issue**: Early convergence
**Solution**: 
```rust
config.selection_pressure = 0.2;
config.diversity_weight = 0.3;
```

---

## Extension Points

### Add New Selection Strategy
```rust
// In src/runtime/selection.rs
SelectionType::Custom { parameter: f32 }

impl Selector {
    fn custom_select(&self, population: &[Individual], param: f32) -> Option<&Individual> {
        // Implementation
    }
}
```

### Add New Evolution Direction
```rust
// In src/runtime/constraints.rs
pub enum EvolutionDirection {
    // Existing...
    Creativity,  // New: Optimize for novel solutions
}
```

### Add New Constraint
```rust
// In src/runtime/constraints.rs
pub struct HardConstraints {
    // Existing...
    pub max_memory_usage: usize,  // New constraint
}
```

---

## Performance Benchmarks

### Expected Performance

| Metric | Target |
|--------|--------|
| Generations/minute | 5-10 (depends on LLM latency) |
| Population size | 10-50 (configurable) |
| Archive capacity | 100 (with compression) |
| Memory usage | < 500MB |

### Optimization Tips

1. **Parallel execution**: Use `tokio::spawn` for concurrent agent evaluation
2. **Archive compression**: Cluster similar agents, keep only representatives
3. **LLM batching**: Batch multiple LLM calls when possible
4. **Caching**: Cache LLM responses for identical prompts

---

## API Reference

### EvolutionLoop

```rust
pub struct EvolutionLoop<C: LLMClient> {
    executor: Executor<C>,
    evaluator: Evaluator<C>,
    mutator: Mutator<C>,
    meta_mutator: MetaMutator<C>,
    state: RuntimeState,
}

impl<C: LLMClient + Clone> EvolutionLoop<C> {
    pub fn new(client: C, config: RuntimeConfig) -> Self;
    pub async fn run(&mut self, task: &str) -> Result<RuntimeState>;
    pub async fn run_with_iterations(&mut self, task: &str, iterations: usize) -> Result<RuntimeState>;
}
```

### RuntimeState

```rust
pub struct RuntimeState {
    pub config: RuntimeConfig,
    pub phase: RuntimePhase,
    pub current_generation: u32,
    pub best_score: f32,
    pub best_agent: Option<Agent>,
    pub archive: Archive,
    pub lineage: Lineage,
}
```

---

## Version History

- **v0.1.0**: Basic evolution loop
- **v0.2.0**: Thermodynamic framework added
- **v0.3.0**: 6 selection strategies, constraint system
- **Current**: Full self-evolving system with diagnostics

---

## Quick Commands

```bash
# Build
cargo build

# Test
cargo test

# Run basic example
cargo run

# Run self-evolving example
cargo run --example self_evolving

# Check code
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

---

**End of Skills Document**

For questions, refer to:
- `docs/arc2.md` - Theoretical foundation
- `docs/QUICKSTART.md` - Practical guide
- GitHub Issues - Community support
