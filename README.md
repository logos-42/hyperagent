# Hyperagent

A Rust-based **self-evolving agent system** that iteratively improves AI agents through an evolutionary loop: execute, evaluate, mutate, and meta-mutate. Uses LLMs (via `rig-core`) to both perform tasks and improve the agents themselves -- including improving the mutation strategy (meta-learning).

## Architecture

```
                    +-----------------+
                    |  EvolutionLoop   |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
        +-----+-----+ +-----+-----+ +------+------+
        | Executor   | | Evaluator  | | Mutator    |
        | (LLM call) | | (score 0-10)| | (LLM call) |
        +-----------+ +------------+ +------------+
                             |
                    +--------+--------+
                    |  MetaMutator    |
                    | (evolves the    |
                    |  mutation strat)|
                    +--------+--------+
                             |
                    +--------+--------+
                    |  Memory Layer   |
                    |  Archive +      |
                    |  Lineage        |
                    +-----------------+
```

Each evolution iteration follows the phase cycle:

1. **Execute** -- Run the current agent on the task via LLM
2. **Evaluate** -- Score the output (correctness, efficiency, robustness, 0-10)
3. **Archive** -- Store the result; record in lineage tree
4. **Mutate** -- Generate a new agent variant using past failures as feedback
5. **Meta-mutate** (periodic) -- Evolve the mutation strategy itself
6. **Select** -- Compare against archive best; keep the better agent

## Project Structure

```
src/
├── lib.rs                  # Library root, re-exports all public types
├── main.rs                 # Binary entry point
├── agent/
│   ├── mod.rs              # Agent, MutationStrategy structs
│   ├── executor.rs         # Task execution via LLM
│   ├── mutator.rs          # Agent mutation based on failures
│   └── meta_mutator.rs     # Meta-learning: evolves mutation strategy
├── eval/
│   ├── mod.rs              # Re-exports
│   ├── evaluator.rs        # LLM / rule-based / ensemble evaluators
│   └── benchmark.rs        # Benchmark task suites and reports
├── llm/
│   ├── mod.rs              # Re-exports
│   ├── client.rs           # LLMClient trait + RigClient (OpenAI-compatible)
│   └── prompts.rs          # Prompt templates (execute/mutate/meta/evaluate)
├── memory/
│   ├── mod.rs              # Re-exports, Record struct
│   ├── archive.rs          # Bounded archive with compression & top-k
│   └── lineage.rs          # Evolution lineage tree
└── runtime/
    ├── mod.rs              # Re-exports
    ├── state.rs            # RuntimeState, RuntimeConfig, RuntimePhase
    └── loop_.rs            # Core EvolutionLoop
```

## Key Features

- **Trait-based abstraction** -- `LLMClient` trait for easy mocking and provider swapping
- **Connection pooling** -- `Arc`-shared HTTP client with `Semaphore` concurrency control
- **Bounded memory** -- Archive auto-compresses when capacity is reached
- **Meta-learning** -- The mutation strategy itself is improved over generations
- **Three evaluator strategies** -- LLM-based, rule-based heuristic, ensemble (multi-evaluator average)
- **Full observability** -- Structured logging via `tracing` with phase-based state machine

## Quick Start

### Prerequisites

- Rust 1.75+
- An OpenAI-compatible API key

### Usage

```rust
use hyperagent::{EvolutionLoop, RuntimeConfig, LLMConfig, RigClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure LLM (reads OPENAI_API_KEY env var by default)
    let llm_config = LLMConfig::default();

    // Configure evolution parameters
    let runtime_config = RuntimeConfig {
        max_generations: 100,
        population_size: 5,
        top_k_selection: 3,
        checkpoint_interval: 10,
        meta_mutation_interval: 20,
    };

    // Build and run
    let client = RigClient::new(&llm_config)?;
    let mut loop_ = EvolutionLoop::new(client, runtime_config);
    let state = loop_.run("Write a function to solve X efficiently").await?;

    println!("{}", state.summary());
    Ok(())
}
```

### Configuration

| Field | Default | Description |
|---|---|---|
| `LLMConfig.model` | `gpt-4o` | Model name |
| `LLMConfig.api_key` | `$OPENAI_API_KEY` | API key |
| `LLMConfig.base_url` | `None` | Custom endpoint (OpenAI-compatible) |
| `LLMConfig.max_concurrent` | `8` | Max concurrent LLM requests |
| `RuntimeConfig.max_generations` | `100` | Max evolution generations |
| `RuntimeConfig.meta_mutation_interval` | `20` | Meta-mutate every N generations |

### Run the Binary

```bash
export OPENAI_API_KEY=sk-...
cargo run
```

## Dependencies

| Crate | Purpose |
|---|---|
| `rig-core` | LLM provider abstraction (OpenAI-compatible) |
| `tokio` | Async runtime |
| `serde` / `serde_json` | Serialization |
| `anyhow` | Error handling |
| `tracing` | Structured logging |
| `chrono` | Timestamps |
| `uuid` | Unique IDs |
| `async-trait` | Async trait support |

## License

MIT
