use anyhow::Result;
use hyperagent::{EvolutionLoop, RuntimeConfig, RuntimeState, LLMClientImpl};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    // Bypass system proxy for local LLM services (Ollama etc.)
    if std::env::var("NO_PROXY").is_err() {
        std::env::set_var("NO_PROXY", "localhost,127.0.0.1");
    }
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("Starting Hyperagent - Self-Evolving Agent System");

    let client = LLMClientImpl::from_env()?;
    tracing::info!("Using provider: {:?}, model: {}", client.provider(), client.model());
    let persist_dir = std::path::Path::new(".hyperagent/data");

    let runtime_config = RuntimeConfig {
        max_generations: 100,
        population_size: 3,
        top_k_selection: 2,
        checkpoint_interval: 5,
        meta_mutation_interval: 3,
        initial_temperature: 1.5,
        annealing_rate: 0.9,
        mutation_rate: 0.1,
        selection_pressure: 0.3,
        num_branches: 3,
        novelty_weight: 0.5,
        diversity_threshold: 0.8,
    };

    let mut evolution_loop = EvolutionLoop::new(client, RuntimeState::with_persistence(runtime_config, persist_dir));

    let task = "Write a Rust function that calculates the Fibonacci number at position n efficiently";
    
    tracing::info!("Task: {}", task);
    
    let final_state: RuntimeState = evolution_loop.run_with_iterations(task, 5).await?;

    tracing::info!("=== Final Results ===");
    tracing::info!("{}", final_state.summary());
    
    if let Some(best_agent) = evolution_loop.get_best_agent() {
        tracing::info!("Best Agent ID: {}", best_agent.id);
        tracing::info!("Best Agent Generation: {}", best_agent.generation);
    }

    tracing::info!("Archive size: {}", final_state.archive.size());

    Ok(())
}
