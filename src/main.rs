use anyhow::Result;
use hyperagent::{EvolutionLoop, RuntimeConfig, RuntimeState, LLMConfig, RigClient};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("Starting Hyperagent - Self-Evolving Agent System");

    let llm_config = LLMConfig::default();
    tracing::info!("Using model: {}", llm_config.model);

    let client = RigClient::new(&llm_config)?;
    let runtime_config = RuntimeConfig {
        max_generations: 10,
        population_size: 3,
        top_k_selection: 2,
        checkpoint_interval: 5,
        meta_mutation_interval: 3,
    };

    let mut evolution_loop = EvolutionLoop::new(client, runtime_config);

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
