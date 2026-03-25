use anyhow::Result;
use hyperagent::{LLMClientImpl, SelfEvolutionEngine, SelfEvolutionConfig};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    if std::env::var("NO_PROXY").is_err() {
        std::env::set_var("NO_PROXY", "localhost,127.0.0.1");
    }
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let client = LLMClientImpl::from_env()?;
    tracing::info!(
        "Self-evolution engine starting: provider={:?}, model={}",
        client.provider(),
        client.model()
    );

    let config = SelfEvolutionConfig {
        // 默认 dry_run=true，安全模式：编译+测试通过后回滚，不会真正修改代码
        // 设为 false 才会真正 git commit 修改
        dry_run: std::env::var("SELF_EVOLVE_DRY_RUN")
            .map(|v| v != "false")
            .unwrap_or(true),
        max_iterations: std::env::var("SELF_EVOLVE_ITERATIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5),
        ..Default::default()
    };

    let mut engine = SelfEvolutionEngine::new(client, config);
    let results = engine.run().await?;

    println!("\n=== Summary ===");
    for r in &results {
        let icon = match r.status {
            hyperagent::SelfEvolutionStatus::Accepted => "+",
            hyperagent::SelfEvolutionStatus::Rejected => "-",
            hyperagent::SelfEvolutionStatus::Skipped => "~",
            hyperagent::SelfEvolutionStatus::Failed => "!",
        };
        println!(
            "  [{}] Iter {}: {} - {}",
            icon, r.iteration, r.file, r.description
        );
    }

    Ok(())
}
