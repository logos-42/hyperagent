use anyhow::Result;
use hyperagent::{LLMClientImpl, SelfEvolutionEngine, SelfEvolutionConfig, SelfEvolutionStatus};
use std::time::Instant;

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

    let start = Instant::now();
    let mut engine = SelfEvolutionEngine::new(client, config);
    let results = engine.run().await?;
    let duration = start.elapsed();

    // Calculate summary statistics
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for r in &results {
        match r.status {
            SelfEvolutionStatus::Accepted => accepted += 1,
            SelfEvolutionStatus::Rejected => rejected += 1,
            SelfEvolutionStatus::Skipped => skipped += 1,
            SelfEvolutionStatus::Failed => failed += 1,
        }
    }

    let total = results.len();
    let success_rate = if total > 0 {
        (accepted as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };

    println!("\n=== Summary ===");
    println!("  Total iterations: {}", total);
    println!("  ✓ Accepted: {} ({:.1}%)", accepted, accepted as f64 / total as f64 * 100.0);
    println!("  ✗ Rejected: {} ({:.1}%)", rejected, rejected as f64 / total as f64 * 100.0);
    println!("  ~ Skipped:  {} ({:.1}%)", skipped, skipped as f64 / total as f64 * 100.0);
    println!("  ! Failed:   {} ({:.1}%)", failed, failed as f64 / total as f64 * 100.0);
    println!("  Success rate: {}%", success_rate);
    println!("  Duration: {:.2}s", duration.as_secs_f64());

    // Print individual results
    println!("\n=== Details ===");
    for r in &results {
        let icon = match r.status {
            SelfEvolutionStatus::Accepted => "✓",
            SelfEvolutionStatus::Rejected => "✗",
            SelfEvolutionStatus::Skipped => "~",
            SelfEvolutionStatus::Failed => "!",
        };
        println!(
            "  [{}] Iter {}: {} - {}",
            icon, r.iteration, r.file, r.description
        );
    }

    // Exit with appropriate status code
    // 0 = at least one accepted improvement
    // 1 = no accepted improvements (all rejected/skipped/failed)
    // 2 = all iterations failed (critical error)
    if failed == total {
        tracing::error!("All iterations failed - critical error");
        std::process::exit(2);
    } else if accepted == 0 {
        tracing::warn!("No accepted improvements in this run");
        std::process::exit(1);
    }

    tracing::info!("Self-evolution completed successfully with {} accepted improvements", accepted);
    Ok(())
}