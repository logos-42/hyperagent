//! 统一自动研究入口 — Karpathy 风格 + 结构化自改进 + Web 搜索
//!
//! 用法:
//!   cargo run --bin research                           # 默认：自动 commit + push + web 搜索
//!   RESEARCH_DRY_RUN=true cargo run --bin research      # 安全模式：只观察，不提交
//!   RESEARCH_STRICT=true cargo run --bin research       # 严格模式：测试 100% 通过才接受
//!   RESEARCH_AUTO_PUSH=false cargo run --bin research   # 只 commit 不 push
//!   RESEARCH_ITERATIONS=10 cargo run --bin research     # 自定义迭代数
//!   RESEARCH_WEB=false cargo run --bin research         # 禁用 web 搜索

use anyhow::Result;
use hyperagent::{AutoResearch, LLMClientImpl, ResearchConfig};

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
        "Auto Research: provider={:?}, model={}",
        client.provider(),
        client.model()
    );

    let auto_push = std::env::var("RESEARCH_AUTO_PUSH")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(true);  // 默认开启 push

    let dry_run = std::env::var("RESEARCH_DRY_RUN")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let strict = std::env::var("RESEARCH_STRICT")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let max_iterations = std::env::var("RESEARCH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let enable_web = std::env::var("RESEARCH_WEB")
        .map(|v| !(v == "false" || v == "0"))
        .unwrap_or(true);  // 默认开启 web 搜索

    if dry_run {
        tracing::warn!("DRY RUN — changes will be reverted, nothing committed");
    }

    let config = ResearchConfig {
        auto_push,
        dry_run,
        strict,
        max_iterations,
        enable_web,
        ..Default::default()
    };

    let mut engine = AutoResearch::new(client, config);
    let experiments = engine.run().await?;

    println!("\n=== Experiment Log ===");
    for e in &experiments {
        let icon = match e.outcome {
            hyperagent::ExperimentOutcome::Improved => "+",
            hyperagent::ExperimentOutcome::Neutral => "=",
            hyperagent::ExperimentOutcome::Regressed => "-",
            hyperagent::ExperimentOutcome::Failed => "!",
        };
        println!(
            "  [{}] #{} src/{} | {} | {}/{} -> {}/{}",
            icon, e.iteration, e.file,
            &e.hypothesis[..e.hypothesis.len().min(50)],
            e.tests_before.0, e.tests_before.1,
            e.tests_after.0, e.tests_after.1,
        );
    }

    Ok(())
}
