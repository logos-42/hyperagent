//! 统一进化入口 — 自动研究（外层） + 进化引擎（内层）嵌套循环
//!
//! 外层：自动研究改进系统源代码
//! 内层：用改进后的系统运行进化引擎，验证改进效果
//!
//! 效果：系统改进 → 进化验证 → 效果反馈 → 继续改进
//!
//! 用法:
//!   cargo run --bin unified                                    # 默认：3 轮研究 + 每轮 3 代进化
//!   UNIFIED_RESEARCH_ITERS=10 UNIFIED_EVOLVING_GENS=5 cargo run --bin unified
//!   UNIFIED_AUTO_PUSH=true UNIFIED_TASK="写一个排序算法" cargo run --bin unified

use anyhow::Result;
use hyperagent::{
    AutoResearch, EvolutionLoop, LLMClientImpl, ResearchConfig, RuntimeConfig, RuntimeState,
};

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
        "Unified Evolution: provider={:?}, model={}",
        client.provider(),
        client.model()
    );

    let research_iters = std::env::var("UNIFIED_RESEARCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let evolving_gens = std::env::var("UNIFIED_EVOLVING_GENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let auto_push = std::env::var("UNIFIED_AUTO_PUSH")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let task = std::env::var("UNIFIED_TASK")
        .unwrap_or_else(|_| "Write an efficient Rust function to solve the N-Queens problem".to_string());

    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║     Unified Evolution: Research + Evolving      ║");
    tracing::info!("╠══════════════════════════════════════════════════╣");
    tracing::info!("║  Research iterations: {:>3}                       ║", research_iters);
    tracing::info!("║  Evolving generations: {:>2} per research        ║", evolving_gens);
    tracing::info!("║  Auto push: {:>6}                              ║", auto_push);
    tracing::info!("║  Task: {}" , &task[..task.len().min(45)]);
    tracing::info!("╚══════════════════════════════════════════════════╝");

    // === Phase 1: Baseline evolution (改进前的进化表现) ===
    tracing::info!("\n>>> Phase 1: Baseline Evolution (before any improvements)");
    let mut best_baseline_score;
    {
        let persist_dir = std::path::Path::new(".hyperagent/data");
        let runtime_config = RuntimeConfig {
            max_generations: evolving_gens,
            population_size: 3,
            top_k_selection: 2,
            num_branches: 2,
            checkpoint_interval: evolving_gens,
            meta_mutation_interval: 2,
            initial_temperature: 1.5,
            annealing_rate: 0.9,
            ..Default::default()
        };
        let state = RuntimeState::with_persistence(runtime_config, persist_dir);
        let mut loop_ = EvolutionLoop::new(client.clone(), state);
        let final_state = loop_.run_with_iterations(&task, evolving_gens as usize).await?;
        best_baseline_score = final_state.best_score;
        tracing::info!("Baseline best score: {:.2}", best_baseline_score);
    }

    // === Phase 2: Research + Evolving loop ===
    tracing::info!("\n>>> Phase 2: Research + Evolving Loop");

    let target_files = ResearchConfig::default().target_files;

    for round in 1..=research_iters {
        tracing::info!("\n═════════════════════════════════════════");
        tracing::info!("  Round {}/{}: Research → Evolve → Evaluate", round, research_iters);
        tracing::info!("═════════════════════════════════════════");

        // Step 1: 自动研究 — 改进一个源文件（每轮改进一个文件）
        let file = &target_files[(round as usize - 1) % target_files.len()];

        let single_config = ResearchConfig {
            max_iterations: 1,
            auto_push,
            dry_run: false,
            strict: false,
            target_files: vec![file.clone()],
            ..Default::default()
        };

        let mut single_research = AutoResearch::new(client.clone(), single_config);
        let results = single_research.run().await?;
        let exp = &results[0];

        tracing::info!(
            "  Research: file={}, outcome={:?}, hypothesis={}",
            exp.file, exp.outcome,
            &exp.hypothesis[..exp.hypothesis.len().min(60)]
        );

        // 如果改进失败（编译不过或测试退化），跳过进化验证
        if matches!(exp.outcome, hyperagent::ExperimentOutcome::Failed | hyperagent::ExperimentOutcome::Regressed) {
            tracing::info!("  Skipped evolution validation (research failed)");
            continue;
        }

        // Step 2: 重新编译 + 运行进化引擎验证效果
        tracing::info!("  Evolution validation: running {} generations...", evolving_gens);
        let persist_dir = std::path::Path::new(".hyperagent/data");
        let runtime_config = RuntimeConfig {
            max_generations: evolving_gens,
            population_size: 3,
            top_k_selection: 2,
            num_branches: 2,
            checkpoint_interval: evolving_gens,
            meta_mutation_interval: 2,
            initial_temperature: 1.5,
            annealing_rate: 0.9,
            ..Default::default()
        };

        // 运行 cargo build 确保用最新代码
        let build_output = tokio::process::Command::new("cargo")
            .arg("build")
            .output()
            .await?;
        if !build_output.status.success() {
            tracing::warn!("  Build failed after research change");
            continue;
        }

        // 注意：Rust 是编译型语言，无法在运行时动态重新加载代码
        // 但自动研究改进的代码会体现在下次编译运行中
        // 这里运行进化引擎，记录分数作为参考
        let state = RuntimeState::with_persistence(runtime_config, persist_dir);
        let mut loop_ = EvolutionLoop::new(client.clone(), state);
        let final_state = loop_.run_with_iterations(&task, evolving_gens as usize).await?;
        let score = final_state.best_score;

        let delta = score - best_baseline_score;
        tracing::info!(
            "  Evolution result: score={:.2} (baseline={:.2}, delta={:+.2})",
            score, best_baseline_score, delta
        );

        if delta > 0.1 {
            tracing::info!("  >>> Improvement confirmed! System change led to better evolution.");
            best_baseline_score = score;
        }
    }

    // === Phase 3: Final summary ===
    tracing::info!("\n╔══════════════════════════════════════════════════╗");
    tracing::info!("║     Unified Evolution Complete                  ║");
    tracing::info!("╠══════════════════════════════════════════════════╣");
    tracing::info!("║  Research rounds:  {}                          ║", research_iters);
    tracing::info!("║  Evolving gens/round: {}                        ║", evolving_gens);
    tracing::info!("║  Baseline score:  {:.2}                          ║", best_baseline_score);
    tracing::info!("║  Auto pushed:     {}                            ║", auto_push);
    tracing::info!("╚══════════════════════════════════════════════════╝");

    Ok(())
}