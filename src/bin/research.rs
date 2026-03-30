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

    // Compute summary statistics
    let mut improved = 0usize;
    let mut neutral = 0usize;
    let mut regressed = 0usize;
    let mut failed = 0usize;
    
    for e in &experiments {
        match e.outcome {
            hyperagent::ExperimentOutcome::Improved => improved += 1,
            hyperagent::ExperimentOutcome::Neutral => neutral += 1,
            hyperagent::ExperimentOutcome::Regressed => regressed += 1,
            hyperagent::ExperimentOutcome::Failed => failed += 1,
        }
    }
    
    let total = experiments.len();
    
    println!("\n┌───────────────────────────────────────────────────────────────────");
    println!("│ Experiment Log");
    println!("├───────────────────────────────────────────────────────────────────");
    for e in &experiments {
        let icon = match e.outcome {
            hyperagent::ExperimentOutcome::Improved => "✓",
            hyperagent::ExperimentOutcome::Neutral => "○",
            hyperagent::ExperimentOutcome::Regressed => "✗",
            hyperagent::ExperimentOutcome::Failed => "!",
        };
        let hypothesis_truncated = truncate_str(&e.hypothesis, 48);
        println!(
            "│ {} #{:02} src/{}\n│   {} | tests: {}/{} → {}/{}",
            icon, e.iteration, e.file,
            hypothesis_truncated,
            e.tests_before.0, e.tests_before.1,
            e.tests_after.0, e.tests_after.1,
        );
    }
    println!("└───────────────────────────────────────────────────────────────────");

    println!("\n╔═══════════════════════════════════════════════════════════════════");
    println!("║ Summary");
    println!("╠═══════════════════════════════════════════════════════════════════");
    println!("║ Total experiments: {}", total);
    if total > 0 {
        let success_rate = (improved as f64 / total as f64) * 100.0;
        println!("║ Success rate: {:.1}%", success_rate);
        println!("╟───────────────────────────────────────────────────────────────────");
        println!("║ ✓ Improved:  {:3} ({:5.1}%)", improved, pct(improved, total));
        println!("║ ○ Neutral:   {:3} ({:5.1}%)", neutral, pct(neutral, total));
        println!("║ ✗ Regressed: {:3} ({:5.1}%)", regressed, pct(regressed, total));
        println!("║ ! Failed:    {:3} ({:5.1}%)", failed, pct(failed, total));
        println!("╟───────────────────────────────────────────────────────────────────");
        
        // Provide actionable recommendation based on outcome distribution
        let recommendation = if improved > 0 {
            format!("✓ Research produced {} improvement(s). Consider running more iterations.", improved)
        } else if neutral > 0 && failed == 0 {
            "○ No improvements found. Consider adjusting search parameters or hypothesis scope.".to_string()
        } else if failed > (total / 2) {
            format!("✗ High failure rate ({}/{}) suggests build/test environment issues.", failed, total)
        } else {
            "○ Mixed results. Review experiment logs for patterns.".to_string()
        };
        println!("║ {}", recommendation);
    }
    println!("╚═══════════════════════════════════════════════════════════════════");

    // Exit with error if all experiments failed (useful for CI/CD)
    if total > 0 && failed == total {
        anyhow::bail!("All {} experiments failed - check build/test configuration", total);
    }

    Ok(())
}

/// Safely truncate a string to a maximum length, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find safe UTF-8 boundary
        let mut end = max_len;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Calculate percentage as a formatted string.
fn pct(n: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (n as f64 / total as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        assert_eq!(truncate_str("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_str_unicode() {
        // "héllo" has multi-byte characters
        let s = "héllo wörld";
        let truncated = truncate_str(s, 7);
        assert!(truncated.starts_with("h"));
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_truncate_str_empty() {
        assert_eq!(truncate_str("", 10), "");
    }

    #[test]
    fn test_pct_calculation() {
        assert!((pct(5, 10) - 50.0).abs() < 0.01);
        assert!((pct(1, 3) - 33.333).abs() < 0.01);
        assert_eq!(pct(0, 0), 0.0);
    }
}