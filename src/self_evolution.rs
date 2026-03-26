//! 递归自改进模块（基于 Karpathy 假设循环）
//!
//! 核心机制来自 Andrej Karpathy 的研究方法论：
//!   提出假设 → 实验 → 观察 → 反思 → 再提出假设
//!
//! 本模块是 `AutoResearch` 的薄封装，保留向后兼容的公共 API，
//! 内部委托给 Karpathy 循环引擎（`AutoResearch`）执行。

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::auto_research::{AutoResearch, ExperimentOutcome, ResearchConfig};
use crate::llm::LLMClient;

/// 自改进配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionConfig {
    /// 项目根目录（包含 Cargo.toml）
    pub project_root: PathBuf,
    /// 目标源文件列表（相对路径，相对于 src/）
    pub target_files: Vec<String>,
    /// 最大自改进迭代次数
    pub max_iterations: u32,
    /// 只修改，不自动 commit（安全模式）
    pub dry_run: bool,
}

impl Default for SelfEvolutionConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            target_files: ResearchConfig::default().target_files,
            max_iterations: 10,
            dry_run: true,
        }
    }
}

/// 自改进迭代结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionResult {
    pub iteration: u32,
    pub file: String,
    pub status: SelfEvolutionStatus,
    pub score: Option<SelfEvolutionScore>,
    pub error: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SelfEvolutionStatus {
    Accepted,
    Rejected,
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionScore {
    pub compiles: bool,
    pub tests_passed: u32,
    pub tests_total: u32,
    pub test_pass_rate: f32,
    pub compilation_errors: String,
    pub test_output: String,
}

/// 递归自改进引擎（委托给 Karpathy 假设循环）
pub struct SelfEvolutionEngine<C: LLMClient> {
    client: C,
    config: SelfEvolutionConfig,
    results: Vec<SelfEvolutionResult>,
}

impl<C: LLMClient + Clone> SelfEvolutionEngine<C> {
    pub fn new(client: C, config: SelfEvolutionConfig) -> Self {
        Self {
            client,
            config,
            results: Vec::new(),
        }
    }

    /// 将 SelfEvolutionConfig 转换为 ResearchConfig
    fn to_research_config(&self) -> ResearchConfig {
        ResearchConfig {
            project_root: self.config.project_root.clone(),
            target_files: self.config.target_files.clone(),
            max_iterations: self.config.max_iterations,
            auto_push: !self.config.dry_run,
            dry_run: self.config.dry_run,
            ..Default::default()
        }
    }

    /// 运行自改进主循环（委托给 Karpathy 假设循环）
    pub async fn run(&mut self) -> Result<Vec<SelfEvolutionResult>> {
        tracing::info!("╔══════════════════════════════════════════╗");
        tracing::info!("║   Self-Evolution (Karpathy Hypothesis)  ║");
        tracing::info!("╚══════════════════════════════════════════╝");

        let research_config = self.to_research_config();
        let mut engine = AutoResearch::new(self.client.clone(), research_config);
        let experiments = engine.run().await?;

        // 将 Experiment 转换为 SelfEvolutionResult
        for exp in &experiments {
            let status = match exp.outcome {
                ExperimentOutcome::Improved | ExperimentOutcome::Neutral => SelfEvolutionStatus::Accepted,
                ExperimentOutcome::Regressed => SelfEvolutionStatus::Rejected,
                ExperimentOutcome::Failed => SelfEvolutionStatus::Failed,
            };

            let score = Some(SelfEvolutionScore {
                compiles: exp.outcome != ExperimentOutcome::Failed,
                tests_passed: exp.tests_after.0,
                tests_total: exp.tests_after.1,
                test_pass_rate: if exp.tests_after.1 > 0 {
                    exp.tests_after.0 as f32 / exp.tests_after.1 as f32
                } else {
                    0.0
                },
                compilation_errors: if exp.outcome == ExperimentOutcome::Failed {
                    exp.reflection.clone()
                } else {
                    String::new()
                },
                test_output: exp.reflection.clone(),
            });

            let error = if exp.outcome == ExperimentOutcome::Failed {
                Some(format!("Experiment failed: {}", exp.reflection.chars().take(100).collect::<String>()))
            } else if exp.outcome == ExperimentOutcome::Regressed {
                Some(format!("Tests regressed: {}/{}", exp.tests_after.0, exp.tests_after.1))
            } else {
                None
            };

            self.results.push(SelfEvolutionResult {
                iteration: exp.iteration,
                file: exp.file.clone(),
                status,
                score,
                error,
                description: format!(
                    "{} — {}",
                    &exp.hypothesis.chars().take(80).collect::<String>(),
                    &exp.reflection.chars().take(80).collect::<String>(),
                ),
            });
        }

        // 统计摘要
        let accepted = self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Accepted).count();
        let rejected = self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Rejected).count();
        let failed = self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Failed).count();

        tracing::info!("╔══════════════════════════════════════════╗");
        tracing::info!("║   Self-Evolution Complete                ║");
        tracing::info!("╠══════════════════════════════════════════╣");
        tracing::info!("║  Accepted:  {:>3}                           ║", accepted);
        tracing::info!("║  Rejected:  {:>3}                           ║", rejected);
        tracing::info!("║  Failed:    {:>3}                           ║", failed);
        tracing::info!("║  Total:     {:>3}                           ║", self.results.len());
        tracing::info!("╚══════════════════════════════════════════╝");

        Ok(self.results.clone())
    }

    /// 获取所有结果
    pub fn results(&self) -> &[SelfEvolutionResult] {
        &self.results
    }
}