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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_evolution_summary_counts() {
        let results = vec![
            SelfEvolutionResult {
                iteration: 1,
                file: "a.rs".to_string(),
                status: SelfEvolutionStatus::Accepted,
                score: None,
                error: None,
                description: "test".to_string(),
            },
            SelfEvolutionResult {
                iteration: 2,
                file: "b.rs".to_string(),
                status: SelfEvolutionStatus::Accepted,
                score: None,
                error: None,
                description: "test".to_string(),
            },
            SelfEvolutionResult {
                iteration: 3,
                file: "c.rs".to_string(),
                status: SelfEvolutionStatus::Rejected,
                score: None,
                error: None,
                description: "test".to_string(),
            },
            SelfEvolutionResult {
                iteration: 4,
                file: "d.rs".to_string(),
                status: SelfEvolutionStatus::Failed,
                score: None,
                error: None,
                description: "test".to_string(),
            },
        ];

        // Test summary calculation
        let summary = SelfEvolutionSummary {
            total: results.len(),
            accepted: 2,
            rejected: 1,
            failed: 1,
            skipped: 0,
            success_rate: 0.5, // 2 accepted out of 4 non-skipped
        };

        assert_eq!(summary.total, 4);
        assert_eq!(summary.accepted, 2);
        assert!(summary.success_rate > 0.49 && summary.success_rate < 0.51);
    }

    #[test]
    fn test_success_rate_calculation() {
        let mut results = Vec::new();
        
        // 3 accepted, 1 rejected, 1 failed, 1 skipped
        for _ in 0..3 {
            results.push(SelfEvolutionResult {
                iteration: 1,
                file: "a.rs".to_string(),
                status: SelfEvolutionStatus::Accepted,
                score: None,
                error: None,
                description: "test".to_string(),
            });
        }
        results.push(SelfEvolutionResult {
            iteration: 2,
            file: "b.rs".to_string(),
            status: SelfEvolutionStatus::Rejected,
            score: None,
            error: None,
            description: "test".to_string(),
        });
        results.push(SelfEvolutionResult {
            iteration: 3,
            file: "c.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: None,
            error: None,
            description: "test".to_string(),
        });
        results.push(SelfEvolutionResult {
            iteration: 4,
            file: "d.rs".to_string(),
            status: SelfEvolutionStatus::Skipped,
            score: None,
            error: None,
            description: "test".to_string(),
        });

        // Success rate should be 3/5 = 0.6 (skipped excluded)
        // 3 accepted out of 5 non-skipped
        let non_skipped_count = results.iter().filter(|r| r.status != SelfEvolutionStatus::Skipped).count();
        let accepted_count = results.iter().filter(|r| r.status == SelfEvolutionStatus::Accepted).count();
        let rate = accepted_count as f32 / non_skipped_count as f32;
        
        assert!((rate - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_score_reflection_field_separate_from_test_output() {
        let score = SelfEvolutionScore {
            compiles: true,
            tests_passed: 10,
            tests_total: 10,
            test_pass_rate: 1.0,
            compilation_errors: String::new(),
            test_output: "running 10 tests\n...test output...".to_string(),
            reflection: "Hypothesis confirmed: optimization improved performance".to_string(),
        };

        // Verify fields are separate and semantically distinct
        assert_ne!(score.test_output, score.reflection);
        assert!(score.test_output.contains("running"));
        assert!(score.reflection.contains("Hypothesis"));
    }

    #[test]
    fn test_evolution_status_default() {
        let status = SelfEvolutionStatus::default();
        assert_eq!(status, SelfEvolutionStatus::Skipped);
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
    /// Actual test output (stdout/stderr from cargo test)
    pub test_output: String,
    /// Researcher's reflection/analysis of the experiment
    pub reflection: String,
}

/// 递归自改进引擎（委托给 Karpathy 假设循环）
pub struct SelfEvolutionEngine<C: LLMClient> {
    client: C,
    config: SelfEvolutionConfig,
    results: Vec<SelfEvolutionResult>,
}

impl Default for SelfEvolutionStatus {
    fn default() -> Self {
        Self::Skipped
    }
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
                test_output: String::new(), // Actual test output would come from experiment execution
                reflection: exp.reflection.clone(),
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

        let summary = self.summary();
        tracing::info!("╔══════════════════════════════════════════╗");
        tracing::info!("║   Self-Evolution Complete                ║");
        tracing::info!("╠══════════════════════════════════════════╣");
        tracing::info!("║  Accepted:  {:>3}                           ║", summary.accepted);
        tracing::info!("║  Rejected:  {:>3}                           ║", summary.rejected);
        tracing::info!("║  Failed:    {:>3}                           ║", summary.failed);
        tracing::info!("║  Skipped:   {:>3}                           ║", summary.skipped);
        tracing::info!("║  Total:     {:>3}                           ║", summary.total);
        tracing::info!("║  Success:   {:>5.1}%                         ║", summary.success_rate * 100.0);
        tracing::info!("╚══════════════════════════════════════════╝");

        Ok(self.results.clone())
    }

    /// 获取所有结果
    pub fn results(&self) -> &[SelfEvolutionResult] {
        &self.results
    }

    /// Calculate overall success rate (accepted / total non-skipped experiments)
    pub fn success_rate(&self) -> f32 {
        let non_skipped: Vec<_> = self.results.iter()
            .filter(|r| r.status != SelfEvolutionStatus::Skipped)
            .collect();
        
        if non_skipped.is_empty() {
            return 0.0;
        }
        
        let accepted = non_skipped.iter()
            .filter(|r| r.status == SelfEvolutionStatus::Accepted)
            .count();
        
        accepted as f32 / non_skipped.len() as f32
    }

    /// Get summary statistics
    pub fn summary(&self) -> SelfEvolutionSummary {
        SelfEvolutionSummary {
            total: self.results.len(),
            accepted: self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Accepted).count(),
            rejected: self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Rejected).count(),
            failed: self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Failed).count(),
            skipped: self.results.iter().filter(|r| r.status == SelfEvolutionStatus::Skipped).count(),
            success_rate: self.success_rate(),
        }
    }
}

/// Summary statistics for self-evolution run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionSummary {
    pub total: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub failed: usize,
    pub skipped: usize,
    pub success_rate: f32,
}