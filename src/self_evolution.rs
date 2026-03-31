//! 递归自改进模块（基于 Karpathy 假设循环）
//!
//! 核心机制来自 Andrej Karpathy 的研究方法论：
//!   提出假设 → 实验 → 观察 → 反思 → 再提出假设
//!
//! 本模块是 `AutoResearch` 的薄封装，保留向后兼容的公共 API，
//! 内部委托给 Karpathy 循环引擎（`AutoResearch`）执行。

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Truncate a string at word boundaries, ensuring it doesn't exceed max_chars
/// while avoiding cutting words in half. Appends "..." if truncated.
fn truncate_words(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    
    let truncated = &s[..max_chars];
    
    // Find the last space to avoid cutting a word
    if let Some(last_space) = truncated.rfind(' ') {
        // Only use word boundary if it's reasonably close to max_chars (at least 60%)
        if last_space > max_chars / 2 {
            format!("{}...", &truncated[..last_space])
        } else {
            format!("{}...", truncated)
        }
    } else {
        // No spaces found, just truncate at char boundary
        format!("{}...", truncated)
    }
}

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

/// Test metrics before and after an experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub passed: u32,
    pub total: u32,
}

/// 自改进迭代结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEvolutionResult {
    pub iteration: u32,
    pub file: String,
    pub status: SelfEvolutionStatus,
    pub score: Option<SelfEvolutionScore>,
    pub error: Option<String>,
    /// Brief summary of the experiment outcome
    pub description: String,
    /// Complete hypothesis text that was tested
    pub hypothesis: String,
    /// Test metrics before the experiment (if available)
    pub tests_before: Option<(u32, u32)>,
    /// Test metrics after the experiment (if available)
    pub tests_after: Option<(u32, u32)>,
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

impl SelfEvolutionScore {
    /// Returns true if the experiment compiled and all tests passed
    pub fn is_successful(&self) -> bool {
        self.compiles && self.tests_total > 0 && self.tests_passed == self.tests_total
    }

    /// Returns true if any tests were run (total > 0)
    pub fn has_tests(&self) -> bool {
        self.tests_total > 0
    }

    /// Returns true if code compiled successfully
    pub fn is_compilable(&self) -> bool {
        self.compiles
    }
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

            let hypothesis = exp.hypothesis.clone();
            let description = format!(
                "{} — {}",
                truncate_words(&exp.hypothesis, 80),
                truncate_words(&exp.reflection, 80),
            );

            self.results.push(SelfEvolutionResult {
                iteration: exp.iteration,
                file: exp.file.clone(),
                status,
                score,
                error,
                description,
                hypothesis,
                tests_before: Some(exp.tests_before),
                tests_after: Some(exp.tests_after),
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

    /// Filter results by status type
    pub fn filter_by_status(&self, status: SelfEvolutionStatus) -> Vec<&SelfEvolutionResult> {
        self.results.iter().filter(|r| r.status == status).collect()
    }

    /// Get all accepted results (convenience method)
    pub fn accepted(&self) -> Vec<&SelfEvolutionResult> {
        self.filter_by_status(SelfEvolutionStatus::Accepted)
    }

    /// Get all failed results (convenience method)
    pub fn failed(&self) -> Vec<&SelfEvolutionResult> {
        self.filter_by_status(SelfEvolutionStatus::Failed)
    }

    /// Get unique files that were modified (Accepted or Rejected status)
    /// Skips Failed and Skipped experiments since no code changes were applied
    pub fn modified_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.results
            .iter()
            .filter(|r| matches!(r.status, SelfEvolutionStatus::Accepted | SelfEvolutionStatus::Rejected))
            .map(|r| r.file.clone())
            .collect();
        files.sort();
        files.dedup();
        files
    }

    /// Get the most recent result for a specific file
    /// Returns None if no experiments were run for that file
    pub fn get_result_by_file(&self, file: &str) -> Option<&SelfEvolutionResult> {
        self.results.iter().rev().find(|r| r.file == file)
    }

    /// Get all results for a specific file (in iteration order)
    pub fn results_for_file(&self, file: &str) -> Vec<&SelfEvolutionResult> {
        self.results.iter().filter(|r| r.file == file).collect()
    }

    /// Get results that had a measurable impact (tests ran and passed or failed)
    /// Experiments with no tests (tests_total == 0) are filtered out
    pub fn with_test_results(&self) -> Vec<&SelfEvolutionResult> {
        self.results.iter()
            .filter(|r| r.score.as_ref().map_or(false, |s| s.tests_total > 0))
            .collect()
    }

    /// Search through all hypotheses by substring match
    /// Returns results whose hypothesis contains the given pattern (case-insensitive)
    /// Useful for analyzing patterns across experiments (e.g., all "Fix" hypotheses)
    pub fn search_hypotheses(&self, pattern: &str) -> Vec<&SelfEvolutionResult> {
        let pattern_lower = pattern.to_lowercase();
        self.results.iter()
            .filter(|r| r.hypothesis.to_lowercase().contains(&pattern_lower))
            .collect()
    }

    /// Get hypotheses that match a pattern, grouped by their outcome status
    /// Returns (matches_accepted, matches_rejected, matches_failed, matches_skipped)
    pub fn search_hypotheses_by_outcome(&self, pattern: &str) -> (Vec<&SelfEvolutionResult>, Vec<&SelfEvolutionResult>, Vec<&SelfEvolutionResult>, Vec<&SelfEvolutionResult>) {
        let matches = self.search_hypotheses(pattern);
        let accepted: Vec<_> = matches.iter().filter(|r| r.status == SelfEvolutionStatus::Accepted).copied().collect();
        let rejected: Vec<_> = matches.iter().filter(|r| r.status == SelfEvolutionStatus::Rejected).copied().collect();
        let failed: Vec<_> = matches.iter().filter(|r| r.status == SelfEvolutionStatus::Failed).copied().collect();
        let skipped: Vec<_> = matches.iter().filter(|r| r.status == SelfEvolutionStatus::Skipped).copied().collect();
        (accepted, rejected, failed, skipped)
    }

    /// Get top N results sorted by test improvement (descending)
    /// Only includes results with measurable before/after test metrics
    pub fn top_improvements(&self, n: usize) -> Vec<&SelfEvolutionResult> {
        let mut with_improvement: Vec<_> = self.results
            .iter()
            .filter(|r| r.test_improvement().is_some())
            .collect();
        with_improvement.sort_by(|a, b| {
            b.test_improvement().unwrap_or(0).cmp(&a.test_improvement().unwrap_or(0))
        });
        with_improvement.into_iter().take(n).collect()
    }

    /// Get results that improved test count (positive delta)
    pub fn positive_impacts(&self) -> Vec<&SelfEvolutionResult> {
        self.results
            .iter()
            .filter(|r| r.test_improvement().map_or(false, |d| d > 0))
            .collect()
    }

    /// Get results that regressed test count (negative delta)
    pub fn regressions(&self) -> Vec<&SelfEvolutionResult> {
        self.results
            .iter()
            .filter(|r| r.test_improvement().map_or(false, |d| d < 0))
            .collect()
    }

    /// Calculate average test improvement across all results with metrics
    pub fn average_improvement(&self) -> Option<f32> {
        let improvements: Vec<i32> = self.results
            .iter()
            .filter_map(|r| r.test_improvement())
            .collect();
        
        if improvements.is_empty() {
            return None;
        }
        
        Some(improvements.iter().sum::<i32>() as f32 / improvements.len() as f32)
    }

    /// Get a frequency map of how many times each file has been experimented on
    /// Useful for identifying over-explored files and balancing research attention
    pub fn experiment_frequency(&self) -> HashMap<String, usize> {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for result in &self.results {
            *freq.entry(result.file.clone()).or_insert(0) += 1;
        }
        freq
    }

    /// Get files sorted by experiment frequency (most experimented first)
    /// Returns a vector of (file, count) tuples
    pub fn files_by_frequency(&self) -> Vec<(String, usize)> {
        let freq = self.experiment_frequency();
        let mut sorted: Vec<_> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }

    /// Get the least experimented file(s) from the target list
    /// Useful for selecting next research targets to ensure balanced exploration
    pub fn least_experienced_files(&self) -> Vec<String> {
        let freq = self.experiment_frequency();
        let target_files = &self.config.target_files;
        
        if target_files.is_empty() {
            return Vec::new();
        }
        
        let min_count = target_files
            .iter()
            .map(|f| freq.get(f).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);
        
        target_files
            .iter()
            .filter(|f| freq.get(*f).copied().unwrap_or(0) == min_count)
            .cloned()
            .collect()
    }

    /// Categorize each modified file by its cumulative impact across all experiments
    /// Returns (improving_files, neutral_files, regressing_files)
    /// - improving: net positive test improvement (passed more tests after changes)
    /// - regressing: net negative test improvement (passed fewer tests after changes)
    /// - neutral: no measurable impact or no tests run
    pub fn file_impact_summary(&self) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut file_deltas: HashMap<String, i32> = HashMap::new();
        
        for result in &self.results {
            // Only count actual code changes (Accepted or Rejected)
            if !result.is_code_change() {
                continue;
            }
            
            let delta = result.test_improvement().unwrap_or(0);
            *file_deltas.entry(result.file.clone()).or_insert(0) += delta;
        }
        
        let mut improving: Vec<String> = Vec::new();
        let mut neutral: Vec<String> = Vec::new();
        let mut regressing: Vec<String> = Vec::new();
        
        for (file, delta) in file_deltas {
            if delta > 0 {
                improving.push(file);
            } else if delta < 0 {
                regressing.push(file);
            } else {
                neutral.push(file);
            }
        }
        
        // Sort for deterministic output
        improving.sort();
        neutral.sort();
        regressing.sort();
        
        (improving, neutral, regressing)
    }

    /// Get a human-readable summary of file impacts
    /// Format: "improving: X, neutral: Y, regressing: Z"
    pub fn impact_summary_string(&self) -> String {
        let (improving, neutral, regressing) = self.file_impact_summary();
        format!(
            "improving: {}, neutral: {}, regressing: {}",
            improving.len(),
            neutral.len(),
            regressing.len()
        )
    }

    /// Get files that should be prioritized for future research
    /// Returns files that have shown positive impact but haven't been experimented on recently
    /// Useful for "doubling down" on successful improvements
    pub fn high_value_targets(&self) -> Vec<String> {
        let (improving, _, _) = self.file_impact_summary();
        let recent: std::collections::HashSet<_> = self.results
            .iter()
            .rev()
            .take(3) // Last 3 iterations
            .map(|r| r.file.as_str())
            .collect();
        
        improving
            .into_iter()
            .filter(|f| !recent.contains(f.as_str()))
            .collect()
    }

    /// Get all compilation failures (experiments that didn't compile)
    /// These typically indicate syntax errors, type mismatches, or missing imports
    pub fn compilation_failures(&self) -> Vec<&SelfEvolutionResult> {
        self.results.iter()
            .filter(|r| r.is_compilation_failure())
            .collect()
    }

    /// Get all test failures (experiments that compiled but failed tests)
    /// These indicate semantically incorrect implementations
    pub fn test_failures(&self) -> Vec<&SelfEvolutionResult> {
        self.results.iter()
            .filter(|r| r.is_test_failure())
            .collect()
    }

    /// Get a breakdown of failure categories across all results
    /// Returns (compilation_failures, test_failures, unknown_failures)
    pub fn failure_breakdown(&self) -> (usize, usize, usize) {
        let mut compilation = 0;
        let mut test = 0;
        let mut unknown = 0;
        
        for result in &self.results {
            if result.status == SelfEvolutionStatus::Failed {
                match result.failure_category() {
                    Some(FailureCategory::Compilation) => compilation += 1,
                    Some(FailureCategory::Test { .. }) => test += 1,
                    Some(FailureCategory::Unknown) | None => unknown += 1,
                }
            }
        }
        
        (compilation, test, unknown)
    }

    /// Get files that had compilation failures
    /// Useful for identifying files with syntax/type issues that need fixing
    pub fn files_with_compilation_errors(&self) -> Vec<String> {
        let mut files: Vec<String> = self.results
            .iter()
            .filter(|r| r.is_compilation_failure())
            .map(|r| r.file.clone())
            .collect();
        files.sort();
        files.dedup();
        files
    }

    /// Get files that had test failures (but compiled successfully)
    /// Useful for identifying files where the logic needs adjustment
    pub fn files_with_test_failures(&self) -> Vec<String> {
        let mut files: Vec<String> = self.results
            .iter()
            .filter(|r| r.is_test_failure())
            .map(|r| r.file.clone())
            .collect();
        files.sort();
        files.dedup();
        files
    }

    /// Calculate the compilation success rate across all experiments
    /// Returns (compiled_count, total_count, success_rate)
    pub fn compilation_rate(&self) -> (usize, usize, f32) {
        let total = self.results.len();
        if total == 0 {
            return (0, 0, 0.0);
        }
        
        let compiled = self.results.iter()
            .filter(|r| r.compiled_successfully())
            .count();
        
        let rate = compiled as f32 / total as f32;
        (compiled, total, rate)
    }
}

impl SelfEvolutionResult {
    /// Generate a human-readable one-line summary of this result
    /// Format: "[iteration:X] {file} → {status}: {description}"
    pub fn status_summary(&self) -> String {
        let status_str = match &self.status {
            SelfEvolutionStatus::Accepted => "✓ ACCEPTED",
            SelfEvolutionStatus::Rejected => "✗ REJECTED",
            SelfEvolutionStatus::Failed => "⚠ FAILED",
            SelfEvolutionStatus::Skipped => "○ SKIPPED",
        };
        
        let score_str = self.score.as_ref().map(|s| {
            if s.tests_total > 0 {
                format!(" [tests: {}/{}]", s.tests_passed, s.tests_total)
            } else {
                String::new()
            }
        }).unwrap_or_default();
        
        format!(
            "[iteration:{}] {} → {}{}",
            self.iteration,
            self.file,
            status_str,
            score_str
        )
    }

    /// Returns true if this result represents a code change (not Failed or Skipped)
    pub fn is_code_change(&self) -> bool {
        matches!(self.status, SelfEvolutionStatus::Accepted | SelfEvolutionStatus::Rejected)
    }

    /// Calculate the test improvement delta (passed_after - passed_before)
    /// Returns None if either before or after metrics are unavailable
    pub fn test_improvement(&self) -> Option<i32> {
        match (self.tests_before, self.tests_after) {
            (Some((before_passed, _)), Some((after_passed, _))) => {
                Some(after_passed as i32 - before_passed as i32)
            }
            _ => None,
        }
    }

    /// Calculate the test pass rate improvement
    /// Returns None if either metric is unavailable or if totals are zero
    pub fn pass_rate_improvement(&self) -> Option<f32> {
        match (self.tests_before, self.tests_after) {
            (Some((before_passed, before_total)), Some((after_passed, after_total))) => {
                if before_total > 0 && after_total > 0 {
                    let before_rate = before_passed as f32 / before_total as f32;
                    let after_rate = after_passed as f32 / after_total as f32;
                    Some(after_rate - before_rate)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns true if this experiment failed to compile
    /// Only applicable for Failed status - returns false for other statuses
    pub fn is_compilation_failure(&self) -> bool {
        if self.status != SelfEvolutionStatus::Failed {
            return false;
        }
        
        // A compilation failure is a Failed status where the score indicates
        // the code didn't compile (compiles = false or no score)
        self.score.as_ref().map_or(true, |s| !s.compiles)
    }

    /// Returns true if this experiment compiled but tests failed
    /// Distinguishes from compilation failures - only applicable for Failed status
    pub fn is_test_failure(&self) -> bool {
        if self.status != SelfEvolutionStatus::Failed {
            return false;
        }
        
        // A test failure means code compiled (compiles = true) but tests didn't pass
        self.score.as_ref().map_or(false, |s| {
            s.compiles && s.tests_total > 0 && s.tests_passed < s.tests_total
        })
    }

    /// Returns true if this experiment compiled successfully
    /// Useful for quick filtering of syntactically correct code
    pub fn compiled_successfully(&self) -> bool {
        self.score.as_ref().map_or(false, |s| s.compiles)
    }

    /// Get a categorization of the failure type for this result
    /// Returns None for non-failed results
    pub fn failure_category(&self) -> Option<FailureCategory> {
        if self.status != SelfEvolutionStatus::Failed {
            return None;
        }
        
        match self.score.as_ref() {
            None => Some(FailureCategory::Unknown),
            Some(s) if !s.compiles => Some(FailureCategory::Compilation),
            Some(s) if s.tests_total > 0 && s.tests_passed < s.tests_total => {
                Some(FailureCategory::Test { passed: s.tests_passed, total: s.tests_total })
            }
            Some(_) => Some(FailureCategory::Unknown),
        }
    }
}

/// Categorization of why an experiment failed
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailureCategory {
    /// Code failed to compile
    Compilation,
    /// Code compiled but tests failed
    Test { passed: u32, total: u32 },
    /// Failed for unknown reasons (no score data)
    Unknown,
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

impl SelfEvolutionSummary {
    /// Generate a human-readable one-line summary of evolution statistics.
    ///
    /// Format: "{total} total: {accepted} accepted, {rejected} rejected, {failed} failed, {skipped} skipped ({success_rate:.1}% success)"
    ///
    /// # Example
    /// ```
    /// let summary = SelfEvolutionSummary {
    ///     total: 10,
    ///     accepted: 6,
    ///     rejected: 2,
    ///     failed: 1,
    ///     skipped: 1,
    ///     success_rate: 0.6,
    /// };
    /// assert_eq!(summary.summary(), "10 total: 6 accepted, 2 rejected, 1 failed, 1 skipped (60.0% success)");
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "{} total: {} accepted, {} rejected, {} failed, {} skipped ({:.1}% success)",
            self.total,
            self.accepted,
            self.rejected,
            self.failed,
            self.skipped,
            self.success_rate * 100.0
        )
    }
}
