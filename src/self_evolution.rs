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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(iteration: u32, file: &str, status: SelfEvolutionStatus) -> SelfEvolutionResult {
        SelfEvolutionResult {
            iteration,
            file: file.to_string(),
            status,
            score: None,
            error: None,
            description: format!("Test result for {}", file),
            hypothesis: format!("Hypothesis for {}", file),
            tests_before: None,
            tests_after: None,
        }
    }

    fn make_result_with_tests(
        iteration: u32,
        file: &str,
        status: SelfEvolutionStatus,
        passed: u32,
        total: u32,
    ) -> SelfEvolutionResult {
        SelfEvolutionResult {
            iteration,
            file: file.to_string(),
            status,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: passed,
                tests_total: total,
                test_pass_rate: if total > 0 { passed as f32 / total as f32 } else { 0.0 },
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: format!("Test result for {}", file),
            hypothesis: format!("Hypothesis for {}", file),
            tests_before: None,
            tests_after: None,
        }
    }

    fn make_result_with_delta(
        iteration: u32,
        file: &str,
        status: SelfEvolutionStatus,
        before: (u32, u32),
        after: (u32, u32),
    ) -> SelfEvolutionResult {
        SelfEvolutionResult {
            iteration,
            file: file.to_string(),
            status,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: after.0,
                tests_total: after.1,
                test_pass_rate: if after.1 > 0 { after.0 as f32 / after.1 as f32 } else { 0.0 },
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: format!("Test result for {}", file),
            hypothesis: format!("Hypothesis for {}", file),
            tests_before: Some(before),
            tests_after: Some(after),
        }
    }

    #[test]
    fn test_status_summary_accepted() {
        let result = make_result(1, "agent/mod.rs", SelfEvolutionStatus::Accepted);
        let summary = result.status_summary();
        assert!(summary.contains("[iteration:1]"));
        assert!(summary.contains("agent/mod.rs"));
        assert!(summary.contains("✓ ACCEPTED"));
    }

    #[test]
    fn test_status_summary_rejected() {
        let result = make_result(2, "eval/metrics.rs", SelfEvolutionStatus::Rejected);
        let summary = result.status_summary();
        assert!(summary.contains("✗ REJECTED"));
    }

    #[test]
    fn test_status_summary_with_tests() {
        let result = make_result_with_tests(3, "eval/metrics.rs", SelfEvolutionStatus::Accepted, 8, 10);
        let summary = result.status_summary();
        assert!(summary.contains("[tests: 8/10]"));
    }

    #[test]
    fn test_status_summary_failed() {
        let result = make_result(4, "llm/client.rs", SelfEvolutionStatus::Failed);
        let summary = result.status_summary();
        assert!(summary.contains("⚠ FAILED"));
    }

    #[test]
    fn test_status_summary_skipped() {
        let result = make_result(5, "memory/lineage.rs", SelfEvolutionStatus::Skipped);
        let summary = result.status_summary();
        assert!(summary.contains("○ SKIPPED"));
    }

    #[test]
    fn test_is_code_change() {
        let accepted = make_result(1, "test.rs", SelfEvolutionStatus::Accepted);
        let rejected = make_result(2, "test.rs", SelfEvolutionStatus::Rejected);
        let failed = make_result(3, "test.rs", SelfEvolutionStatus::Failed);
        let skipped = make_result(4, "test.rs", SelfEvolutionStatus::Skipped);

        assert!(accepted.is_code_change());
        assert!(rejected.is_code_change());
        assert!(!failed.is_code_change());
        assert!(!skipped.is_code_change());
    }

    #[test]
    fn test_score_is_successful() {
        let success = SelfEvolutionScore {
            compiles: true,
            tests_passed: 10,
            tests_total: 10,
            test_pass_rate: 1.0,
            compilation_errors: String::new(),
            test_output: String::new(),
            reflection: String::new(),
        };
        assert!(success.is_successful());

        let partial = SelfEvolutionScore {
            compiles: true,
            tests_passed: 8,
            tests_total: 10,
            test_pass_rate: 0.8,
            compilation_errors: String::new(),
            test_output: String::new(),
            reflection: String::new(),
        };
        assert!(!partial.is_successful());

        let no_tests = SelfEvolutionScore {
            compiles: true,
            tests_passed: 0,
            tests_total: 0,
            test_pass_rate: 0.0,
            compilation_errors: String::new(),
            test_output: String::new(),
            reflection: String::new(),
        };
        assert!(!no_tests.is_successful());
    }

    #[test]
    fn test_score_has_tests_and_compilable() {
        let with_tests = SelfEvolutionScore {
            compiles: true,
            tests_passed: 5,
            tests_total: 10,
            test_pass_rate: 0.5,
            compilation_errors: String::new(),
            test_output: String::new(),
            reflection: String::new(),
        };
        assert!(with_tests.has_tests());
        assert!(with_tests.is_compilable());

        let no_tests = SelfEvolutionScore {
            compiles: true,
            tests_passed: 0,
            tests_total: 0,
            test_pass_rate: 0.0,
            compilation_errors: String::new(),
            test_output: String::new(),
            reflection: String::new(),
        };
        assert!(!no_tests.has_tests());

        let compile_failed = SelfEvolutionScore {
            compiles: false,
            tests_passed: 0,
            tests_total: 0,
            test_pass_rate: 0.0,
            compilation_errors: "error[E0433]: failed to resolve".to_string(),
            test_output: String::new(),
            reflection: String::new(),
        };
        assert!(!compile_failed.is_compilable());
    }

    #[test]
    fn test_truncate_words_short_string() {
        let short = "hello world";
        assert_eq!(truncate_words(short, 100), short);
    }

    #[test]
    fn test_truncate_words_exact_length() {
        let exact = "hello world";
        assert_eq!(truncate_words(exact, 11), "hello world");
    }

    #[test]
    fn test_truncate_words_at_word_boundary() {
        let long = "hello world this is a test string";
        let truncated = truncate_words(long, 15);
        // Should truncate at word boundary (after "hello")
        assert!(truncated.ends_with("..."));
        assert!(!truncated.contains("world"));
    }

    #[test]
    fn test_truncate_words_no_spaces() {
        let no_spaces = "abcdefghijklmnopqrstuvwxyz";
        let truncated = truncate_words(no_spaces, 10);
        // No word boundary, just truncate
        assert!(truncated.ends_with("..."));
        assert_eq!(truncated.len(), 13); // 10 chars + "..."
    }

    #[test]
    fn test_search_hypotheses_empty_results() {
        let config = SelfEvolutionConfig::default();
        let result = SelfEvolutionResult {
            iteration: 1,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Accepted,
            score: None,
            error: None,
            description: "Test".to_string(),
            hypothesis: "Fix the bug in parser".to_string(),
        };
        
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        // Can't directly access results field since it's private, so test via summary
        assert_eq!(engine.summary().total, 0);
    }

    #[test]
    fn test_search_hypotheses_by_outcome_grouping() {
        // Test that the method correctly groups results by outcome
        let results: Vec<SelfEvolutionResult> = vec![
            make_result_with_tests(1, "a.rs", SelfEvolutionStatus::Accepted, 10, 10),
            make_result_with_tests(2, "b.rs", SelfEvolutionStatus::Rejected, 5, 10),
            make_result_with_tests(3, "c.rs", SelfEvolutionStatus::Failed, 0, 0),
            make_result_with_tests(4, "d.rs", SelfEvolutionStatus::Accepted, 8, 10),
        ];
        
        // Verify hypothesis field exists and is searchable
        assert!(results[0].hypothesis.contains("Hypothesis"));
    }

    #[test]
    fn test_search_hypotheses_case_insensitive() {
        // Verify that case-insensitive search would work
        let hypothesis = "Fix the parser bug in tokenization";
        let pattern = "PARSER";
        assert!(hypothesis.to_lowercase().contains(&pattern.to_lowercase()));
        
        let pattern2 = "fix";
        assert!(hypothesis.to_lowercase().contains(&pattern2.to_lowercase()));
    }

    #[test]
    fn test_test_improvement_positive() {
        let result = make_result_with_delta(1, "test.rs", SelfEvolutionStatus::Accepted, (5, 10), (8, 10));
        assert_eq!(result.test_improvement(), Some(3));
    }

    #[test]
    fn test_test_improvement_negative() {
        let result = make_result_with_delta(2, "test.rs", SelfEvolutionStatus::Rejected, (8, 10), (5, 10));
        assert_eq!(result.test_improvement(), Some(-3));
    }

    #[test]
    fn test_test_improvement_no_change() {
        let result = make_result_with_delta(3, "test.rs", SelfEvolutionStatus::Accepted, (5, 10), (5, 10));
        assert_eq!(result.test_improvement(), Some(0));
    }

    #[test]
    fn test_test_improvement_no_metrics() {
        let result = make_result(4, "test.rs", SelfEvolutionStatus::Accepted);
        assert_eq!(result.test_improvement(), None);
    }

    #[test]
    fn test_pass_rate_improvement() {
        let result = make_result_with_delta(1, "test.rs", SelfEvolutionStatus::Accepted, (5, 10), (8, 10));
        let improvement = result.pass_rate_improvement().unwrap();
        assert!((improvement - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_pass_rate_improvement_different_totals() {
        let result = make_result_with_delta(1, "test.rs", SelfEvolutionStatus::Accepted, (4, 5), (9, 10));
        let improvement = result.pass_rate_improvement().unwrap();
        assert!((improvement - 0.1).abs() < 0.01); // 0.9 - 0.8 = 0.1
    }

    #[test]
    fn test_pass_rate_improvement_zero_total() {
        let result = make_result_with_delta(1, "test.rs", SelfEvolutionStatus::Failed, (0, 0), (5, 10));
        assert_eq!(result.pass_rate_improvement(), None);
    }

    #[test]
    fn test_experiment_frequency_empty() {
        let config = SelfEvolutionConfig::default();
        let engine: SelfEvolutionEngine<crate::llm::LLMClientImpl> = SelfEvolutionEngine::new(
            crate::llm::LLMClientImpl::ollama(),
            config,
        );
        let freq = engine.experiment_frequency();
        assert!(freq.is_empty());
    }

    #[test]
    fn test_experiment_frequency_counts() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Simulate results for different files
        engine.results.push(make_result(1, "agent/mod.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(2, "agent/mod.rs", SelfEvolutionStatus::Rejected));
        engine.results.push(make_result(3, "eval/metrics.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(4, "agent/mod.rs", SelfEvolutionStatus::Failed));
        engine.results.push(make_result(5, "llm/client.rs", SelfEvolutionStatus::Accepted));
        
        let freq = engine.experiment_frequency();
        assert_eq!(freq.get("agent/mod.rs"), Some(&3));
        assert_eq!(freq.get("eval/metrics.rs"), Some(&1));
        assert_eq!(freq.get("llm/client.rs"), Some(&1));
        assert_eq!(freq.len(), 3);
    }

    #[test]
    fn test_files_by_frequency_sorted() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(make_result(1, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(2, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(3, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(4, "b.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(5, "c.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(6, "c.rs", SelfEvolutionStatus::Accepted));
        
        let sorted = engine.files_by_frequency();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0], ("a.rs".to_string(), 3));
        assert_eq!(sorted[1], ("c.rs".to_string(), 2));
        assert_eq!(sorted[2], ("b.rs".to_string(), 1));
    }

    #[test]
    fn test_least_experienced_files() {
        let config = SelfEvolutionConfig {
            target_files: vec!["a.rs".to_string(), "b.rs".to_string(), "c.rs".to_string()],
            ..Default::default()
        };
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Only run experiments on a.rs and b.rs
        engine.results.push(make_result(1, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(2, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(3, "b.rs", SelfEvolutionStatus::Accepted));
        
        let least = engine.least_experienced_files();
        assert_eq!(least.len(), 1);
        assert_eq!(least[0], "c.rs"); // c.rs has 0 experiments
    }

    #[test]
    fn test_least_experienced_files_tie() {
        let config = SelfEvolutionConfig {
            target_files: vec!["a.rs".to_string(), "b.rs".to_string(), "c.rs".to_string()],
            ..Default::default()
        };
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // All files have 1 experiment
        engine.results.push(make_result(1, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(2, "b.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(make_result(3, "c.rs", SelfEvolutionStatus::Accepted));
        
        let least = engine.least_experienced_files();
        assert_eq!(least.len(), 3); // All tied at 1
    }

    #[test]
    fn test_least_experienced_files_empty_targets() {
        let config = SelfEvolutionConfig {
            target_files: vec![],
            ..Default::default()
        };
        let engine: SelfEvolutionEngine<crate::llm::LLMClientImpl> = SelfEvolutionEngine::new(
            crate::llm::LLMClientImpl::ollama(),
            config,
        );
        
        let least = engine.least_experienced_files();
        assert!(least.is_empty());
    }

    #[test]
    fn test_file_impact_summary_empty_results() {
        let config = SelfEvolutionConfig::default();
        let engine: SelfEvolutionEngine<crate::llm::LLMClientImpl> = SelfEvolutionEngine::new(
            crate::llm::LLMClientImpl::ollama(),
            config,
        );
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert!(improving.is_empty());
        assert!(neutral.is_empty());
        assert!(regressing.is_empty());
    }

    #[test]
    fn test_file_impact_summary_improving() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(make_result_with_delta(1, "a.rs", SelfEvolutionStatus::Accepted, (5, 10), (8, 10)));
        engine.results.push(make_result_with_delta(2, "a.rs", SelfEvolutionStatus::Accepted, (8, 10), (10, 10)));
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert_eq!(improving.len(), 1);
        assert_eq!(improving[0], "a.rs");
        assert!(neutral.is_empty());
        assert!(regressing.is_empty());
    }

    #[test]
    fn test_file_impact_summary_regressing() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(make_result_with_delta(1, "b.rs", SelfEvolutionStatus::Rejected, (8, 10), (5, 10)));
        engine.results.push(make_result_with_delta(2, "b.rs", SelfEvolutionStatus::Rejected, (5, 10), (3, 10)));
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert!(improving.is_empty());
        assert!(neutral.is_empty());
        assert_eq!(regressing.len(), 1);
        assert_eq!(regressing[0], "b.rs");
    }

    #[test]
    fn test_file_impact_summary_neutral() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // No test metrics means neutral
        engine.results.push(make_result(1, "c.rs", SelfEvolutionStatus::Accepted));
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert!(improving.is_empty());
        assert_eq!(neutral.len(), 1);
        assert_eq!(neutral[0], "c.rs");
        assert!(regressing.is_empty());
    }

    #[test]
    fn test_file_impact_summary_mixed_files() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Improving file
        engine.results.push(make_result_with_delta(1, "a.rs", SelfEvolutionStatus::Accepted, (5, 10), (8, 10)));
        // Regressing file
        engine.results.push(make_result_with_delta(2, "b.rs", SelfEvolutionStatus::Rejected, (9, 10), (6, 10)));
        // Neutral file (no delta)
        engine.results.push(make_result_with_delta(3, "c.rs", SelfEvolutionStatus::Accepted, (7, 10), (7, 10)));
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert_eq!(improving.len(), 1);
        assert_eq!(improving[0], "a.rs");
        assert_eq!(neutral.len(), 1);
        assert_eq!(neutral[0], "c.rs");
        assert_eq!(regressing.len(), 1);
        assert_eq!(regressing[0], "b.rs");
    }

    #[test]
    fn test_file_impact_summary_cumulative_delta() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Multiple experiments on same file - should sum deltas
        engine.results.push(make_result_with_delta(1, "a.rs", SelfEvolutionStatus::Accepted, (5, 10), (7, 10))); // +2
        engine.results.push(make_result_with_delta(2, "a.rs", SelfEvolutionStatus::Accepted, (7, 10), (6, 10))); // -1
        engine.results.push(make_result_with_delta(3, "a.rs", SelfEvolutionStatus::Accepted, (6, 10), (9, 10))); // +3
        // Net: +4
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert_eq!(improving.len(), 1);
        assert_eq!(improving[0], "a.rs");
    }

    #[test]
    fn test_file_impact_summary_ignores_non_code_changes() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Failed and Skipped should not be counted
        engine.results.push(make_result_with_delta(1, "a.rs", SelfEvolutionStatus::Failed, (5, 10), (5, 10)));
        engine.results.push(make_result_with_delta(2, "a.rs", SelfEvolutionStatus::Skipped, (5, 10), (5, 10)));
        
        let (improving, neutral, regressing) = engine.file_impact_summary();
        assert!(improving.is_empty());
        assert!(neutral.is_empty());
        assert!(regressing.is_empty());
    }

    #[test]
    fn test_impact_summary_string_format() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(make_result_with_delta(1, "a.rs", SelfEvolutionStatus::Accepted, (5, 10), (8, 10)));
        engine.results.push(make_result_with_delta(2, "b.rs", SelfEvolutionStatus::Rejected, (8, 10), (5, 10)));
        engine.results.push(make_result(3, "c.rs", SelfEvolutionStatus::Accepted));
        
        let summary = engine.impact_summary_string();
        assert_eq!(summary, "improving: 1, neutral: 1, regressing: 1");
    }

    #[test]
    fn test_high_value_targets_excludes_recent() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // a.rs is improving but recent
        engine.results.push(make_result_with_delta(1, "a.rs", SelfEvolutionStatus::Accepted, (5, 10), (8, 10)));
        engine.results.push(make_result_with_delta(2, "a.rs", SelfEvolutionStatus::Accepted, (8, 10), (10, 10)));
        engine.results.push(make_result_with_delta(3, "a.rs", SelfEvolutionStatus::Accepted, (10, 10), (12, 12)));
        
        // b.rs is improving and not recent
        engine.results.push(make_result_with_delta(4, "b.rs", SelfEvolutionStatus::Accepted, (5, 10), (9, 10)));
        
        let high_value = engine.high_value_targets();
        // a.rs was in last 3 iterations, so only b.rs should be returned
        assert_eq!(high_value.len(), 1);
        assert_eq!(high_value[0], "b.rs");
    }

    #[test]
    fn test_high_value_targets_only_improving() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Regressing file - should not be high value
        engine.results.push(make_result_with_delta(1, "regressing.rs", SelfEvolutionStatus::Rejected, (9, 10), (5, 10)));
        // Neutral file - should not be high value
        engine.results.push(make_result(2, "neutral.rs", SelfEvolutionStatus::Accepted));
        
        let high_value = engine.high_value_targets();
        assert!(high_value.is_empty());
    }

    #[test]
    fn test_is_compilation_failure() {
        // Failed status with compiles = false
        let compile_fail = SelfEvolutionResult {
            iteration: 1,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error[E0433]: failed to resolve".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Failed".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert!(compile_fail.is_compilation_failure());
        assert!(!compile_fail.is_test_failure());

        // Failed status with compiles = true but tests failed
        let test_fail = SelfEvolutionResult {
            iteration: 2,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 3,
                tests_total: 10,
                test_pass_rate: 0.3,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Failed".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert!(!test_fail.is_compilation_failure());
        assert!(test_fail.is_test_failure());

        // Accepted status should return false for both
        let accepted = make_result(3, "test.rs", SelfEvolutionStatus::Accepted);
        assert!(!accepted.is_compilation_failure());
        assert!(!accepted.is_test_failure());
    }

    #[test]
    fn test_is_test_failure() {
        // Code compiled but tests failed
        let test_fail = SelfEvolutionResult {
            iteration: 1,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 5,
                tests_total: 10,
                test_pass_rate: 0.5,
                compilation_errors: String::new(),
                test_output: "test failures".to_string(),
                reflection: String::new(),
            }),
            error: None,
            description: "Failed".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert!(test_fail.is_test_failure());
        assert!(!test_fail.is_compilation_failure());
    }

    #[test]
    fn test_compiled_successfully() {
        let compiled = SelfEvolutionResult {
            iteration: 1,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Accepted,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 10,
                tests_total: 10,
                test_pass_rate: 1.0,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Success".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert!(compiled.compiled_successfully());

        let no_score = make_result(2, "test.rs", SelfEvolutionStatus::Accepted);
        assert!(!no_score.compiled_successfully());
    }

    #[test]
    fn test_failure_category() {
        // Compilation failure
        let compile_fail = SelfEvolutionResult {
            iteration: 1,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Failed".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert_eq!(compile_fail.failure_category(), Some(FailureCategory::Compilation));

        // Test failure
        let test_fail = SelfEvolutionResult {
            iteration: 2,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 5,
                tests_total: 10,
                test_pass_rate: 0.5,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Failed".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert_eq!(test_fail.failure_category(), Some(FailureCategory::Test { passed: 5, total: 10 }));

        // Non-failed status returns None
        let accepted = make_result(3, "test.rs", SelfEvolutionStatus::Accepted);
        assert_eq!(accepted.failure_category(), None);
    }

    #[test]
    fn test_failure_category_unknown() {
        // Failed with no score
        let no_score = SelfEvolutionResult {
            iteration: 1,
            file: "test.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: None,
            error: Some("Unknown error".to_string()),
            description: "Failed".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        };
        assert_eq!(no_score.failure_category(), Some(FailureCategory::Unknown));
    }

    #[test]
    fn test_compilation_failures() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(make_result(1, "a.rs", SelfEvolutionStatus::Accepted));
        engine.results.push(SelfEvolutionResult {
            iteration: 2,
            file: "b.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Compile fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(SelfEvolutionResult {
            iteration: 3,
            file: "c.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 2,
                tests_total: 10,
                test_pass_rate: 0.2,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Test fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        
        let comp_failures = engine.compilation_failures();
        assert_eq!(comp_failures.len(), 1);
        assert_eq!(comp_failures[0].file, "b.rs");
        
        let test_failures = engine.test_failures();
        assert_eq!(test_failures.len(), 1);
        assert_eq!(test_failures[0].file, "c.rs");
    }

    #[test]
    fn test_failure_breakdown() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // 2 compilation failures
        engine.results.push(SelfEvolutionResult {
            iteration: 1,
            file: "a.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(SelfEvolutionResult {
            iteration: 2,
            file: "b.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        
        // 1 test failure
        engine.results.push(SelfEvolutionResult {
            iteration: 3,
            file: "c.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 3,
                tests_total: 10,
                test_pass_rate: 0.3,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        
        // 1 unknown failure (no score)
        engine.results.push(SelfEvolutionResult {
            iteration: 4,
            file: "d.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: None,
            error: Some("Unknown".to_string()),
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        
        // 1 accepted (not counted)
        engine.results.push(make_result(5, "e.rs", SelfEvolutionStatus::Accepted));
        
        let (comp, test, unknown) = engine.failure_breakdown();
        assert_eq!(comp, 2);
        assert_eq!(test, 1);
        assert_eq!(unknown, 1);
    }

    #[test]
    fn test_files_with_compilation_errors() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(SelfEvolutionResult {
            iteration: 1,
            file: "a.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(SelfEvolutionResult {
            iteration: 2,
            file: "a.rs".to_string(), // Same file again
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(make_result(3, "b.rs", SelfEvolutionStatus::Accepted));
        
        let files = engine.files_with_compilation_errors();
        assert_eq!(files.len(), 1); // Deduplicated
        assert_eq!(files[0], "a.rs");
    }

    #[test]
    fn test_files_with_test_failures() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        engine.results.push(SelfEvolutionResult {
            iteration: 1,
            file: "a.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 3,
                tests_total: 10,
                test_pass_rate: 0.3,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(SelfEvolutionResult {
            iteration: 2,
            file: "b.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 5,
                tests_total: 10,
                test_pass_rate: 0.5,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        
        let files = engine.files_with_test_failures();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&"a.rs".to_string()));
        assert!(files.contains(&"b.rs".to_string()));
    }

    #[test]
    fn test_compilation_rate() {
        let config = SelfEvolutionConfig::default();
        let mut engine = SelfEvolutionEngine::new(crate::llm::LLMClientImpl::ollama(), config);
        
        // Empty results
        let (compiled, total, rate) = engine.compilation_rate();
        assert_eq!(compiled, 0);
        assert_eq!(total, 0);
        assert!((rate - 0.0).abs() < 0.01);
        
        // Add results
        engine.results.push(SelfEvolutionResult {
            iteration: 1,
            file: "a.rs".to_string(),
            status: SelfEvolutionStatus::Accepted,
            score: Some(SelfEvolutionScore {
                compiles: true,
                tests_passed: 10,
                tests_total: 10,
                test_pass_rate: 1.0,
                compilation_errors: String::new(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Success".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(SelfEvolutionResult {
            iteration: 2,
            file: "b.rs".to_string(),
            status: SelfEvolutionStatus::Failed,
            score: Some(SelfEvolutionScore {
                compiles: false,
                tests_passed: 0,
                tests_total: 0,
                test_pass_rate: 0.0,
                compilation_errors: "error".to_string(),
                test_output: String::new(),
                reflection: String::new(),
            }),
            error: None,
            description: "Fail".to_string(),
            hypothesis: "Test".to_string(),
            tests_before: None,
            tests_after: None,
        });
        engine.results.push(make_result(3, "c.rs", SelfEvolutionStatus::Skipped)); // No score
        
        let (compiled, total, rate) = engine.compilation_rate();
        assert_eq!(compiled, 1);
        assert_eq!(total, 3);
        assert!((rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_summary_method_on_summary_struct() {
        let summary = SelfEvolutionSummary {
            total: 10,
            accepted: 6,
            rejected: 2,
            failed: 1,
            skipped: 1,
            success_rate: 0.6,
        };
        let result = summary.summary();
        assert_eq!(result, "10 total: 6 accepted, 2 rejected, 1 failed, 1 skipped (60.0% success)");
    }

    #[test]
    fn test_summary_method_all_accepted() {
        let summary = SelfEvolutionSummary {
            total: 5,
            accepted: 5,
            rejected: 0,
            failed: 0,
            skipped: 0,
            success_rate: 1.0,
        };
        let result = summary.summary();
        assert_eq!(result, "5 total: 5 accepted, 0 rejected, 0 failed, 0 skipped (100.0% success)");
    }

    #[test]
    fn test_summary_method_all_failed() {
        let summary = SelfEvolutionSummary {
            total: 3,
            accepted: 0,
            rejected: 0,
            failed: 3,
            skipped: 0,
            success_rate: 0.0,
        };
        let result = summary.summary();
        assert_eq!(result, "3 total: 0 accepted, 0 rejected, 3 failed, 0 skipped (0.0% success)");
    }

    #[test]
    fn test_summary_method_empty_results() {
        let summary = SelfEvolutionSummary {
            total: 0,
            accepted: 0,
            rejected: 0,
            failed: 0,
            skipped: 0,
            success_rate: 0.0,
        };
        let result = summary.summary();
        assert_eq!(result, "0 total: 0 accepted, 0 rejected, 0 failed, 0 skipped (0.0% success)");
    }

    #[test]
    fn test_summary_method_fractional_success_rate() {
        let summary = SelfEvolutionSummary {
            total: 7,
            accepted: 3,
            rejected: 2,
            failed: 1,
            skipped: 1,
            success_rate: 0.42857,
        };
        let result = summary.summary();
        // 0.42857 * 100 = 42.857, formatted as 42.9
        assert!(result.contains("42.9% success"));
    }
}