use anyhow::Result;
use std::process::Stdio;

use crate::llm::LLMClient;

use super::AutoResearch;

/// Comprehensive quality report from running tests and linting together.
/// Provides a unified view of code health for evaluation.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Number of tests that passed
    pub tests_passed: u32,
    /// Total number of tests run
    pub tests_total: u32,
    /// Whether compilation succeeded
    pub compiles: bool,
    /// Number of compilation errors (blocking)
    pub compilation_errors: u32,
    /// Number of clippy warnings (non-blocking quality issues)
    pub clippy_warnings: u32,
    /// Combined output for debugging
    pub output: String,
}

impl QualityReport {
    /// Returns true if all tests pass and code compiles without errors.
    pub fn is_healthy(&self) -> bool {
        self.tests_passed == self.tests_total && self.compilation_errors == 0
    }

    /// Returns a simple score from 0.0 to 1.0 based on test pass rate and error count.
    /// Uses a weighted formula: test_pass_rate * 0.7 - error_penalty * 0.3
    pub fn health_score(&self) -> f64 {
        if self.tests_total == 0 {
            return if self.compilation_errors == 0 { 0.5 } else { 0.0 };
        }
        let test_rate = self.tests_passed as f64 / self.tests_total as f64;
        let error_penalty = (self.compilation_errors as f64).min(1.0);
        (test_rate * 0.7 + (1.0 - error_penalty) * 0.3).clamp(0.0, 1.0)
    }

    /// Compare this report against another (typically previous/baseline) report.
    /// Returns a QualityDelta showing which metrics improved or regressed.
    /// Positive values indicate improvement, negative values indicate regression.
    pub fn compare(&self, other: &QualityReport) -> QualityDelta {
        QualityDelta {
            tests_passed_delta: self.tests_passed as i64 - other.tests_passed as i64,
            tests_total_delta: self.tests_total as i64 - other.tests_total as i64,
            test_pass_rate_delta: self.test_pass_rate() - other.test_pass_rate(),
            compilation_fixed: other.compilation_errors > 0 && self.compilation_errors == 0,
            compilation_broken: other.compilation_errors == 0 && self.compilation_errors > 0,
            compilation_errors_delta: self.compilation_errors as i64 - other.compilation_errors as i64,
            clippy_warnings_delta: self.clippy_warnings as i64 - other.clippy_warnings as i64,
            health_score_delta: self.health_score() - other.health_score(),
        }
    }

    /// Returns the test pass rate as a percentage (0.0 to 1.0).
    /// Returns 1.0 if there are no tests (nothing to fail).
    fn test_pass_rate(&self) -> f64 {
        if self.tests_total == 0 {
            1.0
        } else {
            self.tests_passed as f64 / self.tests_total as f64
        }
    }
}

/// Delta between two QualityReports, showing what changed.
/// Positive values generally indicate improvement (except for totals/errors/warnings).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityDelta {
    /// Change in number of passing tests (positive = more passing)
    pub tests_passed_delta: i64,
    /// Change in total test count (positive = more tests added)
    pub tests_total_delta: i64,
    /// Change in test pass rate (positive = better rate)
    pub test_pass_rate_delta: f64,
    /// Whether compilation errors were fixed (went from errors to no errors)
    pub compilation_fixed: bool,
    /// Whether compilation was broken (went from no errors to errors)
    pub compilation_broken: bool,
    /// Change in compilation error count (positive = more errors)
    pub compilation_errors_delta: i64,
    /// Change in clippy warnings (positive = more warnings)
    pub clippy_warnings_delta: i64,
    /// Change in overall health score (positive = healthier)
    pub health_score_delta: f64,
}

impl QualityDelta {
    /// Returns true if the change represents a net improvement.
    /// An improvement has: no new compilation errors, same or better test pass rate,
    /// and same or fewer clippy warnings.
    pub fn is_improvement(&self) -> bool {
        !self.compilation_broken
            && self.test_pass_rate_delta >= 0.0
            && self.clippy_warnings_delta <= 0
            && self.compilation_errors_delta <= 0
    }

    /// Returns true if the change represents a clear regression.
    /// A regression has: new compilation errors, or significantly worse test pass rate.
    pub fn is_regression(&self) -> bool {
        self.compilation_broken
            || self.test_pass_rate_delta < -0.1
            || self.compilation_errors_delta > 0
    }

    /// Returns a summary score for this delta (-1.0 to 1.0).
    /// Positive = improvement, negative = regression.
    pub fn net_score(&self) -> f64 {
        let mut score = 0.0;

        // Test pass rate change (weight: 40%)
        score += self.test_pass_rate_delta * 0.4;

        // Health score change (weight: 30%)
        score += self.health_score_delta * 0.3;

        // Compilation status (weight: 20%)
        if self.compilation_fixed {
            score += 0.2;
        } else if self.compilation_broken {
            score -= 0.2;
        }

        // Clippy warnings change (weight: 10%)
        score -= (self.clippy_warnings_delta as f64 * 0.01).min(0.1);

        score.clamp(-1.0, 1.0)
    }

    /// Returns a human-readable one-line summary of the changes.
    /// Example: "2 more tests passing, compilation fixed, 3 fewer warnings"
    /// Example: "1 test failing, compilation broken, 5 more warnings"
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        // Test changes
        if self.tests_passed_delta > 0 {
            parts.push(format!("{} more test{} passing", 
                self.tests_passed_delta, 
                if self.tests_passed_delta == 1 { "" } else { "s" }));
        } else if self.tests_passed_delta < 0 {
            parts.push(format!("{} test{} failing", 
                (-self.tests_passed_delta),
                if self.tests_passed_delta == -1 { "" } else { "s" }));
        }

        // Compilation changes
        if self.compilation_fixed {
            parts.push("compilation fixed".to_string());
        } else if self.compilation_broken {
            parts.push("compilation broken".to_string());
        }

        // Warning changes
        if self.clippy_warnings_delta < 0 {
            parts.push(format!("{} fewer warning{}", 
                (-self.clippy_warnings_delta),
                if self.clippy_warnings_delta == -1 { "" } else { "s" }));
        } else if self.clippy_warnings_delta > 0 {
            parts.push(format!("{} more warning{}", 
                self.clippy_warnings_delta,
                if self.clippy_warnings_delta == 1 { "" } else { "s" }));
        }

        if parts.is_empty() {
            "no changes".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Merge multiple deltas into a single aggregate delta.
    /// This is useful for tracking cumulative quality changes across multiple experiments.
    /// Compilation status is tracked as: fixed if any delta fixed it, broken only if all broke it.
    /// 
    /// # Example
    /// ```
    /// let delta1 = QualityDelta { tests_passed_delta: 2, ... };
    /// let delta2 = QualityDelta { tests_passed_delta: 1, ... };
    /// let merged = QualityDelta::merge(&[delta1, delta2]);
    /// assert_eq!(merged.tests_passed_delta, 3);
    /// ```
    pub fn merge(deltas: &[QualityDelta]) -> QualityDelta {
        if deltas.is_empty() {
            return QualityDelta {
                tests_passed_delta: 0,
                tests_total_delta: 0,
                test_pass_rate_delta: 0.0,
                compilation_fixed: false,
                compilation_broken: false,
                compilation_errors_delta: 0,
                clippy_warnings_delta: 0,
                health_score_delta: 0.0,
            };
        }

        let tests_passed_delta: i64 = deltas.iter().map(|d| d.tests_passed_delta).sum();
        let tests_total_delta: i64 = deltas.iter().map(|d| d.tests_total_delta).sum();
        let test_pass_rate_delta: f64 = deltas.iter().map(|d| d.test_pass_rate_delta).sum();
        let compilation_errors_delta: i64 = deltas.iter().map(|d| d.compilation_errors_delta).sum();
        let clippy_warnings_delta: i64 = deltas.iter().map(|d| d.clippy_warnings_delta).sum();
        let health_score_delta: f64 = deltas.iter().map(|d| d.health_score_delta).sum();

        // Compilation is fixed if any delta fixed it
        let compilation_fixed = deltas.iter().any(|d| d.compilation_fixed);
        
        // Compilation is broken only if the final state is broken
        // (i.e., not fixed and the last delta shows broken)
        let compilation_broken = !compilation_fixed && deltas.iter().last().map(|d| d.compilation_broken).unwrap_or(false);

        QualityDelta {
            tests_passed_delta,
            tests_total_delta,
            test_pass_rate_delta,
            compilation_fixed,
            compilation_broken,
            compilation_errors_delta,
            clippy_warnings_delta,
            health_score_delta,
        }
    }
}

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// cargo test，解析结果
    pub(crate) async fn run_tests(&self) -> Result<(u32, u32, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Test timeout: {}", e))??;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let (passed, total) = Self::parse_test_result(&combined);

        Ok((passed, total, combined))
    }

    /// cargo check
    pub(crate) async fn compile_check(&self) -> Result<(bool, String)> {
        let output = tokio::process::Command::new("cargo")
            .arg("check")
            .arg("--manifest-path")
            .arg(self.config.project_root.join("Cargo.toml"))
            .output()
            .await?;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok((output.status.success(), combined))
    }

    /// 计算编译警告数
    pub(crate) async fn count_warnings(&self) -> u32 {
        let output = tokio::process::Command::new("cargo")
            .arg("check")
            .arg("--manifest-path")
            .arg(self.config.project_root.join("Cargo.toml"))
            .output()
            .await;

        match output {
            Ok(out) => {
                let combined = format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                combined.lines().filter(|l| l.contains("warning:")).count() as u32
            }
            Err(_) => 0,
        }
    }

    /// Phase 2: 检测新代码中是否有新增测试
    pub(crate) fn detect_new_tests(&self, old_code: &str, new_code: &str) -> (bool, u32) {
        let old_test_count = Self::count_test_fns(old_code);
        let new_test_count = Self::count_test_fns(new_code);
        let generated = new_test_count > old_test_count;
        let added = new_test_count.saturating_sub(old_test_count) as u32;
        (generated, added)
    }

    /// 计算代码中 #[test] 函数的数量
    fn count_test_fns(code: &str) -> usize {
        code.matches("#[test]").count()
    }

    /// Parse test results from cargo test output.
    /// Format: "test result: X passed; Y failed; Z ignored; W measured; N filtered out"
    /// Returns (passed, total) where total = passed + failed
    fn parse_test_result(output: &str) -> (u32, u32) {
        for line in output.lines() {
            if line.contains("test result:") {
                return Self::parse_test_line(line);
            }
        }
        (0, 0)
    }

    /// Parse a single "test result:" line to extract passed/failed counts.
    fn parse_test_line(line: &str) -> (u32, u32) {
        let mut passed = 0u32;
        let mut failed = 0u32;

        // Standard format: "test result: X passed; Y failed; ..."
        for part in line.split(';') {
            let part = part.trim();
            if part.ends_with("passed") {
                passed = Self::extract_number(part).unwrap_or(0);
            } else if part.ends_with("failed") {
                failed = Self::extract_number(part).unwrap_or(0);
            }
        }

        (passed, passed + failed)
    }

    /// Extract the leading number from a string like "5 passed" or " 3 failed".
    fn extract_number(s: &str) -> Option<u32> {
        s.split_whitespace()
            .next()
            .and_then(|n| n.parse().ok())
    }

    /// Run tests for a specific module/file using cargo test with --lib flag.
    /// Returns (passed, total, combined_output) like run_tests, but only for the specified module.
    pub(crate) async fn run_tests_for_file(&self, module_name: &str) -> Result<(u32, u32, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--lib")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .arg("--")
                .arg(module_name)
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Module test timeout for {}: {}", module_name, e))??;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let (passed, total) = Self::parse_test_result(&combined);

        Ok((passed, total, combined))
    }

    /// Run tests for a specific module using cargo test's built-in module filtering.
    /// This uses `cargo test --lib -- module_name` which filters tests by module path.
    /// Returns (passed, total, combined_output).
    pub(crate) async fn run_tests_for_module(&self, module_name: &str) -> Result<(u32, u32, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--lib")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .arg("--")
                .arg(&format!("{}::", module_name))
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Module test timeout for {}: {}", module_name, e))??;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let (passed, total) = Self::parse_test_result(&combined);

        Ok((passed, total, combined))
    }

    /// Run tests matching a specific test name pattern.
    /// Uses `cargo test --lib test_name` to filter tests by name.
    /// Returns (passed, total, combined_output).
    pub(crate) async fn run_tests_by_name(&self, test_name: &str) -> Result<(u32, u32, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--lib")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .arg(test_name)
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Test name filter timeout for {}: {}", test_name, e))??;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let (passed, total) = Self::parse_test_result(&combined);

        Ok((passed, total, combined))
    }

    /// Run clippy lints to detect code quality issues and potential bugs.
    /// Returns (warning_count, combined_output) where warning_count is the number of clippy warnings.
    pub(crate) async fn run_clippy_checks(&self) -> Result<(u32, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::process::Command::new("cargo")
                .arg("clippy")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .arg("--")
                .arg("-W")
                .arg("clippy::all")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Clippy timeout: {}", e))??;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let warning_count = Self::parse_clippy_warnings(&combined);
        Ok((warning_count, combined))
    }

    /// Parse clippy output to count the number of warnings.
    /// Clippy warnings typically appear as "warning: <message>" lines.
    /// Does NOT count compilation errors (error[E...]) - those should be
    /// tracked separately as they indicate broken code, not quality issues.
    fn parse_clippy_warnings(output: &str) -> u32 {
        output
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("warning:") ||
                (trimmed.contains("warning[") && trimmed.contains("]:"))
            })
            .count() as u32
    }

    /// Count compilation errors from cargo/clippy output.
    /// These appear as "error[E...]:" lines indicating actual code problems
    /// that prevent compilation, distinct from lint warnings.
    fn count_compilation_errors(output: &str) -> u32 {
        output
            .lines()
            .filter(|line| line.contains("error[E") && line.contains("]:"))
            .count() as u32
    }

    /// Run tests and capture both test results and compilation status.
    /// Returns (passed, total, has_compilation_error, combined_output).
    /// This is more efficient than running compile_check separately.
    pub(crate) async fn run_tests_with_compile_check(&self) -> Result<(u32, u32, bool, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Test timeout: {}", e))??;

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let (passed, total) = Self::parse_test_result(&combined);
        let has_errors = Self::count_compilation_errors(&combined) > 0;

        Ok((passed, total, has_errors, combined))
    }

    /// Run all quality checks (tests + clippy + compilation) in a single pass.
    /// This is more efficient than running each check separately because it
    /// compiles once and captures all metrics from the same output.
    /// 
    /// Returns a QualityReport with comprehensive code health metrics.
    pub(crate) async fn run_quality_checks(&self) -> Result<QualityReport> {
        // Run cargo test first (compiles + runs tests)
        let test_output = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Test timeout: {}", e))??;

        let test_combined = format!(
            "{}{}",
            String::from_utf8_lossy(&test_output.stdout),
            String::from_utf8_lossy(&test_output.stderr)
        );

        let (tests_passed, tests_total) = Self::parse_test_result(&test_combined);
        let compilation_errors = Self::count_compilation_errors(&test_combined);

        // Run clippy separately (lint checks don't run tests)
        let clippy_output = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            tokio::process::Command::new("cargo")
                .arg("clippy")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .arg("--")
                .arg("-W")
                .arg("clippy::all")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Clippy timeout: {}", e))??;

        let clippy_combined = format!(
            "{}{}",
            String::from_utf8_lossy(&clippy_output.stdout),
            String::from_utf8_lossy(&clippy_output.stderr)
        );

        let clippy_warnings = Self::parse_clippy_warnings(&clippy_combined);

        // Combine outputs for debugging
        let full_output = format!("{}\n\n=== CLIPPY ===\n{}", test_combined, clippy_combined);

        Ok(QualityReport {
            tests_passed,
            tests_total,
            compiles: compilation_errors == 0 && test_output.status.success(),
            compilation_errors,
            clippy_warnings,
            output: full_output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_test_result_all_passed() {
        let output = "running 3 tests\ntest test_one ... ok\ntest test_two ... ok\ntest test_three ... ok\n\ntest result: 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (3, 3));
    }

    #[test]
    fn test_parse_test_result_some_failed() {
        let output = "running 5 tests\ntest test_one ... ok\ntest test_two ... FAILED\ntest result: 3 passed; 2 failed; 0 ignored";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (3, 5));
    }

    #[test]
    fn test_parse_test_result_no_tests() {
        let output = "running 0 tests\n\ntest result: 0 passed; 0 failed; 0 ignored";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (0, 0));
    }

    #[test]
    fn test_parse_test_result_with_ignored() {
        let output = "test result: 10 passed; 2 failed; 5 ignored; 0 measured; 3 filtered out";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (10, 12));
    }

    #[test]
    fn test_parse_test_line_various_formats() {
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_line("test result: 1 passed; 0 failed"), (1, 1));
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_line("test result: 0 passed; 5 failed; 3 ignored"), (0, 5));
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_line("test result: 100 passed; 50 failed; 0 ignored; 0 measured"), (100, 150));
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("5 passed"), Some(5));
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("  12 failed"), Some(12));
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("passed"), None);
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("no tests"), None);
    }

    #[test]
    fn test_count_test_fns() {
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(""), 0);
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns("#[test]"), 1);
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns("#[test]\nfn a() {}\n#[test]\nfn b() {}"), 2);
    }

    #[test]
    fn test_detect_new_tests() {
        let old_code = "#[test]\nfn test_one() {}";
        let new_code = "#[test]\nfn test_one() {}\n#[test]\nfn test_two() {}";
        // Note: detect_new_tests is not static, so we can't test it directly here
        // This test validates the underlying count_test_fns works correctly
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(old_code), 1);
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(new_code), 2);
    }

    #[test]
    fn test_parse_test_result_empty_output() {
        let output = "Build succeeded, no tests run.";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (0, 0));
    }

    #[test]
    fn test_parse_test_result_multiple_test_lines() {
        // When running tests for multiple modules, cargo outputs multiple test result lines
        let output = "test result: 2 passed; 0 failed\ntest result: 3 passed; 1 failed";
        // Should parse the first matching line
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (2, 2));
    }

    #[test]
    fn test_parse_test_result_with_error_format() {
        // Cargo can output errors before test results
        let output = "error: could not compile\nnote: try again\ntest result: 5 passed; 1 failed";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (5, 6));
    }

    #[test]
    fn test_parse_test_result_doctest_format() {
        // Doc tests have slightly different output format
        let output = "test result: 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (2, 2));
    }

    #[test]
    fn test_count_warnings_format() {
        // Verify count_test_fns handles various attribute formats
        let code = "#[test]\nfn test_a() {}\n    #[test]\nfn test_b() {}\n#[test]\n#[should_panic]\nfn test_c() {}";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(code), 3);
    }

    #[test]
    fn test_parse_test_result_with_doc_tests() {
        // Doc tests appear separately in cargo output
        let output = "running 2 tests\ntest src/lib.rs - (line 10) ... ok\ntest src/lib.rs - (line 20) ... ok\n\ntest result: 2 passed; 0 failed";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (2, 2));
    }

    #[test]
    fn test_parse_test_result_with_ignored_count() {
        // Tests can be marked as ignored
        let output = "test result: 5 passed; 0 failed; 3 ignored";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_test_result(output), (5, 5));
    }

    #[test]
    fn test_extract_number_edge_cases() {
        // Edge cases for number extraction
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("0 passed"), Some(0));
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("999999 failed"), Some(999999));
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number(""), None);
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("   passed"), None);
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::extract_number("passed 5"), None); // number after word, not before
    }

    #[test]
    fn test_count_test_fns_with_attributes() {
        // Test with various attribute configurations
        let code1 = "#[test]\nfn basic_test() {}";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(code1), 1);

        let code2 = "#[test]\n#[ignore]\nfn ignored_test() {}";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(code2), 1);

        let code3 = "#[cfg(test)]\nmod tests { #[test]\nfn inner() {} }";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_test_fns(code3), 1);
    }

    #[test]
    fn test_parse_clippy_warnings_basic() {
        let output = "warning: unused variable\nnote: consider binding to underscore\nwarning: function too long";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_clippy_warnings(output), 2);
    }

    #[test]
    fn test_parse_clippy_warnings_with_error_codes() {
        // Compilation errors should NOT be counted as clippy warnings
        let output = "error[E0277]: the trait bound is not satisfied\nwarning[clippy::unwrap_used]: called unwrap on an Option";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_clippy_warnings(output), 1);
    }

    #[test]
    fn test_count_compilation_errors_basic() {
        let output = "error[E0277]: the trait bound is not satisfied\nwarning[clippy::unwrap_used]: called unwrap on an Option";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_compilation_errors(output), 1);
    }

    #[test]
    fn test_count_compilation_errors_multiple() {
        let output = "error[E0277]: trait bound\nerror[E0382]: borrow moved\nwarning: unused variable";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_compilation_errors(output), 2);
    }

    #[test]
    fn test_count_compilation_errors_none() {
        let output = "warning: unused variable\nwarning[clippy::let_unit_value]: unit value";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::count_compilation_errors(output), 0);
    }

    #[test]
    fn test_parse_clippy_warnings_empty() {
        let output = "Compiling hyperagent v0.1.0\nFinished dev [unoptimized + debuginfo]";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_clippy_warnings(output), 0);
    }

    #[test]
    fn test_parse_clippy_warnings_mixed() {
        let output = "warning: variable does not need to be mutable\n   --> src/lib.rs:10:5\nhelp: remove this\nwarning[clippy::let_unit_value]: this creates a () value";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_clippy_warnings(output), 2);
    }

    #[test]
    fn test_parse_clippy_warnings_multiline() {
        let output = r#"warning: this function is too long
   --> src/main.rs:15:1
    |
15  | fn long_function() {
    | ^^^^^^^^^^^^^^^^^^
    |
    = note: `-W clippy::too-long-function` implied by `-W clippy::all`
warning: another issue
"#;
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_clippy_warnings(output), 2);
    }

    #[test]
    fn test_quality_report_is_healthy() {
        let healthy = QualityReport {
            tests_passed: 10,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 3,
            output: String::new(),
        };
        assert!(healthy.is_healthy());

        let unhealthy = QualityReport {
            tests_passed: 8,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 0,
            output: String::new(),
        };
        assert!(!unhealthy.is_healthy());

        let broken = QualityReport {
            tests_passed: 10,
            tests_total: 10,
            compiles: false,
            compilation_errors: 2,
            clippy_warnings: 5,
            output: String::new(),
        };
        assert!(!broken.is_healthy());
    }

    #[test]
    fn test_quality_report_health_score() {
        // Perfect score
        let perfect = QualityReport {
            tests_passed: 10,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 0,
            output: String::new(),
        };
        assert!((perfect.health_score() - 1.0).abs() < 0.01);

        // Partial tests, no errors
        let partial = QualityReport {
            tests_passed: 5,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 5,
            output: String::new(),
        };
        let score = partial.health_score();
        assert!(score > 0.35 && score < 0.36, "Expected ~0.35, got {}", score);

        // No tests, no errors (neutral)
        let no_tests = QualityReport {
            tests_passed: 0,
            tests_total: 0,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 3,
            output: String::new(),
        };
        assert!((no_tests.health_score() - 0.5).abs() < 0.01);

        // No tests, has errors (bad)
        let no_tests_errors = QualityReport {
            tests_passed: 0,
            tests_total: 0,
            compiles: false,
            compilation_errors: 1,
            clippy_warnings: 0,
            output: String::new(),
        };
        assert!((no_tests_errors.health_score() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_report_health_score_clamps() {
        // Test that score is clamped between 0 and 1
        let all_fail = QualityReport {
            tests_passed: 0,
            tests_total: 10,
            compiles: false,
            compilation_errors: 5,
            clippy_warnings: 10,
            output: String::new(),
        };
        assert!(all_fail.health_score() >= 0.0);
    }

    #[test]
    fn test_quality_report_compare_improvement() {
        let before = QualityReport {
            tests_passed: 8,
            tests_total: 10,
            compiles: false,
            compilation_errors: 2,
            clippy_warnings: 5,
            output: String::new(),
        };
        let after = QualityReport {
            tests_passed: 10,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 3,
            output: String::new(),
        };

        let delta = after.compare(&before);
        assert_eq!(delta.tests_passed_delta, 2);
        assert_eq!(delta.tests_total_delta, 0);
        assert!(delta.test_pass_rate_delta > 0.0);
        assert!(delta.compilation_fixed);
        assert!(!delta.compilation_broken);
        assert_eq!(delta.compilation_errors_delta, -2);
        assert_eq!(delta.clippy_warnings_delta, -2);
        assert!(delta.health_score_delta > 0.0);
        assert!(delta.is_improvement());
        assert!(!delta.is_regression());
    }

    #[test]
    fn test_quality_report_compare_regression() {
        let before = QualityReport {
            tests_passed: 10,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 2,
            output: String::new(),
        };
        let after = QualityReport {
            tests_passed: 7,
            tests_total: 10,
            compiles: false,
            compilation_errors: 3,
            clippy_warnings: 5,
            output: String::new(),
        };

        let delta = after.compare(&before);
        assert_eq!(delta.tests_passed_delta, -3);
        assert!(delta.test_pass_rate_delta < 0.0);
        assert!(!delta.compilation_fixed);
        assert!(delta.compilation_broken);
        assert_eq!(delta.compilation_errors_delta, 3);
        assert_eq!(delta.clippy_warnings_delta, 3);
        assert!(delta.health_score_delta < 0.0);
        assert!(!delta.is_improvement());
        assert!(delta.is_regression());
    }

    #[test]
    fn test_quality_report_compare_neutral() {
        let before = QualityReport {
            tests_passed: 5,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 3,
            output: String::new(),
        };
        let after = QualityReport {
            tests_passed: 5,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 3,
            output: String::new(),
        };

        let delta = after.compare(&before);
        assert_eq!(delta.tests_passed_delta, 0);
        assert_eq!(delta.test_pass_rate_delta, 0.0);
        assert!(!delta.compilation_fixed);
        assert!(!delta.compilation_broken);
        assert_eq!(delta.compilation_errors_delta, 0);
        assert_eq!(delta.clippy_warnings_delta, 0);
        assert_eq!(delta.health_score_delta, 0.0);
        assert!(delta.is_improvement()); // No regression = acceptable
        assert!(!delta.is_regression());
    }

    #[test]
    fn test_quality_delta_net_score() {
        // Improvement scenario
        let improvement = QualityDelta {
            tests_passed_delta: 2,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.2,
            compilation_fixed: true,
            compilation_broken: false,
            compilation_errors_delta: -2,
            clippy_warnings_delta: -3,
            health_score_delta: 0.15,
        };
        assert!(improvement.net_score() > 0.0);

        // Regression scenario
        let regression = QualityDelta {
            tests_passed_delta: -3,
            tests_total_delta: 0,
            test_pass_rate_delta: -0.3,
            compilation_fixed: false,
            compilation_broken: true,
            compilation_errors_delta: 2,
            clippy_warnings_delta: 5,
            health_score_delta: -0.2,
        };
        assert!(regression.net_score() < 0.0);

        // Neutral scenario
        let neutral = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: 0,
            health_score_delta: 0.0,
        };
        assert_eq!(neutral.net_score(), 0.0);
    }

    #[test]
    fn test_quality_report_test_pass_rate() {
        let report = QualityReport {
            tests_passed: 7,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 0,
            output: String::new(),
        };
        assert!((report.test_pass_rate() - 0.7).abs() < 0.01);

        let no_tests = QualityReport {
            tests_passed: 0,
            tests_total: 0,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 0,
            output: String::new(),
        };
        assert!((no_tests.test_pass_rate() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_report_compare_more_tests() {
        let before = QualityReport {
            tests_passed: 10,
            tests_total: 10,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 0,
            output: String::new(),
        };
        let after = QualityReport {
            tests_passed: 15,
            tests_total: 15,
            compiles: true,
            compilation_errors: 0,
            clippy_warnings: 0,
            output: String::new(),
        };

        let delta = after.compare(&before);
        assert_eq!(delta.tests_passed_delta, 5);
        assert_eq!(delta.tests_total_delta, 5);
        assert_eq!(delta.test_pass_rate_delta, 0.0); // Both have 100% pass rate
        assert!(delta.is_improvement());
    }

    #[test]
    fn test_quality_delta_summary_improvement() {
        let delta = QualityDelta {
            tests_passed_delta: 2,
            tests_total_delta: 2,
            test_pass_rate_delta: 0.1,
            compilation_fixed: true,
            compilation_broken: false,
            compilation_errors_delta: -1,
            clippy_warnings_delta: -3,
            health_score_delta: 0.15,
        };
        assert_eq!(delta.summary(), "2 more tests passing, compilation fixed, 3 fewer warnings");
    }

    #[test]
    fn test_quality_delta_summary_regression() {
        let delta = QualityDelta {
            tests_passed_delta: -1,
            tests_total_delta: 0,
            test_pass_rate_delta: -0.1,
            compilation_fixed: false,
            compilation_broken: true,
            compilation_errors_delta: 2,
            clippy_warnings_delta: 5,
            health_score_delta: -0.2,
        };
        assert_eq!(delta.summary(), "1 test failing, compilation broken, 5 more warnings");
    }

    #[test]
    fn test_quality_delta_summary_single_changes() {
        let single_test = QualityDelta {
            tests_passed_delta: 1,
            tests_total_delta: 1,
            test_pass_rate_delta: 0.1,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: 0,
            health_score_delta: 0.05,
        };
        assert_eq!(single_test.summary(), "1 more test passing");

        let single_warning = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: -1,
            health_score_delta: 0.02,
        };
        assert_eq!(single_warning.summary(), "1 fewer warning");
    }

    #[test]
    fn test_quality_delta_summary_no_changes() {
        let no_change = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: 0,
            health_score_delta: 0.0,
        };
        assert_eq!(no_change.summary(), "no changes");
    }

    #[test]
    fn test_quality_delta_summary_warnings_only() {
        let warnings_only = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: 2,
            health_score_delta: -0.01,
        };
        assert_eq!(warnings_only.summary(), "2 more warnings");
    }

    #[test]
    fn test_quality_delta_merge_empty() {
        let merged = QualityDelta::merge(&[]);
        assert_eq!(merged.tests_passed_delta, 0);
        assert_eq!(merged.tests_total_delta, 0);
        assert_eq!(merged.test_pass_rate_delta, 0.0);
        assert!(!merged.compilation_fixed);
        assert!(!merged.compilation_broken);
        assert_eq!(merged.compilation_errors_delta, 0);
        assert_eq!(merged.clippy_warnings_delta, 0);
        assert_eq!(merged.health_score_delta, 0.0);
    }

    #[test]
    fn test_quality_delta_merge_single() {
        let delta = QualityDelta {
            tests_passed_delta: 5,
            tests_total_delta: 5,
            test_pass_rate_delta: 0.1,
            compilation_fixed: true,
            compilation_broken: false,
            compilation_errors_delta: -2,
            clippy_warnings_delta: -3,
            health_score_delta: 0.15,
        };
        let merged = QualityDelta::merge(&[delta]);
        assert_eq!(merged.tests_passed_delta, 5);
        assert!(merged.compilation_fixed);
    }

    #[test]
    fn test_quality_delta_merge_multiple_improvements() {
        let delta1 = QualityDelta {
            tests_passed_delta: 2,
            tests_total_delta: 2,
            test_pass_rate_delta: 0.1,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: -1,
            clippy_warnings_delta: -2,
            health_score_delta: 0.1,
        };
        let delta2 = QualityDelta {
            tests_passed_delta: 3,
            tests_total_delta: 3,
            test_pass_rate_delta: 0.05,
            compilation_fixed: true,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: -1,
            health_score_delta: 0.05,
        };
        let merged = QualityDelta::merge(&[delta1, delta2]);
        assert_eq!(merged.tests_passed_delta, 5);
        assert_eq!(merged.tests_total_delta, 5);
        assert!((merged.test_pass_rate_delta - 0.15).abs() < 0.001);
        assert!(merged.compilation_fixed); // At least one fixed
        assert!(!merged.compilation_broken);
        assert_eq!(merged.compilation_errors_delta, -1);
        assert_eq!(merged.clippy_warnings_delta, -3);
        assert!((merged.health_score_delta - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_quality_delta_merge_mixed_results() {
        let improvement = QualityDelta {
            tests_passed_delta: 5,
            tests_total_delta: 5,
            test_pass_rate_delta: 0.2,
            compilation_fixed: true,
            compilation_broken: false,
            compilation_errors_delta: -2,
            clippy_warnings_delta: -5,
            health_score_delta: 0.3,
        };
        let regression = QualityDelta {
            tests_passed_delta: -2,
            tests_total_delta: 0,
            test_pass_rate_delta: -0.1,
            compilation_fixed: false,
            compilation_broken: true,
            compilation_errors_delta: 1,
            clippy_warnings_delta: 3,
            health_score_delta: -0.15,
        };
        let merged = QualityDelta::merge(&[improvement, regression]);
        assert_eq!(merged.tests_passed_delta, 3); // 5 - 2
        assert_eq!(merged.tests_total_delta, 5);
        assert!((merged.test_pass_rate_delta - 0.1).abs() < 0.001); // 0.2 - 0.1
        assert!(merged.compilation_fixed); // At least one fixed
        assert!(!merged.compilation_broken); // Last was broken but fixed earlier
        assert_eq!(merged.compilation_errors_delta, -1); // -2 + 1
        assert_eq!(merged.clippy_warnings_delta, -2); // -5 + 3
        assert!((merged.health_score_delta - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_quality_delta_merge_compilation_status() {
        // If compilation was fixed in any delta, it's considered fixed
        let fixed_once = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: true,
            compilation_broken: false,
            compilation_errors_delta: -1,
            clippy_warnings_delta: 0,
            health_score_delta: 0.1,
        };
        let no_fix = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: false,
            compilation_errors_delta: 0,
            clippy_warnings_delta: 0,
            health_score_delta: 0.0,
        };
        let merged = QualityDelta::merge(&[no_fix, fixed_once, no_fix]);
        assert!(merged.compilation_fixed);
        assert!(!merged.compilation_broken);
    }

    #[test]
    fn test_quality_delta_merge_all_broken() {
        let broken1 = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: true,
            compilation_errors_delta: 1,
            clippy_warnings_delta: 0,
            health_score_delta: -0.1,
        };
        let broken2 = QualityDelta {
            tests_passed_delta: 0,
            tests_total_delta: 0,
            test_pass_rate_delta: 0.0,
            compilation_fixed: false,
            compilation_broken: true,
            compilation_errors_delta: 2,
            clippy_warnings_delta: 0,
            health_score_delta: -0.1,
        };
        let merged = QualityDelta::merge(&[broken1, broken2]);
        assert!(!merged.compilation_fixed);
        assert!(merged.compilation_broken); // Last delta is broken, none fixed
    }
}
