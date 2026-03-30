use anyhow::Result;
use std::process::Stdio;

use crate::llm::LLMClient;

use super::AutoResearch;

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
    fn parse_clippy_warnings(output: &str) -> u32 {
        output
            .lines()
            .filter(|line| {
                line.trim().starts_with("warning:") ||
                line.contains("error[E") ||
                (line.contains("warning[") && line.contains("]:"))
            })
            .count() as u32
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
        let output = "error[E0277]: the trait bound is not satisfied\nwarning[clippy::unwrap_used]: called unwrap on an Option";
        assert_eq!(AutoResearch::<crate::llm::LLMClientImpl>::parse_clippy_warnings(output), 2);
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
}
