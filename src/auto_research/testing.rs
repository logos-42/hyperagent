use anyhow::Result;

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

        let mut passed = 0u32;
        let mut total = 0u32;
        for line in combined.lines() {
            if line.contains("test result:") {
                let nums: Vec<u32> = line
                    .split(|c: char| !c.is_ascii_digit())
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if !nums.is_empty() {
                    passed = nums[0];
                    total = nums.get(1).map(|&n| nums[0] + n).unwrap_or(nums[0]);
                }
            }
        }

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
}
