//! 递归自改进模块
//!
//! 让 Hyperagent 修改自己的源代码，通过编译检查和测试验证改进效果。
//!
//! 流程:
//! 1. 选择待改进的源文件
//! 2. 读取当前代码 + 收集改进上下文
//! 3. LLM 生成修改后的代码
//! 4. 写入磁盘 → cargo check → cargo test
//! 5. 通过则 git commit，失败则 git checkout 回滚
//! 6. 重复

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    /// 编译超时（秒）
    pub compile_timeout_secs: u64,
    /// 测试超时（秒）
    pub test_timeout_secs: u64,
    /// 只修改，不自动 commit（安全模式）
    pub dry_run: bool,
}

impl Default for SelfEvolutionConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            target_files: vec![
                "runtime/thermodynamics.rs".to_string(),
                "runtime/loop_.rs".to_string(),
                "agent/mutator.rs".to_string(),
                "agent/meta_mutator.rs".to_string(),
                "eval/evaluator.rs".to_string(),
            ],
            max_iterations: 10,
            compile_timeout_secs: 120,
            test_timeout_secs: 300,
            dry_run: true, // 默认安全模式
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// 递归自改进引擎
pub struct SelfEvolutionEngine<C: LLMClient> {
    client: C,
    config: SelfEvolutionConfig,
    iteration: u32,
    results: Vec<SelfEvolutionResult>,
}

impl<C: LLMClient + Clone> SelfEvolutionEngine<C> {
    pub fn new(client: C, config: SelfEvolutionConfig) -> Self {
        Self {
            client,
            config,
            iteration: 0,
            results: Vec::new(),
        }
    }

    /// 收集自改进上下文：当前代码状态 + 测试结果 + 历史改进
    fn build_context(&self, file_rel: &str) -> Result<String> {
        let src_path = self.config.project_root.join("src").join(file_rel);
        let current_code = std::fs::read_to_string(&src_path)
            .with_context(|| format!("Cannot read {}", src_path.display()))?;

        // 收集最近的自改进历史
        let history = self
            .results
            .iter()
            .rev()
            .take(5)
            .map(|r| {
                format!(
                    "- [Iter {}] {:?} {}: {}",
                    r.iteration, r.status, r.file,
                    r.error.as_deref().unwrap_or("ok")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(
            "=== SELF-EVOLUTION CONTEXT ===\n\
             File: src/{}\n\
             \n\
             === CURRENT CODE ===\n\
             {}\n\
             \n\
             === PAST IMPROVEMENT ATTEMPTS ===\n\
             {}\n\
             \n\
             === INSTRUCTIONS ===\n\
             You are improving this Rust source file as part of a self-evolving agent system.\n\
             Rules:\n\
             1. Output ONLY the complete modified Rust source code, no explanations.\n\
             2. Do NOT change public API signatures (function names, trait methods, struct fields).\n\
             3. Do NOT add new external dependencies.\n\
             4. You MAY improve internal logic, algorithms, error handling, and documentation.\n\
             5. You MAY add new helper functions or refactor internal structure.\n\
             6. Keep all existing use statements and module declarations.\n\
             7. Ensure the code compiles with the existing dependencies.\n",
            file_rel, current_code,
            if history.is_empty() { "(none yet)".to_string() } else { history }
        ))
    }

    /// 调用 LLM 生成修改后的代码
    async fn generate_mutation(&self, context: &str) -> Result<String> {
        let response = self.client.complete(context).await?;
        Ok(response.content)
    }

    /// 提取代码块（处理 markdown code fence 包裹的情况）
    fn extract_code(&self, raw: &str) -> String {
        // 如果被 ```rust ... ``` 包裹，提取内部内容
        if raw.contains("```rust") || raw.contains("```") {
            let start = raw.find("```rust").map(|i| i + 7)
                .or_else(|| raw.find("```").map(|i| i + 3));
            if let Some(start) = start {
                if let Some(end) = raw[start..].find("```") {
                    return raw[start..start + end].trim().to_string();
                }
            }
        }
        // 去除首尾空白行
        raw.trim().to_string()
    }

    /// 编译检查
    async fn compile_check(&self) -> Result<(bool, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.compile_timeout_secs),
            tokio::process::Command::new("cargo")
                .arg("check")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Compile timeout or spawn error: {}", e))?
        .context("Failed to run cargo check")?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let success = output.status.success();

        Ok((success, format!("{}{}", stdout, stderr)))
    }

    /// 运行测试
    async fn run_tests(&self) -> Result<(u32, u32, String)> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.test_timeout_secs),
            tokio::process::Command::new("cargo")
                .arg("test")
                .arg("--manifest-path")
                .arg(self.config.project_root.join("Cargo.toml"))
                .output(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Test timeout or spawn error: {}", e))?
        .context("Failed to run cargo test")?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let combined = format!("{}{}", stdout, stderr);

        // 解析测试结果: "test result: ok. X passed; Y failed"
        let mut passed = 0u32;
        let mut total = 0u32;
        for line in combined.lines() {
            if line.contains("test result:") {
                // 提取数字
                let nums: Vec<u32> = line
                    .split(|c: char| !c.is_ascii_digit())
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() >= 2 {
                    passed = nums[0]; // passed
                    total = nums[0]; // passed + failed
                    if nums.len() >= 3 {
                        total = nums[0] + nums[1];
                    }
                }
            }
        }

        Ok((passed, total, combined))
    }

    /// Git 回滚文件
    fn git_checkout(&self, file_rel: &str) -> Result<()> {
        let src_path = format!("src/{}", file_rel);
        let output = std::process::Command::new("git")
            .arg("checkout")
            .arg("--")
            .arg(&src_path)
            .current_dir(&self.config.project_root)
            .output()
            .context("Failed to run git checkout")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git checkout failed: {}", stderr);
        }
        Ok(())
    }

    /// Git 提交改进
    fn git_commit(&self, file_rel: &str, message: &str) -> Result<()> {
        let src_path = format!("src/{}", file_rel);
        let _ = std::process::Command::new("git")
            .arg("add")
            .arg(&src_path)
            .current_dir(&self.config.project_root)
            .output();

        let output = std::process::Command::new("git")
            .arg("commit")
            .arg("-m")
            .arg(message)
            .current_dir(&self.config.project_root)
            .output()
            .context("Failed to run git commit")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git commit failed: {}", stderr);
        }
        Ok(())
    }

    /// 获取修改前基线（当前测试通过数）
    async fn baseline_score(&self) -> Result<(u32, u32)> {
        let (passed, total, _) = self.run_tests().await?;
        Ok((passed, total))
    }

    /// 执行一次自改进迭代
    async fn evolve_file(&mut self, file_rel: &str) -> Result<SelfEvolutionResult> {
        self.iteration += 1;
        let iter = self.iteration;

        tracing::info!("Self-evolution iter {}: improving src/{}", iter, file_rel);

        // 1. 构建上下文
        let context = self.build_context(file_rel)?;

        // 2. LLM 生成修改
        let raw_code = self.generate_mutation(&context).await?;
        let new_code = self.extract_code(&raw_code);

        if new_code.is_empty() {
            return Ok(SelfEvolutionResult {
                iteration: iter,
                file: file_rel.to_string(),
                status: SelfEvolutionStatus::Skipped,
                score: None,
                error: Some("LLM returned empty code".to_string()),
                description: "Empty response from LLM".to_string(),
            });
        }

        // 3. 写入磁盘
        let src_path = self.config.project_root.join("src").join(file_rel);
        if let Err(e) = std::fs::write(&src_path, &new_code) {
            return Ok(SelfEvolutionResult {
                iteration: iter,
                file: file_rel.to_string(),
                status: SelfEvolutionStatus::Failed,
                score: None,
                error: Some(format!("Write failed: {}", e)),
                description: "Failed to write modified file".to_string(),
            });
        }

        // 4. 编译检查
        tracing::info!("  Compiling...");
        let (compiles, compile_output) = self.compile_check().await?;

        if !compiles {
            tracing::warn!("  Compilation failed, reverting");
            self.git_checkout(file_rel)?;
            return Ok(SelfEvolutionResult {
                iteration: iter,
                file: file_rel.to_string(),
                status: SelfEvolutionStatus::Rejected,
                score: Some(SelfEvolutionScore {
                    compiles: false,
                    tests_passed: 0,
                    tests_total: 0,
                    test_pass_rate: 0.0,
                    compilation_errors: compile_output.chars().take(500).collect(),
                    test_output: String::new(),
                }),
                error: Some("Compilation failed".to_string()),
                description: "Code did not compile".to_string(),
            });
        }

        // 5. 运行测试
        tracing::info!("  Testing...");
        let (tests_passed, tests_total, test_output) = self.run_tests().await?;
        let pass_rate = if tests_total > 0 {
            tests_passed as f32 / tests_total as f32
        } else {
            0.0
        };

        let score = SelfEvolutionScore {
            compiles: true,
            tests_passed,
            tests_total,
            test_pass_rate: pass_rate,
            compilation_errors: String::new(),
            test_output: test_output.chars().take(1000).collect(),
        };

        // 6. 决策：测试不全部通过则回滚
        let all_pass = tests_passed == tests_total && tests_total > 0;
        if !all_pass {
            tracing::warn!(
                "  Tests: {}/{} passed, reverting",
                tests_passed, tests_total
            );
            self.git_checkout(file_rel)?;
            return Ok(SelfEvolutionResult {
                iteration: iter,
                file: file_rel.to_string(),
                status: SelfEvolutionStatus::Rejected,
                score: Some(score),
                error: Some(format!("Tests: {}/{} passed", tests_passed, tests_total)),
                description: format!("Tests regressed: {}/{}", tests_passed, tests_total),
            });
        }

        // 7. 全部通过 → 接受
        if self.config.dry_run {
            tracing::info!(
                "  [DRY RUN] Would accept: {}/{} tests passed",
                tests_passed, tests_total
            );
            // dry_run 模式下仍然回滚
            self.git_checkout(file_rel)?;
            Ok(SelfEvolutionResult {
                iteration: iter,
                file: file_rel.to_string(),
                status: SelfEvolutionStatus::Accepted,
                score: Some(score),
                error: None,
                description: format!(
                    "[DRY RUN] All {} tests passed (change reverted)",
                    tests_total
                ),
            })
        } else {
            let msg = format!(
                "self-evolution: iter {} improved src/{} ({} tests passing)",
                iter, file_rel, tests_total
            );
            self.git_commit(file_rel, &msg)?;
            tracing::info!("  Accepted and committed: {}", msg);
            Ok(SelfEvolutionResult {
                iteration: iter,
                file: file_rel.to_string(),
                status: SelfEvolutionStatus::Accepted,
                score: Some(score),
                error: None,
                description: msg,
            })
        }
    }

    /// 运行自改进主循环
    pub async fn run(&mut self) -> Result<Vec<SelfEvolutionResult>> {
        tracing::info!(
            "=== Starting Recursive Self-Improvement ==="
        );
        tracing::info!(
            "Project: {:?}, Targets: {:?}, Max iterations: {}, Dry run: {}",
            self.config.project_root,
            self.config.target_files,
            self.config.max_iterations,
            self.config.dry_run,
        );

        // 基线测试
        let (baseline_passed, baseline_total) = self.baseline_score().await?;
        tracing::info!(
            "Baseline: {}/{} tests passing",
            baseline_passed, baseline_total
        );

        let mut iteration = 0u32;
        let mut file_idx = 0;

        while iteration < self.config.max_iterations {
            let file = self.config.target_files[file_idx].clone();
            let result = self.evolve_file(&file).await?;
            let status_str = match result.status {
                SelfEvolutionStatus::Accepted => "ACCEPTED",
                SelfEvolutionStatus::Rejected => "REJECTED",
                SelfEvolutionStatus::Skipped => "SKIPPED",
                SelfEvolutionStatus::Failed => "FAILED",
            };

            let score_info = result
                .score
                .as_ref()
                .map(|s| {
                    format!(
                        "compile={} tests={}/{}",
                        s.compiles, s.tests_passed, s.tests_total
                    )
                })
                .unwrap_or_else(|| "N/A".to_string());

            tracing::info!(
                "  [{status_str}] src/{file} | {score_info} | {desc}",
                status_str = status_str,
                file = result.file,
                score_info = score_info,
                desc = result.description,
            );

            self.results.push(result);
            iteration += 1;

            // 轮换目标文件
            file_idx = (file_idx + 1) % self.config.target_files.len();
        }

        // 最终统计
        let accepted = self.results.iter().filter(|r| matches!(r.status, SelfEvolutionStatus::Accepted)).count();
        let rejected = self.results.iter().filter(|r| matches!(r.status, SelfEvolutionStatus::Rejected)).count();
        let failed = self.results.iter().filter(|r| matches!(r.status, SelfEvolutionStatus::Failed)).count();

        tracing::info!("=== Self-Improvement Complete ===");
        tracing::info!(
            "Iterations: {} | Accepted: {} | Rejected: {} | Failed: {}",
            iteration, accepted, rejected, failed,
        );

        // 最终测试
        let (final_passed, final_total, _) = self.run_tests().await?;
        tracing::info!(
            "Final: {}/{} tests passing (baseline: {}/{})",
            final_passed, final_total, baseline_passed, baseline_total,
        );

        Ok(self.results.clone())
    }

    /// 获取所有结果
    pub fn results(&self) -> &[SelfEvolutionResult] {
        &self.results
    }
}
