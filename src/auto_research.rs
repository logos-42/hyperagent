//! 统一自动研究循环（Karpathy 风格 + 结构化自改进）
//!
//! 核心思想（来自 Andrej Karpathy）：
//!   研究不是系统工程，而是一个循环：
//!     提出假设 → 实验 → 观察 → 反思 → 再提出假设
//!
//! 与 hyperagent 进化引擎的区别：
//!   进化引擎：进化"解决任务的代码"
//!   自动研究：进化"系统自身的代码"
//!
//! 循环：
//!   1. 读取当前代码
//!   2. LLM 提出改进（假设）
//!   3. 应用改进
//!   4. cargo check + cargo test（实验）
//!   5. 观察结果
//!   6. LLM 反思 + 记录实验日志
//!   7. git commit + push（存档到 GitHub）
//!   8. 回到 1

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::llm::LLMClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// 项目根目录
    pub project_root: PathBuf,
    /// 目标源文件（相对 src/）
    pub target_files: Vec<String>,
    /// 最大迭代数
    pub max_iterations: u32,
    /// 自动 git push 到远程
    pub auto_push: bool,
    /// 实验日志目录
    pub experiment_log_dir: PathBuf,
    /// 安全模式：只看不改（编译通过也回滚）
    pub dry_run: bool,
    /// 严格模式：测试必须 100% 通过才接受（否则允许测试数不变时接受）
    pub strict: bool,
    /// 每次 push 间隔（0 = 每次成功都 push）
    pub push_interval: u32,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            target_files: vec![
                "runtime/thermodynamics.rs".to_string(),
                "runtime/loop_.rs".to_string(),
                "auto_research.rs".to_string(),
            ],
            max_iterations: 20,
            auto_push: true,
            experiment_log_dir: PathBuf::from(".hyperagent/experiments"),
            dry_run: false,
            strict: false,
            push_interval: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub iteration: u32,
    pub file: String,
    pub hypothesis: String,       // LLM 提出的改进假设
    pub outcome: ExperimentOutcome,
    pub tests_before: (u32, u32),  // (passed, total)
    pub tests_after: (u32, u32),
    pub reflection: String,        // LLM 反思
    pub timestamp: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExperimentOutcome {
    Improved,   // 测试通过数增加
    Neutral,    // 测试通过数不变
    Regressed,  // 测试通过数减少
    Failed,     // 编译失败
}

/// Karpathy 风格的自动研究引擎
pub struct AutoResearch<C: LLMClient> {
    client: C,
    config: ResearchConfig,
    experiments: Vec<Experiment>,
}

impl<C: LLMClient + Clone> AutoResearch<C> {
    pub fn new(client: C, config: ResearchConfig) -> Self {
        Self {
            client,
            config,
            experiments: Vec::new(),
        }
    }

    /// 读取源文件
    fn read_file(&self, rel: &str) -> Result<String> {
        let path = self.config.project_root.join("src").join(rel);
        std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read {}", path.display()))
    }

    /// 写入源文件
    fn write_file(&self, rel: &str, content: &str) -> Result<()> {
        let path = self.config.project_root.join("src").join(rel);
        std::fs::write(&path, content)
            .with_context(|| format!("Cannot write {}", path.display()))
    }

    /// git checkout 回滚
    fn git_revert(&self, file: &str) -> Result<()> {
        let path = format!("src/{}", file);
        std::process::Command::new("git")
            .args(&["checkout", "--", &path])
            .current_dir(&self.config.project_root)
            .output()?;
        Ok(())
    }

    /// git commit
    fn git_commit(&self, msg: &str) -> Result<()> {
        std::process::Command::new("git")
            .args(&["add", "-A"])
            .current_dir(&self.config.project_root)
            .output()?;

        let output = std::process::Command::new("git")
            .args(&["commit", "-m", msg])
            .current_dir(&self.config.project_root)
            .output()?;

        if !output.status.success() {
            // 可能没有变更
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("nothing to commit") {
                tracing::warn!("git commit: {}", stderr);
            }
        }
        Ok(())
    }

    /// git push
    fn git_push(&self) -> Result<()> {
        let output = std::process::Command::new("git")
            .args(&["push", "origin", "HEAD"])
            .current_dir(&self.config.project_root)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git push failed: {}", stderr);
        }
        Ok(())
    }

    /// cargo test，解析结果
    async fn run_tests(&self) -> Result<(u32, u32, String)> {
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
    async fn compile_check(&self) -> Result<(bool, String)> {
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

    /// 提取代码块
    fn extract_code(raw: &str) -> String {
        if raw.contains("```rust") {
            if let Some(start) = raw.find("```rust").map(|i| i + 7) {
                if let Some(end) = raw[start..].find("```") {
                    return raw[start..start + end].trim().to_string();
                }
            }
        }
        if raw.contains("```") {
            if let Some(start) = raw.find("```").map(|i| i + 3) {
                if let Some(end) = raw[start..].find("```") {
                    return raw[start..start + end].trim().to_string();
                }
            }
        }
        raw.trim().to_string()
    }

    /// 构建研究 prompt：Karpathy 风格 — 直接、极简
    fn build_research_prompt(&self, file: &str, code: &str, history: &[Experiment]) -> String {
        let recent_history = history.iter().rev().take(5).map(|e| {
            format!(
                "---\nExp {}: {}\nHypothesis: {}\nOutcome: {:?}\nReflection: {}",
                e.iteration, e.file, e.hypothesis, e.outcome, e.reflection
            )
        }).collect::<Vec<_>>().join("\n");

        format!(
            "You are an AI researcher improving your own codebase. This is a self-research loop.\n\n\
             === FILE: src/{file} ===\n\
             {code}\n\n\
             === PAST EXPERIMENTS ===\n\
             {history}\n\n\
             === YOUR TASK ===\n\
             1. Read the code above carefully.\n\
             2. Identify ONE specific, concrete improvement.\n\
             3. Output in this EXACT format:\n\n\
             HYPOTHESIS: <one sentence describing what you'll improve and why>\n\n\
             IMPROVED_CODE:\n\
             ```rust\n\
             <complete improved file>\n\
             ```\n\n\
             Rules:\n\
             - Do NOT change public API signatures\n\
             - Do NOT add new dependencies\n\
             - Focus on one improvement per iteration\n\
             - Output the COMPLETE file, not a diff\n",
            file = file,
            code = code,
            history = if recent_history.is_empty() { "(no experiments yet)".to_string() } else { recent_history },
        )
    }

    /// 构建反思 prompt
    fn build_reflection_prompt(
        &self,
        file: &str,
        hypothesis: &str,
        tests_before: (u32, u32),
        tests_after: (u32, u32),
        compile_ok: bool,
        test_output: &str,
    ) -> String {
        format!(
            "You are an AI researcher reflecting on an experiment.\n\n\
             File: src/{file}\n\
             Hypothesis: {hypothesis}\n\
             Before: {before_passed}/{before_total} tests passing\n\
             After: {after_passed}/{after_total} tests passing\n\
             Compiled: {compile_ok}\n\n\
             Test output (last 500 chars):\n\
             {output}\n\n\
             Write a 1-2 sentence reflection:\n\
             - What worked or didn't\n\
             - What to try next\n\
             - Keep it specific and actionable\n\n\
             REFLECTION:",
            file = file,
            hypothesis = hypothesis,
            before_passed = tests_before.0,
            before_total = tests_before.1,
            after_passed = tests_after.0,
            after_total = tests_after.1,
            compile_ok = compile_ok,
            output = test_output.chars().take(500).collect::<String>(),
        )
    }

    /// 解析 LLM 响应中的假设和代码
    fn parse_response(&self, response: &str) -> Option<(String, String)> {
        // 提取 HYPOTHESIS
        let hypothesis = if let Some(start) = response.find("HYPOTHESIS:") {
            let start = start + 11;
            let end = response[start..].find("\n\n").unwrap_or(response[start..].len());
            response[start..start + end].trim().to_string()
        } else {
            "No hypothesis stated".to_string()
        };

        // 提取代码块
        let code = Self::extract_code(response);
        if code.is_empty() {
            return None;
        }

        Some((hypothesis, code))
    }

    /// 写入实验日志到 markdown 文件
    fn append_experiment_log(&self, exp: &Experiment) -> Result<()> {
        std::fs::create_dir_all(&self.config.experiment_log_dir)?;
        let log_path = self.config.experiment_log_dir.join("research_log.md");

        let entry = format!(
            "## Experiment {} — {}\n\n\
             - **File**: `src/{}`\n\
             - **Hypothesis**: {}\n\
             - **Outcome**: {:?}\n\
             - **Tests**: {}/{} → {}/{}\n\
             - **Reflection**: {}\n\
             - **Time**: {}\n\n",
            exp.iteration, exp.file, exp.file,
            exp.hypothesis, exp.outcome,
            exp.tests_before.0, exp.tests_before.1,
            exp.tests_after.0, exp.tests_after.1,
            exp.reflection, exp.timestamp,
        );

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        f.write_all(entry.as_bytes())?;
        Ok(())
    }

    /// 单次研究迭代
    async fn run_iteration(&mut self, iteration: u32, file: &str) -> Result<Experiment> {
        tracing::info!("[Research {}] Improving src/{}", iteration, file);

        // 1. 读取当前代码
        let code = self.read_file(file)?;

        // 2. 基线测试
        let tests_before = {
            let (p, t, _) = self.run_tests().await?;
            (p, t)
        };
        tracing::info!("  Baseline: {}/{} tests", tests_before.0, tests_before.1);

        // 3. LLM 提出改进
        let prompt = self.build_research_prompt(file, &code, &self.experiments);
        let response = self.client.complete(&prompt).await?;
        let response_text = response.content;

        let (hypothesis, new_code) = match self.parse_response(&response_text) {
            Some(pair) => pair,
            None => {
                tracing::warn!("  Could not parse LLM response");
                return Ok(Experiment {
                    iteration,
                    file: file.to_string(),
                    hypothesis: "Parse failed".to_string(),
                    outcome: ExperimentOutcome::Failed,
                    tests_before,
                    tests_after: tests_before,
                    reflection: "LLM response could not be parsed".to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                });
            }
        };

        tracing::info!("  Hypothesis: {}", hypothesis);

        // 4. 应用改进
        self.write_file(file, &new_code)?;

        // 5. 编译检查
        let (compile_ok, compile_output) = self.compile_check().await?;
        if !compile_ok {
            tracing::warn!("  Compilation failed, reverting");
            self.git_revert(file)?;

            let reflection = {
                let prompt = self.build_reflection_prompt(
                    file, &hypothesis, tests_before, tests_before, false, &compile_output,
                );
                self.client.complete(&prompt).await.map(|r| r.content).unwrap_or_default()
            };

            return Ok(Experiment {
                iteration,
                file: file.to_string(),
                hypothesis,
                outcome: ExperimentOutcome::Failed,
                tests_before,
                tests_after: tests_before,
                reflection,
                timestamp: chrono::Utc::now().to_rfc3339(),
            });
        }

        // 6. 运行测试
        let (tests_after, total_after, test_output) = self.run_tests().await?;
        tracing::info!("  After: {}/{} tests", tests_after, total_after);

        // 7. 判定结果
        let accept = if self.config.strict {
            // 严格模式：全部通过才接受
            tests_after == total_after && total_after > 0 && tests_after >= tests_before.0
        } else if tests_after < tests_before.0 {
            // 宽松模式：测试数减少则回滚
            tracing::warn!("  Tests regressed, reverting");
            self.git_revert(file)?;
            false
        } else {
            true
        };

        // dry_run 模式：即使接受也回滚
        if accept && self.config.dry_run {
            tracing::info!("  [DRY RUN] Would accept, reverting");
            self.git_revert(file)?;
        }

        let outcome = if !accept {
            ExperimentOutcome::Regressed
        } else if tests_after > tests_before.0 {
            ExperimentOutcome::Improved
        } else {
            ExperimentOutcome::Neutral
        };

        // 8. LLM 反思
        let reflection = {
            let prompt = self.build_reflection_prompt(
                file, &hypothesis, tests_before, (tests_after, total_after), true, &test_output,
            );
            self.client.complete(&prompt).await.map(|r| r.content).unwrap_or_default()
        };

        tracing::info!("  Outcome: {:?}", outcome);

        let experiment = Experiment {
            iteration,
            file: file.to_string(),
            hypothesis,
            outcome,
            tests_before,
            tests_after: (tests_after, total_after),
            reflection,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // 9. 写入实验日志
        self.append_experiment_log(&experiment)?;

        // 10. Git commit + push
        if accept && !self.config.dry_run {
            let msg = format!(
                "research[{}]: {} — {} ({:?})",
                iteration, file,
                experiment.hypothesis.chars().take(60).collect::<String>(),
                outcome,
            );
            self.git_commit(&msg)?;
            tracing::info!("  Committed: {}", msg);

            if self.config.auto_push {
                let interval = self.config.push_interval.max(1);
                if interval == 1 || iteration % interval == 0 {
                    tracing::info!("  Pushing to GitHub...");
                    if let Err(e) = self.git_push() {
                        tracing::warn!("  Push failed: {}", e);
                    }
                }
            }
        }

        Ok(experiment)
    }

    /// 运行完整研究循环
    pub async fn run(&mut self) -> Result<Vec<Experiment>> {
        tracing::info!("╔══════════════════════════════════════════╗");
        tracing::info!("║   Unified Auto Research Loop           ║");
        tracing::info!("╚══════════════════════════════════════════╝");
        tracing::info!("Targets: {:?}", self.config.target_files);
        tracing::info!("Max iterations: {}", self.config.max_iterations);
        tracing::info!("Auto push: {}", self.config.auto_push);
        tracing::info!("Dry run: {}", self.config.dry_run);
        tracing::info!("Strict mode: {}", self.config.strict);

        if self.config.auto_push {
            tracing::warn!("AUTO PUSH enabled — changes will be pushed to GitHub!");
        }

        // 基线
        let (base_passed, base_total, _) = self.run_tests().await?;
        tracing::info!("Baseline: {}/{} tests passing", base_passed, base_total);

        for i in 0..self.config.max_iterations {
            let file = &self.config.target_files[(i as usize) % self.config.target_files.len()];
            let file = file.clone();

            let exp = self.run_iteration(i + 1, &file).await?;
            self.experiments.push(exp.clone());
        }

        // 最终 push
        if self.config.auto_push && !self.config.dry_run {
            tracing::info!("Final push to GitHub...");
            let _ = self.git_push();
        }

        // 统计
        let improved = self.experiments.iter().filter(|e| matches!(e.outcome, ExperimentOutcome::Improved)).count();
        let neutral = self.experiments.iter().filter(|e| matches!(e.outcome, ExperimentOutcome::Neutral)).count();
        let regressed = self.experiments.iter().filter(|e| matches!(e.outcome, ExperimentOutcome::Regressed)).count();
        let failed = self.experiments.iter().filter(|e| matches!(e.outcome, ExperimentOutcome::Failed)).count();

        tracing::info!("╔══════════════════════════════════════════╗");
        tracing::info!("║   Research Complete                     ║");
        tracing::info!("╠══════════════════════════════════════════╣");
        tracing::info!("║  Improved:  {:>3}                        ║", improved);
        tracing::info!("║  Neutral:   {:>3}                        ║", neutral);
        tracing::info!("║  Regressed: {:>3}                        ║", regressed);
        tracing::info!("║  Failed:    {:>3}                        ║", failed);
        tracing::info!("╚══════════════════════════════════════════╝");

        // 最终测试
        let (final_passed, final_total, _) = self.run_tests().await?;
        tracing::info!("Final: {}/{} (was {}/{})", final_passed, final_total, base_passed, base_total);

        Ok(self.experiments.clone())
    }

    pub fn experiments(&self) -> &[Experiment] {
        &self.experiments
    }
}
