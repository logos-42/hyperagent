//! 统一自动研究循环（Karpathy 风格 + 结构化自改进 + 元进化）
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
//!   3. LLM 同时生成测试（Phase 2）
//!   4. 应用改进
//!   5. 多维评估：cargo check + test + 代码指标（Phase 1）
//!   6. 观察结果
//!   7. LLM 反思 + 记录实验日志
//!   8. Git checkpoint + commit + push
//!   9. 回到 1

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::codebase::CodebaseContext;
use crate::eval::metrics::{IterationMetrics, MultiEvalResult};
use crate::llm::LLMClient;
use crate::web::{WebSearchTool, WebFetchTool, WebSearchResult, FetchOutput, build_web_context_prompt};

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
    /// 启用 web 搜索（研究前先搜索相关信息）
    pub enable_web: bool,
    /// Web 搜索结果数量
    pub web_search_limit: usize,
    /// 每次搜索后抓取的页面数（0 = 不抓取）
    pub web_fetch_limit: usize,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            target_files: vec![
                // 顶层核心
                "auto_research.rs".to_string(),
                "self_evolution.rs".to_string(),
                "codebase.rs".to_string(),
                "web.rs".to_string(),
                "tools.rs".to_string(),
                "lib.rs".to_string(),
                "main.rs".to_string(),
                // bin
                "bin/research.rs".to_string(),
                "bin/self_evolve.rs".to_string(),
                "bin/unified.rs".to_string(),
                // agent
                "agent/mod.rs".to_string(),
                "agent/executor.rs".to_string(),
                "agent/mutator.rs".to_string(),
                "agent/meta_mutator.rs".to_string(),
                "agent/population.rs".to_string(),
                // eval
                "eval/mod.rs".to_string(),
                "eval/evaluator.rs".to_string(),
                "eval/benchmark.rs".to_string(),
                // llm
                "llm/mod.rs".to_string(),
                "llm/client.rs".to_string(),
                "llm/prompts.rs".to_string(),
                // memory
                "memory/mod.rs".to_string(),
                "memory/archive.rs".to_string(),
                "memory/lineage.rs".to_string(),
                // runtime
                "runtime/mod.rs".to_string(),
                "runtime/thermodynamics.rs".to_string(),
                "runtime/loop_.rs".to_string(),
                "runtime/selection.rs".to_string(),
                "runtime/constraints.rs".to_string(),
                "runtime/population.rs".to_string(),
                "runtime/environment.rs".to_string(),
                "runtime/state.rs".to_string(),
                "runtime/local_runtime.rs".to_string(),
                "runtime/multi_agent_loop.rs".to_string(),
            ],
            max_iterations: 20,
            auto_push: true,
            experiment_log_dir: PathBuf::from(".hyperagent/experiments"),
            dry_run: false,
            strict: false,
            push_interval: 0,
            enable_web: true,
            web_search_limit: 5,
            web_fetch_limit: 2,
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
    // Phase 1: 多维评估
    pub metrics_before: Option<IterationMetrics>,
    pub metrics_after: Option<IterationMetrics>,
    pub multi_eval: Option<MultiEvalResult>,
    // Phase 2: 自动测试生成
    pub tests_generated: bool,
    pub new_tests_count: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExperimentOutcome {
    Improved,   // 测试通过数增加
    Neutral,    // 测试通过数不变
    Regressed,  // 测试通过数减少
    Failed,     // 编译失败
}

/// Karpathy 风格的自动研究引擎（含全局代码理解 + Web 搜索）
pub struct AutoResearch<C: LLMClient> {
    client: C,
    config: ResearchConfig,
    experiments: Vec<Experiment>,
    codebase: CodebaseContext,
    context_path: PathBuf,
    web_client: WebSearchTool,
}

impl<C: LLMClient + Clone> AutoResearch<C> {
    pub fn new(client: C, config: ResearchConfig) -> Self {
        let context_path = config.project_root.join(".hyperagent/data/codebase_context.json");

        // 加载已有上下文，或重新扫描
        let mut codebase = CodebaseContext::load(&context_path);
        if codebase.total_files == 0 {
            tracing::info!("Scanning codebase for the first time...");
            match CodebaseContext::scan(&config.project_root.to_string_lossy()) {
                Ok(ctx) => {
                    tracing::info!(
                        "Codebase: {} files, {} lines",
                        ctx.total_files, ctx.total_lines
                    );
                    codebase = ctx;
                }
                Err(e) => tracing::warn!("Codebase scan failed: {}", e),
            }
        } else {
            tracing::info!(
                "Loaded codebase context: {} files, {} lines, {} iterations",
                codebase.total_files, codebase.total_lines, codebase.total_iterations
            );
        }

        Self {
            client,
            config,
            experiments: Vec::new(),
            codebase,
            context_path,
            web_client: WebSearchTool::new(),
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

    /// Phase 5: Git checkpoint — 修改核心文件前打标签
    fn git_checkpoint(&self, iteration: u32) -> Result<()> {
        let tag = format!("checkpoint-{}", iteration);
        let _ = std::process::Command::new("git")
            .args(&["tag", "-f", &tag])
            .current_dir(&self.config.project_root)
            .output();
        tracing::info!("  Checkpoint: {}", tag);
        Ok(())
    }

    /// Phase 5: 回滚到上一个 checkpoint
    fn git_rollback(&self, iteration: u32) -> Result<()> {
        let tag = format!("checkpoint-{}", iteration);
        let output = std::process::Command::new("git")
            .args(&["reset", "--hard", &tag])
            .current_dir(&self.config.project_root)
            .output()?;
        if output.status.success() {
            tracing::warn!("  Rolled back to {}", tag);
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

    /// 使用 LLM 生成搜索查询，执行 Web 搜索，并返回上下文字符串
    async fn gather_web_context(&self, file: &str, code: &str) -> Option<String> {
        if !self.config.enable_web {
            return None;
        }

        // 1. LLM 生成搜索查询
        let search_prompt = format!(
            "You are researching improvements for a Rust file. Given the code below, \
             suggest 2-3 web search queries to find best practices, idioms, or solutions.\n\
             Be specific. Output ONLY the queries, one per line. No numbering.\n\n\
             File: src/{file}\n\
             Code (first 2000 chars):\n{code_snippet}",
            file = file,
            code_snippet = code.chars().take(2000).collect::<String>(),
        );

        let search_response = match self.client.complete(&search_prompt).await {
            Ok(r) => r.content,
            Err(e) => {
                tracing::warn!("  Web: failed to generate search queries: {}", e);
                return None;
            }
        };

        let queries: Vec<String> = search_response
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && l.len() < 200)
            .take(3)
            .collect();

        if queries.is_empty() {
            return None;
        }

        tracing::info!("  Web: searching for: {:?}", queries);

        // 2. 并发搜索
        let mut all_results: Vec<WebSearchResult> = Vec::new();
        for query in &queries {
            match self.web_client.search(query, self.config.web_search_limit).await {
                Ok(results) => {
                    tracing::info!("  Web: {} results for '{}'", results.len(), query);
                    all_results.extend(results);
                }
                Err(e) => tracing::warn!("  Web: search failed for '{}': {}", query, e),
            }
        }

        if all_results.is_empty() {
            return None;
        }

        // 3. 抓取前 N 个页面
        let fetch_tool = WebFetchTool::new();
        let fetch_urls: Vec<String> = all_results
            .iter()
            .take(self.config.web_fetch_limit)
            .map(|r| r.url.clone())
            .collect();

        if !fetch_urls.is_empty() {
            tracing::info!("  Web: fetching {} pages...", fetch_urls.len());
            let mut pages: Vec<FetchOutput> = Vec::new();
            for url in &fetch_urls {
                match fetch_tool.fetch(url).await {
                    Ok(page) => {
                        tracing::info!("  Web: fetched {} ({} chars)", url, page.text_length);
                        pages.push(page);
                    }
                    Err(e) => tracing::warn!("  Web: failed to fetch {}: {}", url, e),
                }
            }

            let context = build_web_context_prompt(&all_results, &pages);
            tracing::info!("  Web: gathered {} chars of context", context.len());
            return Some(context);
        }

        Some(build_web_context_prompt(&all_results, &[]))
    }

    /// 构建研究 prompt：注入全局架构上下文 + Web 搜索上下文 + 相关文件（Phase 4）
    fn build_research_prompt(&self, file: &str, code: &str, history: &[Experiment], web_context: Option<&str>) -> String {
        let recent_history = history.iter().rev().take(5).map(|e| {
            format!(
                "---\nExp {}: {}\nHypothesis: {}\nOutcome: {:?}\nReflection: {}\nTests: {}/{} → {}/{}",
                e.iteration, e.file, e.hypothesis, e.outcome, e.reflection,
                e.tests_before.0, e.tests_before.1, e.tests_after.0, e.tests_after.1,
            )
        }).collect::<Vec<_>>().join("\n");

        let codebase_context = self.codebase.build_context_prompt(file);

        // Phase 4: 加载相关文件的签名（被依赖和依赖的文件）
        let related_context = self.build_related_files_context(file);

        // Phase 2: 测试生成指令
        let test_gen_instruction = "\n\
             6. Write NEW TESTS for your improvement inside `#[cfg(test)] mod tests { ... }`.\n\
                - Test the specific behavior you improved\n\
                - Use descriptive test names\n\
                - If the file already has tests, ADD new ones (don't remove existing)\n";

        format!(
            "You are an AI researcher improving your own codebase. This is a self-research loop.\n\n\
             {codebase_context}\n\n\
             {related_context}\n\
             {web_section}\n\
             === CURRENT FILE: src/{file} ===\n\
             {code}\n\n\
             === PAST EXPERIMENTS ===\n\
             {history}\n\n\
             === YOUR TASK ===\n\
             1. Understand the architecture above — how this file fits into the system.\n\
             2. Read the code carefully, including the related files context.\n\
             3. Identify ONE specific, concrete improvement.\n\
             4. Consider cross-file dependencies — don't break other modules.\n\
             5. Output in this EXACT format:\n\n\
             HYPOTHESIS: <one sentence describing what you'll improve and why>\n\n\
             IMPROVED_CODE:\n\
             ```rust\n\
             <complete improved file, including any new #[cfg(test)] tests>\n\
             ```\n\n\
             {test_gen_section}\n\
             Rules:\n\
             - Do NOT change public API signatures (function names, trait methods, struct fields)\n\
             - Do NOT add new external dependencies\n\
             - Do NOT break imports used by other files\n\
             - Focus on one improvement per iteration\n\
             - Output the COMPLETE file, not a diff\n\
             - Include any new tests in the #[cfg(test)] module",
            file = file,
            code = code,
            history = if recent_history.is_empty() { "(no experiments yet)".to_string() } else { recent_history },
            web_section = web_context.map(|ctx| format!("{}\n", ctx)).unwrap_or_default(),
            related_context = related_context,
            test_gen_section = test_gen_instruction,
        )
    }

    /// Phase 4: 构建相关文件上下文 — 被依赖和依赖的文件签名
    fn build_related_files_context(&self, target_file: &str) -> String {
        let mut context = String::new();

        if let Some(target_summary) = self.codebase.files.get(target_file) {
            // 找到依赖目标文件的文件（使用目标文件中导出的东西）
            let dependents: Vec<String> = self
                .codebase
                .files
                .iter()
                .filter(|(path, summary)| {
                    *path != target_file
                        && summary.uses.iter().any(|u| {
                            let module_hint = target_file
                                .strip_suffix(".rs")
                                .unwrap_or(target_file)
                                .replace('/', "::");
                            u.contains(&module_hint)
                                || u.contains(
                                    &target_file.replace('/', "::").replace(".rs", "")
                                        .replace("mod.rs", "")
                                )
                        })
                })
                .map(|(path, _)| path.clone())
                .collect();

            // 找到目标文件依赖的文件（目标文件 use crate::xxx 的文件）
            let dependencies: Vec<String> = target_summary
                .uses
                .iter()
                .filter(|u| u.contains("crate::"))
                .filter_map(|u| {
                    // 从 "crate::module::something" 推导文件路径
                    let binding = u.replace("crate::", "");
                    let module = binding.split("::").next()?;
                    // 查找匹配的文件
                    let candidates: Vec<String> = self
                        .codebase
                        .files
                        .keys()
                        .filter(|p| {
                            p.replace('/', "::")
                                .replace(".rs", "")
                                .replace("mod.rs", "")
                                .contains(module)
                        })
                        .cloned()
                        .collect();
                    candidates.first().cloned()
                })
                .collect();

            let related_files: Vec<String> = dependents
                .into_iter()
                .chain(dependencies.into_iter())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            if !related_files.is_empty() {
                context.push_str("=== RELATED FILES ===\n");
                for rel_file in related_files {
                    if let Some(summary) = self.codebase.files.get(&rel_file) {
                        let types: Vec<String> = summary
                            .structs
                            .iter()
                            .chain(summary.enums.iter())
                            .chain(summary.traits.iter())
                            .take(8)
                            .cloned()
                            .collect();
                        let fns_str: Vec<String> = summary
                            .functions
                            .iter()
                            .take(10)
                            .cloned()
                            .collect();

                        context.push_str(&format!(
                            "\n--- {} ({} lines) ---\n",
                            rel_file, summary.lines
                        ));
                        if !types.is_empty() {
                            context.push_str(&format!("Types: {}\n", types.join(", ")));
                        }
                        if !fns_str.is_empty() {
                            context.push_str(&format!("Functions: {}\n", fns_str.join(", ")));
                        }
                    }
                }
                context.push('\n');
            }
        }

        context
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

    /// 写入实验日志到 markdown 文件（含多维指标）
    fn append_experiment_log(&self, exp: &Experiment) -> Result<()> {
        std::fs::create_dir_all(&self.config.experiment_log_dir)?;
        let log_path = self.config.experiment_log_dir.join("research_log.md");

        let metrics_section = match (&exp.metrics_before, &exp.metrics_after, &exp.multi_eval) {
            (Some(before), Some(after), Some(eval)) => {
                format!(
                    "\n  - **Metrics**: score={:.2}, lines {:+}, warnings {:+}, complexity {:.0}→{:.0}, binary {:+}KB\n",
                    eval.score,
                    after.lines_delta,
                    after.warnings as i32 - before.warnings as i32,
                    before.complexity, after.complexity,
                    (after.binary_size as i64 - before.binary_size as i64) / 1024,
                )
            }
            _ => String::new(),
        };

        let test_section = if exp.tests_generated {
            format!("  - **New Tests**: {} generated\n", exp.new_tests_count)
        } else {
            String::new()
        };

        let entry = format!(
            "## Experiment {} — {}\n\n\
             - **File**: `src/{}`\n\
             - **Hypothesis**: {}\n\
             - **Outcome**: {:?}\n\
             - **Tests**: {}/{} → {}/{}\n\
             {}{}\
             - **Reflection**: {}\n\
             - **Time**: {}\n\n",
            exp.iteration, exp.file, exp.file,
            exp.hypothesis, exp.outcome,
            exp.tests_before.0, exp.tests_before.1,
            exp.tests_after.0, exp.tests_after.1,
            metrics_section,
            test_section,
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

    /// 单次研究迭代（含 Phase 1 多维评估 + Phase 2 测试生成 + Phase 5 安全机制）
    async fn run_iteration(&mut self, iteration: u32, file: &str) -> Result<Experiment> {
        tracing::info!("[Research {}] Improving src/{}", iteration, file);

        // Phase 5: 核心文件修改前打 checkpoint
        let is_core_file = file == "auto_research.rs"
            || file == "self_evolution.rs"
            || file == "codebase.rs"
            || file == "lib.rs";
        if is_core_file {
            self.git_checkpoint(iteration)?;
        }

        // 1. 读取当前代码
        let code = self.read_file(file)?;

        // 2. 基线测试 + 基线指标
        let tests_before = {
            let (p, t, _) = self.run_tests().await?;
            (p, t)
        };
        tracing::info!("  Baseline: {}/{} tests", tests_before.0, tests_before.1);

        // Phase 1: 收集基线指标
        let (before_lines, before_complexity, before_nesting) = IterationMetrics::from_code(&code);
        let before_binary = IterationMetrics::get_binary_size(&self.config.project_root, "research")
            .unwrap_or(0);
        let before_warnings = self.count_warnings().await;
        let metrics_before = IterationMetrics {
            tests_passed: tests_before.0,
            tests_total: tests_before.1,
            code_lines: before_lines,
            code_lines_before: before_lines,
            lines_delta: 0,
            warnings: before_warnings,
            complexity: before_complexity,
            max_nesting: before_nesting,
            binary_size: before_binary,
            binary_size_before: before_binary,
            binary_delta: 0,
        };

        // 3. Web 搜索（获取外部知识）
        let web_context = self.gather_web_context(file, &code).await;

        // 4. LLM 提出改进（含 Phase 2 测试生成指令 + Phase 4 相关文件上下文）
        let prompt = self.build_research_prompt(file, &code, &self.experiments, web_context.as_deref());
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
                    metrics_before: Some(metrics_before),
                    metrics_after: None,
                    multi_eval: None,
                    tests_generated: false,
                    new_tests_count: 0,
                });
            }
        };

        tracing::info!("  Hypothesis: {}", hypothesis);

        // Phase 2: 检测是否生成了新测试
        let (tests_generated, new_tests_count) = self.detect_new_tests(&code, &new_code);

        // 5. 应用改进
        self.write_file(file, &new_code)?;

        // 6. 编译检查
        let (compile_ok, compile_output) = self.compile_check().await?;
        if !compile_ok {
            tracing::warn!("  Compilation failed, reverting");
            self.git_revert(file)?;

            // Phase 5: 核心文件编译失败回滚到 checkpoint
            if is_core_file {
                self.git_rollback(iteration)?;
            }

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
                metrics_before: Some(metrics_before),
                metrics_after: None,
                multi_eval: None,
                tests_generated: false,
                new_tests_count: 0,
            });
        }

        // 7. 运行测试
        let (tests_after, total_after, test_output) = self.run_tests().await?;
        tracing::info!("  After: {}/{} tests", tests_after, total_after);

        // Phase 1: 收集修改后指标
        let (after_lines, after_complexity, after_nesting) = IterationMetrics::from_code(&new_code);
        let after_binary = IterationMetrics::get_binary_size(&self.config.project_root, "research")
            .unwrap_or(before_binary);
        let after_warnings = self.count_warnings().await;
        let metrics_after = IterationMetrics {
            tests_passed: tests_after,
            tests_total: total_after,
            code_lines: after_lines,
            code_lines_before: before_lines,
            lines_delta: after_lines as i32 - before_lines as i32,
            warnings: after_warnings,
            complexity: after_complexity,
            max_nesting: after_nesting,
            binary_size: after_binary,
            binary_size_before: before_binary,
            binary_delta: after_binary as i64 - before_binary as i64,
        };

        // Phase 1: 多维评估
        let multi_eval = MultiEvalResult::compare(&metrics_before, &metrics_after);
        tracing::info!("  Multi-eval: {}", multi_eval.summary);

        // 8. 判定结果（使用多维评估）
        let accept = multi_eval.should_accept(self.config.strict);

        if !accept {
            tracing::warn!("  Regressed (multi-eval), reverting");
            self.git_revert(file)?;
        }

        // dry_run 模式：即使接受也回滚
        if accept && self.config.dry_run {
            tracing::info!("  [DRY RUN] Would accept, reverting");
            self.git_revert(file)?;
        }

        let outcome = if !accept {
            ExperimentOutcome::Regressed
        } else if multi_eval.score > 0.2 {
            ExperimentOutcome::Improved
        } else {
            ExperimentOutcome::Neutral
        };

        // 9. LLM 反思（注入多维评估结果）
        let reflection = {
            let prompt = format!(
                "You are an AI researcher reflecting on an experiment.\n\n\
                 File: src/{file}\n\
                 Hypothesis: {hypothesis}\n\
                 Before: {before_passed}/{before_total} tests\n\
                 After: {after_passed}/{after_total} tests\n\
                 {metrics_section}\n\
                 New tests generated: {tests_gen} ({new_count})\n\n\
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
                after_passed = tests_after,
                after_total = total_after,
                metrics_section = format!(
                    "Multi-eval score: {:.2}\n{}",
                    multi_eval.score, multi_eval.summary
                ),
                tests_gen = tests_generated,
                new_count = new_tests_count,
                output = test_output.chars().take(500).collect::<String>(),
            );
            self.client.complete(&prompt).await.map(|r| r.content).unwrap_or_default()
        };

        tracing::info!("  Outcome: {:?}", outcome);
        if tests_generated {
            tracing::info!("  Tests generated: {} new", new_tests_count);
        }

        let experiment = Experiment {
            iteration,
            file: file.to_string(),
            hypothesis: hypothesis.clone(),
            outcome,
            tests_before,
            tests_after: (tests_after, total_after),
            reflection,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics_before: Some(metrics_before),
            metrics_after: Some(metrics_after),
            multi_eval: Some(multi_eval),
            tests_generated,
            new_tests_count,
        };

        // 10. 记录改进到全局上下文
        self.codebase.record_improvement(
            file,
            &hypothesis,
            &format!("{:?} (score={:.2})", outcome,
                experiment.multi_eval.as_ref().map(|m| m.score).unwrap_or(0.0)),
        );

        // 11. 写入实验日志（含多维指标）
        self.append_experiment_log(&experiment)?;

        // 12. Git commit + push
        if accept && !self.config.dry_run {
            let extra = if tests_generated {
                format!(" [+{} tests]", new_tests_count)
            } else {
                String::new()
            };
            let msg = format!(
                "research[{}]: {} — {} ({:?}){}",
                iteration, file,
                experiment.hypothesis.chars().take(60).collect::<String>(),
                outcome, extra,
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

    /// Phase 2: 检测新代码中是否有新增测试
    fn detect_new_tests(&self, old_code: &str, new_code: &str) -> (bool, u32) {
        let old_test_count = Self::count_test_fns(old_code);
        let new_test_count = Self::count_test_fns(new_code);
        let generated = new_test_count > old_test_count;
        let added = (new_test_count - old_test_count) as u32;
        (generated, added)
    }

    /// 计算代码中 #[test] 函数的数量
    fn count_test_fns(code: &str) -> usize {
        code.matches("#[test]").count()
    }

    /// 计算编译警告数
    async fn count_warnings(&self) -> u32 {
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
        tracing::info!("Web search: {}", self.config.enable_web);

        if self.config.auto_push {
            tracing::warn!("AUTO PUSH enabled — changes will be pushed to GitHub!");
        }

        // 基线
        let (base_passed, base_total, _) = self.run_tests().await?;
        tracing::info!("Baseline: {}/{} tests passing", base_passed, base_total);

        for i in 0..self.config.max_iterations {
            let file = &self.config.target_files[(i as usize) % self.config.target_files.len()];
            let file = file.clone();

            let exp = match self.run_iteration(i + 1, &file).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!("[Research {}] FAILED on {}: {} — skipping", i + 1, file, e);
                    // Phase 5: 核心文件失败时尝试回滚到 checkpoint
                    let is_core_file = file == "auto_research.rs"
                        || file == "self_evolution.rs"
                        || file == "codebase.rs"
                        || file == "lib.rs";
                    if is_core_file {
                        let _ = self.git_rollback(i + 1);
                    }
                    continue;
                }
            };
            self.experiments.push(exp.clone());

            // 每 N 轮刷新代码库上下文（源文件可能已改变）
            if (i + 1) % 5 == 0 {
                self.codebase.refresh(&self.config.project_root.to_string_lossy());
            }
        }

        // 保存全局上下文到磁盘
        if let Err(e) = self.codebase.save(&self.context_path) {
            tracing::warn!("Failed to save codebase context: {}", e);
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
        let tests_gen_count = self.experiments.iter().filter(|e| e.tests_generated).count();
        let total_new_tests: u32 = self.experiments.iter().map(|e| e.new_tests_count).sum();

        tracing::info!("╔══════════════════════════════════════════╗");
        tracing::info!("║   Research Complete                     ║");
        tracing::info!("╠══════════════════════════════════════════╣");
        tracing::info!("║  Improved:  {:>3}                        ║", improved);
        tracing::info!("║  Neutral:   {:>3}                        ║", neutral);
        tracing::info!("║  Regressed: {:>3}                        ║", regressed);
        tracing::info!("║  Failed:    {:>3}                        ║", failed);
        tracing::info!("╠══════════════════════════════════════════╣");
        tracing::info!("║  Tests Gen: {:>3} rounds (+{} tests)     ║", tests_gen_count, total_new_tests);
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
