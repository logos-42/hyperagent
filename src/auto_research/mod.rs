//! 统一自动研究循环（Karpathy 风格 + 结构化自改进 + 元进化）
//!
//! 核心思想（来自 Andrej Karpathy）：
//!   研究不是系统工程，而是一个循环：
//!     提出假设 → 实验 → 观察 → 反思 → 再提出假设
//!
//! 与 hyperagent 进化引擎的区别：
//!   进化引擎：进化"解决任务的代码"
//!   自动研究：进化"系统自身的代码"

mod git;
mod log;
mod parsers;
mod prompts;
mod testing;
mod types;
mod web_search;

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::codebase::CodebaseContext;
use crate::eval::metrics::{IterationMetrics, MultiEvalResult};
use crate::llm::LLMClient;
use crate::strategy::{StrategyConfig, StrategyEvolver};
use crate::web::WebSearchTool;

pub use types::{Experiment, ExperimentOutcome, FileChange, ResearchConfig};

/// Karpathy 风格的自动研究引擎（含全局代码理解 + Web 搜索）
pub struct AutoResearch<C: LLMClient> {
    pub(crate) client: C,
    pub(crate) config: ResearchConfig,
    pub(crate) experiments: Vec<Experiment>,
    pub(crate) codebase: CodebaseContext,
    pub(crate) context_path: PathBuf,
    pub(crate) web_client: WebSearchTool,
    /// Phase 6: 可进化的策略参数
    pub(crate) strategy: Option<StrategyConfig>,
    pub(crate) strategy_path: PathBuf,
}

impl<C: LLMClient + Clone> AutoResearch<C> {
    pub fn new(client: C, config: ResearchConfig) -> Self {
        let context_path = config.project_root.join(".hyperagent/data/codebase_context.json");
        let strategy_path = config.project_root.join(".hyperagent/strategy.json");

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

        // Phase 6: 加载策略配置
        let strategy = Some(StrategyConfig::load(&strategy_path));
        tracing::info!(
            "Strategy v{}: temp={:.2}, target={:?}",
            strategy.as_ref().unwrap().version,
            strategy.as_ref().unwrap().research_temperature,
            strategy.as_ref().unwrap().target_selection,
        );

        Self {
            client,
            config,
            experiments: Vec::new(),
            codebase,
            context_path,
            web_client: WebSearchTool::new(),
            strategy,
            strategy_path,
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

        // Phase 3: 使用多文件解析器
        let (hypothesis, file_changes) = match self.parse_response_multi(&response_text) {
            Some(pair) => pair,
            None => {
                tracing::warn!("  Could not parse LLM response");
                return Ok(Experiment {
                    iteration,
                    file: file.to_string(),
                    files_changed: vec![],
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

        // Phase 3: 解析多文件变更 — 第一个条目的 file_path 为空则用主文件
        let mut resolved_changes: Vec<(String, String)> = Vec::new();
        for (file_path, code_content) in &file_changes {
            let resolved = if file_path.is_empty() {
                file.to_string()
            } else {
                file_path.trim().to_string()
            };
            resolved_changes.push((resolved, code_content.clone()));
        }

        if resolved_changes.is_empty() {
            tracing::warn!("  No file changes parsed");
            return Ok(Experiment {
                iteration,
                file: file.to_string(),
                files_changed: vec![],
                hypothesis: "No changes".to_string(),
                outcome: ExperimentOutcome::Failed,
                tests_before,
                tests_after: tests_before,
                reflection: "No file changes were parsed from the LLM response".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                metrics_before: Some(metrics_before),
                metrics_after: None,
                multi_eval: None,
                tests_generated: false,
                new_tests_count: 0,
            });
        }

        // 记录修改了哪些文件
        let mut files_changed: Vec<FileChange> = Vec::new();
        for (ref target, _) in &resolved_changes {
            let old_code = self.read_file(target).unwrap_or_default();
            files_changed.push(FileChange {
                file: target.clone(),
                old_lines: old_code.lines().count(),
                new_lines: 0, // 写入后更新
            });
        }
        if resolved_changes.len() > 1 {
            tracing::info!("  Phase 3: {} files to modify", resolved_changes.len());
        }

        // Phase 2: 检测是否生成了新测试（主文件）
        let primary_code = resolved_changes[0].1.clone();
        let (tests_generated, new_tests_count) = self.detect_new_tests(&code, &primary_code);

        // Phase 3: 原子写入所有文件
        // 如果任何文件写入失败，全部回滚
        let mut write_ok = true;
        for (target, content) in &resolved_changes {
            if let Err(e) = self.write_file(target, content) {
                tracing::warn!("  Failed to write {}: {}", target, e);
                write_ok = false;
                break;
            }
        }

        if !write_ok {
            // 回滚所有已写入的文件
            for (target, _) in &resolved_changes {
                let _ = self.git_revert(target);
            }
            return Ok(Experiment {
                iteration,
                file: file.to_string(),
                files_changed: vec![],
                hypothesis,
                outcome: ExperimentOutcome::Failed,
                tests_before,
                tests_after: tests_before,
                reflection: "File write failed".to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                metrics_before: Some(metrics_before),
                metrics_after: None,
                multi_eval: None,
                tests_generated: false,
                new_tests_count: 0,
            });
        }

        // Phase 3: 更新 files_changed 的 new_lines
        for fc in &mut files_changed {
            let new_code = self.read_file(&fc.file).unwrap_or_default();
            fc.new_lines = new_code.lines().count();
        }

        // 6. 编译检查
        let (compile_ok, compile_output) = self.compile_check().await?;
        if !compile_ok {
            tracing::warn!("  Compilation failed, reverting all changed files");
            // Phase 3: 回滚所有修改的文件
            for (target, _) in &resolved_changes {
                let _ = self.git_revert(target);
            }

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
                files_changed,
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

        // Phase 1: 收集修改后指标（主文件）
        let (after_lines, after_complexity, after_nesting) = IterationMetrics::from_code(&primary_code);
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

        // Phase 1 + Phase 6: 多维评估（使用策略参数权重）
        let multi_eval = MultiEvalResult::compare_weighted(
            &metrics_before, &metrics_after,
            self.strategy.as_ref(),
        );
        tracing::info!("  Multi-eval: {}", multi_eval.summary);

        // Phase 6: 使用策略的 improved_score_threshold
        let improved_threshold = self.strategy.as_ref()
            .map(|s| s.improved_score_threshold)
            .unwrap_or(0.2);

        // 8. 判定结果
        let accept = multi_eval.should_accept(self.config.strict);

        if !accept {
            tracing::warn!("  Regressed (multi-eval), reverting all changed files");
            for (target, _) in &resolved_changes {
                let _ = self.git_revert(target);
            }
        }

        // dry_run 模式：即使接受也回滚
        if accept && self.config.dry_run {
            tracing::info!("  [DRY RUN] Would accept, reverting");
            for (target, _) in &resolved_changes {
                let _ = self.git_revert(target);
            }
        }

        let outcome = if !accept {
            ExperimentOutcome::Regressed
        } else if multi_eval.score > improved_threshold {
            ExperimentOutcome::Improved
        } else {
            ExperimentOutcome::Neutral
        };

        // 9. LLM 反思
        let reflection = {
            let prompt = format!(
                "You are an AI researcher reflecting on an experiment.\n\n\
                 File: src/{file}\n\
                 Hypothesis: {hypothesis}\n\
                 Before: {before_passed}/{before_total} tests\n\
                 After: {after_passed}/{after_total} tests\n\
                 {metrics_section}\n\
                 New tests generated: {tests_gen} ({new_count})\n\
                 Files changed: {num_files}\n\n\
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
                num_files = resolved_changes.len(),
                output = test_output.chars().take(500).collect::<String>(),
            );
            self.client.complete(&prompt).await.map(|r| r.content).unwrap_or_default()
        };

        tracing::info!("  Outcome: {:?}", outcome);
        if tests_generated {
            tracing::info!("  Tests generated: {} new", new_tests_count);
        }
        if resolved_changes.len() > 1 {
            tracing::info!("  Files changed: {:?}", resolved_changes.iter().map(|(f, _)| f.as_str()).collect::<Vec<_>>());
        }

        let experiment = Experiment {
            iteration,
            file: file.to_string(),
            files_changed,
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
            // Phase 6: 自适应目标选择
            let file = StrategyEvolver::select_next_target(
                self.strategy.as_ref().unwrap_or(&StrategyConfig::default()),
                &self.config.target_files,
                &self.experiments,
                i + 1,
            );

            let exp = match self.run_iteration(i + 1, &file).await {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!("[Research {}] FAILED on {}: {} — skipping", i + 1, file, e);
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

            // 每 N 轮刷新代码库上下文
            if (i + 1) % 5 == 0 {
                self.codebase.refresh(&self.config.project_root.to_string_lossy());
            }

            // Phase 6: 定期进化策略参数
            if let Some(ref current_strategy) = self.strategy {
                if (i + 1) % current_strategy.evolution_interval == 0 {
                    tracing::info!("[Strategy] Evolving strategy based on {} experiments...", self.experiments.len());
                    let new_strategy = StrategyEvolver::evolve(current_strategy, &self.experiments);
                    if let Err(e) = new_strategy.save(&self.strategy_path) {
                        tracing::warn!("Failed to save strategy: {}", e);
                    }
                    self.strategy = Some(new_strategy);
                }
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
