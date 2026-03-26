use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::eval::metrics::IterationMetrics;

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

/// 多文件修改条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub file: String,
    pub old_lines: usize,
    pub new_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub iteration: u32,
    /// 主目标文件（向后兼容）
    pub file: String,
    /// Phase 3: 本次迭代修改的所有文件
    pub files_changed: Vec<FileChange>,
    pub hypothesis: String,
    pub outcome: ExperimentOutcome,
    pub tests_before: (u32, u32),  // (passed, total)
    pub tests_after: (u32, u32),
    pub reflection: String,
    pub timestamp: String,
    // Phase 1: 多维评估
    pub metrics_before: Option<IterationMetrics>,
    pub metrics_after: Option<IterationMetrics>,
    // Phase 1: 多维评估结果
    pub multi_eval: Option<crate::eval::metrics::MultiEvalResult>,
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
