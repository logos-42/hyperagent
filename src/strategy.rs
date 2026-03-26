//! 元进化策略模块 (Phase 6)
//!
//! 核心思想：系统自身的策略参数（评估权重、温度、目标选择策略等）
//! 不是一成不变的，而是可以根据历史实验结果自我调整和进化。
//!
//! 策略参数会被持久化到 `.hyperagent/strategy.json`，
//! 每隔 N 轮迭代后由 StrategyEvolver 分析历史并微调。

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 可进化的策略参数（所有参数都有合理的默认值和边界）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// 版本号（每次进化递增）
    pub version: u32,

    // === 评估权重 ===
    /// 测试通过率权重（最重要的维度）
    pub weight_tests: f64,
    /// 代码行数精简权重
    pub weight_lines: f64,
    /// 编译警告减少权重
    pub weight_warnings: f64,
    /// 复杂度降低权重
    pub weight_complexity: f64,
    /// 二进制体积权重
    pub weight_binary_size: f64,

    // === 维度判定阈值 ===
    /// 行数减少多少算 Improved（行数）
    pub lines_improved_threshold: i32,
    /// 行数增加多少算 Regressed（行数）
    pub lines_regressed_threshold: i32,
    /// 复杂度降低多少算 Improved
    pub complexity_improved_threshold: f64,
    /// 复杂度增加多少算 Regressed
    pub complexity_regressed_threshold: f64,
    /// 二进制缩小百分比算 Improved（如 -0.02 = -2%）
    pub binary_improved_pct: f64,
    /// 二进制膨胀百分比算 Regressed（如 0.05 = +5%）
    pub binary_regressed_pct: f64,

    // === 改进判定 ===
    /// multi-eval score > 此阈值才算 Improved（否则 Neutral）
    pub improved_score_threshold: f64,

    // === LLM 参数 ===
    /// 研究迭代时的 temperature（创造力 vs 稳定性）
    pub research_temperature: f32,
    /// 反思时的 temperature（偏低 = 更保守的反思）
    pub reflection_temperature: f32,

    // === 目标选择策略 ===
    /// 目标文件选择模式：round_robin（轮询）或 adaptive（自适应：优先选历史改进率低的文件）
    pub target_selection: TargetSelectionMode,
    /// 自适应模式时考虑最近 N 次实验
    pub adaptive_window: usize,

    // === 进化调度 ===
    /// 每隔多少轮迭代后触发一次策略进化
    pub evolution_interval: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TargetSelectionMode {
    RoundRobin,
    Adaptive,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            version: 1,
            // 权重（总和归一化在运行时处理）
            weight_tests: 0.4,
            weight_lines: 0.1,
            weight_warnings: 0.1,
            weight_complexity: 0.1,
            weight_binary_size: 0.05,
            // 阈值
            lines_improved_threshold: -5,
            lines_regressed_threshold: 20,
            complexity_improved_threshold: 5.0,
            complexity_regressed_threshold: 10.0,
            binary_improved_pct: -0.02,
            binary_regressed_pct: 0.05,
            // 判定
            improved_score_threshold: 0.2,
            // LLM
            research_temperature: 0.7,
            reflection_temperature: 0.3,
            // 目标选择
            target_selection: TargetSelectionMode::RoundRobin,
            adaptive_window: 10,
            // 进化
            evolution_interval: 10,
        }
    }
}

impl StrategyConfig {
    /// 从磁盘加载策略（不存在则用默认值）
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
                tracing::warn!("Failed to parse strategy config: {}, using defaults", e);
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// 持久化到磁盘
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// 将权重归一化为元组 (weight, -penalty) 供 MultiEvalResult 使用
    pub fn weight_tuple(&self, dim: &str) -> (f64, f64) {
        match dim {
            "tests" => (self.weight_tests, -self.weight_tests * 1.5),
            "lines" => (self.weight_lines, -self.weight_lines * 0.5),
            "warnings" => (self.weight_warnings, -self.weight_warnings),
            "complexity" => (self.weight_complexity, -self.weight_complexity * 0.5),
            "binary_size" => (self.weight_binary_size, -self.weight_binary_size),
            _ => (0.1, -0.1),
        }
    }
}

/// 策略进化器 — 基于历史实验结果微调策略参数
pub struct StrategyEvolver;

impl StrategyEvolver {
    /// 分析历史实验并进化策略参数
    ///
    /// 进化逻辑：
    /// 1. 统计最近 N 轮的改进率、失败率、回滚率
    /// 2. 如果改进率太低 → 降低 temperature（更保守）
    /// 3. 如果失败率太高 → 降低 temperature，提高 weight_tests
    /// 4. 如果回滚率太高但测试没退化 → 放宽 lines/complexity 阈值
    /// 5. 自适应目标选择：如果某些文件反复改进/退化，调整选择策略
    pub fn evolve(
        strategy: &StrategyConfig,
        experiments: &[crate::auto_research::Experiment],
    ) -> StrategyConfig {
        let mut new_strategy = strategy.clone();
        new_strategy.version += 1;

        if experiments.is_empty() {
            return new_strategy;
        }

        // 取最近 experiments
        let window = experiments.len().min(strategy.adaptive_window);
        let recent = &experiments[experiments.len() - window..];

        let total = recent.len() as f64;
        let improved = recent.iter().filter(|e| matches!(e.outcome, crate::auto_research::ExperimentOutcome::Improved)).count() as f64;
        let failed = recent.iter().filter(|e| matches!(e.outcome, crate::auto_research::ExperimentOutcome::Failed)).count() as f64;
        let regressed = recent.iter().filter(|e| matches!(e.outcome, crate::auto_research::ExperimentOutcome::Regressed)).count() as f64;
        let neutral = recent.iter().filter(|e| matches!(e.outcome, crate::auto_research::ExperimentOutcome::Neutral)).count() as f64;

        let improved_rate = improved / total;
        let failed_rate = failed / total;
        let regressed_rate = regressed / total;

        tracing::info!(
            "  [StrategyEvolver] v{} → v{}: improved={:.0}%, failed={:.0}%, regressed={:.0}%, neutral={:.0}%",
            strategy.version, new_strategy.version,
            improved_rate * 100.0, failed_rate * 100.0, regressed_rate * 100.0, neutral * total / total * 100.0
        );

        // === 温度调整 ===
        if failed_rate > 0.4 {
            // 失败太多：大幅降温
            new_strategy.research_temperature = (new_strategy.research_temperature - 0.15).max(0.2);
            tracing::info!("  [StrategyEvolver] High failure rate ({:.0}%), reducing temperature to {:.2}", failed_rate * 100.0, new_strategy.research_temperature);
        } else if regressed_rate > 0.4 {
            // 退化太多：适度降温
            new_strategy.research_temperature = (new_strategy.research_temperature - 0.08).max(0.3);
            tracing::info!("  [StrategyEvolver] High regression rate ({:.0}%), reducing temperature to {:.2}", regressed_rate * 100.0, new_strategy.research_temperature);
        } else if improved_rate < 0.15 && failed_rate < 0.2 {
            // 改进太少但也没怎么失败：稍微升温增加探索
            new_strategy.research_temperature = (new_strategy.research_temperature + 0.05).min(0.95);
            tracing::info!("  [StrategyEvolver] Low improvement rate ({:.0}%), increasing temperature to {:.2}", improved_rate * 100.0, new_strategy.research_temperature);
        }

        // === 权重调整 ===
        if failed_rate > 0.3 {
            // 失败多：提高测试权重，减少复杂度/行数权重
            new_strategy.weight_tests = (new_strategy.weight_tests + 0.1).min(0.7);
            new_strategy.weight_complexity = (new_strategy.weight_complexity - 0.02).max(0.02);
            new_strategy.weight_lines = (new_strategy.weight_lines - 0.02).max(0.02);
        }

        // 如果中性结果太多（score > 0 但不够 Improved）：
        if neutral / total > 0.5 && improved_rate < 0.3 {
            new_strategy.improved_score_threshold = (new_strategy.improved_score_threshold - 0.05).max(0.05);
            tracing::info!("  [StrategyEvolver] Too many neutral outcomes, lowering improved threshold to {:.2}", new_strategy.improved_score_threshold);
        }

        // === 阈值调整 ===
        // 如果 regression 多但主要不是测试退化，说明 lines/complexity 阈值太严
        let non_test_regressions = recent.iter().filter(|e| {
            matches!(e.outcome, crate::auto_research::ExperimentOutcome::Regressed)
                && e.tests_before.0 <= e.tests_after.0 // 测试没退化
        }).count() as f64;

        if non_test_regressions > total * 0.2 {
            // 放宽非测试维度的阈值
            new_strategy.lines_regressed_threshold = (new_strategy.lines_regressed_threshold as f64 * 1.3).min(50.0) as i32;
            new_strategy.complexity_regressed_threshold = (new_strategy.complexity_regressed_threshold * 1.3).min(30.0);
            tracing::info!("  [StrategyEvolver] Non-test regressions detected, relaxing thresholds");
        }

        // === 目标选择 ===
        if improved_rate < 0.2 && total >= 5.0 {
            new_strategy.target_selection = TargetSelectionMode::Adaptive;
            tracing::info!("  [StrategyEvolver] Switching to adaptive target selection");
        } else if improved_rate > 0.5 {
            // 高改进率时可切换回轮询（更均匀的覆盖）
            new_strategy.target_selection = TargetSelectionMode::RoundRobin;
        }

        new_strategy
    }

    /// 自适应选择下一个目标文件（优先选历史改进率低的文件）
    pub fn select_next_target(
        strategy: &StrategyConfig,
        target_files: &[String],
        experiments: &[crate::auto_research::Experiment],
        iteration: u32,
    ) -> String {
        match strategy.target_selection {
            TargetSelectionMode::RoundRobin => {
                target_files[(iteration as usize) % target_files.len()].clone()
            }
            TargetSelectionMode::Adaptive => {
                let window = experiments.len().min(strategy.adaptive_window);
                let recent = if window > 0 {
                    &experiments[experiments.len() - window..]
                } else {
                    return target_files[(iteration as usize) % target_files.len()].clone();
                };

                // 计算每个文件的改进得分（Improved=+1, Neutral=0, Regressed/Failed=-1）
                let mut file_scores: std::collections::HashMap<String, f64> =
                    std::collections::HashMap::new();
                let mut file_counts: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();

                for exp in recent {
                    let score = match exp.outcome {
                        crate::auto_research::ExperimentOutcome::Improved => 1.0,
                        crate::auto_research::ExperimentOutcome::Neutral => 0.3,
                        _ => -0.5,
                    };
                    *file_scores.entry(exp.file.clone()).or_insert(0.0) += score;
                    *file_counts.entry(exp.file.clone()).or_insert(0) += 1;
                }

                // 选择得分最低的文件（最需要改进的）
                // 没有历史的文件默认得分为 -2（优先探索）
                let best_file = target_files
                    .iter()
                    .min_by(|a, b| {
                        let score_a = file_scores.get(*a).copied().unwrap_or(-2.0);
                        let score_b = file_scores.get(*b).copied().unwrap_or(-2.0);
                        score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned()
                    .unwrap_or_else(|| target_files[0].clone());

                tracing::debug!(
                    "  [Adaptive] file_scores={:?}, selected={}",
                    file_scores, best_file
                );
                best_file
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_strategy_is_valid() {
        let s = StrategyConfig::default();
        assert!(s.weight_tests > 0.0);
        assert!(s.research_temperature > 0.0 && s.research_temperature <= 1.0);
        assert!(s.evolution_interval > 0);
    }

    #[test]
    fn test_strategy_serialization() {
        let s = StrategyConfig::default();
        let json = serde_json::to_string(&s).unwrap();
        let parsed: StrategyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, s.version);
        assert_eq!(parsed.weight_tests, s.weight_tests);
    }

    fn make_experiment(iteration: u32, file: &str, outcome: crate::auto_research::ExperimentOutcome) -> crate::auto_research::Experiment {
        crate::auto_research::Experiment {
            iteration,
            file: file.to_string(),
            files_changed: vec![],
            hypothesis: "h".to_string(),
            outcome,
            tests_before: (10, 10),
            tests_after: (10, 10),
            reflection: "r".to_string(),
            timestamp: "t".to_string(),
            metrics_before: None,
            metrics_after: None,
            multi_eval: None,
            tests_generated: false,
            new_tests_count: 0,
        }
    }

    #[test]
    fn test_evolve_high_failure() {
        let strategy = StrategyConfig::default();
        let experiments: Vec<crate::auto_research::Experiment> = (1..=8)
            .map(|i| make_experiment(i, "test.rs", crate::auto_research::ExperimentOutcome::Failed))
            .chain(std::iter::once(make_experiment(9, "test.rs", crate::auto_research::ExperimentOutcome::Improved)))
            .chain(std::iter::once(make_experiment(10, "test.rs", crate::auto_research::ExperimentOutcome::Improved)))
            .collect();
        let new = StrategyEvolver::evolve(&strategy, &experiments);
        assert!(new.research_temperature < strategy.research_temperature);
        assert!(new.weight_tests > strategy.weight_tests);
    }

    #[test]
    fn test_adaptive_selection() {
        let strategy = StrategyConfig {
            target_selection: TargetSelectionMode::Adaptive,
            adaptive_window: 10,
            ..Default::default()
        };
        let targets = vec!["a.rs".to_string(), "b.rs".to_string(), "c.rs".to_string()];
        let experiments = vec![make_experiment(1, "a.rs", crate::auto_research::ExperimentOutcome::Improved)];
        // a.rs has Improved, so should select b.rs or c.rs (no history = -2 priority)
        let selected = StrategyEvolver::select_next_target(&strategy, &targets, &experiments, 2);
        assert_ne!(selected, "a.rs");
    }
}
