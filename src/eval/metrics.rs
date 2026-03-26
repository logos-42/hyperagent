//! 多维评估指标模块
//!
//! 收集每次迭代的多个维度数据，提供比单纯 cargo test 更丰富的评估信号：
//!   - 测试通过/总数
//!   - 代码行数变化
//!   - 编译产物体积
//!   - 代码复杂度（圈复杂度、最大嵌套深度）
//!   - 编译警告数

use serde::{Deserialize, Serialize};
use std::path::Path;

/// 单次迭代的完整指标快照
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationMetrics {
    /// 测试通过数
    pub tests_passed: u32,
    /// 测试总数
    pub tests_total: u32,
    /// 修改后的代码行数
    pub code_lines: usize,
    /// 修改前的代码行数
    pub code_lines_before: usize,
    /// 行数变化（负数=减少）
    pub lines_delta: i32,
    /// 编译警告数
    pub warnings: u32,
    /// 代码圈复杂度（粗略估计）
    pub complexity: f64,
    /// 最大嵌套深度
    pub max_nesting: usize,
    /// 二进制大小（bytes）
    pub binary_size: u64,
    /// 修改前二进制大小
    pub binary_size_before: u64,
    /// 二进制大小变化
    pub binary_delta: i64,
}

/// 指标变化方向
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MetricDirection {
    Improved,
    Neutral,
    Regressed,
}

/// 综合评估结果（多维）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiEvalResult {
    /// 各维度的变化方向
    pub tests: MetricDirection,
    pub lines: MetricDirection,       // 减少=好
    pub warnings: MetricDirection,    // 减少=好
    pub complexity: MetricDirection,  // 减少=好
    pub binary_size: MetricDirection, // 减少=好

    /// 综合分数 (-1.0 到 1.0)
    pub score: f64,

    /// 人类可读摘要
    pub summary: String,
}

impl IterationMetrics {
    /// 收集文件级代码指标
    pub fn from_code(code: &str) -> (usize, f64, usize) {
        let lines = code.lines().count();
        let (complexity, nesting) = Self::analyze_complexity(code);
        (lines, complexity, nesting)
    }

    /// 粗略分析圈复杂度和最大嵌套深度
    pub fn analyze_complexity(code: &str) -> (f64, usize) {
        let mut complexity = 1.0_f64; // 基础复杂度
        let mut max_depth = 0usize;
        let mut current_depth = 0usize;

        for line in code.lines() {
            let trimmed = line.trim();

            // 跳过注释
            if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("//!") {
                continue;
            }

            // 增加复杂度的关键词
            for keyword in &["if ", "else if", "match ", "while ", "for ", "loop ", "?", "&&", "||"] {
                // 避免 "else if" 双重计数
                if keyword == &"else if" {
                    if trimmed.contains("else if") {
                        complexity += 1.0;
                    }
                } else if keyword == &"? " {
                    // ? 运算符算 0.5
                    if trimmed.contains('?') {
                        complexity += 0.5;
                    }
                } else if keyword == &"&&" || keyword == &"||" {
                    let count = trimmed.matches(keyword).count();
                    complexity += count as f64 * 0.3;
                } else if trimmed.contains(keyword) {
                    complexity += 1.0;
                }
            }

            // 嵌套深度追踪（简化版）
            for &open in &['{'] {
                let count = trimmed.matches(open).count();
                if count > 0 && !trimmed.starts_with("//") {
                    current_depth += count;
                    if current_depth > max_depth {
                        max_depth = current_depth;
                    }
                }
            }
            for &close in &['}'] {
                let count = trimmed.matches(close).count();
                if count > 0 && current_depth > 0 {
                    current_depth = current_depth.saturating_sub(count);
                }
            }
        }

        (complexity, max_depth)
    }

    /// 获取编译产物大小
    pub fn get_binary_size(project_root: &Path, bin_name: &str) -> Option<u64> {
        let path = project_root.join("target/debug").join(bin_name);
        std::fs::metadata(&path).ok().map(|m| m.len())
    }
}

impl MultiEvalResult {
    /// 从前后指标对比生成多维评估（使用默认权重）
    pub fn compare(before: &IterationMetrics, after: &IterationMetrics) -> Self {
        Self::compare_weighted(before, after, None)
    }

    /// 从前后指标对比生成多维评估（使用可配权重和阈值）
    ///
    /// `weights` 为 Option<&StrategyConfig>，为 None 时使用默认值
    pub fn compare_weighted(
        before: &IterationMetrics,
        after: &IterationMetrics,
        weights: Option<&crate::strategy::StrategyConfig>,
    ) -> Self {
        let (w_tests_imp, w_tests_reg) = weights
            .map(|w| w.weight_tuple("tests"))
            .unwrap_or((0.4, -0.6));
        let (w_lines_imp, w_lines_reg) = weights
            .map(|w| w.weight_tuple("lines"))
            .unwrap_or((0.1, -0.05));
        let (w_warn_imp, w_warn_reg) = weights
            .map(|w| w.weight_tuple("warnings"))
            .unwrap_or((0.1, -0.1));
        let (w_cplx_imp, w_cplx_reg) = weights
            .map(|w| w.weight_tuple("complexity"))
            .unwrap_or((0.1, -0.05));
        let (w_bin_imp, w_bin_reg) = weights
            .map(|w| w.weight_tuple("binary_size"))
            .unwrap_or((0.05, -0.05));

        let lines_imp_thresh = weights.map(|w| w.lines_improved_threshold).unwrap_or(-5);
        let lines_reg_thresh = weights.map(|w| w.lines_regressed_threshold).unwrap_or(20);
        let cplx_imp_thresh = weights.map(|w| w.complexity_improved_threshold).unwrap_or(5.0);
        let cplx_reg_thresh = weights.map(|w| w.complexity_regressed_threshold).unwrap_or(10.0);
        let bin_imp_pct = weights.map(|w| w.binary_improved_pct).unwrap_or(-0.02);
        let bin_reg_pct = weights.map(|w| w.binary_regressed_pct).unwrap_or(0.05);

        let tests = if after.tests_passed > before.tests_passed {
            MetricDirection::Improved
        } else if after.tests_passed < before.tests_passed {
            MetricDirection::Regressed
        } else {
            MetricDirection::Neutral
        };

        let lines = if after.lines_delta < lines_imp_thresh {
            MetricDirection::Improved
        } else if after.lines_delta > lines_reg_thresh {
            MetricDirection::Regressed
        } else {
            MetricDirection::Neutral
        };

        let warnings = if after.warnings < before.warnings {
            MetricDirection::Improved
        } else if after.warnings > before.warnings {
            MetricDirection::Regressed
        } else {
            MetricDirection::Neutral
        };

        let complexity = if after.complexity < before.complexity - cplx_imp_thresh {
            MetricDirection::Improved
        } else if after.complexity > before.complexity + cplx_reg_thresh {
            MetricDirection::Regressed
        } else {
            MetricDirection::Neutral
        };

        let binary_size = if before.binary_size > 0 && after.binary_size > 0 {
            let pct_change = (after.binary_size as i64 - before.binary_size as i64) as f64
                / before.binary_size as f64;
            if pct_change < bin_imp_pct {
                MetricDirection::Improved
            } else if pct_change > bin_reg_pct {
                MetricDirection::Regressed
            } else {
                MetricDirection::Neutral
            }
        } else {
            MetricDirection::Neutral
        };

        // 综合评分
        let mut score = 0.0_f64;
        score += match tests {
            MetricDirection::Improved => w_tests_imp,
            MetricDirection::Regressed => w_tests_reg,
            MetricDirection::Neutral => 0.0,
        };
        score += match lines {
            MetricDirection::Improved => w_lines_imp,
            MetricDirection::Regressed => w_lines_reg,
            MetricDirection::Neutral => 0.0,
        };
        score += match warnings {
            MetricDirection::Improved => w_warn_imp,
            MetricDirection::Regressed => w_warn_reg,
            MetricDirection::Neutral => 0.0,
        };
        score += match complexity {
            MetricDirection::Improved => w_cplx_imp,
            MetricDirection::Regressed => w_cplx_reg,
            MetricDirection::Neutral => 0.0,
        };
        score += match binary_size {
            MetricDirection::Improved => w_bin_imp,
            MetricDirection::Regressed => w_bin_reg,
            MetricDirection::Neutral => 0.0,
        };

        let summary = format!(
            "Tests: {:?} ({}/{}), Lines: {:?} ({}{}), Warnings: {:?}, Complexity: {:.0}, Binary: {:?} ({:+}KB)",
            tests, after.tests_passed, after.tests_total,
            lines, after.lines_delta,
            if after.lines_delta > 0 { "+" } else { "" },
            warnings,
            after.complexity,
            binary_size,
            if before.binary_size > 0 && after.binary_size > 0 {
                (after.binary_size as i64 - before.binary_size as i64) / 1024
            } else { 0 },
        );

        Self {
            tests,
            lines,
            warnings,
            complexity,
            binary_size,
            score,
            summary,
        }
    }

    /// 是否应该接受这次修改
    pub fn should_accept(&self, strict: bool) -> bool {
        if strict {
            // 严格模式：任一维度退化则拒绝
            !matches!(self.tests, MetricDirection::Regressed)
                && !matches!(self.warnings, MetricDirection::Regressed)
        } else {
            // 宽松模式：只有测试退化才拒绝
            !matches!(self.tests, MetricDirection::Regressed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_complexity_simple() {
        let code = "fn main() {\n    let x = 1;\n    println!(\"{}\");\n}\n";
        let (complexity, nesting) = IterationMetrics::analyze_complexity(code);
        assert!(complexity < 2.0);
        assert!(nesting <= 2);
    }

    #[test]
    fn test_analyze_complexity_branches() {
        let code = r#"fn check(x: i32) {
    if x > 0 {
        println!("positive");
    } else if x < 0 {
        println!("negative");
    } else {
        println!("zero");
    }
    match x {
        1 => println!("one"),
        _ => println!("other"),
    }
}"#;
        let (complexity, nesting) = IterationMetrics::analyze_complexity(code);
        assert!(complexity > 4.0);
        assert!(nesting <= 3);
    }

    #[test]
    fn test_multi_eval_improved() {
        let before = IterationMetrics {
            tests_passed: 80, tests_total: 90, code_lines: 500, code_lines_before: 500,
            lines_delta: 0, warnings: 3, complexity: 50.0, max_nesting: 5,
            binary_size: 100000, binary_size_before: 100000, binary_delta: 0,
        };
        let after = IterationMetrics {
            tests_passed: 85, tests_total: 90, code_lines: 490, code_lines_before: 500,
            lines_delta: -10, warnings: 1, complexity: 42.0, max_nesting: 4,
            binary_size: 98000, binary_size_before: 100000, binary_delta: -2000,
        };
        let result = MultiEvalResult::compare(&before, &after);
        assert_eq!(result.tests, MetricDirection::Improved);
        assert!(result.score > 0.3);
        assert!(result.should_accept(false));
    }

    #[test]
    fn test_multi_eval_regressed() {
        let before = IterationMetrics {
            tests_passed: 85, tests_total: 90, code_lines: 500, code_lines_before: 500,
            lines_delta: 0, warnings: 1, complexity: 50.0, max_nesting: 5,
            binary_size: 100000, binary_size_before: 100000, binary_delta: 0,
        };
        let after = IterationMetrics {
            tests_passed: 80, tests_total: 90, code_lines: 520, code_lines_before: 500,
            lines_delta: 20, warnings: 2, complexity: 65.0, max_nesting: 6,
            binary_size: 105000, binary_size_before: 100000, binary_delta: 5000,
        };
        let result = MultiEvalResult::compare(&before, &after);
        assert_eq!(result.tests, MetricDirection::Regressed);
        assert!(!result.should_accept(false));
    }

    #[test]
    fn test_from_code() {
        let code = "fn test() {\n    if true {\n        println!(\"hello\");\n    }\n}\n";
        let (lines, complexity, nesting) = IterationMetrics::from_code(code);
        assert_eq!(lines, 5);
        assert!(complexity > 1.0);
    }
}
