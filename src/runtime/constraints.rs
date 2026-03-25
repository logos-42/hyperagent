/// 进化约束系统
/// 定义系统可探索的状态空间边界
use serde::{Deserialize, Serialize};

/// 进化方向枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EvolutionDirection {
    /// 优化执行效率
    Efficiency,
    /// 增强鲁棒性
    Robustness,
    /// 提高泛化能力
    Generalization,
    /// 减少代码量（最小化）
    Minimalism,
    /// 增加多样性（探索）
    Exploration,
}

/// 硬约束：不可违反的边界
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardConstraints {
    /// 最大代码长度（字符数）
    pub max_code_length: usize,
    /// 最大圈复杂度
    pub max_cyclomatic_complexity: usize,
    /// 允许的操作白名单
    pub allowed_operations: Vec<String>,
    /// 禁止的模式黑名单
    pub forbidden_patterns: Vec<String>,
    /// 最大嵌套深度
    pub max_nesting_depth: usize,
}

impl Default for HardConstraints {
    fn default() -> Self {
        Self {
            max_code_length: 10000,
            max_cyclomatic_complexity: 15,
            allowed_operations: vec![], // 空表示允许所有
            forbidden_patterns: vec![
                "eval(".to_string(),
                "exec(".to_string(),
                "infinite_loop".to_string(),
            ],
            max_nesting_depth: 5,
        }
    }
}

impl HardConstraints {
    /// 检查是否违反硬约束
    pub fn violates(&self, code: &str) -> Option<String> {
        // 检查代码长度
        if code.len() > self.max_code_length {
            return Some(format!(
                "Code length {} exceeds limit {}",
                code.len(),
                self.max_code_length
            ));
        }

        // 检查禁止模式
        for pattern in &self.forbidden_patterns {
            if code.contains(pattern) {
                return Some(format!("Contains forbidden pattern: {}", pattern));
            }
        }

        // 检查嵌套深度（简化版）
        let mut depth: usize = 0;
        let mut max_depth: usize = 0;
        for ch in code.chars() {
            if ch == '{' || ch == '(' {
                depth = depth + 1;
                if depth > max_depth {
                    max_depth = depth;
                }
            } else if ch == '}' || ch == ')' {
                if depth > 0 {
                    depth = depth - 1;
                }
            }
        }
        if max_depth > self.max_nesting_depth {
            return Some(format!(
                "Nesting depth {} exceeds limit {}",
                max_depth, self.max_nesting_depth
            ));
        }

        None
    }
}

/// 软约束：适应度惩罚项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftConstraints {
    /// 复杂度惩罚系数
    pub complexity_penalty: f32,
    /// 冗余惩罚系数
    pub redundancy_penalty: f32,
    /// 偏离惩罚系数
    pub deviation_penalty: f32,
    /// 进化方向权重
    pub direction_weights: Vec<(EvolutionDirection, f32)>,
}

impl Default for SoftConstraints {
    fn default() -> Self {
        Self {
            complexity_penalty: 0.1,
            redundancy_penalty: 0.05,
            deviation_penalty: 0.2,
            direction_weights: vec![
                (EvolutionDirection::Efficiency, 0.3),
                (EvolutionDirection::Robustness, 0.3),
                (EvolutionDirection::Generalization, 0.2),
                (EvolutionDirection::Minimalism, 0.2),
            ],
        }
    }
}

impl SoftConstraints {
    /// 计算软约束惩罚后的适应度
    pub fn apply_penalty(&self, base_score: f32, code: &str, metrics: &CodeMetrics) -> f32 {
        let mut penalty = 0.0;

        // 复杂度惩罚
        penalty += metrics.cyclomatic_complexity as f32 * self.complexity_penalty;

        // 冗余惩罚（重复代码比例）
        penalty += metrics.redundancy_ratio * self.redundancy_penalty;

        // 代码长度惩罚（鼓励简洁）
        let length_penalty = (code.len() as f32 / 1000.0) * 0.01;
        penalty += length_penalty;

        (base_score - penalty).max(0.0)
    }
}

/// 代码度量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMetrics {
    /// 圈复杂度
    pub cyclomatic_complexity: usize,
    /// 代码行数
    pub lines_of_code: usize,
    /// 冗余比例（0-1）
    pub redundancy_ratio: f32,
    /// 代码熵（信息密度）
    pub code_entropy: f32,
    /// 词汇多样性
    pub vocabulary_diversity: f32,
}

impl CodeMetrics {
    pub fn new(code: &str) -> Self {
        let lines: Vec<&str> = code.lines().collect();
        let loc = lines.len();

        // 简化版圈复杂度计算
        let mut complexity = 1;
        for keyword in &["if", "else", "for", "while", "match", "?", "&&", "||"] {
            complexity += code.matches(*keyword).count();
        }

        // 计算词汇多样性 - 使用排序后计数避免HashSet分配
        let words: Vec<&str> = code
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .collect();
        
        let vocabulary_diversity = if words.is_empty() {
            0.0
        } else if words.len() < 64 {
            // 小输入：使用位图计数（避免分配）
            let mut seen_small = [false; 64];
            let mut unique_count = 0usize;
            for (i, word) in words.iter().enumerate() {
                if !seen_small[i] {
                    seen_small[i] = true;
                    // 检查前面是否已出现
                    let mut is_dup = false;
                    for j in 0..i {
                        if words[j] == *word {
                            is_dup = true;
                            break;
                        }
                    }
                    if !is_dup {
                        unique_count += 1;
                    }
                }
            }
            unique_count as f32 / words.len() as f32
        } else {
            // 大输入：排序后计数唯一元素
            let mut words_sorted: Vec<&str> = words.clone();
            words_sorted.sort_unstable();
            let unique_count = words_sorted.len() - words_sorted
                .windows(2)
                .filter(|w| w[0] == w[1])
                .count();
            unique_count as f32 / words.len() as f32
        };

        // 熵计算 - 使用固定大小数组避免HashMap（ASCII优化）
        let mut char_freq = [0u32; 256];
        let mut total_chars = 0u32;
        for ch in code.chars() {
            let idx = (ch as usize) & 0xFF; // 低8位索引
            char_freq[idx] += 1;
            total_chars += 1;
        }
        
        let entropy: f32 = if total_chars > 0 {
            char_freq
                .iter()
                .filter(|&&count| count > 0)
                .map(|&count| {
                    let p = count as f32 / total_chars as f32;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0
        };

        // 冗余比例 - 使用排序后计数避免HashSet
        let redundancy = if lines.is_empty() {
            0.0
        } else if lines.len() < 32 {
            // 小输入：线性查找
            let mut unique_count = 0usize;
            for (i, line) in lines.iter().enumerate() {
                let mut is_dup = false;
                for j in 0..i {
                    if lines[j] == *line {
                        is_dup = true;
                        break;
                    }
                }
                if !is_dup {
                    unique_count += 1;
                }
            }
            1.0 - (unique_count as f32 / lines.len() as f32)
        } else {
            // 大输入：排序后计数
            let mut lines_sorted = lines.clone();
            lines_sorted.sort_unstable();
            let dup_count = lines_sorted
                .windows(2)
                .filter(|w| w[0] == w[1])
                .count();
            dup_count as f32 / lines.len() as f32
        };

        Self {
            cyclomatic_complexity: complexity,
            lines_of_code: loc,
            redundancy_ratio: redundancy,
            code_entropy: entropy,
            vocabulary_diversity,
        }
    }

    /// 计算代码熵（信息论角度）
    pub fn entropy(&self) -> f32 {
        self.code_entropy
    }
}

/// 拓扑约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalConstraints {
    /// 最小连通分量数
    pub min_connected_components: usize,
    /// 最大洞数（循环依赖）
    pub max_cycles: usize,
    /// 对称性破缺容忍度
    pub symmetry_breaking_tolerance: f32,
}

impl Default for TopologicalConstraints {
    fn default() -> Self {
        Self {
            min_connected_components: 1,
            max_cycles: 3,
            symmetry_breaking_tolerance: 0.1,
        }
    }
}

/// 完整的约束系统
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSystem {
    /// 硬约束
    pub hard: HardConstraints,
    /// 软约束
    pub soft: SoftConstraints,
    /// 拓扑约束
    pub topological: TopologicalConstraints,
    /// 当前进化方向
    pub current_direction: EvolutionDirection,
}

impl Default for ConstraintSystem {
    fn default() -> Self {
        Self {
            hard: HardConstraints::default(),
            soft: SoftConstraints::default(),
            topological: TopologicalConstraints::default(),
            current_direction: EvolutionDirection::Efficiency,
        }
    }
}

impl ConstraintSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// 验证代码是否满足约束
    pub fn validate(&self, code: &str, _metrics: &CodeMetrics) -> Result<f32, String> {
        // 检查硬约束
        if let Some(violation) = self.hard.violates(code) {
            return Err(violation);
        }

        // 返回软约束调整后的分数
        Ok(1.0) // 基础分数，实际由 Evaluator 计算
    }

    /// 切换进化方向
    pub fn set_direction(&mut self, direction: EvolutionDirection) {
        self.current_direction = direction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_constraints_violation() {
        let constraints = HardConstraints::default();
        let long_code = "a".repeat(10001);
        assert!(constraints.violates(&long_code).is_some());
    }

    #[test]
    fn test_code_metrics() {
        let code = "fn main() { if x > 0 { for i in 0..10 {} } }";
        let metrics = CodeMetrics::new(code);
        assert!(metrics.cyclomatic_complexity >= 1);
        assert!(metrics.vocabulary_diversity > 0.0);
    }

    #[test]
    fn test_soft_constraints_penalty() {
        let soft = SoftConstraints::default();
        let code = "fn main() { let x = 1; }";
        let metrics = CodeMetrics::new(code);
        let penalized = soft.apply_penalty(10.0, code, &metrics);
        assert!(penalized <= 10.0);
    }
}