/// 热力学启发的进化框架
/// 基于普利高津耗散结构理论和最大熵产生原理

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// 系统的能量状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyState {
    /// 自由能：可用于进化的"能量"（LLM token 预算）
    pub free_energy: f32,
    /// 熵：系统无序度（代码多样性度量）
    pub entropy: f32,
    /// 熵产生率：每代熵的变化
    pub entropy_production_rate: f32,
    /// 温度：探索/开发平衡参数
    pub temperature: f32,
    /// 初始温度：用于重置和调度
    pub initial_temperature: f32,
    /// 冷却速率：温度衰减率 (0 < alpha < 1)
    pub cooling_rate: f32,
    /// 最小温度：温度下限
    pub min_temperature: f32,
    /// 当前代数
    pub generation: u32,
}

impl EnergyState {
    pub fn new(free_energy: f32, temperature: f32) -> Self {
        Self {
            free_energy,
            entropy: 0.0,
            entropy_production_rate: 0.0,
            temperature,
            initial_temperature: temperature,
            cooling_rate: 0.95, // 默认冷却率
            min_temperature: 0.01, // 最小温度，避免除零
            generation: 0,
        }
    }

    /// 创建带有自定义冷却参数的能量状态
    pub fn with_cooling(free_energy: f32, temperature: f32, cooling_rate: f32, min_temperature: f32) -> Self {
        Self {
            free_energy,
            entropy: 0.0,
            entropy_production_rate: 0.0,
            temperature,
            initial_temperature: temperature,
            cooling_rate: cooling_rate.clamp(0.8, 0.99),
            min_temperature: min_temperature.max(0.001),
            generation: 0,
        }
    }

    /// 计算 Boltzmann 因子：exp(-ΔE / k_B T)
    ///
    /// k_B = 1.0，使温度直接对应分数单位。
    /// 例如 T=1.0 时，差 1 分的解有 ~37% 接受率，差 2 分有 ~13%。
    pub fn boltzmann_factor(&self, energy_diff: f32) -> f32 {
        let kt = self.temperature.max(self.min_temperature); // k_B = 1.0
        if kt > 1e-6 {
            (-energy_diff / kt).exp()
        } else {
            if energy_diff < 0.0 { 1.0 } else { 0.0 }
        }
    }

    /// Metropolis 准则：接受新状态的概率
    pub fn metropolis_criterion(&self, old_score: f32, new_score: f32) -> f32 {
        let delta_e = new_score - old_score;
        if delta_e >= 0.0 {
            1.0
        } else {
            self.boltzmann_factor(-delta_e)
        }
    }

    /// 执行温度冷却
    ///
    /// 使用指数衰减：T(t+1) = alpha * T(t)
    /// 这是模拟退火中常用的冷却策略，保证渐近收敛
    pub fn cool(&mut self) {
        self.temperature = (self.temperature * self.cooling_rate).max(self.min_temperature);
        self.generation += 1;
    }

    /// 自适应冷却：根据系统进展调整冷却速率
    ///
    /// 策略说明：
    /// - 高熵产生率（系统活跃探索）：减缓冷却，维持探索能力
    /// - 低熵产生率（系统已稳定）：加速冷却，收敛到最优解
    ///
    /// 数学推导：
    /// - 温度更新：T(t+1) = T(t) * effective_cooling_rate
    /// - effective_cooling_rate = base_rate ^ adaptation_factor
    /// - adaptation_factor > 1 (高熵)：effective_rate 更大 → 冷却更慢
    /// - adaptation_factor < 1 (低熵)：effective_rate 更小 → 冷却更快
    pub fn adaptive_cool(&mut self) {
        // 计算自适应因子
        // 高熵产生 → 因子 < 1 → 有效冷却率更小 → 温度下降更快 (原逻辑错误)
        // 低熵产生 → 因子 > 1 → 有效冷却率更大 → 温度下降更慢 (原逻辑错误)
        //
        // 正确逻辑：
        // 高熵产生 → 需要保持高温探索 → 冷却要慢 → effective_rate 接近 1
        // 低熵产生 → 可以加速收敛 → 冷却要快 → effective_rate 要小
        //
        // 由于 effective_rate = base_rate ^ factor：
        // - base_rate < 1，所以 factor 越大 → effective_rate 越接近 1 → 冷却越慢
        // - base_rate < 1，所以 factor 越小 → effective_rate 越接近 0 → 冷却越快
        
        let adaptation_factor = if self.entropy_production_rate > 0.1 {
            // 系统活跃探索：减缓冷却
            // entropy_production_rate 范围约 [0.1, 1.0+]
            // factor 范围约 [1.0, 2.0]，使 effective_rate 接近 1
            1.0 + (self.entropy_production_rate - 0.1).min(1.0) * 1.0
        } else if self.entropy_production_rate < 0.01 {
            // 系统已稳定：加速冷却
            // entropy_production_rate 范围约 [0.0, 0.01]
            // factor 范围约 [0.0, 1.0]，使 effective_rate 更小
            0.5 + (self.entropy_production_rate / 0.01) * 0.5
        } else {
            1.0
        };

        // effective_cooling_rate > cooling_rate 表示减慢冷却
        // effective_cooling_rate < cooling_rate 表示加速冷却
        let effective_cooling = self.cooling_rate.powf(adaptation_factor);
        self.temperature = (self.temperature * effective_cooling).max(self.min_temperature);
        self.generation += 1;
    }

    /// 重加热：用于逃离局部最优
    ///
    /// 将温度提升到初始温度的一定比例
    pub fn reheat(&mut self, factor: f32) {
        self.temperature = (self.initial_temperature * factor.clamp(0.1, 1.0))
            .max(self.temperature)
            .min(self.initial_temperature);
    }

    /// 检查是否需要重加热
    ///
    /// 当温度过低且系统陷入停滞时返回 true
    pub fn should_reheat(&self, stagnation_generations: u32, threshold: u32) -> bool {
        self.temperature <= self.min_temperature * 1.1 && stagnation_generations >= threshold
    }

    /// 计算当前的冷却进度 (0 到 1)
    ///
    /// 返回 0 表示刚开始，1 表示完全冷却
    pub fn cooling_progress(&self) -> f32 {
        if self.initial_temperature <= self.min_temperature {
            1.0
        } else {
            let progress = (self.initial_temperature - self.temperature)
                / (self.initial_temperature - self.min_temperature);
            progress.clamp(0.0, 1.0)
        }
    }

    /// 更新熵和熵产生率
    pub fn update_entropy(&mut self, new_entropy: f32) {
        let old_entropy = self.entropy;
        self.entropy = new_entropy;
        // 使用指数移动平均平滑熵产生率
        let rate = if self.generation > 0 {
            (self.entropy - old_entropy).abs()
        } else {
            0.0
        };
        // 指数移动平均，alpha = 0.3
        self.entropy_production_rate = 0.3 * rate + 0.7 * self.entropy_production_rate;
    }
}

/// 耗散尺度参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissipationScale {
    /// 松弛时间：系统回到稳态的时间（代数）
    pub relaxation_time: f32,
    /// 扩散长度：变异在种群中传播的距离
    pub diffusion_length: f32,
    /// 边界层厚度：选择压力的作用范围
    pub boundary_layer: f32,
    /// 德博拉数：响应时间 / 驱动时间
    pub deborah_number: f32,
}

impl DissipationScale {
    pub fn new(
        population_size: usize,
        mutation_rate: f32,
        selection_pressure: f32,
    ) -> Self {
        // 扩散松弛时间：τ_diff ~ L²/D（扩散主导时）
        // L = 种群大小，D = 变异率
        let diffusion_time = if mutation_rate > 1e-6 {
            (population_size as f32).powi(2) / mutation_rate
        } else {
            1000.0
        };

        // 选择松弛时间：τ_sel ~ 1/s（选择主导时）
        // 选择压力驱动系统向适应度高峰移动
        let selection_time = if selection_pressure > 1e-6 {
            1.0 / selection_pressure
        } else {
            1000.0
        };

        // 实际松弛时间由较快的过程主导
        // τ ≈ (τ_diff × τ_sel) / (τ_diff + τ_sel)
        let relaxation_time = (diffusion_time * selection_time) / (diffusion_time + selection_time);

        // 扩散长度：l_D = √(D τ)
        // 表示在松弛时间内变异能传播的距离
        let diffusion_length = (mutation_rate * relaxation_time).sqrt();

        // 边界层：δ ~ D/v（扩散/选择比值）
        // 类似流体边界层，表示选择影响范围
        let boundary_layer = if selection_pressure > 1e-6 {
            mutation_rate / selection_pressure
        } else {
            10.0
        };

        // 德博拉数：De = τ_response / τ_drive
        // De < 1: 系统快速响应，类流体行为
        // De > 1: 系统响应慢，类固体行为
        // De ≈ 1: 临界态，耗散结构形成
        let deborah_number = relaxation_time / 10.0; // 假设驱动周期为 10 代

        Self {
            relaxation_time,
            diffusion_length,
            boundary_layer,
            deborah_number,
        }
    }

    /// 判断是否接近临界点（相变）
    pub fn near_critical(&self, threshold: f32) -> bool {
        // 当 De ~ 1 时系统处于临界状态
        (self.deborah_number - 1.0).abs() < threshold
    }
}

/// 信息 - 能量耦合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoEnergyCoupling {
    /// 互信息：系统与环境的信息交换
    pub mutual_information: f32,
    /// 预测信息：系统对环境的预测能力
    pub predictive_information: f32,
    /// Landauer 能耗：每比特的最小能耗
    pub landauer_cost: f32,
    /// 信息能：E_info = k_B T I
    pub info_energy: f32,
}

impl InfoEnergyCoupling {
    pub fn new(temperature: f32) -> Self {
        Self {
            mutual_information: 0.0,
            predictive_information: 0.0,
            landauer_cost: temperature * 2.0 * std::f32::consts::LN_2, // k_B T ln 2
            info_energy: 0.0,
        }
    }

    /// 更新互信息（基于适应度 - 基因型相关性）
    pub fn update_mutual_information(
        &mut self,
        fitness_variance: f32,
        genotype_variance: f32,
        covariance: f32,
    ) {
        // I(X;Y) ≈ 0.5 * log(1 + cov²/(var_x * var_y))
        let denom = fitness_variance * genotype_variance;
        if denom > 1e-6 {
            self.mutual_information = 0.5 * (1.0 + covariance.powi(2) / denom).ln();
        }
        self.info_energy = self.mutual_information * self.landauer_cost;
    }
}

/// 进化势函数（自由能景观）
#[derive(Debug, Clone)]
pub struct FitnessLandscape {
    /// 当前适应度
    pub current_fitness: f32,
    /// 局部梯度
    pub gradient: f32,
    /// 曲率（判断是否为局部最优）
    pub curvature: f32,
    /// 逃逸概率（从局部最优跳出）
    pub escape_probability: f32,
    /// 停滞代数（用于判断是否需要重加热）
    pub stagnation_count: u32,
}

impl FitnessLandscape {
    pub fn new() -> Self {
        Self {
            current_fitness: 0.0,
            gradient: 0.0,
            curvature: 0.0,
            escape_probability: 1.0,
            stagnation_count: 0,
        }
    }

    /// 更新景观信息
    pub fn update(&mut self, fitness_history: &[f32]) {
        if fitness_history.len() < 2 {
            return;
        }

        let last_fitness = fitness_history[fitness_history.len() - 1];
        let prev_fitness = fitness_history[fitness_history.len() - 2];

        // 梯度：一阶差分
        self.gradient = last_fitness - prev_fitness;

        // 检测停滞
        if (last_fitness - self.current_fitness).abs() < 0.001 {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
        }
        self.current_fitness = last_fitness;

        // 曲率：二阶差分
        if fitness_history.len() >= 3 {
            let d1 = fitness_history[fitness_history.len() - 1]
                - fitness_history[fitness_history.len() - 2];
            let d2 = fitness_history[fitness_history.len() - 2]
                - fitness_history[fitness_history.len() - 3];
            self.curvature = d1 - d2;
        }

        // 如果曲率为负且梯度接近 0，可能是局部最优
        if self.curvature < 0.0 && self.gradient.abs() < 0.01 {
            // 逃逸概率随"温度"增加
            self.escape_probability = 0.3; // 基础逃逸率
        } else {
            self.escape_probability = 1.0;
        }
    }
}

impl Default for FitnessLandscape {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 新颖度与适应度
// ---------------------------------------------------------------------------

/// 计算 Jaccard 相似度（基于 token 集合）
///
/// 优化：对于小规模输入使用线性扫描避免 HashSet 分配开销
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
    // 快速路径：空字符串处理
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    
    // 快速路径：完全相同
    if a == b {
        return 1.0;
    }
    
    let set_a: HashSet<&str> = a.split_whitespace().collect();
    let set_b: HashSet<&str> = b.split_whitespace().collect();
    
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }
    
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

/// 计算代码新颖度：1 - max_jaccard_similarity(recent_codes)
///
/// - 返回 0.0（与历史完全相同）到 1.0（完全不同）
/// - 如果 recent_codes 为空，返回 1.0（最大新颖度）
/// - 优化：提前退出当发现完全匹配时
pub fn compute_novelty(code: &str, recent_codes: &[String]) -> f32 {
    if recent_codes.is_empty() || code.is_empty() {
        return 1.0;
    }
    
    let mut max_sim = 0.0_f32;
    
    for old in recent_codes {
        // 快速路径：完全相同 → 新颖度为 0，无需继续
        if code == old {
            return 0.0;
        }
        
        let sim = jaccard_similarity(code, old);
        if sim > max_sim {
            max_sim = sim;
            // 早期退出：已找到高相似度，不太可能有更高
            if max_sim > 0.99 {
                break;
            }
        }
    }
    
    1.0 - max_sim
}

/// 适应度函数：fitness = score × (1 + novelty_weight × novelty)
///
/// 新颖度权重使得探索性高的方案获得额外适应度加成。
/// 例如 novelty_weight=0.5 时：
///   score=10, novelty=0.0 → fitness=10.0
///   score=10, novelty=1.0 → fitness=15.0
///   score=7,  novelty=1.0 → fitness=10.5  (超过无新颖度的满分)
pub fn compute_fitness(score: f32, novelty: f32, novelty_weight: f32) -> f32 {
    score * (1.0 + novelty_weight * novelty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzmann_factor() {
        let state = EnergyState::new(100.0, 0.5);
        // 能量降低应该概率高
        assert!(state.boltzmann_factor(-1.0) > 0.5);
        // 能量升高应该概率低
        assert!(state.boltzmann_factor(1.0) < 0.5);
    }

    #[test]
    fn test_dissipation_scale() {
        let scale = DissipationScale::new(100, 0.1, 0.3);
        assert!(scale.relaxation_time > 0.0);
        assert!(scale.diffusion_length > 0.0);
        // 选择压力高时，松弛时间应减小（系统更快稳定）
        let scale_high_sel = DissipationScale::new(100, 0.1, 0.9);
        assert!(scale_high_sel.relaxation_time < scale.relaxation_time);
    }

    #[test]
    fn test_fitness_landscape() {
        let mut landscape = FitnessLandscape::new();
        let history = vec![1.0, 2.0, 2.5, 2.6, 2.5];
        landscape.update(&history);
        assert!(landscape.gradient != 0.0 || landscape.curvature != 0.0);
    }

    #[test]
    fn test_novelty() {
        // 完全相同 → novelty=0
        assert!(
            (compute_novelty("fn foo() { bar() }", &["fn foo() { bar() }".to_string()]) - 0.0).abs()
                < 0.01
        );
        // 完全不同 → novelty≈1
        assert!(
            compute_novelty("aaa bbb ccc", &["xxx yyy zzz".to_string()]) > 0.9
        );
        // 空历史 → novelty=1
        assert_eq!(compute_novelty("anything", &[]), 1.0);
    }

    #[test]
    fn test_fitness() {
        // 高分低新颖 vs 低分高新颖
        let f_high_score = compute_fitness(10.0, 0.0, 0.5); // 10.0
        let f_novel = compute_fitness(7.0, 1.0, 0.5);       // 10.5
        assert!(f_novel > f_high_score);
    }

    #[test]
    fn test_temperature_cooling() {
        let mut state = EnergyState::new(100.0, 1.0);
        assert!((state.temperature - 1.0).abs() < 0.01);

        // 冷却后温度应降低
        state.cool();
        assert!(state.temperature < 1.0);
        assert!(state.temperature >= state.min_temperature);
        assert_eq!(state.generation, 1);

        // 多次冷却
        for _ in 0..100 {
            state.cool();
        }
        // 温度不应低于最小值
        assert!(state.temperature >= state.min_temperature);
    }

    #[test]
    fn test_adaptive_cooling_high_entropy() {
        // 高熵产生率时应该减慢冷却（温度下降更慢）
        let mut state = EnergyState::new(100.0, 1.0);
        state.entropy_production_rate = 0.5; // 高熵产生
        
        let temp_before = state.temperature;
        state.adaptive_cool();
        
        // 温度应该降低（冷却生效）
        assert!(state.temperature < temp_before);
        
        // 与标准冷却比较：高熵时冷却应该更慢
        let mut state_standard = EnergyState::new(100.0, 1.0);
        state_standard.cool();
        
        // 高熵状态的温度应该高于标准冷却后的温度
        // 因为 adaptation_factor > 1 使 effective_cooling 更接近 1.0
        assert!(state.temperature > state_standard.temperature,
            "High entropy should slow cooling: adaptive_temp={} > standard_temp={}",
            state.temperature, state_standard.temperature);
    }

    #[test]
    fn test_adaptive_cooling_low_entropy() {
        // 低熵产生率时应该加速冷却（温度下降更快）
        let mut state = EnergyState::new(100.0, 1.0);
        state.entropy_production_rate = 0.001; // 低熵产生
        
        let temp_before = state.temperature;
        state.adaptive_cool();
        
        // 温度应该降低
        assert!(state.temperature < temp_before);
        
        // 与标准冷却比较：低熵时冷却应该更快
        let mut state_standard = EnergyState::new(100.0, 1.0);
        state_standard.cool();
        
        // 低熵状态的温度应该低于标准冷却后的温度
        // 因为 adaptation_factor < 1 使 effective_cooling 更小
        assert!(state.temperature < state_standard.temperature,
            "Low entropy should accelerate cooling: adaptive_temp={} < standard_temp={}",
            state.temperature, state_standard.temperature);
    }

    #[test]
    fn test_adaptive_cooling_neutral_entropy() {
        // 中等熵产生率时应该接近标准冷却
        let mut state = EnergyState::new(100.0, 1.0);
        state.entropy_production_rate = 0.05; // 中等熵产生（在 0.01 到 0.1 之间）
        
        let temp_before = state.temperature;
        state.adaptive_cool();
        
        // 温度应该降低
        assert!(state.temperature < temp_before);
        
        // 应该接近标准冷却
        let mut state_standard = EnergyState::new(100.0, 1.0);
        state_standard.cool();
        
        // 中等熵时应该接近标准冷却
        let diff = (state.temperature - state_standard.temperature).abs();
        assert!(diff < 0.05,
            "Neutral entropy should be close to standard cooling: diff={}", diff);
    }

    #[test]
    fn test_reheat() {
        let mut state = EnergyState::new(100.0, 1.0);
        // 多次冷却使温度降到很低
        for _ in 0..20 {
            state.cool();
        }
        let temp_after_cool = state.temperature;
        assert!(temp_after_cool < 0.5); // 应该已经冷却到 0.5 以下

        state.reheat(0.5);
        assert!(state.temperature > temp_after_cool);
        assert!(state.temperature <= state.initial_temperature);
    }

    #[test]
    fn test_cooling_progress() {
        let mut state = EnergyState::new(100.0, 10.0);

        // 初始时进度为 0
        assert!((state.cooling_progress() - 0.0).abs() < 0.01);

        // 冷却后进度增加
        state.temperature = 5.0;
        assert!(state.cooling_progress() > 0.0);
        assert!(state.cooling_progress() < 1.0);

        // 完全冷却时进度为 1
        state.temperature = state.min_temperature;
        assert!((state.cooling_progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_should_reheat() {
        let mut state = EnergyState::new(100.0, 1.0);
        state.temperature = state.min_temperature;

        // 刚停滞不久，不需要重加热
        assert!(!state.should_reheat(2, 5));

        // 停滞足够久，需要重加热
        assert!(state.should_reheat(5, 5));
    }
    
    #[test]
    fn test_jaccard_similarity_fast_paths() {
        // 完全相同的字符串应该返回 1.0
        assert!((jaccard_similarity("hello world", "hello world") - 1.0).abs() < 0.001);
        
        // 空字符串
        assert!((jaccard_similarity("", "") - 1.0).abs() < 0.001);
        assert!((jaccard_similarity("hello", "") - 0.0).abs() < 0.001);
        assert!((jaccard_similarity("", "hello") - 0.0).abs() < 0.001);
    }
    
    #[test]
    fn test_compute_novelty_early_exit() {
        // 第一个元素完全匹配应该立即返回 0
        let recent = vec!["exact match".to_string(), "other".to_string()];
        assert!((compute_novelty("exact match", &recent) - 0.0).abs() < 0.001);
        
        // 空代码
        assert!((compute_novelty("", &recent) - 1.0).abs() < 0.001);
    }
}