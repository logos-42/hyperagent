/// 热力学启发的进化框架
/// 基于普利高津耗散结构理论和最大熵产生原理

use serde::{Deserialize, Serialize};

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
}

impl EnergyState {
    pub fn new(free_energy: f32, temperature: f32) -> Self {
        Self {
            free_energy,
            entropy: 0.0,
            entropy_production_rate: 0.0,
            temperature,
        }
    }

    /// 计算 Boltzmann 因子：exp(-E/kT)
    pub fn boltzmann_factor(&self, energy_diff: f32) -> f32 {
        let kt = self.temperature * 0.1; // k_B 归一化
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
        // 扩散主导的松弛时间：τ ~ L²/D
        // L = 种群大小，D = 变异率
        let relaxation_time = if mutation_rate > 1e-6 {
            (population_size as f32).powi(2) / mutation_rate
        } else {
            1000.0
        };

        // 扩散长度：l_D = √(Dt)
        let diffusion_length = (mutation_rate * relaxation_time).sqrt();

        // 边界层：δ ~ 1/√(选择压力)
        let boundary_layer = if selection_pressure > 1e-6 {
            1.0 / selection_pressure.sqrt()
        } else {
            10.0
        };

        // 德博拉数：De = τ_response / τ_drive
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
}

impl FitnessLandscape {
    pub fn new() -> Self {
        Self {
            current_fitness: 0.0,
            gradient: 0.0,
            curvature: 0.0,
            escape_probability: 1.0,
        }
    }

    /// 更新景观信息
    pub fn update(&mut self, fitness_history: &[f32]) {
        if fitness_history.len() < 2 {
            return;
        }

        // 梯度：一阶差分
        self.gradient = fitness_history[fitness_history.len() - 1]
            - fitness_history[fitness_history.len() - 2];

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
    }

    #[test]
    fn test_fitness_landscape() {
        let mut landscape = FitnessLandscape::new();
        let history = vec![1.0, 2.0, 2.5, 2.6, 2.5];
        landscape.update(&history);
        assert!(landscape.gradient != 0.0 || landscape.curvature != 0.0);
    }
}
