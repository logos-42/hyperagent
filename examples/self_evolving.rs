/// 自进化系统启动器
/// 演示如何启动一个热力学一致的自进化系统

use anyhow::Result;
use hyperagent::{
    RuntimeConfig, LLMConfig,
    EnergyState, DissipationScale, ConstraintSystem,
    Selector, SelectionType, Individual,
};
use tracing::{info, warn};

/// 自进化系统配置
#[derive(Debug, Clone)]
pub struct SelfEvolvingConfig {
    /// 运行时配置
    pub runtime: RuntimeConfig,
    /// LLM 配置
    pub llm: LLMConfig,
    
    // === 热力学参数 ===
    /// 初始温度（探索强度）
    pub initial_temperature: f32,
    /// 退火率（每代温度衰减）
    pub annealing_rate: f32,
    /// 最小温度
    pub min_temperature: f32,
    
    // === 进化参数 ===
    /// 种群大小
    pub population_size: usize,
    /// 精英保留率
    pub elite_ratio: f32,
    /// 变异率
    pub mutation_rate: f32,
    /// 选择压力
    pub selection_pressure: f32,
    
    // === 耗散参数 ===
    /// 熵产生阈值（超过则触发多样性注入）
    pub entropy_threshold: f32,
    /// 停滞检测代数
    pub stagnation_threshold: u32,
    
    // === 约束系统 ===
    pub constraints: ConstraintSystem,
}

impl Default for SelfEvolvingConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig {
                max_generations: 100,
                population_size: 10,
                top_k_selection: 5,
                checkpoint_interval: 10,
                meta_mutation_interval: 20,
                initial_temperature: 1.5,
                annealing_rate: 0.9,
                mutation_rate: 0.1,
                selection_pressure: 0.3,
                num_branches: 5,
                novelty_weight: 0.5,
                diversity_threshold: 0.8,
            },
            llm: LLMConfig::default(),
            
            // 热力学参数（推荐起始值）
            initial_temperature: 0.8,  // 高探索性
            annealing_rate: 0.95,       // 每代降温 5%
            min_temperature: 0.1,       // 保留一定探索性
            
            // 进化参数
            population_size: 20,
            elite_ratio: 0.2,           // 保留 top 20%
            mutation_rate: 0.7,         // 70% 变异率
            selection_pressure: 0.3,    // 中等选择压力
            
            // 耗散参数
            entropy_threshold: 0.5,
            stagnation_threshold: 5,
            
            // 约束系统
            constraints: ConstraintSystem::default(),
        }
    }
}

impl SelfEvolvingConfig {
    /// 创建高探索性配置（适合初期）
    pub fn high_exploration() -> Self {
        Self {
            initial_temperature: 0.9,
            annealing_rate: 0.98,
            mutation_rate: 0.8,
            selection_pressure: 0.2,
            ..Default::default()
        }
    }
    
    /// 创建高开发性配置（适合后期）
    pub fn high_exploitation() -> Self {
        Self {
            initial_temperature: 0.3,
            annealing_rate: 0.99,
            mutation_rate: 0.4,
            selection_pressure: 0.5,
            ..Default::default()
        }
    }
    
    /// 计算当前代的温度（退火调度）
    pub fn temperature_at_generation(&self, generation: u32) -> f32 {
        let temp = self.initial_temperature * self.annealing_rate.powi(generation as i32);
        temp.max(self.min_temperature)
    }
    
    /// 计算耗散尺度
    pub fn dissipation_scale(&self) -> DissipationScale {
        DissipationScale::new(
            self.population_size,
            self.mutation_rate,
            self.selection_pressure,
        )
    }
    
    /// 计算能量状态
    pub fn energy_state(&self, generation: u32) -> EnergyState {
        let temp = self.temperature_at_generation(generation);
        // 自由能随代际减少（token 预算）
        let free_energy = 1000.0 * temp;
        
        EnergyState::new(free_energy, temp)
    }
}

/// 自进化系统运行器
pub struct SelfEvolvingSystem {
    config: SelfEvolvingConfig,
    #[allow(dead_code)]
    selector: Selector,
    current_temperature: f32,
    generation: u32,
    stagnation_counter: u32,
    best_fitness_history: Vec<f32>,
}

impl SelfEvolvingSystem {
    pub fn new(config: SelfEvolvingConfig) -> Self {
        let selector = Selector::new(SelectionType::Boltzmann {
            temperature: config.initial_temperature,
        });

        Self {
            config: config.clone(),
            selector,
            current_temperature: config.initial_temperature,
            generation: 0,
            stagnation_counter: 0,
            best_fitness_history: Vec::new(),
        }
    }
    
    /// 运行进化循环
    pub async fn run(&mut self, task: &str) -> Result<()> {
        info!("🚀 Starting Self-Evolving System");
        info!("   Initial Temperature: {}", self.current_temperature);
        info!("   Population Size: {}", self.config.population_size);
        info!("   Task: {}", task);
        
        // 初始化种群（远离平衡态）
        let population = self.initialize_population(task).await?;
        info!("   Initialized population with {} individuals", population.len());
        
        // 计算初始耗散尺度
        let scale = self.config.dissipation_scale();
        info!("   Dissipation Scale:");
        info!("      - Relaxation Time: {:.2} generations", scale.relaxation_time);
        info!("      - Diffusion Length: {:.2}", scale.diffusion_length);
        info!("      - Deborah Number: {:.2}", scale.deborah_number);
        
        if scale.near_critical(0.2) {
            warn!("   ⚠️  System near critical point (De ≈ 1)");
        }
        
        // 主进化循环
        while self.generation < self.config.runtime.max_generations {
            self.run_generation(task).await?;
            
            // 检查停滞
            if self.check_stagnation() {
                warn!("⚠️  Evolution stagnated at generation {}", self.generation);
                self.inject_diversity();
            }
            
            // 更新温度
            self.current_temperature = self.config.temperature_at_generation(self.generation);
            
            // 检查是否接近临界点
            if scale.near_critical(0.1) {
                info!("📊 System at critical point - phase transition possible");
            }
        }
        
        info!("✅ Evolution completed after {} generations", self.generation);
        Ok(())
    }
    
    async fn run_generation(&mut self, _task: &str) -> Result<()> {
        // 这里会集成到 EvolutionLoop
        // 简化版演示
        self.generation += 1;

        if self.generation % 10 == 0 {
            info!("Generation {}: T = {:.3}", self.generation, self.current_temperature);
        }

        Ok(())
    }
    
    async fn initialize_population(&self, task: &str) -> Result<Vec<Individual>> {
        // 创建多样性初始种群
        let mut population = Vec::with_capacity(self.config.population_size);

        for i in 0..self.config.population_size {
            // 添加随机扰动（远离平衡态）
            let agent = hyperagent::Agent::from_prompt(format!(
                "You are an autonomous task-solving agent. Variant: {}. Task: {}",
                i, task
            ));

            population.push(Individual {
                agent,
                fitness: 5.0, // 初始适应度
            });
        }

        Ok(population)
    }
    
    fn check_stagnation(&mut self) -> bool {
        if self.best_fitness_history.len() < 2 {
            return false;
        }
        
        let recent_improvement = self.best_fitness_history
            .iter()
            .rev()
            .take(5)
            .fold(0.0, |acc, &f| acc + f);
        
        let previous_improvement = self.best_fitness_history
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .fold(0.0, |acc, &f| acc + f);
        
        // 如果最近 5 代和前 5 代改进相近，可能停滞
        if (recent_improvement - previous_improvement).abs() < 0.01 {
            self.stagnation_counter += 1;
            self.stagnation_counter >= self.config.stagnation_threshold
        } else {
            self.stagnation_counter = 0;
            false
        }
    }
    
    fn inject_diversity(&mut self) {
        info!("💉 Injecting diversity to escape local optimum");
        // 实际实现会添加随机变异个体
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    
    info!("=== Hyperagent Self-Evolving System ===");
    info!("Based on Prigogine's Dissipative Structure Theory");
    
    // 配置系统
    let config = SelfEvolvingConfig::high_exploration();
    
    // 创建系统
    let mut system = SelfEvolvingSystem::new(config);
    
    // 运行
    let task = "Write a Rust function that efficiently calculates Fibonacci numbers";
    system.run(task).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temperature_annealing() {
        let config = SelfEvolvingConfig::default();
        
        // 温度应该随代际递减
        let t0 = config.temperature_at_generation(0);
        let t10 = config.temperature_at_generation(10);
        let t100 = config.temperature_at_generation(100);
        
        assert!(t0 > t10);
        assert!(t10 > t100);
        assert!(t100 >= config.min_temperature);
    }
    
    #[test]
    fn test_dissipation_scale() {
        let config = SelfEvolvingConfig::default();
        let scale = config.dissipation_scale();
        
        assert!(scale.relaxation_time > 0.0);
        assert!(scale.diffusion_length > 0.0);
    }
    
    #[test]
    fn test_energy_state() {
        let config = SelfEvolvingConfig::default();
        
        let e0 = config.energy_state(0);
        let e10 = config.energy_state(10);
        
        // 自由能应该随温度降低
        assert!(e0.free_energy > e10.free_energy);
    }
}
