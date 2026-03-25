use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::thread;

use crate::agent::{Agent, MutationStrategy};
use crate::memory::{Archive, Lineage};

/// 热力学状态快照，每代更新
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermodynamicSnapshot {
    pub temperature: f32,
    pub entropy: f32,
    pub entropy_production_rate: f32,
    pub free_energy: f32,
    pub landscape_gradient: f32,
    pub landscape_curvature: f32,
    pub escape_probability: f32,
    pub info_energy: f32,
    pub deborah_number: f32,
    pub metropolis_accept_prob: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RuntimePhase {
    Initializing,
    Executing,
    Evaluating,
    Mutating,
    MetaMutating,
    Selecting,
    Checkpointing,
    Finished,
    Error,
}

impl Default for RuntimePhase {
    fn default() -> Self {
        Self::Initializing
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub max_generations: u32,
    pub population_size: usize,
    pub top_k_selection: usize,
    pub checkpoint_interval: u32,
    pub meta_mutation_interval: u32,
    /// 初始温度（控制探索/开发平衡），默认 1.5
    pub initial_temperature: f32,
    /// 退火速率（每代温度乘以这个系数），默认 0.9
    pub annealing_rate: f32,
    /// 变异率（用于耗散尺度计算），默认 0.1
    pub mutation_rate: f32,
    /// 选择压力（用于耗散尺度计算），默认 0.3
    pub selection_pressure: f32,
    /// 每代并行探索分支数，默认 3
    pub num_branches: usize,
    /// 新颖度权重（适应度 = score × (1 + weight × novelty)），默认 0.5
    pub novelty_weight: f32,
    /// 分支多样性阈值（Jaccard 相似度低于此值才允许并行），默认 0.8
    pub diversity_threshold: f32,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_generations: 100,
            population_size: 5,
            top_k_selection: 3,
            checkpoint_interval: 10,
            meta_mutation_interval: 20,
            initial_temperature: 1.5,
            annealing_rate: 0.9,
            mutation_rate: 0.1,
            selection_pressure: 0.3,
            num_branches: 3,
            novelty_weight: 0.5,
            diversity_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    pub phase: RuntimePhase,
    pub current_generation: u32,
    pub current_task: Option<String>,
    pub best_agent: Option<Agent>,
    pub best_score: f32,
    pub best_fitness: f32,
    pub config: RuntimeConfig,
    pub archive: Archive,
    pub lineage: Lineage,
    pub mutation_strategy: MutationStrategy,
    pub errors: Vec<String>,
    pub thermo: ThermodynamicSnapshot,
    #[serde(skip)]
    persist_path: Option<PathBuf>,
}

impl RuntimeState {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            phase: RuntimePhase::Initializing,
            current_generation: 0,
            current_task: None,
            best_agent: None,
            best_score: 0.0,
            best_fitness: 0.0,
            config,
            archive: Archive::new(),
            lineage: Lineage::new(),
            mutation_strategy: MutationStrategy::default(),
            errors: Vec::new(),
            thermo: ThermodynamicSnapshot::default(),
            persist_path: None,
        }
    }

    /// 创建带磁盘持久化的 RuntimeState
    /// 从 `dir/archive.json` 和 `dir/lineage.json` 加载历史数据
    pub fn with_persistence(config: RuntimeConfig, dir: &Path) -> Self {
        let archive_path = dir.join("archive.json");
        let lineage_path = dir.join("lineage.json");

        let archive = Archive::load_from_file(&archive_path);
        let lineage = Lineage::load_from_file(&lineage_path);

        tracing::info!(
            "Loaded archive ({} records) and lineage ({} chains) from {:?}",
            archive.size(),
            lineage.total_chains(),
            dir,
        );

        Self {
            phase: RuntimePhase::Initializing,
            current_generation: 0,
            current_task: None,
            best_agent: archive.get_best().map(|r| r.agent.clone()),
            best_score: archive.get_best().map(|r| r.score.value).unwrap_or(0.0),
            best_fitness: 0.0, // fitness 需要运行时计算
            config,
            archive,
            lineage,
            mutation_strategy: MutationStrategy::default(),
            errors: Vec::new(),
            thermo: ThermodynamicSnapshot::default(),
            persist_path: Some(dir.to_path_buf()),
        }
    }

    /// 保存 archive 和 lineage 到磁盘
    /// 使用并行写入以减少 I/O 延迟
    pub fn save(&self) {
        if let Some(dir) = &self.persist_path {
            let archive_path = dir.join("archive.json");
            let lineage_path = dir.join("lineage.json");

            // Clone data that needs to be moved into threads
            let archive = self.archive.clone();
            let lineage = self.lineage.clone();

            // Use scoped threads to parallelize independent file writes
            thread::scope(|s| {
                let archive_handle = s.spawn(move || {
                    archive.save_to_file(&archive_path)
                });
                let lineage_handle = s.spawn(move || {
                    lineage.save_to_file(&lineage_path)
                });

                // Join and log any errors
                if let Ok(Err(e)) = archive_handle.join() {
                    tracing::warn!("Failed to save archive: {}", e);
                }
                if let Ok(Err(e)) = lineage_handle.join() {
                    tracing::warn!("Failed to save lineage: {}", e);
                }
            });
        }
    }

    pub fn set_phase(&mut self, phase: RuntimePhase) {
        self.phase = phase;
    }

    pub fn increment_generation(&mut self) {
        self.current_generation += 1;
    }

    pub fn update_best(&mut self, agent: Agent, score: f32, fitness: f32) {
        if score > self.best_score {
            self.best_score = score;
        }
        if fitness > self.best_fitness {
            self.best_fitness = fitness;
            self.best_agent = Some(agent);
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn is_finished(&self) -> bool {
        self.phase == RuntimePhase::Finished
            || self.current_generation >= self.config.max_generations
    }

    pub fn should_meta_mutate(&self) -> bool {
        self.current_generation > 0
            && self.current_generation % self.config.meta_mutation_interval == 0
    }

    pub fn should_checkpoint(&self) -> bool {
        self.current_generation > 0
            && self.current_generation % self.config.checkpoint_interval == 0
    }

    pub fn summary(&self) -> String {
        let t = &self.thermo;
        format!(
            "Generation: {}/{}, Best Score: {:.2}, Best Fitness: {:.2}, Phase: {:?}, Errors: {}\n  Thermodynamics: T={:.3}, S={:.3}, dS/dt={:.3}, F={:.1}\n  Landscape: ∇={:.2}, κ={:.2}, P_esc={:.2}, E_info={:.3}\n  Dissipation: De={:.1}",
            self.current_generation,
            self.config.max_generations,
            self.best_score,
            self.best_fitness,
            self.phase,
            self.errors.len(),
            t.temperature, t.entropy, t.entropy_production_rate, t.free_energy,
            t.landscape_gradient, t.landscape_curvature, t.escape_probability, t.info_energy,
            t.deborah_number,
        )
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new(RuntimeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_state_creation() {
        let state = RuntimeState::default();
        assert_eq!(state.phase, RuntimePhase::Initializing);
        assert_eq!(state.current_generation, 0);
    }

    #[test]
    fn test_update_best() {
        let mut state = RuntimeState::default();
        let agent = Agent::new("code".to_string(), "prompt".to_string());

        state.update_best(agent.clone(), 8.5, 8.5);

        assert_eq!(state.best_score, 8.5);
        assert_eq!(state.best_fitness, 8.5);
        assert!(state.best_agent.is_some());
    }

    #[test]
    fn test_should_meta_mutate() {
        let config = RuntimeConfig {
            meta_mutation_interval: 10,
            ..Default::default()
        };
        let mut state = RuntimeState::new(config);

        state.current_generation = 10;
        assert!(state.should_meta_mutate());

        state.current_generation = 5;
        assert!(!state.should_meta_mutate());
    }
}