/// 多智能体种群进化系统
/// 实现种群级别的并行进化，支持多方向专业化分工

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, Executor, Mutator, MetaMutator};
use crate::eval::Evaluator;
use crate::llm::LLMClient;
use crate::memory::Archive;

use super::selection::{Selector, SelectionType, Individual, PopulationStats};
use super::constraints::{EvolutionDirection, ConstraintSystem};

/// 种群中每个智能体的角色标签
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentRole {
    /// 通用型（均衡发展）
    Generalist,
    /// 效率专家
    EfficiencyExpert,
    /// 鲁棒性专家
    RobustnessExpert,
    /// 泛化专家
    GeneralizationExpert,
    /// 最小化专家
    MinimalistExpert,
    /// 探索者（偏向创新）
    Explorer,
}

impl AgentRole {
    pub fn display_name(&self) -> &str {
        match self {
            AgentRole::Generalist => "通用",
            AgentRole::EfficiencyExpert => "效率专家",
            AgentRole::RobustnessExpert => "鲁棒专家",
            AgentRole::GeneralizationExpert => "泛化专家",
            AgentRole::MinimalistExpert => "极简专家",
            AgentRole::Explorer => "探索者",
        }
    }

    pub fn direction(&self) -> EvolutionDirection {
        match self {
            AgentRole::Generalist => EvolutionDirection::Efficiency,
            AgentRole::EfficiencyExpert => EvolutionDirection::Efficiency,
            AgentRole::RobustnessExpert => EvolutionDirection::Robustness,
            AgentRole::GeneralizationExpert => EvolutionDirection::Generalization,
            AgentRole::MinimalistExpert => EvolutionDirection::Minimalism,
            AgentRole::Explorer => EvolutionDirection::Exploration,
        }
    }

    pub fn mutation_prompt_suffix(&self) -> &str {
        match self {
            AgentRole::Generalist => "Focus on overall quality and correctness.",
            AgentRole::EfficiencyExpert => "Focus on making the solution faster and more efficient. Optimize algorithms and reduce overhead.",
            AgentRole::RobustnessExpert => "Focus on handling edge cases, error handling, and making the solution reliable under all conditions.",
            AgentRole::GeneralizationExpert => "Focus on making the solution work for a wide range of inputs and be adaptable to similar problems.",
            AgentRole::MinimalistExpert => "Focus on reducing code size and complexity while maintaining correctness.",
            AgentRole::Explorer => "Try creative and unconventional approaches. Don't be afraid to use novel patterns.",
        }
    }

    pub fn all_roles() -> Vec<AgentRole> {
        vec![
            AgentRole::Generalist,
            AgentRole::EfficiencyExpert,
            AgentRole::RobustnessExpert,
            AgentRole::GeneralizationExpert,
            AgentRole::MinimalistExpert,
            AgentRole::Explorer,
        ]
    }
}

/// 种群配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    /// 每个角色的初始个体数
    pub individuals_per_role: usize,
    /// 启用的角色列表
    pub roles: Vec<AgentRole>,
    /// 选择策略
    pub selection_type: SelectionType,
    /// 每代精英保留数量
    pub elite_count: usize,
    /// 角色间知识共享比例（0.0-1.0）
    pub knowledge_sharing_rate: f32,
    /// 是否启用跨角色变异
    pub cross_role_mutation: bool,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            individuals_per_role: 2,
            roles: AgentRole::all_roles(),
            selection_type: SelectionType::Tournament { tournament_size: 3 },
            elite_count: 2,
            knowledge_sharing_rate: 0.3,
            cross_role_mutation: true,
        }
    }
}

/// 种群中一个带角色的个体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationMember {
    pub agent: Agent,
    pub role: AgentRole,
    pub fitness: f32,
    pub correctness: f32,
    pub efficiency: f32,
    pub robustness: f32,
    pub generation: u32,
    pub is_elite: bool,
}

impl PopulationMember {
    pub fn new(agent: Agent, role: AgentRole) -> Self {
        Self {
            agent,
            role,
            fitness: 0.0,
            correctness: 0.0,
            efficiency: 0.0,
            robustness: 0.0,
            generation: 0,
            is_elite: false,
        }
    }

    pub fn to_individual(&self) -> Individual {
        Individual {
            agent: self.agent.clone(),
            fitness: self.fitness,
        }
    }
}

/// 每代进化的汇总统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: u32,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub role_best: HashMap<String, f32>,
    pub population_size: usize,
    pub elite_count: usize,
    pub innovations: usize,
}

/// 消息类型（智能体间通信）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from_role: AgentRole,
    pub to_role: AgentRole,
    pub content: String,
    pub message_type: MessageType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// 分享最优解
    BestSolutionShared,
    /// 分享失败经验
    FailureShared,
    /// 请求协助
    HelpRequest,
    /// 提供建议
    Suggestion,
}

/// 多智能体种群进化器
pub struct PopulationEvolution<C: LLMClient> {
    /// 种群中的所有成员
    members: Vec<PopulationMember>,
    /// 各角色的执行器（共享 LLM client）
    executor: Executor<C>,
    /// 评估器
    evaluator: Evaluator<C>,
    /// 各角色的变异器
    mutators: HashMap<AgentRole, Mutator<C>>,
    /// 元变异器
    meta_mutator: MetaMutator<C>,
    /// 选择器
    selector: Selector,
    /// 共享存档
    archive: Archive,
    /// 种群配置
    config: PopulationConfig,
    /// 约束系统
    constraints: ConstraintSystem,
    /// 进化历史统计
    history: Vec<GenerationStats>,
    /// 消息队列（智能体间通信）
    message_queue: Vec<AgentMessage>,
    /// 当前全局最优
    global_best: Option<PopulationMember>,
}

impl<C: LLMClient + Clone> PopulationEvolution<C> {
    pub fn new(client: C, pop_config: PopulationConfig) -> Result<Self> {
        let executor = Executor::new(client.clone());
        let evaluator = Evaluator::new(client.clone());
        let meta_mutator = MetaMutator::new(client.clone());
        let mutators = HashMap::new();

        let selector = Selector::new(pop_config.selection_type.clone());

        Ok(Self {
            members: Vec::new(),
            executor,
            evaluator,
            mutators,
            meta_mutator,
            selector,
            archive: Archive::new(),
            config: pop_config,
            constraints: ConstraintSystem::default(),
            history: Vec::new(),
            message_queue: Vec::new(),
            global_best: None,
        })
    }

    /// 初始化种群：为每个角色创建初始个体
    pub fn initialize(&mut self, task: &str) {
        self.members.clear();

        for role in &self.config.roles {
            for i in 0..self.config.individuals_per_role {
                let prompt = format!(
                    "You are a {} AI agent solving this task: {}.\n{}",
                    role.display_name(),
                    task,
                    role.mutation_prompt_suffix()
                );
                let agent = Agent::from_prompt(prompt);
                let mut member = PopulationMember::new(agent, role.clone());
                // 给每个个体一个微小的随机扰动以保持多样性
                member.generation = 0;
                member.fitness = 0.0;
                let _ = i; // 用于未来多样性初始化
                self.members.push(member);
            }
        }

        tracing::info!(
            "Population initialized: {} members across {} roles",
            self.members.len(),
            self.config.roles.len()
        );
    }

    /// 运行种群进化
    pub async fn evolve(&mut self, task: &str, generations: usize) -> Result<PopulationEvolutionResult> {
        self.initialize(task);

        for gen in 0..generations {
            tracing::info!("=== Population Generation {} ===", gen + 1);

            // 阶段 1: 执行 + 评估
            let mut evaluated = Vec::new();
            let mut member_updates = Vec::new();
            
            // First pass: evaluate all members
            for (idx, member) in self.members.iter().enumerate() {
                tracing::info!(
                    "  [{}] Evaluating {} #{}...",
                    idx + 1,
                    member.role.display_name(),
                    member.generation + 1
                );

                let result = self.executor.run(&member.agent, task).await;
                let eval_result: Option<crate::eval::evaluator::EvaluationResult> = match result {
                    Ok(r) => match self.evaluator.score(task, &r).await {
                        Ok(e) => Some(e),
                        Err(e) => {
                            tracing::warn!("  [{}] Evaluation failed: {}", idx + 1, e);
                            None
                        }
                    },
                    Err(e) => {
                        tracing::warn!("  [{}] Failed: {}", idx + 1, e);
                        None
                    }
                };

                member_updates.push((idx, eval_result));
            }

            // Second pass: update members with results
            for (idx, eval_result) in member_updates {
                let member = &mut self.members[idx];
                
                if let Some(eval) = eval_result {
                    
                    member.correctness = eval.score.correctness;
                    member.efficiency = eval.score.efficiency;
                    member.robustness = eval.score.robustness;
                    member.fitness = (eval.score.correctness + eval.score.efficiency + eval.score.robustness) / 3.0;
                    member.generation += 1;

                    tracing::info!(
                        "  [{}] {} #{} score: {:.2} (c={:.1} e={:.1} r={:.1})",
                        idx + 1,
                        member.role.display_name(),
                        member.generation,
                        member.fitness,
                        eval.score.correctness,
                        eval.score.efficiency,
                        eval.score.robustness,
                    );

                    self.archive.store(
                        member.agent.clone(),
                        crate::eval::evaluator::Score::new(eval.score.correctness, eval.score.efficiency, eval.score.robustness),
                        task.to_string(),
                        format!("[{}] gen {}", member.role.display_name(), member.generation),
                    );

                    if self.global_best.is_none() || member.fitness > self.global_best.as_ref().unwrap().fitness {
                        self.global_best = Some(member.clone());
                    }

                    evaluated.push(member.clone());
                } else {
                    evaluated.push(member.clone());
                }
            }

            // 阶段 2: 角色间知识共享
            self.share_knowledge();

            // 阶段 3: 变异（精英保留）
            let mut next_gen = self.elite_selection(&evaluated);

            // 对非精英个体进行变异
            for member in &mut next_gen {
                if !member.is_elite {
                    match self.mutate_member(member, task).await {
                        Ok(new_agent) => {
                            member.agent = new_agent;
                        }
                        Err(e) => {
                            tracing::warn!("  Mutation failed for {}: {}", member.role.display_name(), e);
                        }
                    }
                }
            }

            // 阶段 4: 统计
            let stats = self.compute_stats(gen as u32 + 1);
            tracing::info!(
                "  Gen {} stats: best={:.2}, avg={:.2}, elites={}, innovations={}",
                stats.generation,
                stats.best_fitness,
                stats.avg_fitness,
                stats.elite_count,
                stats.innovations,
            );
            self.history.push(stats);

            self.members = next_gen;
        }

        // 汇总结果
        let best_by_role = self.compute_best_by_role();
        Ok(PopulationEvolutionResult {
            members: self.members.clone(),
            global_best: self.global_best.clone(),
            best_by_role,
            history: self.history.clone(),
            archive_size: self.archive.size(),
            total_generations: generations as u32,
        })
    }

    /// 评估单个成员
    async fn evaluate_member(
        &self,
        member: &PopulationMember,
        task: &str,
    ) -> Result<(f32, f32, f32)> {
        let result = self.executor.run(&member.agent, task).await?;
        let eval_result = self.evaluator.score(task, &result).await?;
        Ok((
            eval_result.score.correctness,
            eval_result.score.efficiency,
            eval_result.score.robustness,
        ))
    }

    /// 精英选择：保留每个角色的 top-k
    fn elite_selection(&self, evaluated: &[PopulationMember]) -> Vec<PopulationMember> {
        let mut next_gen: Vec<PopulationMember> = Vec::new();

        for role in &self.config.roles {
            let mut role_members: Vec<&PopulationMember> = evaluated
                .iter()
                .filter(|m| &m.role == role)
                .collect();

            // 按适应度降序排列
            role_members.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

            // 保留精英
            for (i, member) in role_members.iter().enumerate() {
                let mut selected = (*member).clone();
                selected.is_elite = i < self.config.elite_count;
                next_gen.push(selected);
            }
        }

        next_gen
    }

    /// 变异单个成员
    async fn mutate_member(
        &self,
        member: &mut PopulationMember,
        task: &str,
    ) -> Result<Agent> {
        // 获取同角色的失败经验
        let failures = self.archive.get_failures_text();
        let failures_vec: Vec<String> = if failures.is_empty() {
            vec![]
        } else {
            failures.lines().take(5).map(|s| s.to_string()).collect()
        };

        // 添加角色特化指令
        let role_failures: Vec<String> = failures_vec
            .iter()
            .map(|f| format!("[{}] {}", member.role.display_name(), f))
            .collect();

        // 获取其他角色的最佳解作为跨角色参考
        let cross_ref = if self.config.cross_role_mutation {
            self.get_cross_role_reference(&member.role, task)
        } else {
            String::new()
        };

        let combined_failures: Vec<String> = if cross_ref.is_empty() {
            role_failures
        } else {
            let mut v = role_failures;
            v.push(format!("Cross-role reference:\n{}", cross_ref));
            v
        };

        // 使用角色对应的变异器
        if let Some(mutator) = self.mutators.get(&member.role) {
            mutator.mutate(&member.agent, &combined_failures).await
        } else {
            // fallback: 使用通用变异
            anyhow::bail!("No mutator for role {:?}", member.role)
        }
    }

    /// 获取其他角色的最佳解作为参考
    fn get_cross_role_reference(&self, current_role: &AgentRole, _task: &str) -> String {
        let other_best: Vec<String> = self.members
            .iter()
            .filter(|m| &m.role != current_role && m.fitness > 0.0)
            .map(|m| format!(
                "{} (fitness {:.2}): {}",
                m.role.display_name(),
                m.fitness,
                &m.agent.prompt[..m.agent.prompt.len().min(100)]
            ))
            .collect();

        if other_best.is_empty() {
            String::new()
        } else {
            other_best.join("\n")
        }
    }

    /// 角色间知识共享
    fn share_knowledge(&mut self) {
        if self.members.is_empty() {
            return;
        }

        // 找到每个角色的最佳个体
        let mut role_best: HashMap<AgentRole, &PopulationMember> = HashMap::new();
        for member in &self.members {
            if member.fitness > 0.0 {
                let entry = role_best.entry(member.role.clone()).or_insert(member);
                if member.fitness > entry.fitness {
                    *entry = member;
                }
            }
        }

        // 生成消息
        self.message_queue.clear();
        let rate = self.config.knowledge_sharing_rate;

        for (role, best) in &role_best {
            for other_role in &self.config.roles {
                if role != other_role && rand::random::<f32>() < rate {
                    self.message_queue.push(AgentMessage {
                        from_role: role.clone(),
                        to_role: other_role.clone(),
                        content: best.agent.prompt.clone(),
                        message_type: MessageType::BestSolutionShared,
                    });
                }
            }
        }

        tracing::debug!(
            "Knowledge sharing: {} messages exchanged between roles",
            self.message_queue.len()
        );
    }

    /// 计算每代统计
    fn compute_stats(&self, generation: u32) -> GenerationStats {
        let n = self.members.len();
        if n == 0 {
            return GenerationStats {
                generation,
                best_fitness: 0.0,
                avg_fitness: 0.0,
                role_best: HashMap::new(),
                population_size: 0,
                elite_count: 0,
                innovations: 0,
            };
        }

        let best_fitness = self.members.iter().map(|m| m.fitness).fold(0.0f32, f32::max);
        let avg_fitness = self.members.iter().map(|m| m.fitness).sum::<f32>() / n as f32;
        let elite_count = self.members.iter().filter(|m| m.is_elite).count();

        let mut role_best: HashMap<String, f32> = HashMap::new();
        for member in &self.members {
            let key = member.role.display_name().to_string();
            let entry = role_best.entry(key).or_insert(0.0);
            if member.fitness > *entry {
                *entry = member.fitness;
            }
        }

        let innovations = self.message_queue
            .iter()
            .filter(|m| matches!(m.message_type, MessageType::BestSolutionShared))
            .count();

        GenerationStats {
            generation,
            best_fitness,
            avg_fitness,
            role_best,
            population_size: n,
            elite_count,
            innovations,
        }
    }

    /// 计算每个角色的最佳成员
    fn compute_best_by_role(&self) -> HashMap<AgentRole, PopulationMember> {
        let mut best: HashMap<AgentRole, PopulationMember> = HashMap::new();
        for member in &self.members {
            let entry = best.entry(member.role.clone()).or_insert_with(|| member.clone());
            if member.fitness > entry.fitness {
                *entry = member.clone();
            }
        }
        best
    }

    /// 获取全局最优
    pub fn get_global_best(&self) -> Option<&PopulationMember> {
        self.global_best.as_ref()
    }

    /// 获取种群统计
    pub fn get_population_stats(&self) -> PopulationStats {
        let individuals: Vec<Individual> = self.members.iter().map(|m| m.to_individual()).collect();
        PopulationStats::calculate(&individuals)
    }
}

/// 种群进化结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationEvolutionResult {
    /// 最终种群
    pub members: Vec<PopulationMember>,
    /// 全局最优个体
    pub global_best: Option<PopulationMember>,
    /// 各角色最佳个体
    pub best_by_role: HashMap<AgentRole, PopulationMember>,
    /// 进化历史
    pub history: Vec<GenerationStats>,
    /// 存档大小
    pub archive_size: usize,
    /// 总代数
    pub total_generations: u32,
}

impl std::fmt::Display for PopulationEvolutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Population Evolution Result ===")?;
        writeln!(f, "Total generations: {}", self.total_generations)?;
        writeln!(f, "Population size: {}", self.members.len())?;
        writeln!(f, "Archive size: {}", self.archive_size)?;
        writeln!(f)?;

        if let Some(best) = &self.global_best {
            writeln!(f, "Global Best:")?;
            writeln!(f, "  Role: {}", best.role.display_name())?;
            writeln!(f, "  Fitness: {:.2}", best.fitness)?;
            writeln!(f, "  Correctness: {:.1}", best.correctness)?;
            writeln!(f, "  Efficiency: {:.1}", best.efficiency)?;
            writeln!(f, "  Robustness: {:.1}", best.robustness)?;
            writeln!(f)?;
        }

        writeln!(f, "Best by Role:")?;
        for (role, member) in &self.best_by_role {
            writeln!(
                f, "  {}: fitness={:.2} (c={:.1} e={:.1} r={:.1})",
                role.display_name(),
                member.fitness,
                member.correctness,
                member.efficiency,
                member.robustness,
            )?;
        }

        if let Some(last) = self.history.last() {
            writeln!(f, "\nFinal Generation Stats:")?;
            writeln!(f, "  Best fitness: {:.2}", last.best_fitness)?;
            writeln!(f, "  Avg fitness: {:.2}", last.avg_fitness)?;
        }

        Ok(())
    }
}

/// 特化的 Mutator 扩展（支持插入角色提示）
pub trait MutatorExt {
    fn insert(&mut self, role: AgentRole, prompt: String) -> Result<()>;
}

impl<C: LLMClient> MutatorExt for Mutator<C> {
    fn insert(&mut self, _role: AgentRole, prompt: String) -> Result<()> {
        self.update_strategy(prompt);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_roles() {
        let roles = AgentRole::all_roles();
        assert_eq!(roles.len(), 6);
        assert_eq!(roles[0], AgentRole::Generalist);
    }

    #[test]
    fn test_population_member() {
        let agent = Agent::from_prompt("test prompt".to_string());
        let member = PopulationMember::new(agent, AgentRole::EfficiencyExpert);
        assert_eq!(member.role.display_name(), "效率专家");
        assert_eq!(member.role.direction(), EvolutionDirection::Efficiency);
    }

    #[test]
    fn test_population_config_default() {
        let config = PopulationConfig::default();
        assert_eq!(config.individuals_per_role, 2);
        assert_eq!(config.roles.len(), 6);
        assert_eq!(config.elite_count, 2);
    }

    #[test]
    fn test_generation_stats() {
        let stats = GenerationStats {
            generation: 1,
            best_fitness: 8.5,
            avg_fitness: 6.0,
            role_best: HashMap::new(),
            population_size: 12,
            elite_count: 4,
            innovations: 3,
        };
        assert_eq!(stats.generation, 1);
        assert_eq!(stats.best_fitness, 8.5);
    }

    #[test]
    fn test_population_result_display() {
        let result = PopulationEvolutionResult {
            members: vec![],
            global_best: None,
            best_by_role: HashMap::new(),
            history: vec![],
            archive_size: 5,
            total_generations: 3,
        };
        let display = format!("{}", result);
        assert!(display.contains("Population Evolution Result"));
        assert!(display.contains("Archive size: 5"));
    }
}
