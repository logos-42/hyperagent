use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::Agent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Planner,
    Executor,
    Evaluator,
    Reflector,
    Synthesizer,
}

impl AgentRole {
    pub fn all() -> Vec<AgentRole> {
        vec![
            AgentRole::Planner,
            AgentRole::Executor,
            AgentRole::Evaluator,
            AgentRole::Reflector,
            AgentRole::Synthesizer,
        ]
    }

    pub fn system_prompt(&self) -> &str {
        match self {
            AgentRole::Planner => {
                "You are a strategic planner. Break down complex tasks into step-by-step plans."
            }
            AgentRole::Executor => {
                "You are a task executor. Execute plans with precision and attention to detail."
            }
            AgentRole::Evaluator => {
                "You are a quality evaluator. Assess outputs critically and identify flaws."
            }
            AgentRole::Reflector => {
                "You are a reflective thinker. Analyze past actions and extract insights."
            }
            AgentRole::Synthesizer => {
                "You are a synthesis specialist. Combine diverse inputs into coherent outputs."
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub id: String,
    pub from_agent: String,
    pub to_agent: Option<String>,
    pub role: AgentRole,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AgentMessage {
    pub fn new(from_agent: String, role: AgentRole, content: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from_agent,
            to_agent: None,
            role,
            content,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn broadcast(from_agent: String, role: AgentRole, content: String) -> Self {
        Self::new(from_agent, role, content)
    }

    pub fn direct(from_agent: String, to_agent: String, role: AgentRole, content: String) -> Self {
        let mut msg = Self::new(from_agent, role, content);
        msg.to_agent = Some(to_agent);
        msg
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub population_size: usize,
    pub roles: Vec<AgentRole>,
    pub cooperation_enabled: bool,
    pub competition_enabled: bool,
    pub crossover_enabled: bool,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            population_size: 5,
            roles: AgentRole::all(),
            cooperation_enabled: true,
            competition_enabled: true,
            crossover_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationAgent {
    pub id: String,
    pub agent: Agent,
    pub role: AgentRole,
    pub fitness: f64,
    pub messages: Vec<AgentMessage>,
    pub specialization: String,
}

impl PopulationAgent {
    pub fn new(agent: Agent, role: AgentRole) -> Self {
        let prompt = format!("{}\n\n{}", role.system_prompt(), agent.prompt);

        Self {
            id: Uuid::new_v4().to_string(),
            agent: Agent::new(agent.code, prompt).with_generation(agent.generation),
            role,
            fitness: 0.0,
            messages: Vec::new(),
            specialization: role.system_prompt().to_string(),
        }
    }

    pub fn receive_message(&mut self, msg: AgentMessage) {
        self.messages.push(msg);
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
    }
}

/// Statistics about inter-agent communication patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStats {
    /// Total number of messages in shared memory
    pub total_messages: usize,
    /// Number of broadcast messages (to_agent is None)
    pub broadcast_count: usize,
    /// Number of direct messages (to_agent is Some)
    pub direct_message_count: usize,
    /// Average message length in characters
    pub avg_message_length: f64,
    /// Messages per role
    pub messages_by_role: std::collections::HashMap<String, usize>,
}

/// Distribution of agents across roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDistribution {
    /// Number of planners
    pub planners: usize,
    /// Number of executors
    pub executors: usize,
    /// Number of evaluators
    pub evaluators: usize,
    /// Number of reflectors
    pub reflectors: usize,
    /// Number of synthesizers
    pub synthesizers: usize,
    /// Total agents
    pub total: usize,
}

impl RoleDistribution {
    pub fn count_for_role(&self, role: AgentRole) -> usize {
        match role {
            AgentRole::Planner => self.planners,
            AgentRole::Executor => self.executors,
            AgentRole::Evaluator => self.evaluators,
            AgentRole::Reflector => self.reflectors,
            AgentRole::Synthesizer => self.synthesizers,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAgentSystem {
    pub id: String,
    pub population: Vec<PopulationAgent>,
    pub config: PopulationConfig,
    pub generation: u32,
    pub shared_memory: Vec<AgentMessage>,
    pub best_overall: Option<PopulationAgent>,
}

impl MultiAgentSystem {
    pub fn new(config: PopulationConfig) -> Self {
        let population = (0..config.population_size)
            .enumerate()
            .map(|(_, i)| {
                let role = config.roles[i % config.roles.len()].clone();
                let agent = Agent::from_prompt(
                    "You are a multi-agent team member. Collaborate effectively.".to_string(),
                );
                PopulationAgent::new(agent, role)
            })
            .collect();

        Self {
            id: Uuid::new_v4().to_string(),
            population,
            config,
            generation: 0,
            shared_memory: Vec::new(),
            best_overall: None,
        }
    }

    pub fn from_prompts(prompts: Vec<String>, config: PopulationConfig) -> Self {
        let population = prompts
            .into_iter()
            .enumerate()
            .map(|(i, prompt)| {
                let role = config.roles[i % config.roles.len()].clone();
                let agent = Agent::new(String::new(), prompt);
                PopulationAgent::new(agent, role)
            })
            .collect();

        Self {
            id: Uuid::new_v4().to_string(),
            population,
            config,
            generation: 0,
            shared_memory: Vec::new(),
            best_overall: None,
        }
    }

    pub fn agents_by_role(&self, role: AgentRole) -> Vec<&PopulationAgent> {
        self.population.iter().filter(|a| a.role == role).collect()
    }

    pub fn broadcast_message(&mut self, from_id: &str, content: String) {
        let from_agent = self
            .population
            .iter()
            .find(|a| a.id == from_id)
            .map(|a| a.role.clone())
            .unwrap_or(AgentRole::Executor);

        let msg = AgentMessage::broadcast(from_id.to_string(), from_agent, content);

        for agent in &mut self.population {
            if agent.id != from_id {
                agent.receive_message(msg.clone());
            }
        }
        self.shared_memory.push(msg);
    }

    pub fn send_message(&mut self, from_id: &str, to_id: &str, content: String) {
        let from_agent = self
            .population
            .iter()
            .find(|a| a.id == from_id)
            .map(|a| a.role.clone())
            .unwrap_or(AgentRole::Executor);

        let msg = AgentMessage::direct(from_id.to_string(), to_id.to_string(), from_agent, content);

        if let Some(agent) = self.population.iter_mut().find(|a| a.id == to_id) {
            agent.receive_message(msg.clone());
        }
        self.shared_memory.push(msg);
    }

    pub fn update_fitness(&mut self, agent_id: &str, fitness: f64) {
        if let Some(agent) = self.population.iter_mut().find(|a| a.id == agent_id) {
            agent.fitness = fitness;

            if let Some(ref best) = self.best_overall {
                if fitness > best.fitness {
                    self.best_overall = Some(agent.clone());
                }
            } else {
                self.best_overall = Some(agent.clone());
            }
        }
    }

    pub fn get_best(&self) -> Option<&PopulationAgent> {
        self.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn get_best_by_role(&self, role: AgentRole) -> Option<&PopulationAgent> {
        self.agents_by_role(role)
            .into_iter()
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a)
    }

    pub fn select_parents(&self, count: usize) -> Vec<PopulationAgent> {
        let mut sorted = self.population.clone();
        sorted.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_fitness: f64 = sorted.iter().map(|a| a.fitness).sum();
        if total_fitness <= 0.0 {
            return sorted.into_iter().take(count).collect();
        }

        let mut parents = Vec::new();
        for _ in 0..count {
            let mut r = rand_simple(total_fitness);
            for agent in &sorted {
                r -= agent.fitness;
                if r <= 0.0 {
                    parents.push(agent.clone());
                    break;
                }
            }
        }
        parents
    }

    pub fn clear_round(&mut self) {
        for agent in &mut self.population {
            agent.clear_messages();
        }
    }

    pub fn increment_generation(&mut self) {
        self.generation += 1;
        for agent in &mut self.population {
            agent.agent.generation = self.generation;
        }
    }

    pub fn population_fitness_stats(&self) -> (f64, f64, f64) {
        if self.population.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let sum: f64 = self.population.iter().map(|a| a.fitness).sum();
        let mean = sum / self.population.len() as f64;

        let variance = self
            .population
            .iter()
            .map(|a| (a.fitness - mean).powi(2))
            .sum::<f64>()
            / self.population.len() as f64;

        (
            mean,
            variance.sqrt(),
            self.get_best().map(|a| a.fitness).unwrap_or(0.0),
        )
    }

    /// Get the distribution of agents across roles
    pub fn role_distribution(&self) -> RoleDistribution {
        let mut planners = 0;
        let mut executors = 0;
        let mut evaluators = 0;
        let mut reflectors = 0;
        let mut synthesizers = 0;

        for agent in &self.population {
            match agent.role {
                AgentRole::Planner => planners += 1,
                AgentRole::Executor => executors += 1,
                AgentRole::Evaluator => evaluators += 1,
                AgentRole::Reflector => reflectors += 1,
                AgentRole::Synthesizer => synthesizers += 1,
            }
        }

        RoleDistribution {
            planners,
            executors,
            evaluators,
            reflectors,
            synthesizers,
            total: self.population.len(),
        }
    }

    /// Get statistics about inter-agent communication patterns
    pub fn communication_stats(&self) -> CommunicationStats {
        let total_messages = self.shared_memory.len();
        let broadcast_count = self.shared_memory.iter().filter(|m| m.to_agent.is_none()).count();
        let direct_message_count = total_messages - broadcast_count;

        let avg_message_length = if total_messages > 0 {
            self.shared_memory.iter().map(|m| m.content.len()).sum::<usize>() as f64
                / total_messages as f64
        } else {
            0.0
        };

        let mut messages_by_role = std::collections::HashMap::new();
        for msg in &self.shared_memory {
            let role_name = format!("{:?}", msg.role);
            *messages_by_role.entry(role_name).or_insert(0) += 1;
        }

        CommunicationStats {
            total_messages,
            broadcast_count,
            direct_message_count,
            avg_message_length,
            messages_by_role,
        }
    }
}

fn rand_simple(max: f64) -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let x = splitmix64(seed);
    x as f64 % max
}

fn splitmix64(mut seed: u64) -> u64 {
    seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_creation() {
        let config = PopulationConfig::default();
        let system = MultiAgentSystem::new(config);
        assert_eq!(system.population.len(), 5);
    }

    #[test]
    fn test_role_assignment() {
        let config = PopulationConfig::default();
        let system = MultiAgentSystem::new(config);
        let roles: Vec<_> = system.population.iter().map(|a| a.role.clone()).collect();
        assert!(roles.contains(&AgentRole::Planner));
    }

    #[test]
    fn test_fitness_update() {
        let config = PopulationConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let agent_id = system.population[0].id.clone();
        system.update_fitness(&agent_id, 0.8);
        assert_eq!(system.population[0].fitness, 0.8);
    }

    #[test]
    fn test_role_distribution() {
        let config = PopulationConfig::default();
        let system = MultiAgentSystem::new(config);
        let dist = system.role_distribution();
        
        assert_eq!(dist.total, 5);
        assert!(dist.planners >= 1);
        assert!(dist.executors >= 1);
    }

    #[test]
    fn test_communication_stats_empty() {
        let config = PopulationConfig::default();
        let system = MultiAgentSystem::new(config);
        let stats = system.communication_stats();
        
        assert_eq!(stats.total_messages, 0);
        assert_eq!(stats.broadcast_count, 0);
        assert_eq!(stats.direct_message_count, 0);
        assert_eq!(stats.avg_message_length, 0.0);
    }

    #[test]
    fn test_communication_stats_with_messages() {
        let config = PopulationConfig::default();
        let mut system = MultiAgentSystem::new(config);

        let from_id = system.population[0].id.clone();
        let to_id = system.population[1].id.clone();

        system.broadcast_message(&from_id, "Hello everyone".to_string());
        system.send_message(&from_id, &to_id, "Direct message".to_string());

        let stats = system.communication_stats();
        assert_eq!(stats.total_messages, 2);
        assert_eq!(stats.broadcast_count, 1);
        assert_eq!(stats.direct_message_count, 1);
        assert!(stats.avg_message_length > 0.0);
    }

    #[test]
    fn test_role_distribution_count_for_role() {
        let config = PopulationConfig::default();
        let system = MultiAgentSystem::new(config);
        let dist = system.role_distribution();
        
        // Verify count_for_role returns correct counts
        for role in AgentRole::all() {
            let count = dist.count_for_role(role);
            assert!(count <= dist.total);
        }
    }
}