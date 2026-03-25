use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::agent::Agent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub agent_id: String,
    pub parent_id: Option<String>,
    pub generation: u32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionChain {
    pub root_id: String,
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    chains: HashMap<String, EvolutionChain>,
    children_map: HashMap<String, Vec<String>>,
}

impl Lineage {
    pub fn new() -> Self {
        Self {
            chains: HashMap::new(),
            children_map: HashMap::new(),
        }
    }

    pub fn add(&mut self, agent: &Agent, parent_id: Option<&str>, score: f32) {
        let node = Node {
            agent_id: agent.id.clone(),
            parent_id: parent_id.map(|s| s.to_string()),
            generation: agent.generation,
            score,
        };

        if let Some(parent) = parent_id {
            self.children_map
                .entry(parent.to_string())
                .or_insert_with(Vec::new)
                .push(agent.id.clone());
        }

        if parent_id.is_none() {
            let chain = EvolutionChain {
                root_id: agent.id.clone(),
                nodes: vec![node],
            };
            self.chains.insert(agent.id.clone(), chain);
        } else {
            for chain in self.chains.values_mut() {
                if chain.nodes.iter().any(|n| n.agent_id == parent_id.unwrap()) {
                    chain.nodes.push(node);
                    break;
                }
            }
        }
    }

    pub fn get_chain(&self, agent_id: &str) -> Option<&EvolutionChain> {
        self.chains.get(agent_id)
    }

    pub fn get_children(&self, agent_id: &str) -> Vec<&Node> {
        let children_ids = self.children_map.get(agent_id);

        match children_ids {
            Some(ids) => {
                let mut children = Vec::new();
                for chain in self.chains.values() {
                    for node in &chain.nodes {
                        if ids.contains(&node.agent_id) {
                            children.push(node);
                        }
                    }
                }
                children
            }
            None => Vec::new(),
        }
    }

    pub fn get_best_lineage(&self) -> Option<EvolutionChain> {
        let mut best_chain: Option<&EvolutionChain> = None;
        let mut best_score = f32::MIN;

        for chain in self.chains.values() {
            if let Some(last_node) = chain.nodes.last() {
                if last_node.score > best_score {
                    best_score = last_node.score;
                    best_chain = Some(chain);
                }
            }
        }

        best_chain.map(|c| c.clone())
    }

    pub fn get_generation(&self, agent_id: &str) -> Option<u32> {
        for chain in self.chains.values() {
            for node in &chain.nodes {
                if node.agent_id == agent_id {
                    return Some(node.generation);
                }
            }
        }
        None
    }

    pub fn get_ancestors(&self, agent_id: &str) -> Vec<Node> {
        let mut ancestors = Vec::new();
        let mut current_id = Some(agent_id.to_string());

        loop {
            let id = match &current_id {
                Some(id) => id.clone(),
                None => break,
            };

            let mut found = false;
            for chain in self.chains.values() {
                for node in &chain.nodes {
                    if node.agent_id == id {
                        ancestors.push(node.clone());
                        current_id = node.parent_id.clone();
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                break;
            }
        }

        ancestors
    }

    pub fn total_chains(&self) -> usize {
        self.chains.len()
    }

    /// 持久化到 JSON 文件
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// 从 JSON 文件加载，失败则返回空 lineage
    pub fn load_from_file(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|e| {
                tracing::warn!("Failed to parse lineage from {:?}: {}", path, e);
                Self::new()
            }),
            Err(_) => Self::new(),
        }
    }
}

impl Default for Lineage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Score;

    #[test]
    fn test_lineage_add() {
        let mut lineage = Lineage::new();

        let parent = Agent::new("parent_code".to_string(), "prompt".to_string());
        lineage.add(&parent, None, 5.0);

        let child = Agent::new("child_code".to_string(), "prompt".to_string()).with_generation(1);
        lineage.add(&child, Some(&parent.id), 8.0);

        assert_eq!(lineage.total_chains(), 1);
    }

    #[test]
    fn test_get_ancestors() {
        let mut lineage = Lineage::new();

        let root = Agent::new("root".to_string(), "prompt".to_string());
        lineage.add(&root, None, 5.0);

        let child = Agent::new("child".to_string(), "prompt".to_string()).with_generation(1);
        lineage.add(&child, Some(&root.id), 7.0);

        let grandchild =
            Agent::new("grandchild".to_string(), "prompt".to_string()).with_generation(2);
        lineage.add(&grandchild, Some(&child.id), 9.0);

        let ancestors = lineage.get_ancestors(&grandchild.id);
        assert!(ancestors.len() >= 2);
    }
}
