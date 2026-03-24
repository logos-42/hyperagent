use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::Record;
use crate::agent::Agent;
use crate::eval::Score;

const DEFAULT_MAX_SIZE: usize = 1000;
const DEFAULT_TOP_K: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveConfig {
    pub max_size: usize,
    pub top_k: usize,
    pub compression_enabled: bool,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            max_size: DEFAULT_MAX_SIZE,
            top_k: DEFAULT_TOP_K,
            compression_enabled: true,
        }
    }
}

pub struct Archive {
    records: VecDeque<Record>,
    config: ArchiveConfig,
}

impl Archive {
    pub fn new() -> Self {
        Self {
            records: VecDeque::new(),
            config: ArchiveConfig::default(),
        }
    }

    pub fn with_config(config: ArchiveConfig) -> Self {
        Self {
            records: VecDeque::new(),
            config,
        }
    }

    pub fn store(&mut self, agent: Agent, score: Score, task: String, output: String) {
        let record = Record::new(agent, score, task, output);

        if self.records.len() >= self.config.max_size {
            if self.config.compression_enabled {
                self.compress();
            }

            if self.records.len() >= self.config.max_size {
                self.records.pop_back();
            }
        }

        self.records.push_front(record);
    }

    pub fn get(&self, id: &str) -> Option<&Record> {
        self.records.iter().find(|r| r.agent.id == id)
    }

    pub fn get_all(&self) -> Vec<&Record> {
        self.records.iter().collect()
    }

    pub fn top_k(&self, k: usize) -> Vec<&Record> {
        let mut sorted: Vec<_> = self.records.iter().collect();
        sorted.sort_by(|a, b| {
            b.score
                .value
                .partial_cmp(&a.score.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(k).collect()
    }

    pub fn get_best(&self) -> Option<&Record> {
        self.records.iter().max_by(|a, b| {
            a.score
                .value
                .partial_cmp(&b.score.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn get_recent(&self, n: usize) -> Vec<&Record> {
        self.records.iter().take(n).collect()
    }

    pub fn get_failures(&self) -> Vec<&Record> {
        self.records
            .iter()
            .filter(|r| !r.score.is_passing())
            .collect()
    }

    pub fn get_failures_text(&self) -> String {
        self.get_failures()
            .iter()
            .map(|r| {
                format!(
                    "Task: {}, Score: {:.2}, Output: {}",
                    r.task, r.score.value, r.output
                )
            })
            .collect::<Vec<_>>()
            .join("\n---\n")
    }

    pub fn compress(&mut self) {
        let mut sorted_records: Vec<Record> = self.records.iter().cloned().collect();
        sorted_records.sort_by(|a, b| {
            b.score.value
                .partial_cmp(&a.score.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.records.clear();
        for record in sorted_records.into_iter().take(self.config.top_k) {
            self.records.push_back(record);
        }
    }
    }

    pub fn size(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn average_score(&self) -> f32 {
        if self.records.is_empty() {
            return 0.0;
        }
        let total: f32 = self.records.iter().map(|r| r.score.value).sum();
        total / self.records.len() as f32
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl Default for Archive {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archive_store() {
        let mut archive = Archive::new();
        archive.store(
            Agent::new("code".to_string(), "prompt".to_string()),
            Score::new(8.0, 7.0, 6.0),
            "test task".to_string(),
            "output".to_string(),
        );
        assert_eq!(archive.size(), 1);
    }

    #[test]
    fn test_archive_top_k() {
        let mut archive = Archive::new();
        archive.store(
            Agent::new("c1".to_string(), "p1".to_string()),
            Score::new(5.0, 5.0, 5.0),
            "t1".to_string(),
            "o1".to_string(),
        );
        archive.store(
            Agent::new("c2".to_string(), "p2".to_string()),
            Score::new(9.0, 9.0, 9.0),
            "t2".to_string(),
            "o2".to_string(),
        );
        archive.store(
            Agent::new("c3".to_string(), "p3".to_string()),
            Score::new(7.0, 7.0, 7.0),
            "t3".to_string(),
            "o3".to_string(),
        );

        let top = archive.top_k(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].score.value >= top[1].score.value);
    }

    #[test]
    fn test_archive_best() {
        let mut archive = Archive::new();
        archive.store(
            Agent::new("c1".to_string(), "p1".to_string()),
            Score::new(5.0, 5.0, 5.0),
            "t1".to_string(),
            "o1".to_string(),
        );
        archive.store(
            Agent::new("c2".to_string(), "p2".to_string()),
            Score::new(9.0, 9.0, 9.0),
            "t2".to_string(),
            "o2".to_string(),
        );

        let best = archive.get_best().unwrap();
        assert_eq!(best.score.value, 9.0);
    }
}
