pub mod archive;
pub mod lineage;

pub use archive::Archive;
pub use lineage::Lineage;

use serde::{Deserialize, Serialize};

use crate::agent::Agent;
use crate::eval::Score;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub agent: Agent,
    pub score: Score,
    pub task: String,
    pub output: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Record {
    pub fn new(agent: Agent, score: Score, task: String, output: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            agent,
            score,
            task,
            output,
            timestamp: chrono::Utc::now(),
        }
    }
}
