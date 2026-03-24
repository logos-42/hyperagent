use serde::{Deserialize, Serialize};

use super::Score;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    pub id: String,
    pub description: String,
    pub prompt: String,
    pub expected_criteria: Vec<String>,
}

impl BenchmarkTask {
    pub fn new(description: String, prompt: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description,
            prompt,
            expected_criteria: Vec::new(),
        }
    }

    pub fn with_criteria(mut self, criteria: Vec<String>) -> Self {
        self.expected_criteria = criteria;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub task_id: String,
    pub agent_id: String,
    pub score: Score,
    pub execution_time_ms: u64,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub benchmark_name: String,
    pub total_tasks: usize,
    pub successful_tasks: usize,
    pub average_score: f32,
    pub average_time_ms: u64,
    pub results: Vec<BenchmarkResult>,
}

pub struct Benchmark {
    name: String,
    tasks: Vec<BenchmarkTask>,
}

impl Benchmark {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tasks: Vec::new(),
        }
    }

    pub fn add_task(&mut self, task: BenchmarkTask) {
        self.tasks.push(task);
    }

    pub fn with_code_generation_tasks(mut self) -> Self {
        self.tasks.push(BenchmarkTask::new(
            "Simple function".to_string(),
            "Write a Rust function that adds two numbers".to_string(),
        ));

        self.tasks.push(BenchmarkTask::new(
            "Error handling".to_string(),
            "Write a Rust function that reads a file and handles errors gracefully".to_string(),
        ));

        self.tasks.push(BenchmarkTask::new(
            "Data structure".to_string(),
            "Implement a Stack data structure in Rust with push, pop, and is_empty methods"
                .to_string(),
        ));

        self
    }

    pub fn with_algorithm_tasks(mut self) -> Self {
        self.tasks.push(BenchmarkTask::new(
            "Binary search".to_string(),
            "Implement binary search in Rust".to_string(),
        ));

        self.tasks.push(BenchmarkTask::new(
            "Sorting".to_string(),
            "Implement quicksort in Rust".to_string(),
        ));

        self.tasks.push(BenchmarkTask::new(
            "String manipulation".to_string(),
            "Write a function to reverse a string in Rust".to_string(),
        ));

        self
    }

    pub fn tasks(&self) -> &[BenchmarkTask] {
        &self.tasks
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn generate_report(&self, results: Vec<BenchmarkResult>) -> BenchmarkReport {
        let total_tasks = self.tasks.len();
        let successful_tasks = results.iter().filter(|r| r.success).count();

        let total_score: f32 = results.iter().map(|r| r.score.value).sum();
        let average_score = if !results.is_empty() {
            total_score / results.len() as f32
        } else {
            0.0
        };

        let total_time: u64 = results.iter().map(|r| r.execution_time_ms).sum();
        let average_time_ms = if !results.is_empty() {
            total_time / results.len() as u64
        } else {
            0
        };

        BenchmarkReport {
            benchmark_name: self.name.clone(),
            total_tasks,
            successful_tasks,
            average_score,
            average_time_ms,
            results,
        }
    }
}

impl Default for Benchmark {
    fn default() -> Self {
        Self::new("default").with_code_generation_tasks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_creation() {
        let benchmark = Benchmark::new("test");
        assert_eq!(benchmark.name(), "test");
    }

    #[test]
    fn test_benchmark_with_tasks() {
        let benchmark = Benchmark::new("test").with_code_generation_tasks();
        assert!(!benchmark.tasks().is_empty());
    }

    #[test]
    fn test_benchmark_report() {
        let benchmark = Benchmark::new("test");
        let results = vec![
            BenchmarkResult {
                task_id: "1".to_string(),
                agent_id: "a1".to_string(),
                score: Score::new(8.0, 7.0, 6.0),
                execution_time_ms: 100,
                success: true,
            },
            BenchmarkResult {
                task_id: "2".to_string(),
                agent_id: "a2".to_string(),
                score: Score::new(6.0, 8.0, 7.0),
                execution_time_ms: 150,
                success: true,
            },
        ];

        let report = benchmark.generate_report(results);
        assert_eq!(report.total_tasks, 0);
        assert_eq!(report.successful_tasks, 2);
    }
}
