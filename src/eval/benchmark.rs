use serde::{Deserialize, Serialize};

use super::Score;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    pub id: String,
    pub description: String,
    pub prompt: String,
    pub expected_criteria: Vec<String>,
    pub category: String,
}

impl BenchmarkTask {
    pub fn new(description: String, prompt: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description,
            prompt,
            expected_criteria: Vec::new(),
            category: "general".to_string(),
        }
    }

    pub fn with_criteria(mut self, criteria: Vec<String>) -> Self {
        self.expected_criteria = criteria;
        self
    }

    pub fn with_category(mut self, category: &str) -> Self {
        self.category = category.to_string();
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
pub struct CategoryStats {
    pub category: String,
    pub total_tasks: usize,
    pub successful_tasks: usize,
    pub average_score: f32,
    pub success_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub benchmark_name: String,
    pub total_tasks: usize,
    pub successful_tasks: usize,
    pub average_score: f32,
    pub average_time_ms: u64,
    pub success_rate: f32,
    pub results: Vec<BenchmarkResult>,
    pub category_stats: Vec<CategoryStats>,
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
        self.tasks.push(
            BenchmarkTask::new(
                "Simple function".to_string(),
                "Write a Rust function that adds two numbers".to_string(),
            )
            .with_category("code_generation"),
        );

        self.tasks.push(
            BenchmarkTask::new(
                "Error handling".to_string(),
                "Write a Rust function that reads a file and handles errors gracefully".to_string(),
            )
            .with_category("code_generation"),
        );

        self.tasks.push(
            BenchmarkTask::new(
                "Data structure".to_string(),
                "Implement a Stack data structure in Rust with push, pop, and is_empty methods"
                    .to_string(),
            )
            .with_category("code_generation"),
        );

        self
    }

    pub fn with_algorithm_tasks(mut self) -> Self {
        self.tasks.push(
            BenchmarkTask::new(
                "Binary search".to_string(),
                "Implement binary search in Rust".to_string(),
            )
            .with_category("algorithm"),
        );

        self.tasks.push(
            BenchmarkTask::new(
                "Sorting".to_string(),
                "Implement quicksort in Rust".to_string(),
            )
            .with_category("algorithm"),
        );

        self.tasks.push(
            BenchmarkTask::new(
                "String manipulation".to_string(),
                "Write a function to reverse a string in Rust".to_string(),
            )
            .with_category("algorithm"),
        );

        self
    }

    pub fn tasks(&self) -> &[BenchmarkTask] {
        &self.tasks
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn calculate_category_stats(
        &self,
        results: &[BenchmarkResult],
    ) -> Vec<CategoryStats> {
        use std::collections::HashMap;
        
        let mut category_data: HashMap<String, (usize, usize, f32)> = HashMap::new();
        
        // Initialize categories from tasks
        for task in &self.tasks {
            category_data
                .entry(task.category.clone())
                .or_insert((0, 0, 0.0));
        }
        
        // Aggregate results by category
        for result in results {
            if let Some(task) = self.tasks.iter().find(|t| t.id == result.task_id) {
                let entry = category_data.entry(task.category.clone()).or_insert((0, 0, 0.0));
                entry.0 += 1;
                if result.success {
                    entry.1 += 1;
                }
                entry.2 += result.score.value;
            }
        }
        
        // Convert to CategoryStats
        category_data
            .into_iter()
            .map(|(category, (total, successful, score_sum))| {
                let average_score = if total > 0 {
                    score_sum / total as f32
                } else {
                    0.0
                };
                let success_rate = if total > 0 {
                    successful as f32 / total as f32
                } else {
                    0.0
                };
                CategoryStats {
                    category,
                    total_tasks: total,
                    successful_tasks: successful,
                    average_score,
                    success_rate,
                }
            })
            .collect()
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

        let success_rate = if !results.is_empty() {
            successful_tasks as f32 / results.len() as f32
        } else {
            0.0
        };

        let category_stats = self.calculate_category_stats(&results);

        BenchmarkReport {
            benchmark_name: self.name.clone(),
            total_tasks,
            successful_tasks,
            average_score,
            average_time_ms,
            success_rate,
            results,
            category_stats,
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
    fn test_task_category() {
        let task = BenchmarkTask::new(
            "Test".to_string(),
            "Test prompt".to_string(),
        )
        .with_category("custom_category");
        
        assert_eq!(task.category, "custom_category");
    }

    #[test]
    fn test_benchmark_report_with_success_rate() {
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
                success: false,
            },
        ];

        let report = benchmark.generate_report(results);
        assert_eq!(report.total_tasks, 0);
        assert_eq!(report.successful_tasks, 1);
        assert!((report.success_rate - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_category_stats_calculation() {
        let mut benchmark = Benchmark::new("test");
        benchmark.add_task(
            BenchmarkTask::new("Task 1".to_string(), "Prompt 1".to_string())
                .with_category("cat_a"),
        );
        benchmark.add_task(
            BenchmarkTask::new("Task 2".to_string(), "Prompt 2".to_string())
                .with_category("cat_b"),
        );

        let results = vec![
            BenchmarkResult {
                task_id: benchmark.tasks()[0].id.clone(),
                agent_id: "a1".to_string(),
                score: Score::new(8.0, 7.0, 6.0),
                execution_time_ms: 100,
                success: true,
            },
            BenchmarkResult {
                task_id: benchmark.tasks()[1].id.clone(),
                agent_id: "a2".to_string(),
                score: Score::new(6.0, 8.0, 7.0),
                execution_time_ms: 150,
                success: false,
            },
        ];

        let report = benchmark.generate_report(results);
        assert_eq!(report.category_stats.len(), 2);
        
        let cat_a_stats = report.category_stats.iter()
            .find(|s| s.category == "cat_a")
            .expect("cat_a should exist");
        assert_eq!(cat_a_stats.success_rate, 1.0);
        
        let cat_b_stats = report.category_stats.iter()
            .find(|s| s.category == "cat_b")
            .expect("cat_b should exist");
        assert_eq!(cat_b_stats.success_rate, 0.0);
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