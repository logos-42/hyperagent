use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::llm::LLMClient;
use crate::agent::executor::ExecutionResult;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub value: f32,
    pub correctness: f32,
    pub efficiency: f32,
    pub robustness: f32,
}

impl Score {
    pub fn new(correctness: f32, efficiency: f32, robustness: f32) -> Self {
        let value = (correctness + efficiency + robustness) / 3.0;
        Self {
            value: value.min(10.0).max(0.0),
            correctness: correctness.min(10.0).max(0.0),
            efficiency: efficiency.min(10.0).max(0.0),
            robustness: robustness.min(10.0).max(0.0),
        }
    }

    pub fn zero() -> Self {
        Self {
            value: 0.0,
            correctness: 0.0,
            efficiency: 0.0,
            robustness: 0.0,
        }
    }

    pub fn is_passing(&self) -> bool {
        self.value >= 5.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub task: String,
    pub output: String,
    pub score: Score,
    pub justification: String,
    pub agent_id: String,
}

pub struct Evaluator<C: LLMClient> {
    client: C,
    strictness: f32,
}

impl<C: LLMClient> Evaluator<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            strictness: 1.0,
        }
    }

    pub fn with_strictness(mut self, strictness: f32) -> Self {
        self.strictness = strictness;
        self
    }

    pub async fn score(&self, task: &str, result: &ExecutionResult) -> Result<EvaluationResult> {
        let prompt = format!(
            r#"You are a strict evaluator.

Task:
{}

Agent output:
{}

Score from 0 to 10 based on:
- correctness (does it solve the task?)
- efficiency (is it optimal?)
- robustness (does it handle edge cases?)

Return your response in this exact format:
SCORE: <correctness>/10, <efficiency>/10, <robustness>/10
JUSTIFICATION: <short explanation of your scoring>"#,
            task,
            result.output
        );

        let response = self.client.complete(&prompt).await?;
        self.parse_response(task, result, &response.content)
    }

    fn parse_response(
        &self,
        task: &str,
        result: &ExecutionResult,
        response: &str,
    ) -> Result<EvaluationResult> {
        let mut correctness = 5.0;
        let mut efficiency = 5.0;
        let mut robustness = 5.0;
        let mut justification = String::new();

        for line in response.lines() {
            if line.starts_with("SCORE:") {
                let parts = line.trim_start_matches("SCORE:")
                    .trim()
                    .split(',')
                    .collect::<Vec<_>>();

                if parts.len() >= 3 {
                    correctness = Self::parse_score(parts[0]);
                    efficiency = Self::parse_score(parts[1]);
                    robustness = Self::parse_score(parts[2]);
                }
            } else if line.starts_with("JUSTIFICATION:") {
                justification = line.trim_start_matches("JUSTIFICATION:").trim().to_string();
            }
        }

        let score = Score::new(
            correctness * self.strictness,
            efficiency * self.strictness,
            robustness * self.strictness,
        );

        Ok(EvaluationResult {
            task: task.to_string(),
            output: result.output.clone(),
            score,
            justification,
            agent_id: result.agent_id.clone(),
        })
    }

    fn parse_score(s: &str) -> f32 {
        s.split('/')
            .next()
            .and_then(|n| n.trim().parse::<f32>().ok())
            .unwrap_or(5.0)
    }
}

pub struct RuleBasedEvaluator;

impl RuleBasedEvaluator {
    pub fn evaluate(task: &str, output: &str) -> Score {
        let correctness = Self::evaluate_correctness(task, output);
        let efficiency = Self::evaluate_efficiency(output);
        let robustness = Self::evaluate_robustness(output);

        Score::new(correctness, efficiency, robustness)
    }

    fn evaluate_correctness(task: &str, output: &str) -> f32 {
        if output.is_empty() {
            return 0.0;
        }

        if task.contains("code") || task.contains("function") {
            if output.contains("fn ") || output.contains("function ") || output.contains("def ") {
                return 8.0;
            }
        }

        5.0
    }

    fn evaluate_efficiency(output: &str) -> f32 {
        let len = output.len();
        if len < 100 {
            8.0
        } else if len < 500 {
            6.0
        } else {
            4.0
        }
    }

    fn evaluate_robustness(output: &str) -> f32 {
        if output.contains("error") || output.contains("Error") || output.contains("fail") {
            return 3.0;
        }
        7.0
    }
}

pub struct EnsembleEvaluator<C: LLMClient> {
    evaluators: Vec<Evaluator<C>>,
}

impl<C: LLMClient> EnsembleEvaluator<C> {
    pub fn new(clients: Vec<C>) -> Self {
        let evaluators = clients
            .into_iter()
            .map(Evaluator::new)
            .collect();

        Self { evaluators }
    }

    pub async fn evaluate(
        &self,
        task: &str,
        result: &ExecutionResult,
    ) -> Result<EvaluationResult> {
        let mut results = Vec::new();

        for evaluator in &self.evaluators {
            match evaluator.score(task, result).await {
                Ok(eval_result) => results.push(eval_result),
                Err(e) => {
                    tracing::warn!("Evaluator failed: {}", e);
                }
            }
        }

        if results.is_empty() {
            return Ok(EvaluationResult {
                task: task.to_string(),
                output: result.output.clone(),
                score: Score::zero(),
                justification: "All evaluators failed".to_string(),
                agent_id: result.agent_id.clone(),
            });
        }

        let avg_correctness = results.iter().map(|r| r.score.correctness).sum::<f32>() / results.len() as f32;
        let avg_efficiency = results.iter().map(|r| r.score.efficiency).sum::<f32>() / results.len() as f32;
        let avg_robustness = results.iter().map(|r| r.score.robustness).sum::<f32>() / results.len() as f32;

        let justification = format!(
            "Ensemble of {} evaluators",
            results.len()
        );

        Ok(EvaluationResult {
            task: task.to_string(),
            output: result.output.clone(),
            score: Score::new(avg_correctness, avg_efficiency, avg_robustness),
            justification,
            agent_id: result.agent_id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_creation() {
        let score = Score::new(8.0, 7.0, 6.0);
        assert!((score.value - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_score_bounds() {
        let score = Score::new(15.0, -5.0, 5.0);
        assert_eq!(score.correctness, 10.0);
        assert_eq!(score.efficiency, 0.0);
    }

    #[test]
    fn test_rule_based_evaluator() {
        let score = RuleBasedEvaluator::evaluate("Write a function", "fn main() {}");
        assert!(score.correctness > 5.0);
    }
}
