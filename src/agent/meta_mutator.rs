use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::llm::LLMClient;

use super::MutationStrategy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionHistory {
    pub generation: u32,
    pub strategy_version: u32,
    pub mutation_prompt: String,
    pub improvement_rate: f32,
    pub notes: String,
}

pub struct MetaMutator<C: LLMClient> {
    client: C,
    current_strategy: MutationStrategy,
    history: Vec<EvolutionHistory>,
    generation: u32,
}

impl<C: LLMClient> MetaMutator<C> {
    pub fn new(client: C) -> Self {
        Self {
            client,
            current_strategy: MutationStrategy::default(),
            history: Vec::new(),
            generation: 0,
        }
    }

    pub fn with_strategy(client: C, strategy: MutationStrategy) -> Self {
        Self {
            client,
            current_strategy: strategy,
            history: Vec::new(),
            generation: 0,
        }
    }

    /// Calculate the average improvement rate from history
    fn calculate_average_improvement(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        
        let total: f32 = self.history.iter()
            .map(|h| h.improvement_rate)
            .sum();
        total / self.history.len() as f32
    }

    /// Extract key learnings from history for notes
    fn extract_learnings(&self, new_strategy_prompt: &str) -> String {
        if self.history.is_empty() {
            return "Initial strategy evolution".to_string();
        }
        
        let successful_count = self.history.iter()
            .filter(|h| h.improvement_rate > 0.0)
            .count();
        let total_count = self.history.len();
        let avg_rate = self.calculate_average_improvement();
        
        // Identify patterns in successful strategies
        let avg_prompt_len = self.history.iter()
            .map(|h| h.mutation_prompt.len())
            .sum::<usize>() / self.history.len().max(1);
        
        let new_prompt_len = new_strategy_prompt.len();
        let prompt_change = if new_prompt_len > avg_prompt_len {
            "longer"
        } else if new_prompt_len < avg_prompt_len {
            "shorter"
        } else {
            "similar length"
        };
        
        format!(
            "Gen {}: {} successful of {} total (avg rate: {:.3}). New strategy is {}.",
            self.generation,
            successful_count,
            total_count,
            avg_rate,
            prompt_change
        )
    }

    pub async fn evolve(&mut self, history: &str) -> Result<MutationStrategy> {
        self.generation += 1;

        // Analyze past performance to inform the prompt
        let performance_context = if !self.history.is_empty() {
            let avg_rate = self.calculate_average_improvement();
            let recent_count = self.history.len().min(5);
            let recent_avg: f32 = self.history.iter()
                .rev()
                .take(recent_count)
                .map(|h| h.improvement_rate)
                .sum::<f32>() / recent_count.max(1) as f32;
            
            format!(
                r#"
Performance Analysis:
- Total generations: {}
- Average improvement rate: {:.3}
- Recent {} generations avg: {:.3}
- Strategy versions explored: {}

"#,
                self.history.len(),
                avg_rate,
                recent_count,
                recent_avg,
                self.history.iter().map(|h| h.strategy_version).max().unwrap_or(0)
            )
        } else {
            "No previous evolution history available.\n".to_string()
        };

        let prompt = format!(
            r#"You are a meta-learning system.

Current mutation strategy (version {}):
{}

{}History of improvements:
{}

Problem:
The current mutation strategy is not improving fast enough.

Goal:
Modify the mutation strategy itself.

Think:
- Are we exploring enough?
- Are we exploiting too early?
- Are we missing structural changes?
- What patterns led to successful improvements?

Return ONLY the new mutation strategy prompt, nothing else."#,
            self.current_strategy.version,
            self.current_strategy.prompt,
            performance_context,
            history
        );

        let response = self.client.complete(&prompt).await?;

        let new_strategy = MutationStrategy::new(response.content.clone());
        let old_version = self.current_strategy.version;

        // Calculate improvement rate based on history analysis
        // For a new strategy, we estimate potential based on past performance
        let estimated_improvement_rate = if !self.history.is_empty() {
            // Use weighted average favoring recent performance
            let weights: Vec<f32> = (0..self.history.len())
                .map(|i| (i + 1) as f32) // Linear weights favoring recent
                .collect();
            let weight_sum: f32 = weights.iter().sum();
            let weighted_rate: f32 = self.history.iter()
                .zip(weights.iter())
                .map(|(h, w)| h.improvement_rate * w)
                .sum::<f32>() / weight_sum;
            
            // Apply slight optimism for new strategy exploration
            weighted_rate * 1.05
        } else {
            0.0
        };

        // Extract learnings for notes
        let notes = self.extract_learnings(&new_strategy.prompt);

        self.history.push(EvolutionHistory {
            generation: self.generation,
            strategy_version: old_version,
            mutation_prompt: self.current_strategy.prompt.clone(),
            improvement_rate: estimated_improvement_rate,
            notes,
        });

        self.current_strategy = new_strategy;

        Ok(self.current_strategy.clone())
    }

    pub fn get_strategy(&self) -> &MutationStrategy {
        &self.current_strategy
    }

    pub fn get_history(&self) -> &[EvolutionHistory] {
        &self.history
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn set_strategy(&mut self, strategy: MutationStrategy) {
        self.current_strategy = strategy;
    }

    /// Get the best performing strategy from history
    pub fn get_best_strategy(&self) -> Option<&EvolutionHistory> {
        self.history.iter()
            .max_by(|a, b| a.improvement_rate.partial_cmp(&b.improvement_rate).unwrap())
    }

    /// Get the average improvement rate across all history
    pub fn average_improvement(&self) -> f32 {
        self.calculate_average_improvement()
    }

    /// Calculate diversity score of mutation prompts in history.
    /// Returns a value between 0.0 (all identical) and 1.0 (highly diverse).
    /// Low diversity suggests the system may be stuck in a local optimum.
    pub fn diversity_score(&self) -> f32 {
        if self.history.len() < 2 {
            return 1.0; // Single item is trivially diverse
        }

        let prompts: Vec<&str> = self.history.iter()
            .map(|h| h.mutation_prompt.as_str())
            .collect();

        // Calculate pairwise similarity scores
        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..prompts.len() {
            for j in (i + 1)..prompts.len() {
                let similarity = Self::text_similarity(prompts[i], prompts[j]);
                total_similarity += similarity;
                comparisons += 1;
            }
        }

        if comparisons == 0 {
            return 1.0;
        }

        let avg_similarity = total_similarity / comparisons as f32;
        // Diversity is inverse of similarity
        1.0 - avg_similarity
    }

    /// Calculate text similarity using word overlap (Jaccard-like metric).
    /// Returns value between 0.0 (no overlap) and 1.0 (identical).
    fn text_similarity(a: &str, b: &str) -> f32 {
        let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            return 1.0;
        }

        intersection as f32 / union as f32
    }

    /// Check if the system should explore more aggressively.
    /// Returns true if diversity is low and improvement rate is also low.
    pub fn should_explore(&self) -> bool {
        let diversity = self.diversity_score();
        let avg_improvement = self.average_improvement();
        
        // Low diversity (< 0.3) suggests we're stuck
        // Low improvement (< 0.05) suggests we need new approaches
        diversity < 0.3 && avg_improvement < 0.05
    }
}
