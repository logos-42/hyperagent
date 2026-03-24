use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptType {
    Execute,
    Mutate,
    MetaMutate,
    Evaluate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub name: PromptType,
    pub template: String,
}

impl PromptTemplate {
    pub fn execute() -> Self {
        Self {
            name: PromptType::Execute,
            template: r#"You are an autonomous task-solving agent.

Task:
{task}

Constraints:
- You must produce executable output
- Be concise and correct

Return:
- result
- reasoning (short)"#
                .to_string(),
        }
    }

    pub fn mutate() -> Self {
        Self {
            name: PromptType::Mutate,
            template: r#"You are an AI system that improves agents.

Current agent:
{agent_code}

Past failures:
{failures}

Goal:
Generate a modified version of the agent that performs better.

Rules:
- You may change structure, tools, or reasoning strategy
- Keep it minimal but effective

Return:
NEW_AGENT_CODE"#
                .to_string(),
        }
    }

    pub fn meta_mutate() -> Self {
        Self {
            name: PromptType::MetaMutate,
            template: r#"You are a meta-learning system.

Current mutation strategy:
{mutation_prompt}

History of improvements:
{history}

Problem:
The current mutation strategy is not improving fast enough.

Goal:
Modify the mutation strategy itself.

Think:
- Are we exploring enough?
- Are we exploiting too early?
- Are we missing structural changes?

Return:
NEW_MUTATION_PROMPT"#
                .to_string(),
        }
    }

    pub fn evaluate() -> Self {
        Self {
            name: PromptType::Evaluate,
            template: r#"You are a strict evaluator.

Task:
{task}

Agent output:
{output}

Score from 0 to 10 based on:
- correctness
- efficiency
- robustness

Return:
score + short justification"#
                .to_string(),
        }
    }

    pub fn render(&self, variables: &HashMap<&str, String>) -> String {
        let mut result = self.template.clone();
        for (key, value) in variables {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }
}

pub struct PromptManager;

impl PromptManager {
    pub fn execute_task(task: &str) -> String {
        let mut vars = HashMap::new();
        vars.insert("task", task.to_string());
        PromptTemplate::execute().render(&vars)
    }

    pub fn mutate_agent(agent_code: &str, failures: &str) -> String {
        let mut vars = HashMap::new();
        vars.insert("agent_code", agent_code.to_string());
        vars.insert("failures", failures.to_string());
        PromptTemplate::mutate().render(&vars)
    }

    pub fn meta_mutate(mutation_prompt: &str, history: &str) -> String {
        let mut vars = HashMap::new();
        vars.insert("mutation_prompt", mutation_prompt.to_string());
        vars.insert("history", history.to_string());
        PromptTemplate::meta_mutate().render(&vars)
    }

    pub fn evaluate(task: &str, output: &str) -> String {
        let mut vars = HashMap::new();
        vars.insert("task", task.to_string());
        vars.insert("output", output.to_string());
        PromptTemplate::evaluate().render(&vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_execute() {
        let prompt = PromptManager::execute_task("Solve this problem");
        assert!(prompt.contains("Solve this problem"));
    }

    #[test]
    fn test_render_mutate() {
        let prompt = PromptManager::mutate_agent("code", "failure1\nfailure2");
        assert!(prompt.contains("code"));
        assert!(prompt.contains("failure1"));
    }
}
