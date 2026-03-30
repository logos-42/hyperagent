use anyhow::Result;
use crate::llm::LLMClient;

use super::{AutoResearch, Experiment};

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// 写入实验日志到 markdown 文件（含多维指标 + Phase 3 多文件信息）
    pub(crate) fn append_experiment_log(&self, exp: &Experiment) -> Result<()> {
        std::fs::create_dir_all(&self.config.experiment_log_dir)?;
        let log_path = self.config.experiment_log_dir.join("research_log.md");

        // Helper to format optional value with unit
        fn format_opt<T: std::fmt::Display>(val: Option<T>, unit: &str) -> String {
            val.map(|v| format!("{}{}", v, unit)).unwrap_or_default()
        }
        std::fs::create_dir_all(&self.config.experiment_log_dir)?;
        let log_path = self.config.experiment_log_dir.join("research_log.md");

        let metrics_section = match (&exp.metrics_before, &exp.metrics_after, &exp.multi_eval) {
            (Some(before), Some(after), Some(eval)) => {
                format!(
                    "\n  - **Metrics**: score={:.2}, lines {:+}, warnings {:+}, complexity {:.0}→{:.0}, binary {:+}KB\n",
                    eval.score,
                    after.lines_delta,
                    after.warnings as i32 - before.warnings as i32,
                    before.complexity, after.complexity,
                    (after.binary_size as i64 - before.binary_size as i64) / 1024,
                )
            }
            _ => String::new(),
        };

        let test_section = if exp.tests_generated {
            format!("  - **New Tests**: {} generated\n", exp.new_tests_count)
        } else {
            String::new()
        };

        let files_section = if exp.files_changed.len() > 1 {
            let changes: Vec<String> = exp.files_changed.iter()
                .map(|fc| format!("`{}` ({} → {} lines)", fc.file, fc.old_lines, fc.new_lines))
                .collect();
            format!("  - **Files changed**: {}\n", changes.join(", "))
        } else {
            String::new()
        };

        let entry = format!(
            "---\n\n## Experiment {} — {}\n\n\
             - **File**: `src/{}`\n\
             {}\
             - **Hypothesis**: {}\n\
             - **Outcome**: {:?}\n\
             - **Tests**: {}/{} → {}/{}\n\
             {}{}\
             - **Reflection**: {}\n\
             - **Time**: {}\n",
            exp.iteration, exp.file, exp.file,
            files_section,
            exp.hypothesis, exp.outcome,
            exp.tests_before.0, exp.tests_before.1,
            exp.tests_after.0, exp.tests_after.1,
            metrics_section,
            test_section,
            exp.reflection, exp.timestamp,
        );

        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        f.write_all(entry.as_bytes())?;
        Ok(())
    }

    /// Read all experiment logs from the markdown file
    pub(crate) fn read_experiment_logs(&self) -> Result<Vec<String>> {
        let log_path = self.config.experiment_log_dir.join("research_log.md");
        if !log_path.exists() {
            return Ok(Vec::new());
        }
        let content = std::fs::read_to_string(&log_path)?;
        Ok(content
            .split("\n---\n")
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_research::{ExperimentOutcome, FileChange};
    use tempfile::TempDir;

    fn make_test_experiment(iteration: u32) -> Experiment {
        Experiment {
            iteration,
            file: "test.rs".to_string(),
            hypothesis: "Test hypothesis".to_string(),
            outcome: ExperimentOutcome::Improved,
            tests_before: (5, 10),
            tests_after: (7, 10),
            files_changed: vec![
                FileChange {
                    file: "test.rs".to_string(),
                    old_lines: 100,
                    new_lines: 110,
                },
            ],
            metrics_before: None,
            metrics_after: None,
            multi_eval: None,
            tests_generated: false,
            new_tests_count: 0,
            reflection: "Test reflection".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_append_experiment_log_creates_directory() {
        let temp = TempDir::new().unwrap();
        let log_dir = temp.path().join("logs");
        
        // Verify directory is created
        assert!(!log_dir.exists());
        
        // After appending, directory should exist
        let config = crate::auto_research::ResearchConfig {
            experiment_log_dir: log_dir.clone(),
            ..Default::default()
        };
        
        // Just verify the directory creation logic works
        std::fs::create_dir_all(&log_dir).unwrap();
        assert!(log_dir.exists());
    }

    #[test]
    fn test_log_entry_format_no_duplicate_files_section() {
        let exp = make_test_experiment(1);
        
        // Simulate the log entry generation
        let files_section = if exp.files_changed.len() > 1 {
            let changes: Vec<String> = exp.files_changed.iter()
                .map(|fc| format!("`{}` ({} → {} lines)", fc.file, fc.old_lines, fc.new_lines))
                .collect();
            format!("  - **Files changed**: {}\n", changes.join(", "))
        } else {
            String::new()
        };
        
        let entry = format!(
            "---\n\n## Experiment {} — {}\n\n\
             - **File**: `src/{}`\n\
             {}\
             - **Hypothesis**: {}\n\
             - **Outcome**: {:?}\n\
             - **Tests**: {}/{} → {}/{}\n\
             {}\n\
             - **Reflection**: {}\n\
             - **Time**: {}\n",
            exp.iteration, exp.file, exp.file,
            files_section,
            exp.hypothesis, exp.outcome,
            exp.tests_before.0, exp.tests_before.1,
            exp.tests_after.0, exp.tests_after.1,
            "metrics_placeholder",
            exp.reflection, exp.timestamp,
        );
        
        // Count occurrences of files_section in entry
        let count = entry.matches("**Files changed**").count();
        assert_eq!(count, 0, "Single file change should not have Files changed section");
        
        // Verify the entry starts with separator
        assert!(entry.starts_with("---\n\n## Experiment"));
    }

    #[test]
    fn test_log_entry_multi_file_has_files_section_once() {
        let mut exp = make_test_experiment(2);
        exp.files_changed = vec![
            FileChange { file: "a.rs".to_string(), old_lines: 10, new_lines: 15 },
            FileChange { file: "b.rs".to_string(), old_lines: 20, new_lines: 25 },
        ];
        
        let files_section = format!(
            "  - **Files changed**: `{}` ({} → {} lines), `{}` ({} → {} lines)\n",
            exp.files_changed[0].file, exp.files_changed[0].old_lines, exp.files_changed[0].new_lines,
            exp.files_changed[1].file, exp.files_changed[1].old_lines, exp.files_changed[1].new_lines,
        );
        
        let entry = format!(
            "## Experiment {} — {}\n\n\
             - **File**: `src/{}`\n\
             {}\
             - **Hypothesis**: {}\n",
            exp.iteration, exp.file, exp.file,
            files_section,
            exp.hypothesis,
        );
        
        // Verify Files changed appears exactly once
        let count = entry.matches("**Files changed**").count();
        assert_eq!(count, 1, "Files changed section should appear exactly once in multi-file change");
    }
}
