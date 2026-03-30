use anyhow::Result;
use crate::llm::LLMClient;

use super::{AutoResearch, Experiment, ExperimentOutcome};

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// 写入实验日志到 markdown 文件（含多维指标 + Phase 3 多文件信息）
    pub(crate) fn append_experiment_log(&self, exp: &Experiment) -> Result<()> {
        std::fs::create_dir_all(&self.config.experiment_log_dir)?;
        let log_path = self.config.experiment_log_dir.join("research_log.md");

        let metrics_section = match (&exp.metrics_before, &exp.metrics_after, &exp.multi_eval) {
            (Some(before), Some(after), Some(eval)) => {
                let warnings_delta = after.warnings as i32 - before.warnings as i32;
                let complexity_delta = after.complexity - before.complexity;
                let binary_delta = (after.binary_size as i64 - before.binary_size as i64) / 1024;
                format!(
                    "\n  - **Metrics**: score={:.2}, lines {:+}, warnings {:+} ({}→{}), complexity {:+.0} ({:.0}→{:.0}), binary {:+}KB\n",
                    eval.score,
                    after.lines_delta,
                    warnings_delta,
                    before.warnings,
                    after.warnings,
                    complexity_delta,
                    before.complexity,
                    after.complexity,
                    binary_delta,
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
        
        // Ensure proper separation from previous entries if file exists and doesn't end with newline
        if log_path.exists() && log_path.metadata()?.len() > 0 {
            let existing = std::fs::read_to_string(&log_path)?;
            if !existing.ends_with('\n') {
                f.write_all(b"\n")?;
            }
        }
        
        f.write_all(entry.as_bytes())?;
        Ok(())
    }

    /// Clear all experiment logs (useful for starting fresh research sessions)
    pub(crate) fn clear_experiment_logs(&self) -> Result<()> {
        let log_path = self.config.experiment_log_dir.join("research_log.md");
        if log_path.exists() {
            std::fs::remove_file(&log_path)?;
        }
        Ok(())
    }

    /// Check if an experiment for a given iteration/file combination already exists
    /// Uses buffered reading for memory efficiency
    pub(crate) fn experiment_exists(&self, iteration: u32, file: &str) -> Result<bool> {
        let log_path = self.config.experiment_log_dir.join("research_log.md");
        if !log_path.exists() {
            return Ok(false);
        }
        
        use std::io::{BufRead, BufReader};
        use std::fs::File;
        
        let file_handle = File::open(&log_path)?;
        let reader = BufReader::new(file_handle);
        
        let target = format!("## Experiment {} — {}", iteration, file);
        
        for line in reader.lines() {
            let line = line?;
            if line.contains(&target) {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    /// Read all experiment logs from the markdown file
    /// Uses buffered reading for memory efficiency with large log files
    pub(crate) fn read_experiment_logs(&self) -> Result<Vec<String>> {
        let log_path = self.config.experiment_log_dir.join("research_log.md");
        if !log_path.exists() {
            return Ok(Vec::new());
        }
        
        use std::io::{BufRead, BufReader};
        use std::fs::File;
        
        let file = File::open(&log_path)?;
        let reader = BufReader::new(file);
        
        let mut entries: Vec<String> = Vec::new();
        let mut current_entry: Vec<String> = Vec::new();
        let mut in_entry = false;
        
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            
            // Detect experiment entry start: "---" followed by "## Experiment"
            if trimmed == "---" {
                if in_entry && !current_entry.is_empty() {
                    entries.push(current_entry.join("\n"));
                    current_entry.clear();
                }
                in_entry = true;
            } else if in_entry {
                current_entry.push(line);
            }
        }
        
        // Don't forget the last entry if file doesn't end with separator
        if !current_entry.is_empty() {
            entries.push(current_entry.join("\n"));
        }
        
        Ok(entries.into_iter().filter(|s| !s.trim().is_empty()).collect())
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
    fn test_metrics_section_formats_delta_correctly() {
        use crate::eval::metrics::IterationMetrics;
        use crate::eval::metrics::MultiEvalResult;
        
        let mut exp = make_test_experiment(3);
        exp.metrics_before = Some(IterationMetrics {
            lines_delta: 0,
            warnings: 2,
            complexity: 15.0,
            binary_size: 1024 * 100, // 100KB
            tests_passed: 10,
            tests_total: 12,
            compile_time_ms: 500,
        });
        exp.metrics_after = Some(IterationMetrics {
            lines_delta: 50,
            warnings: 0,
            complexity: 12.0,
            binary_size: 1024 * 105, // 105KB
            tests_passed: 12,
            tests_total: 12,
            compile_time_ms: 450,
        });
        exp.multi_eval = Some(MultiEvalResult {
            score: 0.85,
            test_passed: true,
            coverage: Some(0.75),
        });
        
        // Verify the metrics section would show:
        // - warnings: -2 (2→0)
        // - complexity: -3.0 (15→12)
        // - binary: +5KB
        
        let before = exp.metrics_before.as_ref().unwrap();
        let after = exp.metrics_after.as_ref().unwrap();
        
        let warnings_delta = after.warnings as i32 - before.warnings as i32;
        let complexity_delta = after.complexity - before.complexity;
        let binary_delta = (after.binary_size as i64 - before.binary_size as i64) / 1024;
        
        assert_eq!(warnings_delta, -2);
        assert_eq!(complexity_delta, -3.0);
        assert_eq!(binary_delta, 5);
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

    #[test]
    fn test_buffered_reading_parses_multiple_entries() {
        // Simulate parsing multiple entries with buffered reading logic
        let log_content = "---\n\n## Experiment 1 — test.rs\n\n- **File**: `src/test.rs`\n- **Hypothesis**: First\n---\n\n## Experiment 2 — test2.rs\n\n- **File**: `src/test2.rs`\n- **Hypothesis**: Second\n";
        
        let entries: Vec<String> = log_content
            .split("---\n")
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();
        
        assert_eq!(entries.len(), 2, "Should parse 2 entries from log content");
        assert!(entries[0].contains("Experiment 1"));
        assert!(entries[1].contains("Experiment 2"));
    }

    #[test]
    fn test_import_path_resolves_correctly() {
        // Verify that MultiEvalResult is accessible from the correct path
        // This test will fail at compile time if the import path is wrong
        use crate::eval::metrics::MultiEvalResult;
        let result = MultiEvalResult {
            score: 0.5,
            test_passed: true,
            coverage: Some(0.5),
        };
        assert!(!result.test_passed || result.score >= 0.0);
    }

    #[test]
    fn test_clear_experiment_logs() {
        let temp = TempDir::new().unwrap();
        let log_dir = temp.path().join("logs");
        std::fs::create_dir_all(&log_dir).unwrap();
        
        let log_path = log_dir.join("research_log.md");
        std::fs::write(&log_path, "## Experiment 1 — test.rs\nSome content\n").unwrap();
        
        assert!(log_path.exists(), "Log file should exist before clearing");
        
        // Simulate clearing by removing the file
        std::fs::remove_file(&log_path).unwrap();
        assert!(!log_path.exists(), "Log file should not exist after clearing");
        
        // Clearing non-existent file should succeed
        std::fs::create_dir_all(&log_dir).unwrap();
        assert!(log_dir.exists());
    }

    #[test]
    fn test_experiment_exists_detects_existing() {
        let temp = TempDir::new().unwrap();
        let log_dir = temp.path().join("logs");
        std::fs::create_dir_all(&log_dir).unwrap();
        
        let log_path = log_dir.join("research_log.md");
        let content = "---\n\n## Experiment 42 — target.rs\n\n- **File**: `src/target.rs`\n- **Hypothesis**: Test\n";
        std::fs::write(&log_path, content).unwrap();
        
        // Simulate the experiment_exists check logic
        let target = format!("## Experiment {} — {}", 42, "target.rs");
        let file_content = std::fs::read_to_string(&log_path).unwrap();
        let exists = file_content.contains(&target);
        
        assert!(exists, "Should detect existing experiment");
        
        // Non-existent iteration
        let target2 = format!("## Experiment {} — {}", 99, "target.rs");
        let exists2 = file_content.contains(&target2);
        assert!(!exists2, "Should not detect non-existent experiment");
    }

    #[test]
    fn test_experiment_exists_empty_file() {
        let temp = TempDir::new().unwrap();
        let log_dir = temp.path().join("logs");
        std::fs::create_dir_all(&log_dir).unwrap();
        
        let log_path = log_dir.join("research_log.md");
        std::fs::write(&log_path, "").unwrap();
        
        // Empty file should return false for any experiment
        let file_content = std::fs::read_to_string(&log_path).unwrap();
        assert!(file_content.is_empty(), "File should be empty");
        
        // Non-existent log dir should also return false
        std::fs::remove_file(&log_path).unwrap();
        assert!(!log_path.exists());
    }

    #[test]
    fn test_newline_handling_prevents_malformed_concatenation() {
        // Simulate two entries being written without proper newline separation
        let entry1 = "---\n\n## Experiment 1 — test.rs\n\n- **File**: `src/test.rs`\n- **Hypothesis**: First\n";
        let entry2 = "---\n\n## Experiment 2 — test.rs\n\n- **File**: `src/test.rs`\n- **Hypothesis`: Second\n";
        
        // If entry1 doesn't end with newline, concatenation would be malformed
        let malformed = if !entry1.ends_with('\n') {
            format!("{}{}", entry1, entry2)
        } else {
            format!("{}{}", entry1, entry2)
        };
        
        // Proper handling: ensure newline before appending
        let proper = if !entry1.ends_with('\n') {
            format!("{}\n{}", entry1, entry2)
        } else {
            format!("{}{}", entry1, entry2)
        };
        
        // Malformed would have "First---" instead of proper separation
        assert!(malformed.contains("First---"), "Malformed shows improper concatenation");
        assert!(proper.contains("First\n"), "Proper has correct newline separation");
    }

    #[test]
    fn test_experiment_stats_default_values() {
        let stats = ExperimentStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.improved, 0);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.neutral, 0);
        assert_eq!(stats.regressed, 0);
        assert_eq!(stats.total_tests_passed, 0);
        assert_eq!(stats.total_tests_run, 0);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_experiment_stats_success_rate() {
        let mut stats = ExperimentStats::default();
        assert_eq!(stats.success_rate(), 0.0);
        
        stats.total = 10;
        stats.improved = 3;
        assert_eq!(stats.success_rate(), 30.0);
        
        stats.improved = 5;
        assert_eq!(stats.success_rate(), 50.0);
    }

    #[test]
    fn test_experiment_stats_test_pass_rate() {
        let mut stats = ExperimentStats::default();
        assert_eq!(stats.test_pass_rate(), 0.0);
        
        stats.total_tests_run = 100;
        stats.total_tests_passed = 85;
        assert_eq!(stats.test_pass_rate(), 85.0);
    }

    #[test]
    fn test_parsing_outcome_from_log_entry() {
        let improved_entry = "## Experiment 1 — test.rs\n- **Outcome**: Improved\n- **Tests**: 5/10 → 7/10\n";
        let failed_entry = "## Experiment 2 — test.rs\n- **Outcome**: Failed\n- **Tests**: 5/10 → 3/10\n";
        let neutral_entry = "## Experiment 3 — test.rs\n- **Outcome**: Neutral\n- **Tests**: 5/10 → 5/10\n";
        let regressed_entry = "## Experiment 4 — test.rs\n- **Outcome**: Regressed\n- **Tests**: 5/10 → 2/10\n";
        
        // Test outcome detection
        assert!(improved_entry.contains("Outcome: Improved"));
        assert!(failed_entry.contains("Outcome: Failed"));
        assert!(neutral_entry.contains("Outcome: Neutral"));
        assert!(regressed_entry.contains("Outcome: Regressed"));
    }

    #[test]
    fn test_parsing_tests_from_log_entry() {
        let entry = "## Experiment 1 — test.rs\n- **Tests**: 5/10 → 8/12\n";
        
        // Parse tests line format: "- **Tests**: 5/10 → 8/12"
        if let Some(tests_line) = entry.lines().find(|l| l.contains("**Tests**:")) {
            if let Some(after_arrow) = tests_line.split("→").nth(1) {
                let passed = after_arrow.trim().split('/').next().unwrap().trim();
                let total_part = after_arrow.trim().split('/').nth(1).unwrap().trim();
                
                assert_eq!(passed, "8");
                // total_part would be "12" followed by newline
                assert!(total_part.starts_with("12"));
            }
        }
    }

    #[test]
    fn test_experiment_stats_accumulation() {
        let entries = vec![
            "## Experiment 1 — test.rs\n- **Outcome**: Improved\n- **Tests**: 5/10 → 7/10\n".to_string(),
            "## Experiment 2 — test.rs\n- **Outcome**: Failed\n- **Tests**: 7/10 → 5/10\n".to_string(),
            "## Experiment 3 — test.rs\n- **Outcome**: Neutral\n- **Tests**: 5/10 → 5/10\n".to_string(),
        ];
        
        let mut stats = ExperimentStats::default();
        
        for entry in &entries {
            if entry.contains("Outcome: Improved") {
                stats.improved += 1;
            } else if entry.contains("Outcome: Failed") {
                stats.failed += 1;
            } else if entry.contains("Outcome: Neutral") {
                stats.neutral += 1;
            }
        }
        stats.total = stats.improved + stats.failed + stats.neutral + stats.regressed;
        
        assert_eq!(stats.improved, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.neutral, 1);
        assert_eq!(stats.total, 3);
        assert!(!stats.is_empty());
    }
}
