use anyhow::Result;
use crate::llm::LLMClient;

use super::{AutoResearch, Experiment, ExperimentOutcome};

/// Statistics aggregated from experiment logs
#[derive(Debug, Clone, Default)]
pub struct ExperimentStats {
    pub total: usize,
    pub improved: usize,
    pub failed: usize,
    pub neutral: usize,
    pub regressed: usize,
    pub total_tests_passed: usize,
    pub total_tests_run: usize,
}

/// Delta between two experiment statistics snapshots
#[derive(Debug, Clone)]
pub struct StatsDelta {
    pub total_delta: i64,
    pub improved_delta: i64,
    pub failed_delta: i64,
    pub neutral_delta: i64,
    pub regressed_delta: i64,
    pub tests_passed_delta: i64,
    pub tests_run_delta: i64,
    pub success_rate_delta: f64,
    pub test_pass_rate_delta: f64,
}

impl ExperimentStats {
    /// Returns true if no experiments have been recorded
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Returns the percentage of improved experiments out of total
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.improved as f64 / self.total as f64) * 100.0
        }
    }

    /// Returns the percentage of tests passed across all experiments
    pub fn test_pass_rate(&self) -> f64 {
        if self.total_tests_run == 0 {
            0.0
        } else {
            (self.total_tests_passed as f64 / self.total_tests_run as f64) * 100.0
        }
    }

    /// Returns a human-readable one-line summary of the statistics
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "No experiments recorded".to_string();
        }
        
        let success_pct = self.success_rate();
        let test_pct = self.test_pass_rate();
        
        format!(
            "{} experiments: {} improved ({:.0}%), {} failed, {} neutral, {} regressed, {:.0}% tests passing ({}/{})",
            self.total,
            self.improved,
            success_pct,
            self.failed,
            self.neutral,
            self.regressed,
            test_pct,
            self.total_tests_passed,
            self.total_tests_run
        )
    }

    /// Compare this stats snapshot with another, computing the delta
    pub fn compare(&self, other: &ExperimentStats) -> StatsDelta {
        StatsDelta {
            total_delta: self.total as i64 - other.total as i64,
            improved_delta: self.improved as i64 - other.improved as i64,
            failed_delta: self.failed as i64 - other.failed as i64,
            neutral_delta: self.neutral as i64 - other.neutral as i64,
            regressed_delta: self.regressed as i64 - other.regressed as i64,
            tests_passed_delta: self.total_tests_passed as i64 - other.total_tests_passed as i64,
            tests_run_delta: self.total_tests_run as i64 - other.total_tests_run as i64,
            success_rate_delta: self.success_rate() - other.success_rate(),
            test_pass_rate_delta: self.test_pass_rate() - other.test_pass_rate(),
        }
    }
}

impl StatsDelta {
    /// Returns true if the delta shows overall improvement
    pub fn is_improvement(&self) -> bool {
        self.improved_delta > 0 && self.success_rate_delta > 0.0
    }

    /// Returns true if the delta shows regression
    pub fn is_regression(&self) -> bool {
        self.regressed_delta > 0 || self.success_rate_delta < 0.0
    }

    /// Returns a net score combining all deltas (positive = improvement)
    pub fn net_score(&self) -> f64 {
        (self.improved_delta as f64 * 1.0)
            + (self.regressed_delta as f64 * -2.0)
            + (self.failed_delta as f64 * -1.5)
            + (self.success_rate_delta * 0.5)
            + (self.test_pass_rate_delta * 0.3)
    }
}

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

    /// Get aggregate statistics from all experiment logs
    /// Parses log entries to count outcomes and test results
    pub fn get_experiment_stats(&self) -> Result<ExperimentStats> {
        let entries = self.read_experiment_logs()?;
        let mut stats = ExperimentStats::default();

        for entry in &entries {
            stats.total += 1;

            // Parse outcome
            if entry.contains("Outcome: Improved") {
                stats.improved += 1;
            } else if entry.contains("Outcome: Failed") {
                stats.failed += 1;
            } else if entry.contains("Outcome: Neutral") {
                stats.neutral += 1;
            } else if entry.contains("Outcome: Regressed") {
                stats.regressed += 1;
            }

            // Parse tests: "Tests: X/Y → Z/W"
            if let Some(tests_line) = entry.lines().find(|l| l.contains("**Tests**:")) {
                if let Some(after_arrow) = tests_line.split("→").nth(1) {
                    let parts: Vec<&str> = after_arrow.trim().split('/').collect();
                    if parts.len() >= 2 {
                        if let Ok(passed) = parts[0].trim().parse::<usize>() {
                            stats.total_tests_passed += passed;
                        }
                        // Extract total from second part (may have trailing content)
                        let total_str = parts[1].split_whitespace().next().unwrap_or("0");
                        if let Ok(total) = total_str.parse::<usize>() {
                            stats.total_tests_run += total;
                        }
                    }
                }
            }
        }

        Ok(stats)
    }
}
