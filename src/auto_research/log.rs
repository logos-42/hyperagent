use anyhow::Result;
use crate::llm::LLMClient;
use chrono::{DateTime, Utc, Duration};

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

    /// Returns a human-readable trend description based on the net score
    /// Useful for quickly communicating research momentum direction
    pub fn trend(&self) -> &'static str {
        let score = self.net_score();
        if score > 1.0 {
            "improving"
        } else if score < -1.0 {
            "declining"
        } else {
            "stable"
        }
    }

    /// Returns a human-readable one-line summary of the statistics delta
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        if self.total_delta != 0 {
            parts.push(format!("{:+} experiments", self.total_delta));
        }
        if self.improved_delta != 0 {
            parts.push(format!("{:+} improved", self.improved_delta));
        }
        if self.regressed_delta != 0 {
            parts.push(format!("{:+} regressed", self.regressed_delta));
        }
        if self.failed_delta != 0 {
            parts.push(format!("{:+} failed", self.failed_delta));
        }
        if self.success_rate_delta.abs() > 0.01 {
            parts.push(format!("{:+.1}% success rate", self.success_rate_delta));
        }
        if self.test_pass_rate_delta.abs() > 0.01 {
            parts.push(format!("{:+.1}% test pass rate", self.test_pass_rate_delta));
        }

        if parts.is_empty() {
            "no change".to_string()
        } else {
            parts.join(", ")
        }
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
            if let Some(entry_stats) = self.parse_entry_stats(entry) {
                stats.total += entry_stats.total;
                stats.improved += entry_stats.improved;
                stats.failed += entry_stats.failed;
                stats.neutral += entry_stats.neutral;
                stats.regressed += entry_stats.regressed;
                stats.total_tests_passed += entry_stats.total_tests_passed;
                stats.total_tests_run += entry_stats.total_tests_run;
            }
        }

        Ok(stats)
    }

    /// Get statistics for experiments within a time window (last N hours)
    /// Useful for detecting recent research momentum and trends
    pub fn time_window_stats(&self, hours: u64) -> Result<ExperimentStats> {
        let entries = self.read_experiment_logs()?;
        let cutoff = Utc::now() - Duration::hours(hours as i64);
        let mut stats = ExperimentStats::default();

        for entry in &entries {
            // Parse timestamp from entry: "- **Time**: 2024-01-15T10:30:00Z"
            if let Some(timestamp) = self.parse_entry_timestamp(entry) {
                if timestamp >= cutoff {
                    if let Some(entry_stats) = self.parse_entry_stats(entry) {
                        stats.total += entry_stats.total;
                        stats.improved += entry_stats.improved;
                        stats.failed += entry_stats.failed;
                        stats.neutral += entry_stats.neutral;
                        stats.regressed += entry_stats.regressed;
                        stats.total_tests_passed += entry_stats.total_tests_passed;
                        stats.total_tests_run += entry_stats.total_tests_run;
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Parse timestamp from experiment log entry
    fn parse_entry_timestamp(&self, entry: &str) -> Option<DateTime<Utc>> {
        entry.lines()
            .find(|l| l.contains("**Time**:"))
            .and_then(|line| {
                // Extract timestamp after "- **Time**: "
                line.split("**Time**:").nth(1)
                    .map(|s| s.trim())
                    .and_then(|ts| DateTime::parse_from_rfc3339(ts).ok())
                    .map(|dt| dt.with_timezone(&Utc))
            })
    }

    /// Parse statistics from a single experiment log entry
    fn parse_entry_stats(&self, entry: &str) -> Option<ExperimentStats> {
        let mut stats = ExperimentStats::default();
        stats.total = 1;

        // Parse outcome
        if entry.contains("Outcome: Improved") {
            stats.improved = 1;
        } else if entry.contains("Outcome: Failed") {
            stats.failed = 1;
        } else if entry.contains("Outcome: Neutral") {
            stats.neutral = 1;
        } else if entry.contains("Outcome: Regressed") {
            stats.regressed = 1;
        } else {
            return None;
        }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_delta_summary_no_change() {
        let delta = StatsDelta {
            total_delta: 0,
            improved_delta: 0,
            failed_delta: 0,
            neutral_delta: 0,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: 0.0,
            test_pass_rate_delta: 0.0,
        };
        assert_eq!(delta.summary(), "no change");
    }

    #[test]
    fn test_stats_delta_summary_with_improvements() {
        let delta = StatsDelta {
            total_delta: 5,
            improved_delta: 3,
            failed_delta: 0,
            neutral_delta: 2,
            regressed_delta: 0,
            tests_passed_delta: 15,
            tests_run_delta: 15,
            success_rate_delta: 12.5,
            test_pass_rate_delta: 5.0,
        };
        assert_eq!(delta.summary(), "+5 experiments, +3 improved, +12.5% success rate, +5.0% test pass rate");
    }

    #[test]
    fn test_stats_delta_summary_with_regressions() {
        let delta = StatsDelta {
            total_delta: 3,
            improved_delta: 0,
            failed_delta: 1,
            neutral_delta: 1,
            regressed_delta: 2,
            tests_passed_delta: -5,
            tests_run_delta: 0,
            success_rate_delta: -8.3,
            test_pass_rate_delta: -3.2,
        };
        assert_eq!(delta.summary(), "+3 experiments, +2 regressed, +1 failed, -8.3% success rate, -3.2% test pass rate");
    }

    #[test]
    fn test_stats_delta_summary_partial_fields() {
        let delta = StatsDelta {
            total_delta: 0,
            improved_delta: 2,
            failed_delta: 0,
            neutral_delta: 0,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: 5.0,
            test_pass_rate_delta: 0.0,
        };
        assert_eq!(delta.summary(), "+2 improved, +5.0% success rate");
    }

    #[test]
    fn test_stats_delta_is_improvement() {
        let improving = StatsDelta {
            total_delta: 5,
            improved_delta: 3,
            failed_delta: 0,
            neutral_delta: 2,
            regressed_delta: 0,
            tests_passed_delta: 10,
            tests_run_delta: 10,
            success_rate_delta: 10.0,
            test_pass_rate_delta: 5.0,
        };
        assert!(improving.is_improvement());

        let not_improving = StatsDelta {
            total_delta: 2,
            improved_delta: 0,
            failed_delta: 1,
            neutral_delta: 1,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: -5.0,
            test_pass_rate_delta: 0.0,
        };
        assert!(!not_improving.is_improvement());
    }

    #[test]
    fn test_stats_delta_is_regression() {
        let regressing = StatsDelta {
            total_delta: 3,
            improved_delta: 0,
            failed_delta: 1,
            neutral_delta: 1,
            regressed_delta: 2,
            tests_passed_delta: -5,
            tests_run_delta: 0,
            success_rate_delta: -10.0,
            test_pass_rate_delta: 0.0,
        };
        assert!(regressing.is_regression());

        let not_regressing = StatsDelta {
            total_delta: 5,
            improved_delta: 3,
            failed_delta: 0,
            neutral_delta: 2,
            regressed_delta: 0,
            tests_passed_delta: 10,
            tests_run_delta: 10,
            success_rate_delta: 10.0,
            test_pass_rate_delta: 5.0,
        };
        assert!(!not_regressing.is_regression());
    }

    #[test]
    fn test_experiment_stats_summary_empty() {
        let stats = ExperimentStats::default();
        assert_eq!(stats.summary(), "No experiments recorded");
    }

    #[test]
    fn test_experiment_stats_summary_with_data() {
        let stats = ExperimentStats {
            total: 10,
            improved: 4,
            failed: 2,
            neutral: 3,
            regressed: 1,
            total_tests_passed: 45,
            total_tests_run: 50,
        };
        let summary = stats.summary();
        assert!(summary.contains("10 experiments"));
        assert!(summary.contains("4 improved"));
        assert!(summary.contains("40%")); // success rate
        assert!(summary.contains("90%")); // test pass rate
    }

    #[test]
    fn test_stats_delta_trend_improving() {
        let delta = StatsDelta {
            total_delta: 5,
            improved_delta: 4,
            failed_delta: 0,
            neutral_delta: 1,
            regressed_delta: 0,
            tests_passed_delta: 20,
            tests_run_delta: 20,
            success_rate_delta: 15.0,
            test_pass_rate_delta: 10.0,
        };
        // net_score = 4*1.0 + 0*-2.0 + 0*-1.5 + 15.0*0.5 + 10.0*0.3 = 4 + 7.5 + 3 = 14.5
        assert_eq!(delta.trend(), "improving");
    }

    #[test]
    fn test_stats_delta_trend_declining() {
        let delta = StatsDelta {
            total_delta: 3,
            improved_delta: 0,
            failed_delta: 2,
            neutral_delta: 0,
            regressed_delta: 2,
            tests_passed_delta: -10,
            tests_run_delta: 0,
            success_rate_delta: -5.0,
            test_pass_rate_delta: -2.0,
        };
        // net_score = 0*1.0 + 2*-2.0 + 2*-1.5 + (-5.0)*0.5 + (-2.0)*0.3 = -4 - 3 - 2.5 - 0.6 = -10.1
        assert_eq!(delta.trend(), "declining");
    }

    #[test]
    fn test_stats_delta_trend_stable() {
        let delta = StatsDelta {
            total_delta: 2,
            improved_delta: 1,
            failed_delta: 0,
            neutral_delta: 1,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: 0.5,
            test_pass_rate_delta: 0.0,
        };
        // net_score = 1*1.0 + 0 + 0 + 0.5*0.5 + 0 = 1.25, but still close to threshold
        // Let's use a clearly stable case
        let stable_delta = StatsDelta {
            total_delta: 1,
            improved_delta: 0,
            failed_delta: 0,
            neutral_delta: 1,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: 0.0,
            test_pass_rate_delta: 0.0,
        };
        // net_score = 0, clearly stable
        assert_eq!(stable_delta.trend(), "stable");
    }

    #[test]
    fn test_stats_delta_trend_threshold_boundary() {
        // Test the exact threshold at 1.0
        let just_above = StatsDelta {
            total_delta: 0,
            improved_delta: 1,
            failed_delta: 0,
            neutral_delta: 0,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: 0.1,
            test_pass_rate_delta: 0.0,
        };
        // net_score = 1.0 + 0.05 = 1.05 > 1.0
        assert_eq!(just_above.trend(), "improving");

        let just_below = StatsDelta {
            total_delta: 0,
            improved_delta: 0,
            failed_delta: 0,
            neutral_delta: 1,
            regressed_delta: 0,
            tests_passed_delta: 0,
            tests_run_delta: 0,
            success_rate_delta: -0.5,
            test_pass_rate_delta: 0.0,
        };
        // net_score = -0.25, which is > -1.0, so stable
        assert_eq!(just_below.trend(), "stable");
    }

    #[test]
    fn test_parse_entry_timestamp_valid() {
        let research = create_test_research();
        let entry = "## Experiment 1 — test.rs\n- **Time**: 2024-01-15T10:30:00Z\n";
        let ts = research.parse_entry_timestamp(entry);
        assert!(ts.is_some());
    }

    #[test]
    fn test_parse_entry_timestamp_missing() {
        let research = create_test_research();
        let entry = "## Experiment 1 — test.rs\nNo timestamp here\n";
        let ts = research.parse_entry_timestamp(entry);
        assert!(ts.is_none());
    }

    #[test]
    fn test_parse_entry_stats_improved() {
        let research = create_test_research();
        let entry = "## Experiment 1 — test.rs\n- **Outcome**: Improved\n- **Tests**: 3/5 → 5/5\n";
        let stats = research.parse_entry_stats(entry);
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert_eq!(s.total, 1);
        assert_eq!(s.improved, 1);
        assert_eq!(s.total_tests_passed, 5);
        assert_eq!(s.total_tests_run, 5);
    }

    #[test]
    fn test_parse_entry_stats_failed() {
        let research = create_test_research();
        let entry = "## Experiment 2 — test.rs\n- **Outcome**: Failed\n";
        let stats = research.parse_entry_stats(entry);
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert_eq!(s.failed, 1);
    }

    #[test]
    fn test_parse_entry_stats_regressed() {
        let research = create_test_research();
        let entry = "## Experiment 3 — test.rs\n- **Outcome**: Regressed\n";
        let stats = research.parse_entry_stats(entry);
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert_eq!(s.regressed, 1);
    }

    #[test]
    fn test_parse_entry_stats_neutral() {
        let research = create_test_research();
        let entry = "## Experiment 4 — test.rs\n- **Outcome**: Neutral\n";
        let stats = research.parse_entry_stats(entry);
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert_eq!(s.neutral, 1);
    }

    

    /// Helper to create a test AutoResearch instance
    fn create_test_research() -> crate::auto_research::AutoResearch<crate::llm::OpenAI> {
        use crate::auto_research::ResearchConfig;
        use crate::llm::OpenAI;
        use std::path::PathBuf;
        
        let config = ResearchConfig {
            experiment_log_dir: PathBuf::from("/tmp/test_research_logs"),
            ..Default::default()
        };
        crate::auto_research::AutoResearch::new(OpenAI::default(), config)
    }
}
        // Parse tests: "Tests: X/Y → Z/W"
        if let Some(tests_line) = entry.lines().find(|l| l.contains("**Tests**:")) {
            if let Some(after_arrow) = tests_line.split("→").nth(1) {
                let parts: Vec<&str> = after_arrow.trim().split('/').collect();
                if parts.len() >= 2 {
                    if let Ok(passed) = parts[0].trim().parse::<usize>() {
                        stats.total_tests_passed = passed;
                    }
                    // Extract total from second part (may have trailing content)
                    let total_str = parts[1].split_whitespace().next().unwrap_or("0");
                    if let Ok(total) = total_str.parse::<usize>() {
                        stats.total_tests_run = total;
                    }
                }
            }
        }

        Some(stats)
    }
}
