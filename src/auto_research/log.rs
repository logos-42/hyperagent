use anyhow::Result;
use crate::llm::LLMClient;

use super::{AutoResearch, Experiment};

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// 写入实验日志到 markdown 文件（含多维指标 + Phase 3 多文件信息）
    pub(crate) fn append_experiment_log(&self, exp: &Experiment) -> Result<()> {
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
            "## Experiment {} — {}\n\n\
             - **File**: `src/{}`\n\
             {}\
             - **Hypothesis**: {}\n\
             - **Outcome**: {:?}\n\
             - **Tests**: {}/{} → {}/{}\n\
             {}{}{}\
             - **Reflection**: {}\n\
             - **Time**: {}\n\n",
            exp.iteration, exp.file, exp.file,
            files_section,
            exp.hypothesis, exp.outcome,
            exp.tests_before.0, exp.tests_before.1,
            exp.tests_after.0, exp.tests_after.1,
            metrics_section,
            test_section,
            files_section,
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
}
