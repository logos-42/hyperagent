use anyhow::Result;

use crate::llm::LLMClient;

use super::AutoResearch;

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// git checkout 回滚
    pub(crate) fn git_revert(&self, file: &str) -> Result<()> {
        // 支持项目根目录下的任意文件（不仅是 src/）
        let path = if file.starts_with("src/") || file.contains('/') {
            file.to_string()
        } else {
            format!("src/{}", file)
        };
        std::process::Command::new("git")
            .args(&["checkout", "--", &path])
            .current_dir(&self.config.project_root)
            .output()?;
        Ok(())
    }

    /// git commit
    pub(crate) fn git_commit(&self, msg: &str) -> Result<()> {
        std::process::Command::new("git")
            .args(&["add", "-A"])
            .current_dir(&self.config.project_root)
            .output()?;

        let output = std::process::Command::new("git")
            .args(&["commit", "-m", msg])
            .current_dir(&self.config.project_root)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("nothing to commit") {
                tracing::warn!("git commit: {}", stderr);
            }
        }
        Ok(())
    }

    /// git push
    pub(crate) fn git_push(&self) -> Result<()> {
        let output = std::process::Command::new("git")
            .args(&["push", "origin", "HEAD"])
            .current_dir(&self.config.project_root)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("git push failed: {}", stderr);
        }
        Ok(())
    }

    /// Phase 5: Git checkpoint — 修改核心文件前打标签
    pub(crate) fn git_checkpoint(&self, iteration: u32) -> Result<()> {
        let tag = format!("checkpoint-{}", iteration);
        let _ = std::process::Command::new("git")
            .args(&["tag", "-f", &tag])
            .current_dir(&self.config.project_root)
            .output();
        tracing::info!("  Checkpoint: {}", tag);
        Ok(())
    }

    /// Phase 5: 回滚到上一个 checkpoint
    pub(crate) fn git_rollback(&self, iteration: u32) -> Result<()> {
        let tag = format!("checkpoint-{}", iteration);
        let output = std::process::Command::new("git")
            .args(&["reset", "--hard", &tag])
            .current_dir(&self.config.project_root)
            .output()?;
        if output.status.success() {
            tracing::warn!("  Rolled back to {}", tag);
        }
        Ok(())
    }
}
