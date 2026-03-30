use anyhow::Result;

use crate::llm::LLMClient;

use super::AutoResearch;

/// Git status representation for observability
#[derive(Debug, Clone)]
pub struct GitStatus {
    pub staged: Vec<String>,
    pub unstaged: Vec<String>,
    pub untracked: Vec<String>,
    pub is_clean: bool,
}

/// Git diff representation
#[derive(Debug, Clone)]
pub struct GitDiff {
    pub files_changed: Vec<String>,
    pub insertions: u32,
    pub deletions: u32,
    pub diff_output: String,
}

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

    /// Get current git status for observability
    pub(crate) fn git_status(&self) -> Result<GitStatus> {
        let output = std::process::Command::new("git")
            .args(&["status", "--porcelain"])
            .current_dir(&self.config.project_root)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut staged = Vec::new();
        let mut unstaged = Vec::new();
        let mut untracked = Vec::new();

        for line in stdout.lines() {
            if line.len() < 2 {
                continue;
            }
            let index_status = line.chars().next().unwrap_or(' ');
            let work_tree_status = line.chars().nth(1).unwrap_or(' ');
            let file_path = line[3..].to_string();

            match (index_status, work_tree_status) {
                ('?', '?') => untracked.push(file_path),
                (' ', ' ') => {} // clean file, not displayed in porcelain
                (' ', _) | (_, ' ') if index_status != ' ' && work_tree_status == ' ' => {
                    staged.push(file_path.clone());
                }
                (_, ' ') if index_status != ' ' => staged.push(file_path),
                (' ', _) if work_tree_status != ' ' => unstaged.push(file_path),
                _ => {
                    // Both have changes - staged and unstaged portions
                    staged.push(format!("{} (staged)", file_path));
                    unstaged.push(format!("{} (unstaged)", file_path));
                }
            }
        }

        let is_clean = staged.is_empty() && unstaged.is_empty() && untracked.is_empty();

        Ok(GitStatus {
            staged,
            unstaged,
            untracked,
            is_clean,
        })
    }

    /// Get git diff statistics and content for observability
    pub(crate) fn git_diff(&self, staged_only: bool) -> Result<GitDiff> {
        let args = if staged_only {
            vec!["diff", "--cached", "--stat", "--patch"]
        } else {
            vec!["diff", "--stat", "--patch"]
        };

        let output = std::process::Command::new("git")
            .args(&args)
            .current_dir(&self.config.project_root)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let diff_output = stdout.to_string();

        // Parse files changed from diff output
        let mut files_changed = Vec::new();
        let mut insertions = 0;
        let mut deletions = 0;

        // Look for summary line like: "3 files changed, 10 insertions(+), 5 deletions(-)"
        for line in stdout.lines() {
            if line.contains("files changed") || line.contains("file changed") {
                let parts: Vec<&str> = line.split(',').collect();
                for part in parts {
                    let trimmed = part.trim();
                    if trimmed.contains("insertion") {
                        insertions = trimmed
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                    } else if trimmed.contains("deletion") {
                        deletions = trimmed
                            .split_whitespace()
                            .next()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0);
                    }
                }
            }
            // Parse file names from diff headers like "--- a/src/file.rs"
            if line.starts_with("diff --git") {
                if let Some(file) = line.split(' ').nth(2) {
                    let file = file.strip_prefix("b/").unwrap_or(file);
                    if !files_changed.contains(&file.to_string()) {
                        files_changed.push(file.to_string());
                    }
                }
            }
        }

        Ok(GitDiff {
            files_changed,
            insertions,
            deletions,
            diff_output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_git_status_parse() {
        // Test that GitStatus can be constructed
        let status = GitStatus {
            staged: vec!["src/main.rs".to_string()],
            unstaged: vec!["src/lib.rs".to_string()],
            untracked: vec!["new_file.rs".to_string()],
            is_clean: false,
        };

        assert!(!status.is_clean);
        assert_eq!(status.staged.len(), 1);
        assert_eq!(status.unstaged.len(), 1);
        assert_eq!(status.untracked.len(), 1);
    }

    #[test]
    fn test_git_status_clean() {
        let status = GitStatus {
            staged: vec![],
            unstaged: vec![],
            untracked: vec![],
            is_clean: true,
        };

        assert!(status.is_clean);
    }

    #[test]
    fn test_git_diff_parse() {
        let diff = GitDiff {
            files_changed: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            insertions: 10,
            deletions: 5,
            diff_output: String::new(),
        };

        assert_eq!(diff.files_changed.len(), 2);
        assert_eq!(diff.insertions, 10);
        assert_eq!(diff.deletions, 5);
    }

    #[test]
    fn test_git_diff_empty() {
        let diff = GitDiff {
            files_changed: vec![],
            insertions: 0,
            deletions: 0,
            diff_output: String::new(),
        };

        assert!(diff.files_changed.is_empty());
    }
}
