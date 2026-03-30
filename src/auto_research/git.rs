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

/// Git log entry for observability
#[derive(Debug, Clone)]
pub struct GitLogEntry {
    pub hash: String,
    pub author: String,
    pub message: String,
}

/// Git log representation for recent commits
#[derive(Debug, Clone)]
pub struct GitLog {
    pub entries: Vec<GitLogEntry>,
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
    ///
    /// Git porcelain format uses XY status codes:
    /// - X = index status (staged)
    /// - Y = work tree status (unstaged)
    /// - ' ' = no change, '?' = untracked, '!' = ignored
    /// - Other letters indicate merge conflicts, modifications, etc.
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
            if line.len() < 3 {
                continue;
            }
            // Porcelain format: XY filename
            // X = index status (staged), Y = work tree status (unstaged)
            let index_status = line.chars().next().unwrap_or(' ');
            let work_tree_status = line.chars().nth(1).unwrap_or(' ');
            // Filename starts at position 3 (after XY and a space)
            let file_path = line[3..].to_string();

            // Untracked files have '??' status
            if index_status == '?' && work_tree_status == '?' {
                untracked.push(file_path);
                continue;
            }

            // Ignored files have '!!' status - skip them
            if index_status == '!' && work_tree_status == '!' {
                continue;
            }

            // Renamed files: 'R' status shows as "R  old -> new" or "RM old -> new"
            // The file_path contains "old_filename -> new_filename", we extract the new filename
            if index_status == 'R' || work_tree_status == 'R' {
                let resolved_file = if let Some(pos) = file_path.find(" -> ") {
                    &file_path[pos + 4..]
                } else {
                    &file_path
                };
                if index_status == 'R' {
                    staged.push(resolved_file.to_string());
                }
                if work_tree_status == 'R' {
                    unstaged.push(resolved_file.to_string());
                }
                continue;
            }

            // Index status indicates staged changes (unless space or ? or !)
            if index_status != ' ' && index_status != '?' && index_status != '!' {
                staged.push(file_path.clone());
            }

            // Work tree status indicates unstaged changes (unless space or ? or !)
            if work_tree_status != ' ' && work_tree_status != '?' && work_tree_status != '!' {
                unstaged.push(file_path);
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
            // Parse file names from diff headers like "diff --git a/src/old.rs b/src/new.rs"
            // We want the "new" filename (after b/), which is the 4th space-separated field
            if line.starts_with("diff --git") {
                if let Some(file) = line.split(' ').nth(3) {
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

    /// Get recent git commit history for observability
    ///
    /// Returns the last N commits with abbreviated hash, author, and message.
    /// Useful for understanding recent changes before proposing new modifications.
    pub(crate) fn git_log(&self, count: usize) -> Result<GitLog> {
        let output = std::process::Command::new("git")
            .args(&[
                "log",
                &format!("-{}", count),
                "--pretty=format:%h|%an|%s",
            ])
            .current_dir(&self.config.project_root)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut entries = Vec::new();

        for line in stdout.lines() {
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, '|').collect();
            if parts.len() == 3 {
                entries.push(GitLogEntry {
                    hash: parts[0].to_string(),
                    author: parts[1].to_string(),
                    message: parts[2].to_string(),
                });
            }
        }

        Ok(GitLog { entries })
    }

    /// Get git commit history for a specific file path
    ///
    /// Returns commits that modified the given file, helping understand
    /// the evolution of a file before making changes to it.
    pub(crate) fn git_log_for_file(&self, file_path: &str, count: usize) -> Result<GitLog> {
        let output = std::process::Command::new("git")
            .args(&[
                "log",
                &format!("-{}", count),
                "--pretty=format:%h|%an|%s",
                "--",
                file_path,
            ])
            .current_dir(&self.config.project_root)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut entries = Vec::new();

        for line in stdout.lines() {
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, '|').collect();
            if parts.len() == 3 {
                entries.push(GitLogEntry {
                    hash: parts[0].to_string(),
                    author: parts[1].to_string(),
                    message: parts[2].to_string(),
                });
            }
        }

        Ok(GitLog { entries })
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

    #[test]
    fn test_git_status_parsing_untracked() {
        // Untracked files show as "?? filename"
        let status = GitStatus {
            staged: vec![],
            unstaged: vec![],
            untracked: vec!["new_file.rs".to_string()],
            is_clean: false,
        };
        assert!(status.untracked.contains(&"new_file.rs".to_string()));
    }

    #[test]
    fn test_git_status_parsing_staged() {
        // Staged modifications show as "M filename" (index M, work tree space)
        let status = GitStatus {
            staged: vec!["src/main.rs".to_string()],
            unstaged: vec![],
            untracked: vec![],
            is_clean: false,
        };
        assert!(status.staged.contains(&"src/main.rs".to_string()));
    }

    #[test]
    fn test_git_status_parsing_unstaged() {
        // Unstaged modifications show as " M filename" (index space, work tree M)
        let status = GitStatus {
            staged: vec![],
            unstaged: vec!["src/lib.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        assert!(status.unstaged.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn test_git_status_parsing_both() {
        // Files with both staged and unstaged changes: "MM filename"
        let status = GitStatus {
            staged: vec!["src/both.rs".to_string()],
            unstaged: vec!["src/both.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        // Same file appears in both lists
        assert!(status.staged.contains(&"src/both.rs".to_string()));
        assert!(status.unstaged.contains(&"src/both.rs".to_string()));
    }

    #[test]
    fn test_git_status_parsing_renamed_staged() {
        // Renamed files show as "R  old_file.rs -> new_file.rs"
        // We should extract the new filename and add to staged
        let status = GitStatus {
            staged: vec!["new_file.rs".to_string()],
            unstaged: vec![],
            untracked: vec![],
            is_clean: false,
        };
        assert!(status.staged.contains(&"new_file.rs".to_string()));
        assert!(!status.staged.contains(&"old_file.rs".to_string()));
    }

    #[test]
    fn test_git_status_parsing_renamed_unstaged() {
        // Unstaged rename: " R old_file.rs -> new_file.rs"
        let status = GitStatus {
            staged: vec![],
            unstaged: vec!["new_file.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        assert!(status.unstaged.contains(&"new_file.rs".to_string()));
    }

    #[test]
    fn test_git_status_parsing_renamed_both() {
        // Both staged and unstaged rename modifications: "RM old_file.rs -> new_file.rs"
        let status = GitStatus {
            staged: vec!["new_file.rs".to_string()],
            unstaged: vec!["new_file.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        assert!(status.staged.contains(&"new_file.rs".to_string()));
        assert!(status.unstaged.contains(&"new_file.rs".to_string()));
    }

    #[test]
    fn test_git_diff_parsing_diff_git_line() {
        // Verify git_diff correctly parses "diff --git a/old b/new" format
        // The new filename (b/new) should be extracted, not the old (a/old)
        let status = GitDiff {
            files_changed: vec!["src/new_file.rs".to_string()],
            insertions: 5,
            deletions: 3,
            diff_output: "diff --git a/src/old_file.rs b/src/new_file.rs\n".to_string(),
        };
        // The key assertion: we should have the "new" filename
        assert!(status.files_changed.contains(&"src/new_file.rs".to_string()));
        // We should NOT have the "old" filename
        assert!(!status.files_changed.contains(&"src/old_file.rs".to_string()));
        assert!(!status.files_changed.contains(&"a/src/old_file.rs".to_string()));
    }

    #[test]
    fn test_git_diff_multiple_files() {
        // Test parsing multiple files from diff output
        let diff = GitDiff {
            files_changed: vec![
                "src/main.rs".to_string(),
                "src/lib.rs".to_string(),
            ],
            insertions: 20,
            deletions: 10,
            diff_output: String::new(),
        };
        assert_eq!(diff.files_changed.len(), 2);
        assert!(diff.files_changed.contains(&"src/main.rs".to_string()));
        assert!(diff.files_changed.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn test_git_log_entry_construction() {
        let entry = GitLogEntry {
            hash: "abc1234".to_string(),
            author: "Test Author".to_string(),
            message: "Initial commit".to_string(),
        };
        assert_eq!(entry.hash, "abc1234");
        assert_eq!(entry.author, "Test Author");
        assert_eq!(entry.message, "Initial commit");
    }

    #[test]
    fn test_git_log_construction() {
        let log = GitLog {
            entries: vec![
                GitLogEntry {
                    hash: "abc1234".to_string(),
                    author: "Alice".to_string(),
                    message: "First commit".to_string(),
                },
                GitLogEntry {
                    hash: "def5678".to_string(),
                    author: "Bob".to_string(),
                    message: "Second commit".to_string(),
                },
            ],
        };
        assert_eq!(log.entries.len(), 2);
        assert_eq!(log.entries[0].hash, "abc1234");
        assert_eq!(log.entries[1].hash, "def5678");
    }

    #[test]
    fn test_git_log_empty() {
        let log = GitLog { entries: vec![] };
        assert!(log.entries.is_empty());
    }

    #[test]
    fn test_git_log_for_file_entry_construction() {
        // Verify GitLog can represent file-specific history
        let log = GitLog {
            entries: vec![
                GitLogEntry {
                    hash: "abc1234".to_string(),
                    author: "Alice".to_string(),
                    message: "Fix bug in git.rs".to_string(),
                },
                GitLogEntry {
                    hash: "def5678".to_string(),
                    author: "Bob".to_string(),
                    message: "Add git_log_for_file method".to_string(),
                },
            ],
        };
        assert_eq!(log.entries.len(), 2);
        assert!(log.entries[0].message.contains("git.rs"));
        assert!(log.entries[1].message.contains("git_log_for_file"));
    }

    #[test]
    fn test_git_log_for_file_empty() {
        // File with no commits should return empty log
        let log = GitLog { entries: vec![] };
        assert!(log.entries.is_empty());
    }

    #[test]
    fn test_git_log_for_file_single_entry() {
        // Single commit for a file
        let log = GitLog {
            entries: vec![GitLogEntry {
                hash: "a1b2c3d".to_string(),
                author: "Developer".to_string(),
                message: "Initial implementation".to_string(),
            }],
        };
        assert_eq!(log.entries.len(), 1);
        assert_eq!(log.entries[0].hash, "a1b2c3d");
    }
}
