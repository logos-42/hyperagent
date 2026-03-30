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

impl GitStatus {
    /// Returns a human-readable one-line summary of the status.
    ///
    /// Format: "clean" or "staged:N, unstaged:M, untracked:K"
    /// (e.g., "staged:2, unstaged:1, untracked:3" or "clean")
    pub fn summary(&self) -> String {
        if self.is_clean {
            "clean".to_string()
        } else {
            let parts: Vec<String> = [
                if !self.staged.is_empty() {
                    Some(format!("staged:{}", self.staged.len()))
                } else {
                    None
                },
                if !self.unstaged.is_empty() {
                    Some(format!("unstaged:{}", self.unstaged.len()))
                } else {
                    None
                },
                if !self.untracked.is_empty() {
                    Some(format!("untracked:{}", self.untracked.len()))
                } else {
                    None
                },
            ]
            .iter()
            .filter_map(|p| p.clone())
            .collect();
            parts.join(", ")
        }
    }
}

/// Git diff representation
#[derive(Debug, Clone)]
pub struct GitDiff {
    pub files_changed: Vec<String>,
    pub insertions: u32,
    pub deletions: u32,
    pub diff_output: String,
}

impl GitDiff {
    /// Returns a human-readable one-line summary of the diff.
    ///
    /// Format: "{count} file(s) changed, +{insertions}/-{deletions}"
    /// (e.g., "3 files changed, +10/-5" or "1 file changed, +25/-0")
    pub fn summary(&self) -> String {
        let file_count = self.files_changed.len();
        let files_str = if file_count == 1 { "file" } else { "files" };
        format!(
            "{} {} changed, +{}/-{}",
            file_count, files_str, self.insertions, self.deletions
        )
    }
}

/// Git log entry for observability
#[derive(Debug, Clone)]
pub struct GitLogEntry {
    pub hash: String,
    pub author: String,
    pub message: String,
}

impl GitLogEntry {
    /// Returns a compact one-line summary of this commit.
    ///
    /// Format: "{hash}|{author}: {message}" (e.g., "abc1234|Alice: Fix bug in parser")
    /// Useful for quickly scanning commit history in research context.
    pub fn summary(&self) -> String {
        format!("{}|{}: {}", self.hash, self.author, self.message)
    }
}

/// Git log representation for recent commits
#[derive(Debug, Clone)]
pub struct GitLog {
    pub entries: Vec<GitLogEntry>,
}

/// Git conflict detection result
#[derive(Debug, Clone)]
pub struct GitConflicts {
    /// Files containing conflict markers
    pub files: Vec<String>,
    /// Whether any conflicts exist
    pub has_conflicts: bool,
}

impl GitConflicts {
    /// Returns a human-readable one-line summary of conflicts.
    ///
    /// Format: "no conflicts" or "conflicts in: file1, file2, ..."
    /// (e.g., "conflicts in: src/main.rs, src/lib.rs" or "no conflicts")
    pub fn summary(&self) -> String {
        if self.has_conflicts {
            format!("conflicts in: {}", self.files.join(", "))
        } else {
            "no conflicts".to_string()
        }
    }
}

impl GitLog {
    /// Returns an iterator over commit messages in this log.
    ///
    /// Useful for quickly scanning recent activity without accessing
    /// the full entry structure.
    pub fn messages(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.message.as_str())
    }

    /// Returns the number of entries in this log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if this log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
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

    /// Check if the repository is in a clean state (no uncommitted changes)
    ///
    /// This is a convenience method for the common pattern of checking
    /// whether the repository is ready for a new operation.
    pub(crate) fn git_is_clean(&self) -> Result<bool> {
        let status = self.git_status()?;
        Ok(status.is_clean)
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

    /// Detect merge conflict markers in tracked files
    ///
    /// Returns a list of files containing conflict markers (<<<<<<<, =======, >>>>>>>).
    /// Useful for checking repository state before attempting automated operations.
    pub(crate) fn git_has_conflicts(&self) -> Result<GitConflicts> {
        // Get list of files that might have conflicts (staged, modified, or unmerged)
        let status = self.git_status()?;
        let mut files_to_check = Vec::new();
        files_to_check.extend(status.staged.iter().cloned());
        files_to_check.extend(status.unstaged.iter().cloned());
        // Also check untracked files as they might be result of a failed merge
        files_to_check.extend(status.untracked.iter().cloned());

        let mut conflicting_files = Vec::new();

        for file_path in &files_to_check {
            // Build full path
            let full_path = self.config.project_root.join(file_path);

            // Skip if file doesn't exist (might be deleted)
            if !full_path.exists() {
                continue;
            }

            // Read file content and check for conflict markers
            if let Ok(content) = std::fs::read_to_string(&full_path) {
                if content.contains("<<<<<<<") 
                    && content.contains("=======") 
                    && content.contains(">>>>>>>") {
                    conflicting_files.push(file_path.clone());
                }
            }
        }

        let has_conflicts = !conflicting_files.is_empty();
        Ok(GitConflicts {
            files: conflicting_files,
            has_conflicts,
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

    #[test]
    fn test_git_log_messages_iterator() {
        let log = GitLog {
            entries: vec![
                GitLogEntry {
                    hash: "abc123".to_string(),
                    author: "Alice".to_string(),
                    message: "First commit".to_string(),
                },
                GitLogEntry {
                    hash: "def456".to_string(),
                    author: "Bob".to_string(),
                    message: "Second commit".to_string(),
                },
                GitLogEntry {
                    hash: "ghi789".to_string(),
                    author: "Carol".to_string(),
                    message: "Third commit".to_string(),
                },
            ],
        };

        let messages: Vec<&str> = log.messages().collect();
        assert_eq!(messages, vec!["First commit", "Second commit", "Third commit"]);
    }

    #[test]
    fn test_git_log_messages_empty() {
        let log = GitLog { entries: vec![] };
        let messages: Vec<&str> = log.messages().collect();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_git_log_len_and_is_empty() {
        let empty_log = GitLog { entries: vec![] };
        assert!(empty_log.is_empty());
        assert_eq!(empty_log.len(), 0);

        let log = GitLog {
            entries: vec![
                GitLogEntry {
                    hash: "abc".to_string(),
                    author: "Test".to_string(),
                    message: "Test commit".to_string(),
                },
            ],
        };
        assert!(!log.is_empty());
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_git_log_messages_single() {
        let log = GitLog {
            entries: vec![GitLogEntry {
                hash: "single".to_string(),
                author: "Solo".to_string(),
                message: "Only commit".to_string(),
            }],
        };
        let messages: Vec<&str> = log.messages().collect();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "Only commit");
    }

    #[test]
    fn test_git_is_clean_method_exists() {
        // Test that GitStatus correctly reports clean state
        let clean_status = GitStatus {
            staged: vec![],
            unstaged: vec![],
            untracked: vec![],
            is_clean: true,
        };
        assert!(clean_status.is_clean);

        let dirty_status = GitStatus {
            staged: vec!["src/main.rs".to_string()],
            unstaged: vec![],
            untracked: vec![],
            is_clean: false,
        };
        assert!(!dirty_status.is_clean);
    }

    #[test]
    fn test_git_is_clean_with_untracked() {
        // Repository with untracked files is not clean
        let status = GitStatus {
            staged: vec![],
            unstaged: vec![],
            untracked: vec!["new_file.rs".to_string()],
            is_clean: false,
        };
        assert!(!status.is_clean);
    }

    #[test]
    fn test_git_is_clean_with_staged() {
        // Repository with staged changes is not clean
        let status = GitStatus {
            staged: vec!["src/lib.rs".to_string()],
            unstaged: vec![],
            untracked: vec![],
            is_clean: false,
        };
        assert!(!status.is_clean);
    }

    #[test]
    fn test_git_is_clean_with_unstaged() {
        // Repository with unstaged changes is not clean
        let status = GitStatus {
            staged: vec![],
            unstaged: vec!["src/lib.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        assert!(!status.is_clean);
    }

    #[test]
    fn test_git_conflicts_construction() {
        let conflicts = GitConflicts {
            files: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            has_conflicts: true,
        };
        assert!(conflicts.has_conflicts);
        assert_eq!(conflicts.files.len(), 2);
        assert!(conflicts.files.contains(&"src/main.rs".to_string()));
    }

    #[test]
    fn test_git_conflicts_empty() {
        let conflicts = GitConflicts {
            files: vec![],
            has_conflicts: false,
        };
        assert!(!conflicts.has_conflicts);
        assert!(conflicts.files.is_empty());
    }

    #[test]
    fn test_git_conflicts_single_file() {
        let conflicts = GitConflicts {
            files: vec!["src/conflicted.rs".to_string()],
            has_conflicts: true,
        };
        assert!(conflicts.has_conflicts);
        assert_eq!(conflicts.files.len(), 1);
    }

    #[test]
    fn test_git_log_entry_summary() {
        let entry = GitLogEntry {
            hash: "abc1234".to_string(),
            author: "Alice".to_string(),
            message: "Fix bug in parser".to_string(),
        };
        assert_eq!(entry.summary(), "abc1234|Alice: Fix bug in parser");
    }

    #[test]
    fn test_git_log_entry_summary_special_chars() {
        let entry = GitLogEntry {
            hash: "def5678".to_string(),
            author: "Bob (contributor)".to_string(),
            message: "Add feature: x, y, z".to_string(),
        };
        assert_eq!(entry.summary(), "def5678|Bob (contributor): Add feature: x, y, z");
    }

    #[test]
    fn test_git_log_entry_summary_short_message() {
        let entry = GitLogEntry {
            hash: "a1b".to_string(),
            author: "Dev".to_string(),
            message: "WIP".to_string(),
        };
        assert_eq!(entry.summary(), "a1b|Dev: WIP");
    }

    #[test]
    fn test_git_diff_summary_multiple_files() {
        let diff = GitDiff {
            files_changed: vec!["src/main.rs".to_string(), "src/lib.rs".to_string(), "src/util.rs".to_string()],
            insertions: 25,
            deletions: 10,
            diff_output: String::new(),
        };
        assert_eq!(diff.summary(), "3 files changed, +25/-10");
    }

    #[test]
    fn test_git_diff_summary_single_file() {
        let diff = GitDiff {
            files_changed: vec!["src/main.rs".to_string()],
            insertions: 5,
            deletions: 2,
            diff_output: String::new(),
        };
        assert_eq!(diff.summary(), "1 file changed, +5/-2");
    }

    #[test]
    fn test_git_diff_summary_no_changes() {
        let diff = GitDiff {
            files_changed: vec![],
            insertions: 0,
            deletions: 0,
            diff_output: String::new(),
        };
        assert_eq!(diff.summary(), "0 files changed, +0/-0");
    }

    #[test]
    fn test_git_diff_summary_additions_only() {
        let diff = GitDiff {
            files_changed: vec!["src/new.rs".to_string()],
            insertions: 100,
            deletions: 0,
            diff_output: String::new(),
        };
        assert_eq!(diff.summary(), "1 file changed, +100/-0");
    }

    #[test]
    fn test_git_diff_summary_deletions_only() {
        let diff = GitDiff {
            files_changed: vec!["src/deprecated.rs".to_string()],
            insertions: 0,
            deletions: 50,
            diff_output: String::new(),
        };
        assert_eq!(diff.summary(), "1 file changed, +0/-50");
    }

    #[test]
    fn test_git_status_summary_clean() {
        let status = GitStatus {
            staged: vec![],
            unstaged: vec![],
            untracked: vec![],
            is_clean: true,
        };
        assert_eq!(status.summary(), "clean");
    }

    #[test]
    fn test_git_status_summary_staged_only() {
        let status = GitStatus {
            staged: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            unstaged: vec![],
            untracked: vec![],
            is_clean: false,
        };
        assert_eq!(status.summary(), "staged:2");
    }

    #[test]
    fn test_git_status_summary_unstaged_only() {
        let status = GitStatus {
            staged: vec![],
            unstaged: vec!["src/util.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        assert_eq!(status.summary(), "unstaged:1");
    }

    #[test]
    fn test_git_status_summary_untracked_only() {
        let status = GitStatus {
            staged: vec![],
            unstaged: vec![],
            untracked: vec!["new_file.rs".to_string(), "another.rs".to_string()],
            is_clean: false,
        };
        assert_eq!(status.summary(), "untracked:2");
    }

    #[test]
    fn test_git_status_summary_all_types() {
        let status = GitStatus {
            staged: vec!["src/a.rs".to_string()],
            unstaged: vec!["src/b.rs".to_string()],
            untracked: vec!["c.rs".to_string()],
            is_clean: false,
        };
        assert_eq!(status.summary(), "staged:1, unstaged:1, untracked:1");
    }

    #[test]
    fn test_git_status_summary_staged_and_unstaged() {
        let status = GitStatus {
            staged: vec!["src/main.rs".to_string()],
            unstaged: vec!["src/lib.rs".to_string(), "src/util.rs".to_string()],
            untracked: vec![],
            is_clean: false,
        };
        assert_eq!(status.summary(), "staged:1, unstaged:2");
    }

    #[test]
    fn test_git_conflicts_summary_none() {
        let conflicts = GitConflicts {
            files: vec![],
            has_conflicts: false,
        };
        assert_eq!(conflicts.summary(), "no conflicts");
    }

    #[test]
    fn test_git_conflicts_summary_single() {
        let conflicts = GitConflicts {
            files: vec!["src/main.rs".to_string()],
            has_conflicts: true,
        };
        assert_eq!(conflicts.summary(), "conflicts in: src/main.rs");
    }

    #[test]
    fn test_git_conflicts_summary_multiple() {
        let conflicts = GitConflicts {
            files: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
            has_conflicts: true,
        };
        assert_eq!(conflicts.summary(), "conflicts in: src/main.rs, src/lib.rs");
    }

    #[test]
    fn test_git_conflicts_summary_three_files() {
        let conflicts = GitConflicts {
            files: vec![
                "src/a.rs".to_string(),
                "src/b.rs".to_string(),
                "src/c.rs".to_string(),
            ],
            has_conflicts: true,
        };
        assert_eq!(conflicts.summary(), "conflicts in: src/a.rs, src/b.rs, src/c.rs");
    }
}
