//! Local codebase tools for the auto research agent (rig Tool trait).
//!
//! Implements rig `Tool` trait so the LLM agent can directly call:
//!   - `codebase_grep`   — regex search across all source files
//!   - `codebase_search` — find files by name/glob pattern
//!   - `codebase_read`   — read file content
//!   - `codebase_tree`   — list directory structure
//!
//! These give the agent full introspection into its own codebase.

use std::future::Future;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared: project root
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ProjectRoot {
    path: PathBuf,
}

impl ProjectRoot {
    fn new() -> Self {
        Self {
            path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }

    fn with_root(root: PathBuf) -> Self {
        Self { path: root }
    }

    fn src_dir(&self) -> PathBuf {
        self.path.join("src")
    }
}

impl Default for ProjectRoot {
    fn default() -> Self {
        Self::new()
    }
}

fn default_project_root() -> ProjectRoot {
    ProjectRoot::new()
}

// ---------------------------------------------------------------------------
// rig Tool: codebase_grep
// ---------------------------------------------------------------------------

/// Arguments for the codebase_grep tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct GrepArgs {
    /// Regex pattern to search for in file contents
    pub pattern: String,
    /// File extension filter (e.g. "rs", "toml"). Empty = all files.
    #[serde(default)]
    pub file_ext: String,
    /// Max results to return (default: 20)
    #[serde(default = "default_grep_max")]
    pub max_results: usize,
    /// Whether to show surrounding context lines (default: 0)
    #[serde(default)]
    pub context_lines: usize,
}

fn default_grep_max() -> usize {
    20
}

/// A single grep match.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GrepMatch {
    pub file: String,
    pub line_number: usize,
    pub line: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

/// Output of the codebase_grep tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct GrepOutput {
    pub pattern: String,
    pub matches: Vec<GrepMatch>,
    pub total_matches: usize,
    pub files_searched: usize,
    pub truncated: bool,
}

/// Error type for codebase tools.
#[derive(Debug, thiserror::Error)]
pub enum CodebaseToolError {
    #[error("File error: {0}")]
    FileError(String),
    #[error("Regex error: {0}")]
    RegexError(String),
    #[error("Path error: {0}")]
    PathError(String),
}

/// Regex search across all source files in the codebase.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CodebaseGrepTool {
    #[serde(skip, default = "default_project_root")]
    root: ProjectRoot,
}

impl CodebaseGrepTool {
    pub fn new() -> Self {
        Self {
            root: ProjectRoot::new(),
        }
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root: ProjectRoot::with_root(root),
        }
    }

    /// Direct grep (without rig Tool machinery).
    pub fn grep(
        &self,
        pattern: &str,
        file_ext: &str,
        max_results: usize,
        context_lines: usize,
    ) -> Result<GrepOutput> {
        let regex =
            regex::Regex::new(pattern).map_err(|e| CodebaseToolError::RegexError(e.to_string()))?;

        let src_dir = self.root.src_dir();
        let root_path = self.root.path.clone();
        let mut matches = Vec::new();
        let mut files_searched = 0usize;

        Self::walk_files(&src_dir, &root_path, file_ext, &mut |path, lines| {
            files_searched += 1;
            if matches.len() >= max_results {
                return;
            }
            
            // Use a rolling window for context extraction
            for (idx, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    // Extract context using slicing (lines are now owned Strings)
                    let before: Vec<String> = lines[idx.saturating_sub(context_lines)..idx]
                        .iter()
                        .cloned()
                        .collect();
                    let after: Vec<String> = lines[idx + 1..]
                        .iter()
                        .take(context_lines)
                        .cloned()
                        .collect();

                    matches.push(GrepMatch {
                        file: path.clone(),
                        line_number: idx + 1,
                        line: line.clone(),
                        context_before: before,
                        context_after: after,
                    });
                    if matches.len() >= max_results {
                        return;
                    }
                }
            }
        })?;

        let truncated = matches.len() >= max_results;
        let total_matches = matches.len();

        Ok(GrepOutput {
            pattern: pattern.to_string(),
            matches,
            total_matches,
            files_searched,
            truncated,
        })
    }

    fn walk_files<F>(
        dir: &Path,
        root: &Path,
        file_ext: &str,
        callback: &mut F,
    ) -> Result<()>
    where
        F: FnMut(String, Vec<String>),
    {
        if !dir.exists() {
            return Ok(());
        }
        Self::walk_dir_recursive(dir, root, file_ext, callback)
    }

    fn walk_dir_recursive<F>(dir: &Path, root: &Path, file_ext: &str, callback: &mut F) -> Result<()>
    where
        F: FnMut(String, Vec<String>),
    {
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("Cannot read dir {}", dir.display()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip target/ and hidden dirs
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if file_name == "target" || file_name.starts_with('.') {
                    continue;
                }
                Self::walk_dir_recursive(&path, root, file_ext, callback)?;
            } else if path.is_file() {
                // Check extension
                if !file_ext.is_empty() {
                    let ext = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    if ext != file_ext {
                        continue;
                    }
                }

                // Use the original root for consistent relative paths
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();

                // Read lines efficiently using BufReader
                if let Ok(file) = std::fs::File::open(&path) {
                    let reader = BufReader::new(file);
                    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
                    callback(rel, lines);
                }
            }
        }
        Ok(())
    }
}

impl Default for CodebaseGrepTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CodebaseGrepTool {
    const NAME: &'static str = "codebase_grep";

    type Error = CodebaseToolError;
    type Args = GrepArgs;
    type Output = GrepOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "codebase_grep".to_string(),
            description: "Search across all source files in the codebase using a regex pattern. \
                          Returns matching lines with file paths and line numbers. \
                          Use this to find function definitions, usages, patterns, or \
                          any text in the code.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "file_ext": {
                        "type": "string",
                        "description": "File extension filter (e.g. 'rs', 'toml'). Empty = all files.",
                        "default": ""
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matches to return (default 20)",
                        "default": 20
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around each match (default 0)",
                        "default": 0
                    }
                },
                "required": ["pattern"]
            }),
        };
        async move { def }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        let pattern = args.pattern.clone();
        let file_ext = args.file_ext.clone();
        let max = args.max_results;
        let ctx = args.context_lines;
        let root = self.root.clone();
        async move {
            let tool = CodebaseGrepTool { root };
            tool.grep(&pattern, &file_ext, max, ctx)
                .map_err(|e| CodebaseToolError::FileError(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// rig Tool: codebase_search
// ---------------------------------------------------------------------------

/// Arguments for the codebase_search tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct SearchFilesArgs {
    /// Glob-style pattern to match file names (e.g. "*.rs", "*config*", "*/test*.rs")
    pub pattern: String,
    /// Max results to return (default: 30)
    #[serde(default = "default_search_max")]
    pub max_results: usize,
}

fn default_search_max() -> usize {
    30
}

/// A found file entry.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileEntry {
    pub path: String,
    pub size_bytes: u64,
}

/// Output of the codebase_search tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct SearchFilesOutput {
    pub pattern: String,
    pub files: Vec<FileEntry>,
    pub total_found: usize,
}

/// Find files by name pattern in the codebase.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CodebaseSearchTool {
    #[serde(skip, default = "default_project_root")]
    root: ProjectRoot,
}

impl CodebaseSearchTool {
    pub fn new() -> Self {
        Self {
            root: ProjectRoot::new(),
        }
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root: ProjectRoot::with_root(root),
        }
    }

    /// Direct search (without rig Tool machinery).
    pub fn search_files(&self, pattern: &str, max_results: usize) -> Result<SearchFilesOutput> {
        let src_dir = self.root.src_dir();
        let root_path = self.root.path.clone();
        let mut files = Vec::new();

        Self::find_files_recursive(&src_dir, &root_path, pattern, &mut files, max_results)?;

        let total_found = files.len();
        Ok(SearchFilesOutput {
            pattern: pattern.to_string(),
            files,
            total_found,
        })
    }

    fn find_files_recursive(
        dir: &Path,
        root: &Path,
        pattern: &str,
        files: &mut Vec<FileEntry>,
        max_results: usize,
    ) -> Result<()> {
        if !dir.exists() || files.len() >= max_results {
            return Ok(());
        }

        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("Cannot read dir {}", dir.display()))?;

        for entry in entries.flatten() {
            if files.len() >= max_results {
                break;
            }
            let path = entry.path();
            if path.is_dir() {
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if file_name == "target" || file_name.starts_with('.') {
                    continue;
                }
                Self::find_files_recursive(&path, root, pattern, files, max_results)?;
            } else if path.is_file() {
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if glob_match(pattern, file_name) {
                    // Use the original root for consistent relative paths
                    let rel = path
                        .strip_prefix(root)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();
                    let metadata = entry.metadata();
                    let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
                    files.push(FileEntry {
                        path: rel,
                        size_bytes: size,
                    });
                }
            }
        }
        Ok(())
    }
}

impl Default for CodebaseSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CodebaseSearchTool {
    const NAME: &'static str = "codebase_search";

    type Error = CodebaseToolError;
    type Args = SearchFilesArgs;
    type Output = SearchFilesOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "codebase_search".to_string(),
            description: "Find files in the codebase by name pattern. Supports simple glob \
                          wildcards (* = any chars, ? = single char). Returns file paths \
                          relative to src/ and their sizes. Use this to discover what files \
                          exist or find files matching a name pattern.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern with glob wildcards (e.g. '*.rs', '*test*', '*/mod.rs')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max files to return (default 30)",
                        "default": 30
                    }
                },
                "required": ["pattern"]
            }),
        };
        async move { def }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        let pattern = args.pattern.clone();
        let max = args.max_results;
        let root = self.root.clone();
        async move {
            let tool = CodebaseSearchTool { root };
            tool.search_files(&pattern, max)
                .map_err(|e| CodebaseToolError::FileError(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// rig Tool: codebase_read
// ---------------------------------------------------------------------------

/// Arguments for the codebase_read tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct ReadFileArgs {
    /// File path relative to project root (e.g. "src/lib.rs", "Cargo.toml")
    pub path: String,
    /// Start line (1-indexed, default: 1)
    #[serde(default = "default_start_line")]
    pub start_line: usize,
    /// Max number of lines to return (default: 100)
    #[serde(default = "default_max_lines")]
    pub max_lines: usize,
}

fn default_start_line() -> usize {
    1
}

fn default_max_lines() -> usize {
    100
}

/// Output of the codebase_read tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct ReadFileOutput {
    pub path: String,
    pub total_lines: usize,
    pub returned_lines: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
}

/// Read a file's content.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CodebaseReadTool {
    #[serde(skip, default = "default_project_root")]
    root: ProjectRoot,
}

impl CodebaseReadTool {
    pub fn new() -> Self {
        Self {
            root: ProjectRoot::new(),
        }
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root: ProjectRoot::with_root(root),
        }
    }

    /// Direct read (without rig Tool machinery).
    pub fn read_file(&self, path: &str, start_line: usize, max_lines: usize) -> Result<ReadFileOutput> {
        let full_path = self.root.path.join(path);
        
        // Canonicalize both paths to resolve symlinks and .. components
        let canonical_root = self.root.path.canonicalize()
            .unwrap_or_else(|_| self.root.path.clone());
        let canonical_full = full_path.canonicalize()
            .map_err(|e| CodebaseToolError::PathError(
                format!("Cannot resolve path '{}': {}", path, e)
            ))?;
        
        // Security check: ensure resolved path is within project root
        if !canonical_full.starts_with(&canonical_root) {
            return Err(CodebaseToolError::PathError(
                format!("Path '{}' resolves outside project root", path)
            ).into());
        }
        
        let content = std::fs::read_to_string(&canonical_full)
            .with_context(|| format!("Cannot read {}", full_path.display()))?;

        let all_lines: Vec<&str> = content.lines().collect();
        let total_lines = all_lines.len();

        let start = if start_line < 1 { 0 } else { start_line - 1 };
        let end = (start + max_lines).min(total_lines);
        let slice: Vec<&str> = all_lines[start..end].to_vec();
        let returned_content = slice.join("\n");

        Ok(ReadFileOutput {
            path: path.to_string(),
            total_lines,
            returned_lines: slice.len(),
            start_line: start + 1,
            end_line: end,
            content: returned_content,
        })
    }
}

impl Default for CodebaseReadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CodebaseReadTool {
    const NAME: &'static str = "codebase_read";

    type Error = CodebaseToolError;
    type Args = ReadFileArgs;
    type Output = ReadFileOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "codebase_read".to_string(),
            description: "Read a file's content from the codebase. Returns the text content \
                          with line numbers. Supports pagination (start_line + max_lines) \
                          for large files. Paths are relative to the project root.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root (e.g. 'src/lib.rs', 'Cargo.toml')"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-indexed, default 1)",
                        "default": 1
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to return (default 100)",
                        "default": 100
                    }
                },
                "required": ["path"]
            }),
        };
        async move { def }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        let path = args.path.clone();
        let start = args.start_line;
        let max = args.max_lines;
        let root = self.root.clone();
        async move {
            let tool = CodebaseReadTool { root };
            tool.read_file(&path, start, max)
                .map_err(|e| CodebaseToolError::FileError(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// rig Tool: codebase_tree
// ---------------------------------------------------------------------------

/// Arguments for the codebase_tree tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct TreeArgs {
    /// Directory path relative to project root (e.g. "src", "src/runtime")
    #[serde(default = "default_tree_dir")]
    pub dir: String,
    /// Max depth to recurse (default: 3)
    #[serde(default = "default_tree_depth")]
    pub max_depth: usize,
}

fn default_tree_dir() -> String {
    "src".to_string()
}

fn default_tree_depth() -> usize {
    3
}

/// A tree entry.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TreeEntry {
    pub path: String,
    pub is_dir: bool,
    pub size_bytes: Option<u64>,
}

/// Output of the codebase_tree tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct TreeOutput {
    pub base_dir: String,
    pub entries: Vec<TreeEntry>,
    pub total_files: usize,
    pub total_dirs: usize,
}

/// List directory structure.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CodebaseTreeTool {
    #[serde(skip, default = "default_project_root")]
    root: ProjectRoot,
}

impl CodebaseTreeTool {
    pub fn new() -> Self {
        Self {
            root: ProjectRoot::new(),
        }
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root: ProjectRoot::with_root(root),
        }
    }

    /// Direct tree (without rig Tool machinery).
    pub fn tree(&self, dir: &str, max_depth: usize) -> Result<TreeOutput> {
        let full_path = self.root.path.join(dir);
        if !full_path.exists() {
            anyhow::bail!("Directory does not exist: {}", full_path.display());
        }

        let mut entries = Vec::new();
        let mut total_files = 0usize;
        let mut total_dirs = 0usize;

        self.build_tree(&full_path, dir, max_depth, 0, &mut entries, &mut total_files, &mut total_dirs)?;

        Ok(TreeOutput {
            base_dir: dir.to_string(),
            entries,
            total_files,
            total_dirs,
        })
    }

    fn build_tree(
        &self,
        dir: &Path,
        rel_prefix: &str,
        max_depth: usize,
        current_depth: usize,
        entries: &mut Vec<TreeEntry>,
        total_files: &mut usize,
        total_dirs: &mut usize,
    ) -> Result<()> {
        if current_depth >= max_depth {
            return Ok(());
        }

        let read_entries = std::fs::read_dir(dir)
            .with_context(|| format!("Cannot read {}", dir.display()))?;

        let mut items: Vec<_> = read_entries
            .flatten()
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                !name.starts_with('.') && name != "target"
            })
            .collect();

        // Sort: directories first, then files, alphabetically
        items.sort_by_key(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = e.path().is_dir();
            (std::cmp::Reverse(is_dir), name.to_lowercase())
        });

        for entry in items {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            let rel = format!("{}/{}", rel_prefix, file_name);

            if path.is_dir() {
                *total_dirs += 1;
                entries.push(TreeEntry {
                    path: rel.clone(),
                    is_dir: true,
                    size_bytes: None,
                });
                self.build_tree(&path, &rel, max_depth, current_depth + 1, entries, total_files, total_dirs)?;
            } else {
                *total_files += 1;
                let size = entry.metadata().ok().map(|m| m.len());
                entries.push(TreeEntry {
                    path: rel,
                    is_dir: false,
                    size_bytes: size,
                });
            }
        }

        Ok(())
    }
}

impl Default for CodebaseTreeTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CodebaseTreeTool {
    const NAME: &'static str = "codebase_tree";

    type Error = CodebaseToolError;
    type Args = TreeArgs;
    type Output = TreeOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "codebase_tree".to_string(),
            description: "List the directory structure of the codebase. Returns files and \
                          directories in a tree-like format with sizes. Useful for understanding \
                          the project layout and discovering modules.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "dir": {
                        "type": "string",
                        "description": "Directory path relative to project root (default 'src')"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max recursion depth (default 3)",
                        "default": 3
                    }
                }
            }),
        };
        async move { def }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        let dir = args.dir.clone();
        let max = args.max_depth;
        let root = self.root.clone();
        async move {
            let tool = CodebaseTreeTool { root };
            tool.tree(&dir, max)
                .map_err(|e| CodebaseToolError::FileError(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple glob matching (supports * and ?) using dynamic programming.
/// Achieves O(m*n) time complexity where m = pattern length, n = text length.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let m = p.len();
    let n = t.len();

    // dp[i][j] = true if pattern[0..i] matches text[0..j]
    let mut dp = vec![vec![false; n + 1]; m + 1];

    // Empty pattern matches empty text
    dp[0][0] = true;

    // Handle leading wildcards: * can match empty string
    for i in 1..=m {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        } else {
            break;
        }
    }

    // Fill the DP table
    for i in 1..=m {
        for j in 1..=n {
            match p[i - 1] {
                '*' => {
                    // * can match:
                    // - empty string (dp[i-1][j])
                    // - one or more characters (dp[i][j-1])
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
                '?' => {
                    // ? matches exactly one character
                    dp[i][j] = dp[i - 1][j - 1];
                }
                c => {
                    // Literal character must match
                    dp[i][j] = dp[i - 1][j - 1] && c == t[j - 1];
                }
            }
        }
    }

    dp[m][n]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hyperagent_test_{}", name));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(dir.join("src/runtime"));
        let _ = fs::create_dir_all(dir.join("src/agent"));
        let _ = fs::write(dir.join("src/lib.rs"), "pub mod agent;\npub mod runtime;\n");
        let _ = fs::write(dir.join("src/runtime/loop_.rs"), "pub fn run() {}\npub async fn step() {}\n");
        let _ = fs::write(dir.join("src/agent/mod.rs"), "pub struct Agent;\nimpl Agent { fn new() -> Self { Agent } }\n");
        let _ = fs::write(dir.join("Cargo.toml"), "[package]\nname = \"test\"\n");
        dir
    }

    #[test]
    fn test_glob_match_star() {
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(glob_match("*.rs", "mod.rs"));
        assert!(!glob_match("*.rs", "lib.toml"));
    }

    #[test]
    fn test_glob_match_question() {
        assert!(glob_match("?.rs", "a.rs"));
        assert!(!glob_match("?.rs", "ab.rs"));
    }

    #[test]
    fn test_glob_match_complex() {
        assert!(glob_match("*test*.rs", "test_mod.rs"));
        assert!(glob_match("*/mod.rs", "agent/mod.rs"));
        assert!(!glob_match("*/mod.rs", "agent/test.rs"));
    }

    #[test]
    fn test_glob_match_multiple_stars() {
        // Test patterns with multiple * wildcards
        assert!(glob_match("a*b*c", "abc"));
        assert!(glob_match("a*b*c", "aXbYc"));
        assert!(glob_match("a*b*c", "aXXbYYc"));
        assert!(glob_match("**test**", "test"));
        assert!(glob_match("**test**", "my_test_file.rs"));
        assert!(!glob_match("a*b*c", "ac"));
        assert!(!glob_match("a*b*c", "ab"));
    }

    #[test]
    fn test_glob_match_edge_cases() {
        // Empty patterns
        assert!(glob_match("", ""));
        assert!(!glob_match("", "a"));
        
        // Only wildcards
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("**", "anything"));
        assert!(glob_match("?", ""));
        assert!(!glob_match("?", ""));
        assert!(glob_match("?", "x"));
        assert!(!glob_match("?", "xy"));
    }

    #[test]
    fn test_grep_tool() {
        let dir = setup_test_dir("grep");
        let tool = CodebaseGrepTool::with_root(dir.clone());

        let result = tool.grep("pub fn", "rs", 20, 0).unwrap();
        assert!(result.total_matches >= 1);
        assert!(result.files_searched >= 2);
        // Should find "pub fn run()" and "pub async fn step()"
        let has_run = result.matches.iter().any(|m| m.line.contains("pub fn run"));
        assert!(has_run);
    }

    #[test]
    fn test_grep_with_context() {
        let dir = setup_test_dir("grep_ctx");
        let tool = CodebaseGrepTool::with_root(dir.clone());

        let result = tool.grep("pub fn run", "rs", 10, 1).unwrap();
        if let Some(m) = result.matches.iter().find(|m| m.line.contains("pub fn run")) {
            assert!(!m.context_before.is_empty() || !m.context_after.is_empty());
        }
    }

    #[test]
    fn test_grep_by_ext() {
        let dir = setup_test_dir("grep_ext");
        let tool = CodebaseGrepTool::with_root(dir.clone());

        let rs_result = tool.grep("pub", "rs", 20, 0).unwrap();
        let toml_result = tool.grep("pub", "toml", 20, 0).unwrap();
        assert!(rs_result.total_matches > toml_result.total_matches);
    }

    #[test]
    fn test_grep_consistent_paths() {
        let dir = setup_test_dir("grep_paths");
        let tool = CodebaseGrepTool::with_root(dir.clone());

        let result = tool.grep("pub", "rs", 20, 0).unwrap();
        // All paths should start with "src/" since we strip from the project root
        for m in &result.matches {
            assert!(m.file.starts_with("src/"), "Path '{}' should start with 'src/'", m.file);
        }
    }

    #[test]
    fn test_search_files_tool() {
        let dir = setup_test_dir("search");
        let tool = CodebaseSearchTool::with_root(dir.clone());

        let result = tool.search_files("*.rs", 30).unwrap();
        assert!(result.total_found >= 3); // lib.rs + loop_.rs + mod.rs
        assert!(result.files.iter().any(|f| f.path.contains("lib.rs")));
    }

    #[test]
    fn test_search_files_pattern() {
        let dir = setup_test_dir("search_pat");
        let tool = CodebaseSearchTool::with_root(dir.clone());

        let result = tool.search_files("*mod*", 30).unwrap();
        assert!(result.files.iter().any(|f| f.path.contains("mod.rs")));
    }

    #[test]
    fn test_search_files_consistent_paths() {
        let dir = setup_test_dir("search_paths");
        let tool = CodebaseSearchTool::with_root(dir.clone());

        let result = tool.search_files("*.rs", 30).unwrap();
        // All paths should start with "src/"
        for f in &result.files {
            assert!(f.path.starts_with("src/"), "Path '{}' should start with 'src/'", f.path);
        }
    }

    #[test]
    fn test_read_file_tool() {
        let dir = setup_test_dir("read");
        let tool = CodebaseReadTool::with_root(dir.clone());

        let result = tool.read_file("src/lib.rs", 1, 100).unwrap();
        assert_eq!(result.total_lines, 2);
        assert_eq!(result.returned_lines, 2);
        assert!(result.content.contains("pub mod agent"));
    }

    #[test]
    fn test_read_file_pagination() {
        let dir = setup_test_dir("read_page");
        let tool = CodebaseReadTool::with_root(dir.clone());

        let result = tool.read_file("src/lib.rs", 1, 1).unwrap();
        assert_eq!(result.returned_lines, 1);
        assert_eq!(result.start_line, 1);
        assert_eq!(result.end_line, 1);
    }

    #[test]
    fn test_tree_tool() {
        let dir = setup_test_dir("tree");
        let tool = CodebaseTreeTool::with_root(dir.clone());

        let result = tool.tree("src", 3).unwrap();
        assert!(result.total_files >= 3);
        assert!(result.total_dirs >= 2);
        assert!(result.entries.iter().any(|e| e.path.contains("lib.rs")));
        assert!(result.entries.iter().any(|e| e.is_dir && e.path.contains("runtime")));
    }

    #[tokio::test]
    async fn test_grep_tool_trait() {
        let tool = CodebaseGrepTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "codebase_grep");
        assert!(!def.description.is_empty());
    }

    #[tokio::test]
    async fn test_search_tool_trait() {
        let tool = CodebaseSearchTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "codebase_search");
    }

    #[tokio::test]
    async fn test_read_tool_trait() {
        let tool = CodebaseReadTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "codebase_read");
    }

    #[tokio::test]
    async fn test_tree_tool_trait() {
        let tool = CodebaseTreeTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "codebase_tree");
    }

    #[tokio::test]
    async fn test_grep_tool_call() {
        let dir = setup_test_dir("grep_call");
        let tool = CodebaseGrepTool::with_root(dir);
        let args = GrepArgs {
            pattern: "pub fn".to_string(),
            file_ext: "rs".to_string(),
            max_results: 10,
            context_lines: 0,
        };
        let result = tool.call(args).await.unwrap();
        assert!(result.total_matches >= 1);
    }

    #[tokio::test]
    async fn test_read_tool_call() {
        let dir = setup_test_dir("read_call");
        let tool = CodebaseReadTool::with_root(dir);
        let args = ReadFileArgs {
            path: "src/lib.rs".to_string(),
            start_line: 1,
            max_lines: 100,
        };
        let result = tool.call(args).await.unwrap();
        assert!(result.content.contains("pub mod"));
    }

    #[test]
    fn test_read_file_blocks_traversal_attack() {
        let dir = setup_test_dir("traversal");
        let tool = CodebaseReadTool::with_root(dir.clone());
        
        // Attempt to read a file outside project root using ../
        let result = tool.read_file("../../../etc/passwd", 1, 100);
        assert!(result.is_err(), "Should reject path traversal attempt");
        
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("outside project root") || err_msg.contains("resolve"), 
                "Error message should indicate path resolution issue");
    }

    #[test]
    fn test_read_file_blocks_absolute_path() {
        let dir = setup_test_dir("absolute");
        let tool = CodebaseReadTool::with_root(dir.clone());
        
        // Attempt to read an absolute path
        let result = tool.read_file("/etc/passwd", 1, 100);
        assert!(result.is_err(), "Should reject absolute path outside project");
    }

    #[test]
    fn test_read_file_valid_path() {
        let dir = setup_test_dir("valid_path");
        let tool = CodebaseReadTool::with_root(dir.clone());
        
        // Valid path within project should work
        let result = tool.read_file("Cargo.toml", 1, 100);
        assert!(result.is_ok(), "Should read valid path within project");
        
        let content = result.unwrap();
        assert!(content.content.contains("[package]"));
    }

    #[test]
    fn test_read_file_nested_path() {
        let dir = setup_test_dir("nested");
        let tool = CodebaseReadTool::with_root(dir.clone());
        
        // Nested path within project should work
        let result = tool.read_file("src/runtime/loop_.rs", 1, 100);
        assert!(result.is_ok(), "Should read nested path within project");
        
        let content = result.unwrap();
        assert!(content.content.contains("pub fn run"));
    }

    #[test]
    fn test_grep_large_context_window() {
        let dir = std::env::temp_dir().join("hyperagent_test_grep_large_ctx");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(dir.join("src"));
        
        // Create a file with many lines
        let mut content = String::new();
        for i in 1..=100 {
            content.push_str(&format!("line {}\n", i));
        }
        content.push_str("TARGET_LINE\n");
        for i in 102..=200 {
            content.push_str(&format!("line {}\n", i));
        }
        let _ = fs::write(dir.join("src/large.rs"), &content);
        
        let tool = CodebaseGrepTool::with_root(dir);
        let result = tool.grep("TARGET_LINE", "rs", 10, 5).unwrap();
        
        assert_eq!(result.total_matches, 1);
        let m = &result.matches[0];
        assert_eq!(m.line_number, 101);
        assert_eq!(m.context_before.len(), 5);
        assert_eq!(m.context_after.len(), 5);
        assert_eq!(m.context_before[0], "line 96");
        assert_eq!(m.context_after[4], "line 106");
    }

    #[test]
    fn test_grep_context_at_file_boundaries() {
        let dir = std::env::temp_dir().join("hyperagent_test_grep_boundary");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(dir.join("src"));
        
        // Match at the beginning of file
        let _ = fs::write(dir.join("src/start.rs"), "FIRST_LINE\nsecond\nthird\n");
        // Match at the end of file
        let _ = fs::write(dir.join("src/end.rs"), "first\nsecond\nLAST_LINE\n");
        
        let tool = CodebaseGrepTool::with_root(dir.clone());
        
        // Test match at start
        let result_start = tool.grep("FIRST_LINE", "rs", 10, 2).unwrap();
        assert_eq!(result_start.matches[0].context_before.len(), 0);
        assert_eq!(result_start.matches[0].context_after.len(), 2);
        
        // Test match at end
        let result_end = tool.grep("LAST_LINE", "rs", 10, 2).unwrap();
        assert_eq!(result_end.matches[0].context_before.len(), 2);
        assert_eq!(result_end.matches[0].context_after.len(), 0);
    }

    #[test]
    fn test_grep_multiple_matches_same_file() {
        let dir = setup_test_dir("grep_multi");
        let tool = CodebaseGrepTool::with_root(dir.clone());
        
        // The test file has "pub fn run" and "pub async fn step"
        let result = tool.grep("pub", "rs", 20, 0).unwrap();
        
        // Should find matches in multiple files (lib.rs, runtime/loop_.rs, agent/mod.rs)
        assert!(result.total_matches >= 3, "Should find at least 3 'pub' occurrences");
    }
}