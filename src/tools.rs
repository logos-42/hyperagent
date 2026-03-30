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
    /// Context lines before the match, formatted as "line_num: content"
    pub context_before: Vec<String>,
    /// Context lines after the match, formatted as "line_num: content"  
    pub context_after: Vec<String>,
    /// Character ranges where the pattern matched (start, end positions in the line)
    pub match_ranges: Vec<(usize, usize)>,
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
                // Find all match ranges in the line
                let match_ranges: Vec<(usize, usize)> = regex
                    .find_iter(line)
                    .map(|m| (m.start(), m.end()))
                    .collect();
                
                if !match_ranges.is_empty() {
                    // Extract context with line numbers for easier navigation
                    let before: Vec<String> = lines[idx.saturating_sub(context_lines)..idx]
                        .iter()
                        .enumerate()
                        .map(|(i, l)| format!("{}: {}", idx.saturating_sub(context_lines) + i + 1, l))
                        .collect();
                    let after: Vec<String> = lines[idx + 1..]
                        .iter()
                        .take(context_lines)
                        .enumerate()
                        .map(|(i, l)| format!("{}: {}", idx + i + 2, l))
                        .collect();

                    matches.push(GrepMatch {
                        file: path.clone(),
                        line_number: idx + 1,
                        line: line.clone(),
                        context_before: before,
                        context_after: after,
                        match_ranges,
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
    /// Whether results were truncated due to max_results limit
    pub truncated: bool,
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

        let truncated = Self::find_files_recursive(&src_dir, &root_path, pattern, &mut files, max_results)?;

        let total_found = files.len();
        Ok(SearchFilesOutput {
            pattern: pattern.to_string(),
            files,
            total_found,
            truncated,
        })
    }

    fn find_files_recursive(
        dir: &Path,
        root: &Path,
        pattern: &str,
        files: &mut Vec<FileEntry>,
        max_results: usize,
    ) -> Result<bool> {
        if !dir.exists() || files.len() >= max_results {
            return Ok(files.len() >= max_results);
        }

        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("Cannot read dir {}", dir.display()))?;

        let mut truncated = false;
        for entry in entries.flatten() {
            if files.len() >= max_results {
                truncated = true;
                break;
            }
            let path = entry.path();
            if path.is_dir() {
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if file_name == "target" || file_name.starts_with('.') {
                    continue;
                }
                let was_truncated = Self::find_files_recursive(&path, root, pattern, files, max_results)?;
                if was_truncated {
                    truncated = true;
                    break;
                }
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
        Ok(truncated)
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
    /// Whether there are more lines beyond the requested range
    pub truncated: bool,
    /// Content with line numbers prefixed (e.g., "1: fn main() {")
    pub content_with_lines: String,
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
        
        // Build content with line numbers for easier reference
        let content_with_lines = slice
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{}: {}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");
        
        let truncated = end < total_lines;

        Ok(ReadFileOutput {
            path: path.to_string(),
            total_lines,
            returned_lines: slice.len(),
            start_line: start + 1,
            end_line: end,
            content: returned_content,
            truncated,
            content_with_lines,
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
    /// Nesting depth (0 = root level, 1 = one level deep, etc.)
    pub depth: usize,
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
                    depth: current_depth,
                });
                self.build_tree(&path, &rel, max_depth, current_depth + 1, entries, total_files, total_dirs)?;
            } else {
                *total_files += 1;
                let size = entry.metadata().ok().map(|m| m.len());
                entries.push(TreeEntry {
                    path: rel,
                    is_dir: false,
                    size_bytes: size,
                    depth: current_depth,
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
    use tempfile::TempDir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_read_file_with_line_numbers() {
        let temp_dir = TempDir::new().unwrap();
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        create_test_file(temp_dir.path(), "test.rs", content);

        let tool = CodebaseReadTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.read_file("test.rs", 1, 100).unwrap();

        assert_eq!(result.total_lines, 3);
        assert_eq!(result.content, content.trim_end());
        assert_eq!(
            result.content_with_lines,
            "1: fn main() {\n2:     println!(\"hello\");\n3: }"
        );
    }

    #[test]
    fn test_read_file_pagination_with_line_numbers() {
        let temp_dir = TempDir::new().unwrap();
        let content = "line one\nline two\nline three\nline four\nline five\n";
        create_test_file(temp_dir.path(), "test.txt", content);

        let tool = CodebaseReadTool::with_root(temp_dir.path().to_path_buf());
        
        // Read lines 2-3
        let result = tool.read_file("test.txt", 2, 2).unwrap();
        assert_eq!(result.start_line, 2);
        assert_eq!(result.end_line, 3);
        assert_eq!(result.returned_lines, 2);
        assert!(result.truncated);
        assert_eq!(result.content_with_lines, "2: line two\n3: line three");
    }

    #[test]
    fn test_read_file_empty() {
        let temp_dir = TempDir::new().unwrap();
        create_test_file(temp_dir.path(), "empty.txt", "");

        let tool = CodebaseReadTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.read_file("empty.txt", 1, 100).unwrap();

        assert_eq!(result.total_lines, 0);
        assert_eq!(result.content, "");
        assert_eq!(result.content_with_lines, "");
        assert!(!result.truncated);
    }

    #[test]
    fn test_glob_match_basic() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(!glob_match("*.rs", "main.txt"));
        assert!(glob_match("*test*", "my_test_file.rs"));
        assert!(glob_match("?", "a"));
        assert!(!glob_match("?", "ab"));
        assert!(glob_match("*", ""));
        assert!(glob_match("*.rs", "test.rs"));
        assert!(glob_match("src/**/*.rs", "src/main.rs"));
    }

    #[test]
    fn test_grep_with_match_ranges() {
        let temp_dir = TempDir::new().unwrap();
        let content = "fn hello() {\n    println!(\"hello world\");\n}\n";
        create_test_file(temp_dir.path(), "test.rs", content);

        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("hello", "rs", 10, 0).unwrap();

        assert_eq!(result.matches.len(), 2);
        
        // Check first match (fn hello)
        let first_match = &result.matches[0];
        assert_eq!(first_match.line_number, 1);
        assert!(!first_match.match_ranges.is_empty());
        assert_eq!(first_match.match_ranges[0], (3, 8)); // "hello" position in "fn hello()"
    }

    #[test]
    fn test_grep_context_with_line_numbers() {
        let temp_dir = TempDir::new().unwrap();
        let content = "line one\nfn target() {\n    println!(\"hello\");\n}\nline five\n";
        create_test_file(temp_dir.path(), "test.rs", content);

        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("target", "rs", 10, 2).unwrap();

        assert_eq!(result.matches.len(), 1);
        
        let m = &result.matches[0];
        assert_eq!(m.line_number, 2);
        assert_eq!(m.line, "fn target() {");
        
        // Context before should have line numbers
        assert_eq!(m.context_before.len(), 1);
        assert_eq!(m.context_before[0], "1: line one");
        
        // Context after should have line numbers
        assert_eq!(m.context_after.len(), 2);
        assert_eq!(m.context_after[0], "3:     println!(\"hello\");");
        assert_eq!(m.context_after[1], "4: }");
    }

    #[test]
    fn test_grep_context_at_file_start() {
        let temp_dir = TempDir::new().unwrap();
        let content = "fn first() {}\nfn second() {}\nfn third() {}\n";
        create_test_file(temp_dir.path(), "test.rs", content);

        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("first", "rs", 10, 5).unwrap();

        assert_eq!(result.matches.len(), 1);
        
        let m = &result.matches[0];
        assert_eq!(m.line_number, 1);
        
        // Context before should be empty (no lines before line 1)
        assert!(m.context_before.is_empty());
        
        // Context after should have line numbers
        assert_eq!(m.context_after.len(), 2);
        assert_eq!(m.context_after[0], "2: fn second() {}");
        assert_eq!(m.context_after[1], "3: fn third() {}");
    }

    #[test]
    fn test_search_files_truncated() {
        let temp_dir = TempDir::new().unwrap();
        for i in 0..50 {
            create_test_file(temp_dir.path(), &format!("file{}.rs", i), "content");
        }

        let tool = CodebaseSearchTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.search_files("file*.rs", 10).unwrap();

        assert!(result.truncated);
        assert_eq!(result.files.len(), 10);
    }
}
