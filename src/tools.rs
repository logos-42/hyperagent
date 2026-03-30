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
use std::io::{BufRead, BufReader, Read};
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

/// Check if a file is binary by sampling the first 8KB.
/// Returns true if null bytes are found (common in binary files).
/// This is a heuristic - some binary files may not have nulls in the first 8KB,
/// but this catches most common cases efficiently.
fn is_binary_file(path: &Path) -> bool {
    if let Ok(mut file) = std::fs::File::open(path) {
        let mut sample = [0u8; 8192];
        if let Ok(n) = file.read(&mut sample) {
            // Check for null bytes in the sample
            return sample[..n].contains(&b'\0');
        }
    }
    false
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
        let mut should_stop = false;

        Self::walk_files(&src_dir, &root_path, file_ext, &mut |path, lines| {
            files_searched += 1;
            if should_stop {
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
                        should_stop = true;
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
    ) -> Result<bool>
    where
        F: FnMut(String, Vec<String>),
    {
        if !dir.exists() {
            return Ok(false);
        }
        Self::walk_dir_recursive(dir, root, file_ext, callback)
    }

    fn walk_dir_recursive<F>(dir: &Path, root: &Path, file_ext: &str, callback: &mut F) -> Result<bool>
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
                if Self::walk_dir_recursive(&path, root, file_ext, callback)? {
                    return Ok(true);
                }
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

                // Skip binary files early (before reading all lines into memory)
                // This avoids loading large binary files entirely
                if is_binary_file(&path) {
                    continue;
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
        Ok(false)
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
// rig Tool: codebase_write
// ---------------------------------------------------------------------------

/// Arguments for the codebase_write tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct WriteFileArgs {
    /// File path relative to project root (e.g. "src/lib.rs", "Cargo.toml")
    pub path: String,
    /// Content to write to the file
    pub content: String,
    /// Whether to create parent directories if they don't exist (default: true)
    #[serde(default = "default_create_dirs")]
    pub create_dirs: bool,
    /// Whether to append to existing file (default: false = overwrite)
    #[serde(default)]
    pub append: bool,
}

fn default_create_dirs() -> bool {
    true
}

/// Output of the codebase_write tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct WriteFileOutput {
    pub path: String,
    pub bytes_written: usize,
    pub created: bool,
    pub appended: bool,
}

/// Arguments for the codebase_delete tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct DeleteFileArgs {
    /// File path relative to project root (e.g. "src/deprecated.rs")
    pub path: String,
}

/// Output of the codebase_delete tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct DeleteFileOutput {
    pub path: String,
    pub deleted: bool,
    pub bytes_freed: u64,
}

/// Delete a file from the codebase.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CodebaseDeleteTool {
    #[serde(skip, default = "default_project_root")]
    root: ProjectRoot,
}

impl CodebaseDeleteTool {
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

    /// Direct delete (without rig Tool machinery).
    pub fn delete_file(&self, path: &str) -> Result<DeleteFileOutput> {
        let full_path = self.root.path.join(path);
        
        // Canonicalize both paths for security check
        let canonical_root = self.root.path.canonicalize()
            .unwrap_or_else(|_| self.root.path.clone());
        
        // Check if file exists before canonicalizing
        if !full_path.exists() {
            return Ok(DeleteFileOutput {
                path: path.to_string(),
                deleted: false,
                bytes_freed: 0,
            });
        }
        
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
        
        // Only allow file deletion, not directories
        if canonical_full.is_dir() {
            return Err(CodebaseToolError::FileError(
                "Cannot delete directories, only files".to_string()
            ).into());
        }
        
        // Get file size before deletion
        let metadata = std::fs::metadata(&canonical_full)
            .with_context(|| format!("Cannot read metadata for {}", path))?;
        let bytes_freed = metadata.len();
        
        // Delete the file
        std::fs::remove_file(&canonical_full)
            .with_context(|| format!("Cannot delete {}", path))?;
        
        Ok(DeleteFileOutput {
            path: path.to_string(),
            deleted: true,
            bytes_freed,
        })
    }
}

impl Default for CodebaseDeleteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CodebaseDeleteTool {
    const NAME: &'static str = "codebase_delete";

    type Error = CodebaseToolError;
    type Args = DeleteFileArgs;
    type Output = DeleteFileOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "codebase_delete".to_string(),
            description: "Delete a file from the codebase. Use this to remove deprecated code, \
                          clean up dead code, or restructure modules. Only works on files, not \
                          directories. Paths are relative to project root.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root (e.g. 'src/deprecated.rs')"
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
        let root = self.root.clone();
        async move {
            let tool = CodebaseDeleteTool { root };
            tool.delete_file(&path)
                .map_err(|e| CodebaseToolError::FileError(e.to_string()))
        }
    }
}

/// Write content to a file in the codebase.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct CodebaseWriteTool {
    #[serde(skip, default = "default_project_root")]
    root: ProjectRoot,
}

impl CodebaseWriteTool {
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

    /// Direct write (without rig Tool machinery).
    pub fn write_file(&self, path: &str, content: &str, create_dirs: bool, append: bool) -> Result<WriteFileOutput> {
        let full_path = self.root.path.join(path);
        
        // Canonicalize root path for security check
        let canonical_root = self.root.path.canonicalize()
            .unwrap_or_else(|_| self.root.path.clone());
        
        // Check if file already exists before we create it
        let existed = full_path.exists();
        
        // Create parent directories if needed
        if create_dirs {
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Cannot create directories for {}", full_path.display()))?;
            }
        }
        
        // Security check: ensure resolved path is within project root
        // Only check if the path exists (can't canonicalize non-existent files)
        if existed {
            let canonical_full = full_path.canonicalize()
                .map_err(|e| CodebaseToolError::PathError(
                    format!("Cannot resolve path '{}': {}", path, e)
                ))?;
            
            if !canonical_full.starts_with(&canonical_root) {
                return Err(CodebaseToolError::PathError(
                    format!("Path '{}' resolves outside project root", path)
                ).into());
            }
        } else {
            // For new files, check that the parent is within the project root
            if let Some(parent) = full_path.parent() {
                if parent.exists() {
                    let canonical_parent = parent.canonicalize()
                        .map_err(|e| CodebaseToolError::PathError(
                            format!("Cannot resolve parent directory: {}", e)
                        ))?;
                    
                    if !canonical_parent.starts_with(&canonical_root) {
                        return Err(CodebaseToolError::PathError(
                            format!("Path '{}' would be outside project root", path)
                        ).into());
                    }
                }
            }
        }
        
        // Write or append content
        let bytes_written = if append {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&full_path)
                .with_context(|| format!("Cannot open {} for appending", full_path.display()))?;
            file.write_all(content.as_bytes())
                .with_context(|| format!("Cannot append to {}", full_path.display()))?;
            content.len()
        } else {
            std::fs::write(&full_path, content)
                .with_context(|| format!("Cannot write to {}", full_path.display()))?;
            content.len()
        };

        Ok(WriteFileOutput {
            path: path.to_string(),
            bytes_written,
            created: !existed,
            appended: append,
        })
    }
}

impl Default for CodebaseWriteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for CodebaseWriteTool {
    const NAME: &'static str = "codebase_write";

    type Error = CodebaseToolError;
    type Args = WriteFileArgs;
    type Output = WriteFileOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "codebase_write".to_string(),
            description: "Write content to a file in the codebase. Creates new files or overwrites \
                          existing ones. Use this to implement code changes, create new modules, \
                          or update configuration. Paths are relative to project root.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root (e.g. 'src/lib.rs', 'Cargo.toml')"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist (default true)",
                        "default": true
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Append to existing file instead of overwriting (default false)",
                        "default": false
                    }
                },
                "required": ["path", "content"]
            }),
        };
        async move { def }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = std::result::Result<Self::Output, Self::Error>> + Send {
        let path = args.path.clone();
        let content = args.content.clone();
        let create_dirs = args.create_dirs;
        let append = args.append;
        let root = self.root.clone();
        async move {
            let tool = CodebaseWriteTool { root };
            tool.write_file(&path, &content, create_dirs, append)
                .map_err(|e| CodebaseToolError::FileError(e.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple glob matching (supports * and ?) using dynamic programming.
/// Achieves O(m*n) time complexity and O(n) space complexity
/// where m = pattern length, n = text length.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let m = p.len();
    let n = t.len();

    // Use two 1D arrays instead of a 2D matrix to reduce space from O(m*n) to O(n)
    // prev[j] = whether pattern[0..i-1] matches text[0..j]
    // curr[j] = whether pattern[0..i] matches text[0..j]
    let mut prev = vec![false; n + 1];
    let mut curr = vec![false; n + 1];

    // Empty pattern matches empty text
    prev[0] = true;

    // Fill first row: handle leading wildcards (pattern matching empty text)
    // This is computed in the main loop now, but we need to track if all
    // pattern chars so far are '*'
    for i in 1..=m {
        // Reset curr[0]: pattern[0..i] matches empty text iff all chars are '*'
        curr[0] = p[i - 1] == '*' && prev[0];
        
        for j in 1..=n {
            match p[i - 1] {
                '*' => {
                    // * can match:
                    // - empty string (prev[j] = dp[i-1][j])
                    // - one or more characters (curr[j-1] = dp[i][j-1])
                    curr[j] = prev[j] || curr[j - 1];
                }
                '?' => {
                    // ? matches exactly one character
                    curr[j] = prev[j - 1];
                }
                c => {
                    // Literal character must match
                    curr[j] = prev[j - 1] && c == t[j - 1];
                }
            }
        }
        
        // Swap prev and curr for next iteration
        std::mem::swap(&mut prev, &mut curr);
    }

    // After last swap, result is in prev (not curr)
    prev[n]
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

    #[test]
    fn test_write_new_file() {
        let temp_dir = TempDir::new().unwrap();
        let tool = CodebaseWriteTool::with_root(temp_dir.path().to_path_buf());
        
        let result = tool.write_file("src/new_file.rs", "fn main() {}", true, false).unwrap();
        
        assert_eq!(result.bytes_written, 14);
        assert!(result.created);
        assert!(!result.appended);
        
        // Verify file was created
        let content = std::fs::read_to_string(temp_dir.path().join("src/new_file.rs")).unwrap();
        assert_eq!(content, "fn main() {}");
    }

    #[test]
    fn test_write_overwrites_existing() {
        let temp_dir = TempDir::new().unwrap();
        create_test_file(temp_dir.path(), "test.rs", "old content");
        
        let tool = CodebaseWriteTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.write_file("test.rs", "new content", true, false).unwrap();
        
        assert!(!result.created);
        
        // Verify file was overwritten
        let content = std::fs::read_to_string(temp_dir.path().join("test.rs")).unwrap();
        assert_eq!(content, "new content");
    }

    #[test]
    fn test_write_append() {
        let temp_dir = TempDir::new().unwrap();
        create_test_file(temp_dir.path(), "test.rs", "line one\n");
        
        let tool = CodebaseWriteTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.write_file("test.rs", "line two\n", true, true).unwrap();
        
        assert!(result.appended);
        
        let content = std::fs::read_to_string(temp_dir.path().join("test.rs")).unwrap();
        assert_eq!(content, "line one\nline two\n");
    }

    #[test]
    fn test_write_creates_directories() {
        let temp_dir = TempDir::new().unwrap();
        let tool = CodebaseWriteTool::with_root(temp_dir.path().to_path_buf());
        
        let result = tool.write_file("deep/nested/path.rs", "content", true, false).unwrap();
        assert!(result.created);
        
        // Verify nested path was created
        assert!(temp_dir.path().join("deep/nested/path.rs").exists());
    }

    #[test]
    fn test_write_security_path_traversal() {
        let temp_dir = TempDir::new().unwrap();
        let tool = CodebaseWriteTool::with_root(temp_dir.path().to_path_buf());
        
        // Attempt to write outside project root
        let result = tool.write_file("../outside.rs", "content", true, false);
        assert!(result.is_err());
        
        let err = result.unwrap_err().to_string();
        assert!(err.contains("outside project root") || err.contains("Cannot resolve"));
    }

    #[test]
    fn test_grep_skips_binary_files() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create a binary file with null bytes
        let binary_content = "fn binary\x00test\x00content".as_bytes();
        fs::write(temp_dir.path().join("binary.rs"), binary_content).unwrap();
        
        // Create a normal text file with a match
        create_test_file(temp_dir.path(), "normal.rs", "fn hello() { println!(\"hello\"); }");
        
        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("hello", "rs", 10, 0).unwrap();
        
        // Should find match in normal.rs but skip binary.rs
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].file, "normal.rs");
        // files_searched only counts text files (binary files are skipped before callback)
        assert_eq!(result.files_searched, 1);
    }

    #[test]
    fn test_grep_binary_file_no_matches() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create a file with null bytes that would match the pattern
        let binary_content = "fn hello\x00world\x00".as_bytes();
        fs::write(temp_dir.path().join("binary.rs"), binary_content).unwrap();
        
        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("hello", "rs", 10, 0).unwrap();
        
        // Binary file should be skipped entirely
        assert_eq!(result.matches.len(), 0);
        // Binary file is not counted in files_searched since it's skipped early
        assert_eq!(result.files_searched, 0);
    }

    #[test]
    fn test_is_binary_file_detects_nulls() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create a text file
        create_test_file(temp_dir.path(), "text.rs", "fn main() {}");
        assert!(!is_binary_file(&temp_dir.path().join("text.rs")));
        
        // Create a binary file with null bytes at start
        let binary_content = "\x00\x01\x02\x03".as_bytes();
        fs::write(temp_dir.path().join("binary1.bin"), binary_content).unwrap();
        assert!(is_binary_file(&temp_dir.path().join("binary1.bin")));
        
        // Create a binary file with null bytes later (within 8KB)
        let large_binary = format!("{}\x00{}", "x".repeat(100), "y".repeat(100));
        fs::write(temp_dir.path().join("binary2.bin"), large_binary.as_bytes()).unwrap();
        assert!(is_binary_file(&temp_dir.path().join("binary2.bin")));
        
        // Create a binary file with null bytes beyond 8KB (won't be detected)
        let large_text = format!("{}\x00{}", "x".repeat(9000), "y".repeat(100));
        fs::write(temp_dir.path().join("large.bin"), large_text.as_bytes()).unwrap();
        // This should NOT be detected as binary (null is beyond 8KB sample)
        assert!(!is_binary_file(&temp_dir.path().join("large.bin")));
    }

    #[test]
    fn test_is_binary_file_handles_missing_file() {
        // Non-existent file should return false (not binary)
        let non_existent = PathBuf::from("/nonexistent/file/path");
        assert!(!is_binary_file(&non_existent));
    }

    #[test]
    fn test_is_binary_file_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        create_test_file(temp_dir.path(), "empty.txt", "");
        
        // Empty file is not binary
        assert!(!is_binary_file(&temp_dir.path().join("empty.txt")));
    }

    #[test]
    fn test_delete_existing_file() {
        let temp_dir = TempDir::new().unwrap();
        create_test_file(temp_dir.path(), "to_delete.rs", "content to delete");
        
        let tool = CodebaseDeleteTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.delete_file("to_delete.rs").unwrap();
        
        assert!(result.deleted);
        assert_eq!(result.bytes_freed, 18);
        assert_eq!(result.path, "to_delete.rs");
        
        // Verify file no longer exists
        assert!(!temp_dir.path().join("to_delete.rs").exists());
    }

    #[test]
    fn test_delete_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        
        let tool = CodebaseDeleteTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.delete_file("nonexistent.rs").unwrap();
        
        assert!(!result.deleted);
        assert_eq!(result.bytes_freed, 0);
    }

    #[test]
    fn test_delete_security_path_traversal() {
        let temp_dir = TempDir::new().unwrap();
        
        let tool = CodebaseDeleteTool::with_root(temp_dir.path().to_path_buf());
        
        // Attempt to delete file outside project root
        let result = tool.delete_file("../outside.rs");
        assert!(result.is_err());
        
        let err = result.unwrap_err().to_string();
        assert!(err.contains("outside project root") || err.contains("Cannot resolve"));
    }

    #[test]
    fn test_delete_directory_fails() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create a directory
        fs::create_dir(temp_dir.path().join("mydir")).unwrap();
        
        let tool = CodebaseDeleteTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.delete_file("mydir");
        
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Cannot delete directories"));
    }

    #[test]
    fn test_delete_nested_file() {
        let temp_dir = TempDir::new().unwrap();
        fs::create_dir_all(temp_dir.path().join("deep/nested")).unwrap();
        create_test_file(temp_dir.path(), "deep/nested/file.rs", "nested content");
        
        let tool = CodebaseDeleteTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.delete_file("deep/nested/file.rs").unwrap();
        
        assert!(result.deleted);
        assert_eq!(result.bytes_freed, 15);
        
        // Verify file deleted but parent dirs still exist
        assert!(!temp_dir.path().join("deep/nested/file.rs").exists());
        assert!(temp_dir.path().join("deep/nested").exists());
    }

    #[test]
    fn test_glob_match_empty_pattern_and_text() {
        assert!(glob_match("", ""));
        assert!(!glob_match("", "text"));
        assert!(!glob_match("a", ""));
    }

    #[test]
    fn test_glob_match_only_wildcards() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("**", "anything"));
        assert!(glob_match("***", "anything"));
    }

    #[test]
    fn test_glob_match_consecutive_wildcards() {
        assert!(glob_match("**.rs", "test.rs"));
        assert!(glob_match("src/**/*.rs", "src/deep/nested/file.rs"));
        assert!(glob_match("a**b", "axb"));
        assert!(glob_match("a**b", "axxxyyyb"));
    }

    #[test]
    fn test_glob_match_complex_patterns() {
        // Pattern with wildcards and literals mixed
        assert!(glob_match("test*.rs", "test.rs"));
        assert!(glob_match("test*.rs", "test_file.rs"));
        assert!(glob_match("test*backup.rs", "test_backup.rs"));
        assert!(glob_match("test*backup.rs", "test_old_backup.rs"));
        
        // Question marks
        assert!(glob_match("test?.rs", "test1.rs"));
        assert!(glob_match("test?.rs", "testA.rs"));
        assert!(!glob_match("test?.rs", "test.rs"));
        assert!(!glob_match("test?.rs", "test12.rs"));
    }

    #[test]
    fn test_glob_match_edge_cases() {
        // Single character match
        assert!(glob_match("?", "a"));
        assert!(!glob_match("?", "ab"));
        assert!(!glob_match("?", ""));
        
        // Pattern longer than text
        assert!(!glob_match("abcdef", "abc"));
        assert!(!glob_match("a?b?c?", "ab"));
        
        // Text longer than pattern without wildcards
        assert!(!glob_match("abc", "abcdef"));
        
        // Exact match
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "Exact")); // Case sensitive
    }

    #[test]
    fn test_glob_match_unicode() {
        assert!(glob_match("*.rs", "файл.rs"));
        assert!(glob_match("test*", "test文字"));
        assert!(glob_match("*测试*", "前置测试后置"));
    }

    #[test]
    fn test_grep_files_searched_count_accurate() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create multiple files, only one has match
        create_test_file(temp_dir.path(), "file1.rs", "fn one() {}");
        create_test_file(temp_dir.path(), "file2.rs", "fn two() {}");
        create_test_file(temp_dir.path(), "file3.rs", "fn target() {}");
        create_test_file(temp_dir.path(), "file4.rs", "fn four() {}");
        create_test_file(temp_dir.path(), "file5.rs", "fn five() {}");

        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("target", "rs", 10, 0).unwrap();

        // Should count all 5 files searched, even though only 1 matched
        assert_eq!(result.files_searched, 5);
        assert_eq!(result.matches.len(), 1);
    }

    #[test]
    fn test_grep_early_termination_saves_work() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create many files with matches
        for i in 0..100 {
            create_test_file(temp_dir.path(), &format!("file{}.rs", i), &format!("fn match_{}() {{}}", i));
        }

        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("match", "rs", 5, 0).unwrap();

        // Should stop after finding 5 matches
        assert_eq!(result.matches.len(), 5);
        assert!(result.truncated);
        // files_searched should be less than 100 due to early termination
        assert!(result.files_searched < 100, "Expected early termination but searched {} files", result.files_searched);
    }

    #[test]
    fn test_grep_files_searched_includes_non_matching() {
        let temp_dir = TempDir::new().unwrap();
        
        // Create files where only last file has match
        create_test_file(temp_dir.path(), "a.rs", "fn a() {}");
        create_test_file(temp_dir.path(), "b.rs", "fn b() {}");
        create_test_file(temp_dir.path(), "c.rs", "fn found_it() {}");

        let tool = CodebaseGrepTool::with_root(temp_dir.path().to_path_buf());
        let result = tool.grep("found_it", "rs", 10, 0).unwrap();

        // All 3 files should be counted as searched
        assert_eq!(result.files_searched, 3);
        assert_eq!(result.matches.len(), 1);
    }
}
