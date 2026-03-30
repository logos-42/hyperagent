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

impl GrepMatch {
    /// Extract the matched text from the line using match_ranges.
    /// Returns all matched substrings concatenated together.
    /// 
    /// # Example
    /// ```
    /// let m = GrepMatch { line: "fn hello() {}".to_string(), match_ranges: vec![(3, 8)], ... };
    /// assert_eq!(m.matched_text(), vec!["hello"]);
    /// ```
    pub fn matched_text(&self) -> Vec<&str> {
        self.match_ranges
            .iter()
            .map(|(start, end)| {
                self.line.get(*start..*end).unwrap_or("")
            })
            .collect()
    }
    
    /// Format the match with full context as a single string.
    /// Includes context lines before and after with proper indentation.
    /// 
    /// # Example output:
    /// ```text
    /// src/lib.rs:42:     fn example() {
    /// src/lib.rs:43: >>> fn target_function() {  <<< MATCH
    /// src/lib.rs:44:         println!("hello");
    /// ```
    pub fn format_context(&self) -> String {
        let mut result = String::new();
        
        // Add context before with file reference
        for ctx_line in &self.context_before {
            result.push_str(&format!("{}:{}:   {}\n", self.file, self.line_number.saturating_sub(self.context_before.len()), ctx_line));
        }
        
        // Add the matched line with marker
        result.push_str(&format!("{}:{}: >>> {} <<< MATCH\n", 
            self.file, self.line_number, self.line));
        
        // Add context after
        for (i, ctx_line) in self.context_after.iter().enumerate() {
            result.push_str(&format!("{}:{}:   {}\n", 
                self.file, 
                self.line_number + i + 1, 
                ctx_line));
        }
        
        result
    }
    
    /// Check if this match has any match ranges.
    pub fn has_matches(&self) -> bool {
        !self.match_ranges.is_empty()
    }
    
    /// Get the total length of all matched text.
    pub fn matched_length(&self) -> usize {
        self.match_ranges.iter().map(|(start, end)| end - start).sum()
    }
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

impl GrepOutput {
    /// Returns a human-readable one-line summary of the grep results.
    /// 
    /// # Example outputs:
    /// - `"Found 15 matches in 3 files for 'fn main'"`
    /// - `"Found 5 matches in 1 file for 'TODO' (truncated, showing first 5 of 20)"`
    /// - `"No matches found for 'nonexistent_pattern' in 10 files"`
    pub fn summary(&self) -> String {
        let truncated_note = if self.truncated {
            format!(" (truncated, showing first {} of {})", self.matches.len(), self.total_matches)
        } else {
            String::new()
        };
        
        let files_word = if self.files_searched == 1 { "file" } else { "files" };
        
        if self.matches.is_empty() {
            format!("No matches found for '{}' in {} {}", self.pattern, self.files_searched, files_word)
        } else {
            let matches_word = if self.matches.len() == 1 { "match" } else { "matches" };
            let unique_files: std::collections::HashSet<&str> = self.matches.iter()
                .map(|m| m.file.as_str())
                .collect();
            let files_with_matches = unique_files.len();
            let files_word2 = if files_with_matches == 1 { "file" } else { "files" };
            
            format!("Found {} {} in {} {} for '{}'{}", 
                self.matches.len(), matches_word, files_with_matches, files_word2, self.pattern, truncated_note)
        }
    }
    
    /// Returns the unique files that contain matches.
    /// Useful for quickly identifying which files need attention.
    pub fn files_with_matches(&self) -> Vec<&str> {
        let unique_files: std::collections::HashSet<&str> = self.matches.iter()
            .map(|m| m.file.as_str())
            .collect();
        let mut files: Vec<&str> = unique_files.into_iter().collect();
        files.sort();
        files
    }
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

impl SearchFilesOutput {
    /// Returns a human-readable one-line summary of the search results.
    ///
    /// # Example outputs:
    /// - `"Found 15 files matching '*.rs'"`
    /// - `"Found 3 files matching '*config*' (truncated, showing first 3 of 50)"`
    /// - `"No files found matching 'nonexistent*.txt'"`
    pub fn summary(&self) -> String {
        let truncated_note = if self.truncated {
            format!(" (truncated, showing first {} of {})", self.files.len(), self.total_found)
        } else {
            String::new()
        };

        if self.files.is_empty() {
            format!("No files found matching '{}'", self.pattern)
        } else {
            let files_word = if self.files.len() == 1 { "file" } else { "files" };
            format!("Found {} {} matching '{}'{}", self.files.len(), files_word, self.pattern, truncated_note)
        }
    }

    /// Returns files sorted by size (smallest first).
    /// Useful for identifying large files that may need attention.
    pub fn files_by_size(&self) -> Vec<&FileEntry> {
        let mut files: Vec<&FileEntry> = self.files.iter().collect();
        files.sort_by_key(|f| f.size_bytes);
        files
    }

    /// Returns files sorted by size (largest first).
    /// Useful for quickly identifying the largest matching files.
    pub fn files_by_size_desc(&self) -> Vec<&FileEntry> {
        let mut files: Vec<&FileEntry> = self.files.iter().collect();
        files.sort_by_key(|f| std::cmp::Reverse(f.size_bytes));
        files
    }

    /// Returns the total size in bytes of all matched files.
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
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

impl TreeOutput {
    /// Format the tree as a visual string with indentation, similar to `tree` command.
    /// Returns a string representation that makes hierarchy clear at a glance.
    ///
    /// # Example output:
    /// ```text
    /// src/
    /// ├── agent/
    /// │   ├── mod.rs
    /// │   └── executor.rs
    /// └── lib.rs
    /// ```
    pub fn display_tree(&self) -> String {
        let mut result = String::new();
        
        for entry in &self.entries {
            // Indent based on depth
            let indent = "    ".repeat(entry.depth);
            
            // Choose prefix character based on entry type
            let prefix = if entry.is_dir { "📁 " } else { "📄 " };
            
            // Extract just the file/directory name from the path
            let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
            
            // Format with size for files
            let size_str = match (entry.is_dir, entry.size_bytes) {
                (false, Some(size)) => format!(" ({} bytes)", format_size(size)),
                _ => String::new(),
            };
            
            result.push_str(&format!("{}{}{}{}\n", indent, prefix, name, size_str));
        }
        
        // Add summary
        result.push_str(&format!(
            "\n{} directories, {} files",
            self.total_dirs, self.total_files
        ));
        
        result
    }
}

/// Format byte size in human-readable form
fn format_size(size: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if size >= GB {
        format!("{:.1} GB", size as f64 / GB as f64)
    } else if size >= MB {
        format!("{:.1} MB", size as f64 / MB as f64)
    } else if size >= KB {
        format!("{:.1} KB", size as f64 / KB as f64)
    } else {
        format!("{} B", size)
    }
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
    fn test_search_files_output_summary_empty() {
        let output = SearchFilesOutput {
            pattern: "*.nonexistent".to_string(),
            files: vec![],
            total_found: 0,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "No files found matching '*.nonexistent'");
    }

    #[test]
    fn test_search_files_output_summary_single_file() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![
                FileEntry { path: "src/main.rs".to_string(), size_bytes: 100 },
            ],
            total_found: 1,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "Found 1 file matching '*.rs'");
    }

    #[test]
    fn test_search_files_output_summary_multiple_files() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![
                FileEntry { path: "src/a.rs".to_string(), size_bytes: 100 },
                FileEntry { path: "src/b.rs".to_string(), size_bytes: 200 },
            ],
            total_found: 2,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "Found 2 files matching '*.rs'");
    }

    #[test]
    fn test_search_files_output_summary_truncated() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![
                FileEntry { path: "src/a.rs".to_string(), size_bytes: 100 },
                FileEntry { path: "src/b.rs".to_string(), size_bytes: 200 },
            ],
            total_found: 50,
            truncated: true,
        };
        
        assert_eq!(output.summary(), "Found 2 files matching '*.rs' (truncated, showing first 2 of 50)");
    }

    #[test]
    fn test_search_files_output_files_by_size() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![
                FileEntry { path: "large.rs".to_string(), size_bytes: 1000 },
                FileEntry { path: "small.rs".to_string(), size_bytes: 100 },
                FileEntry { path: "medium.rs".to_string(), size_bytes: 500 },
            ],
            total_found: 3,
            truncated: false,
        };
        
        let sorted = output.files_by_size();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].path, "small.rs");
        assert_eq!(sorted[0].size_bytes, 100);
        assert_eq!(sorted[1].path, "medium.rs");
        assert_eq!(sorted[2].path, "large.rs");
    }

    #[test]
    fn test_search_files_output_files_by_size_desc() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![
                FileEntry { path: "large.rs".to_string(), size_bytes: 1000 },
                FileEntry { path: "small.rs".to_string(), size_bytes: 100 },
                FileEntry { path: "medium.rs".to_string(), size_bytes: 500 },
            ],
            total_found: 3,
            truncated: false,
        };
        
        let sorted = output.files_by_size_desc();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].path, "large.rs");
        assert_eq!(sorted[0].size_bytes, 1000);
        assert_eq!(sorted[1].path, "medium.rs");
        assert_eq!(sorted[2].path, "small.rs");
    }

    #[test]
    fn test_search_files_output_total_size() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![
                FileEntry { path: "a.rs".to_string(), size_bytes: 100 },
                FileEntry { path: "b.rs".to_string(), size_bytes: 200 },
                FileEntry { path: "c.rs".to_string(), size_bytes: 300 },
            ],
            total_found: 3,
            truncated: false,
        };
        
        assert_eq!(output.total_size(), 600);
    }

    #[test]
    fn test_search_files_output_total_size_empty() {
        let output = SearchFilesOutput {
            pattern: "*.rs".to_string(),
            files: vec![],
            total_found: 0,
            truncated: false,
        };
        
        assert_eq!(output.total_size(), 0);
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

    #[test]
    fn test_grep_match_matched_text() {
        let temp = GrepMatch {
            file: "test.rs".to_string(),
            line_number: 10,
            line: "fn hello_world() {".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_ranges: vec![(3, 8), (9, 14)],
        };
        
        let matched = temp.matched_text();
        assert_eq!(matched, vec!["hello", "_world"]);
    }

    #[test]
    fn test_grep_match_matched_text_empty_ranges() {
        let temp = GrepMatch {
            file: "test.rs".to_string(),
            line_number: 10,
            line: "fn hello() {".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_ranges: vec![],
        };
        
        let matched = temp.matched_text();
        assert!(matched.is_empty());
    }

    #[test]
    fn test_grep_match_has_matches() {
        let with_matches = GrepMatch {
            file: "test.rs".to_string(),
            line_number: 1,
            line: "fn test()".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_ranges: vec![(0, 2)],
        };
        assert!(with_matches.has_matches());
        
        let without_matches = GrepMatch {
            file: "test.rs".to_string(),
            line_number: 1,
            line: "fn test()".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_ranges: vec![],
        };
        assert!(!without_matches.has_matches());
    }

    #[test]
    fn test_grep_match_matched_length() {
        let temp = GrepMatch {
            file: "test.rs".to_string(),
            line_number: 1,
            line: "fn hello_world() {}".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_ranges: vec![(3, 8), (9, 14)],
        };
        
        // "hello" is 5 chars, "_world" is 6 chars, total = 11
        assert_eq!(temp.matched_length(), 11);
    }

    #[test]
    fn test_grep_match_format_context() {
        let temp = GrepMatch {
            file: "src/lib.rs".to_string(),
            line_number: 43,
            line: "fn target_function() {".to_string(),
            context_before: vec!["41:     fn example() {".to_string(), "42: }".to_string()],
            context_after: vec!["44:     println!(\"hello\");".to_string()],
            match_ranges: vec![(3, 19)],
        };
        
        let formatted = temp.format_context();
        
        // Should contain file reference
        assert!(formatted.contains("src/lib.rs"));
        // Should contain the match marker
        assert!(formatted.contains(">>>"));
        assert!(formatted.contains("MATCH"));
        // Should contain context before
        assert!(formatted.contains("41:"));
        assert!(formatted.contains("example"));
        // Should contain context after
        assert!(formatted.contains("44:"));
        assert!(formatted.contains("println"));
    }

    #[test]
    fn test_grep_match_format_context_no_context() {
        let temp = GrepMatch {
            file: "test.rs".to_string(),
            line_number: 1,
            line: "fn main() {}".to_string(),
            context_before: vec![],
            context_after: vec![],
            match_ranges: vec![(3, 7)],
        };
        
        let formatted = temp.format_context();
        
        // Should still show the match line
        assert!(formatted.contains("test.rs:1"));
        assert!(formatted.contains(">>> fn main() {} <<< MATCH"));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(1023), "1023 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1048576), "1.0 MB");
        assert_eq!(format_size(1572864), "1.5 MB");
        assert_eq!(format_size(1073741824), "1.0 GB");
    }

    #[test]
    fn test_tree_output_display_tree_empty() {
        let output = TreeOutput {
            base_dir: "src".to_string(),
            entries: vec![],
            total_files: 0,
            total_dirs: 0,
        };
        
        let display = output.display_tree();
        assert!(display.contains("0 directories, 0 files"));
    }

    #[test]
    fn test_tree_output_display_tree_with_files() {
        let output = TreeOutput {
            base_dir: "src".to_string(),
            entries: vec![
                TreeEntry { path: "src/agent".to_string(), is_dir: true, size_bytes: None, depth: 0 },
                TreeEntry { path: "src/agent/mod.rs".to_string(), is_dir: false, size_bytes: Some(100), depth: 1 },
                TreeEntry { path: "src/lib.rs".to_string(), is_dir: false, size_bytes: Some(200), depth: 0 },
            ],
            total_files: 2,
            total_dirs: 1,
        };
        
        let display = output.display_tree();
        
        // Should show directory icon for dirs
        assert!(display.contains("📁"));
        // Should show file icon for files
        assert!(display.contains("📄"));
        // Should show file sizes
        assert!(display.contains("100 bytes"));
        assert!(display.contains("200 bytes"));
        // Should show summary
        assert!(display.contains("1 directories, 2 files"));
    }

    #[test]
    fn test_tree_output_display_tree_indentation() {
        let output = TreeOutput {
            base_dir: "src".to_string(),
            entries: vec![
                TreeEntry { path: "src/top.txt".to_string(), is_dir: false, size_bytes: Some(10), depth: 0 },
                TreeEntry { path: "src/nested".to_string(), is_dir: true, size_bytes: None, depth: 0 },
                TreeEntry { path: "src/nested/inner".to_string(), is_dir: true, size_bytes: None, depth: 1 },
                TreeEntry { path: "src/nested/inner/deep.txt".to_string(), is_dir: false, size_bytes: Some(5), depth: 2 },
            ],
            total_files: 2,
            total_dirs: 2,
        };
        
        let display = output.display_tree();
        let lines: Vec<&str> = display.lines().collect();
        
        // Check indentation increases with depth
        assert!(lines[0].starts_with("📄 top.txt"));  // depth 0, no indent
        assert!(lines[1].starts_with("📁 nested"));   // depth 0, no indent
        assert!(lines[2].starts_with("    📁 inner")); // depth 1, one indent
        assert!(lines[3].starts_with("        📄 deep.txt")); // depth 2, two indents
    }

    #[test]
    fn test_grep_output_summary_no_matches() {
        let output = GrepOutput {
            pattern: "nonexistent".to_string(),
            matches: vec![],
            total_matches: 0,
            files_searched: 10,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "No matches found for 'nonexistent' in 10 files");
    }

    #[test]
    fn test_grep_output_summary_single_match() {
        let output = GrepOutput {
            pattern: "fn main".to_string(),
            matches: vec![
                GrepMatch {
                    file: "src/lib.rs".to_string(),
                    line_number: 1,
                    line: "fn main() {}".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![(0, 7)],
                },
            ],
            total_matches: 1,
            files_searched: 5,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "Found 1 match in 1 file for 'fn main'");
    }

    #[test]
    fn test_grep_output_summary_multiple_matches() {
        let output = GrepOutput {
            pattern: "TODO".to_string(),
            matches: vec![
                GrepMatch {
                    file: "src/a.rs".to_string(),
                    line_number: 10,
                    line: "// TODO fix".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![(3, 7)],
                },
                GrepMatch {
                    file: "src/b.rs".to_string(),
                    line_number: 20,
                    line: "// TODO refactor".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![(3, 7)],
                },
            ],
            total_matches: 2,
            files_searched: 3,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "Found 2 matches in 2 files for 'TODO'");
    }

    #[test]
    fn test_grep_output_summary_truncated() {
        let output = GrepOutput {
            pattern: "fn".to_string(),
            matches: vec![
                GrepMatch {
                    file: "src/a.rs".to_string(),
                    line_number: 1,
                    line: "fn one()".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![(0, 2)],
                },
                GrepMatch {
                    file: "src/a.rs".to_string(),
                    line_number: 5,
                    line: "fn two()".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![(0, 2)],
                },
            ],
            total_matches: 20,
            files_searched: 5,
            truncated: true,
        };
        
        assert_eq!(output.summary(), "Found 2 matches in 1 file for 'fn' (truncated, showing first 2 of 20)");
    }

    #[test]
    fn test_grep_output_summary_single_file_searched() {
        let output = GrepOutput {
            pattern: "test".to_string(),
            matches: vec![],
            total_matches: 0,
            files_searched: 1,
            truncated: false,
        };
        
        assert_eq!(output.summary(), "No matches found for 'test' in 1 file");
    }

    #[test]
    fn test_grep_output_files_with_matches_empty() {
        let output = GrepOutput {
            pattern: "test".to_string(),
            matches: vec![],
            total_matches: 0,
            files_searched: 5,
            truncated: false,
        };
        
        assert!(output.files_with_matches().is_empty());
    }

    #[test]
    fn test_grep_output_files_with_matches_single_file() {
        let output = GrepOutput {
            pattern: "fn".to_string(),
            matches: vec![
                GrepMatch {
                    file: "src/lib.rs".to_string(),
                    line_number: 1,
                    line: "fn one()".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![],
                },
                GrepMatch {
                    file: "src/lib.rs".to_string(),
                    line_number: 10,
                    line: "fn two()".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![],
                },
            ],
            total_matches: 2,
            files_searched: 3,
            truncated: false,
        };
        
        let files = output.files_with_matches();
        assert_eq!(files, vec!["src/lib.rs"]);
    }

    #[test]
    fn test_grep_output_files_with_matches_multiple_files() {
        let output = GrepOutput {
            pattern: "TODO".to_string(),
            matches: vec![
                GrepMatch {
                    file: "src/c.rs".to_string(),
                    line_number: 1,
                    line: "// TODO".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![],
                },
                GrepMatch {
                    file: "src/a.rs".to_string(),
                    line_number: 5,
                    line: "// TODO".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![],
                },
                GrepMatch {
                    file: "src/b.rs".to_string(),
                    line_number: 10,
                    line: "// TODO".to_string(),
                    context_before: vec![],
                    context_after: vec![],
                    match_ranges: vec![],
                },
            ],
            total_matches: 3,
            files_searched: 5,
            truncated: false,
        };
        
        let files = output.files_with_matches();
        // Should be sorted alphabetically
        assert_eq!(files, vec!["src/a.rs", "src/b.rs", "src/c.rs"]);
    }
}
