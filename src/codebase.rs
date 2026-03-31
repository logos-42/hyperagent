//! 代码库理解模块 — 给系统全局架构感知和上下文记忆
//!
//! 功能：
//! 1. 扫描所有源文件，提取模块结构、公开 API、依赖关系
//! 2. 构建架构图（哪个文件做什么、文件之间怎么调用）
//! 3. 持久化到 .hyperagent/codebase_context.json
//! 4. 每次研究迭代时注入上下文，让 LLM "看见全局"

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// A single improvement record with structured data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementRecord {
    /// When the improvement was made
    pub timestamp: DateTime<Utc>,
    /// File that was modified
    pub file: String,
    /// The hypothesis being tested
    pub hypothesis: String,
    /// Outcome: "Improved", "Regressed", "Neutral"
    pub outcome: String,
}

impl ImprovementRecord {
    /// Create a new improvement record
    pub fn new(file: String, hypothesis: String, outcome: String) -> Self {
        Self {
            timestamp: Utc::now(),
            file,
            hypothesis,
            outcome,
        }
    }

    /// Format for display in logs and prompts
    pub fn to_display_string(&self) -> String {
        format!(
            "[{}] {} — {} ({})",
            self.timestamp.format("%H:%M"),
            self.file,
            self.hypothesis.chars().take(80).collect::<String>(),
            self.outcome,
        )
    }

    /// Returns a compact one-line summary of the improvement record.
    ///
    /// Format: `[{time}] {file}: {hypothesis_truncated} → {outcome}`
    /// Useful for compact logging and LLM context windows.
    ///
    /// # Example
    /// ```ignore
    /// let record = ImprovementRecord::new("agent/mod.rs".into(), "Add caching".into(), "Improved".into());
    /// assert_eq!(record.summary(), "[14:30] agent/mod.rs: Add caching → Improved");
    /// ```
    pub fn summary(&self) -> String {
        let hypothesis_short: String = self.hypothesis.chars().take(50).collect();
        format!(
            "[{}] {}: {} → {}",
            self.timestamp.format("%H:%M"),
            self.file,
            hypothesis_short,
            self.outcome
        )
    }
}

/// 单个文件的代码摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSummary {
    /// 文件路径（相对 src/）
    pub path: String,
    /// 文件行数
    pub lines: usize,
    /// pub struct 名称（包含泛型参数）
    pub structs: Vec<String>,
    /// pub enum 名称
    pub enums: Vec<String>,
    /// pub fn / pub async fn 名称
    pub functions: Vec<String>,
    /// pub trait 名称
    pub traits: Vec<String>,
    /// impl 块（impl Xxx）
    pub impls: Vec<String>,
    /// use 语句（外部依赖）
    pub uses: Vec<String>,
    /// mod 声明（子模块）
    pub mods: Vec<String>,
    /// 文件头部的文档注释（//! 或 /// 第一段）
    pub doc_summary: String,
    /// derive 宏属性
    pub derives: Vec<String>,
}

impl FileSummary {
    /// Returns a compact one-line summary of the file's key metadata.
    ///
    /// Format: `{path} ({lines} lines, {structs} structs, {enums} enums, {functions} functions, {traits} traits)`
    /// Useful for logging and compact display in context windows.
    ///
    /// # Example
    /// ```ignore
    /// let summary = FileSummary { path: "agent/mod.rs".into(), lines: 193, ... };
    /// assert_eq!(summary.summary(), "agent/mod.rs (193 lines, 2 structs, 0 enums, 6 functions, 0 traits)");
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "{} ({} lines, {} structs, {} enums, {} functions, {} traits)",
            self.path,
            self.lines,
            self.structs.len(),
            self.enums.len(),
            self.functions.len(),
            self.traits.len()
        )
    }
}

/// 代码库全局上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseContext {
    /// 总文件数
    pub total_files: usize,
    /// 总代码行数
    pub total_lines: usize,
    /// 每个文件的摘要
    pub files: HashMap<String, FileSummary>,
    /// 模块树（从 mod.rs 推导）
    pub module_tree: String,
    /// 架构概要（人类可读）
    pub architecture_summary: String,
    /// 上次扫描时间
    pub last_scanned: String,
    /// 累计迭代次数（跨运行持续增长）
    pub total_iterations: u32,
    /// 改进历史记录（结构化存储）
    pub improvement_history: Vec<ImprovementRecord>,
}

impl CodebaseContext {
    /// 扫描代码库，构建全局上下文
    pub fn scan(project_root: &str) -> Result<Self> {
        let src_dir = PathBuf::from(project_root).join("src");
        let mut files = HashMap::new();
        let mut total_lines = 0usize;

        // 递归扫描所有 .rs 文件（包括子目录）
        Self::scan_dir(&src_dir, &src_dir, &mut files, &mut total_lines);

        let total_files = files.len();
        let module_tree = Self::build_module_tree(&files);
        let architecture_summary = Self::build_architecture_summary(&files);

        Ok(Self {
            total_files,
            total_lines,
            files,
            module_tree,
            architecture_summary,
            last_scanned: chrono::Utc::now().to_rfc3339(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        })
    }

    /// 递归扫描目录下所有 .rs 文件
    fn scan_dir(
        base: &std::path::Path,
        dir: &std::path::Path,
        files: &mut HashMap<String, FileSummary>,
        total_lines: &mut usize,
    ) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::scan_dir(base, &path, files, total_lines);
                } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
                    let rel = path
                        .strip_prefix(base)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();

                    if let Ok(content) = std::fs::read_to_string(&path) {
                        let summary = Self::summarize_file(&rel, &content);
                        *total_lines += summary.lines;
                        files.insert(rel.clone(), summary);
                    }
                }
            }
        }
    }

    /// 分析单个文件，提取结构信息
    fn summarize_file(path: &str, content: &str) -> FileSummary {
        let lines_count = content.lines().count();

        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let mut functions = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut uses = Vec::new();
        let mut mods = Vec::new();
        let mut derives = Vec::new();

        // Process content as a single string for multi-line handling
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();
            let _full_line = lines[i];

            // 提取文档注释摘要（文件头部的 //!）
            if line.starts_with("//!") {
                i += 1;
                continue;
            }

            // Handle derive attributes - collect them for the next item
            if line.starts_with("#[derive(") {
                if let Some(inner) = line
                    .strip_prefix("#[derive(")
                    .and_then(|s| s.strip_suffix(")]"))
                {
                    for derive in inner.split(',') {
                        derives.push(derive.trim().to_string());
                    }
                }
                i += 1;
                continue;
            }

            // Handle outer doc comments
            if line.starts_with("///") {
                i += 1;
                continue;
            }

            // Handle attributes like #[...]
            if line.starts_with("#[") && !line.starts_with("#[derive(") {
                i += 1;
                continue;
            }

            // pub struct - handle multi-line and generic parameters
            if let Some(name) = Self::extract_type_name(line, "pub struct") {
                // Extract generics if present on same line
                let full_signature = if line.contains('{') || line.contains(';') {
                    // Single line struct
                    name
                } else {
                    // Multi-line struct - look for name with generics
                    let next_line = if i + 1 < lines.len() {
                        lines[i + 1].trim()
                    } else {
                        ""
                    };
                    if next_line.starts_with('<') {
                        // Collect generic parameters
                        let mut generic_str = String::new();
                        let _j = i;
                        let mut depth = 0;
                        let mut started = false;

                        // Look backwards and forwards for generic params
                        for ch in line.chars() {
                            if ch == '<' {
                                started = true;
                                depth += 1;
                            }
                            if started {
                                generic_str.push(ch);
                            }
                            if ch == '>' {
                                depth -= 1;
                                if depth == 0 && started {
                                    break;
                                }
                            }
                        }

                        if !generic_str.is_empty() {
                            format!("{}{}", name, generic_str)
                        } else {
                            name
                        }
                    } else {
                        name
                    }
                };
                structs.push(full_signature);
                i += 1;
                continue;
            }

            // pub enum
            if let Some(name) = Self::extract_type_name(line, "pub enum") {
                enums.push(name);
                i += 1;
                continue;
            }

            // pub trait
            if let Some(name) = Self::extract_type_name(line, "pub trait") {
                traits.push(name);
                i += 1;
                continue;
            }

            // pub fn / pub async fn
            if let Some(name) = Self::extract_fn_name(line) {
                functions.push(name);
                i += 1;
                continue;
            }

            // impl Xxx or impl<T> Xxx
            if line.starts_with("impl ") || line.starts_with("impl<") {
                // Handle impl<T> Trait for Type and impl Type
                let impl_content = if line.starts_with("impl<") {
                    // impl<T> Something or impl<T> Trait for Type
                    line.to_string()
                } else {
                    line.strip_prefix("impl ").unwrap_or(line).to_string()
                };

                // Extract the primary type name
                let name = if let Some(stripped) = impl_content.strip_prefix("impl<") {
                    // Generic impl - find the main type after >
                    if let Some(pos) = stripped.find('>') {
                        let after_generic = &stripped[pos + 1..].trim();
                        // Skip "for" if present, get the first type
                        let type_str = after_generic
                            .strip_prefix("for ")
                            .unwrap_or(after_generic)
                            .trim();
                        type_str
                            .split_whitespace()
                            .next()
                            .unwrap_or(type_str)
                            .split('<')
                            .next()
                            .unwrap_or(type_str)
                            .to_string()
                    } else {
                        stripped
                            .split_whitespace()
                            .next()
                            .unwrap_or(stripped)
                            .to_string()
                    }
                } else {
                    impl_content
                        .split_whitespace()
                        .next()
                        .map(|s| s.split('<').next().unwrap_or(s).to_string())
                        .unwrap_or_default()
                };

                if !name.is_empty() && !name.starts_with('{') {
                    impls.push(name);
                }
                i += 1;
                continue;
            }

            // use xxx::yyy
            if line.starts_with("use ") {
                uses.push(line.to_string());
                i += 1;
                continue;
            }

            // mod xxx
            if line.starts_with("pub mod ") || line.starts_with("mod ") {
                let prefix = if line.starts_with("pub mod ") {
                    "pub mod "
                } else {
                    "mod "
                };
                if let Some(rest) = line.strip_prefix(prefix) {
                    mods.push(rest.split(';').next().unwrap_or(rest).trim().to_string());
                }
                i += 1;
                continue;
            }

            i += 1;
        }

        // 提取文件头部文档注释，在字符边界截断以避免截断多字节字符（如中文）
        let doc_summary = {
            let full_doc: String = content
                .lines()
                .take_while(|l| l.trim().starts_with("//!"))
                .map(|l| l.trim().trim_start_matches("//!").trim())
                .collect::<Vec<_>>()
                .join(" ");

            if full_doc.len() <= 200 {
                full_doc
            } else {
                // 按字符截断，确保在字符边界而非字节边界截断
                let chars: String = full_doc.chars().take(180).collect();
                format!("{}...", chars)
            }
        };

        FileSummary {
            path: path.to_string(),
            lines: lines_count,
            structs,
            enums,
            functions,
            traits,
            impls,
            uses,
            mods,
            doc_summary,
            derives,
        }
    }

    /// 从行中提取 `keyword Name` 形式的名称（支持泛型）
    fn extract_type_name(line: &str, keyword: &str) -> Option<String> {
        let trimmed = line.trim();
        if !trimmed.starts_with(keyword) {
            return None;
        }

        let rest = trimmed.strip_prefix(keyword)?.trim();

        // Skip derive attributes that might be inline like #[derive(Debug)]
        let rest = if rest.starts_with("#[derive(") {
            // Find the end of derive
            if let Some(end) = rest.find(")]") {
                rest[end + 2..].trim()
            } else {
                rest
            }
        } else {
            rest
        };

        // Skip other attributes
        let rest = if rest.starts_with("#[") {
            if let Some(end) = rest.find(']') {
                rest[end + 1..].trim()
            } else {
                rest
            }
        } else {
            rest
        };

        // Skip 'pub' if present (for nested visibility)
        let rest = rest.strip_prefix("pub ").unwrap_or(rest).trim();

        // Extract name, stopping at generics, braces, or parens
        let name: String = rest
            .chars()
            .take_while(|&c| c.is_alphanumeric() || c == '_')
            .collect();

        if name.is_empty() {
            return None;
        }

        // Check if there are generic parameters on the same line
        let after_name = rest.strip_prefix(&name)?;
        let after_name = after_name.trim();

        if after_name.starts_with('<') {
            // Extract generic parameters
            let mut generics = String::new();
            let mut depth = 0;
            let mut started = false;

            for ch in after_name.chars() {
                if ch == '<' {
                    depth += 1;
                    started = true;
                }
                if started {
                    generics.push(ch);
                }
                if ch == '>' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }

            if !generics.is_empty() {
                return Some(format!("{}{}", name, generics));
            }
        }

        Some(name)
    }

    /// 提取 pub fn / pub async fn 名称
    fn extract_fn_name(line: &str) -> Option<String> {
        let trimmed = line.trim();

        // Skip attributes
        let trimmed = if trimmed.starts_with("#[") {
            if let Some(pos) = trimmed.find(']') {
                trimmed[pos + 1..].trim()
            } else {
                trimmed
            }
        } else {
            trimmed
        };

        let rest = if trimmed.starts_with("pub async fn ") {
            Some(trimmed.strip_prefix("pub async fn ").unwrap())
        } else if trimmed.starts_with("pub fn ") {
            Some(trimmed.strip_prefix("pub fn ").unwrap())
        } else {
            None
        };

        rest.and_then(|s| {
            // First split at '(' to get the function signature part
            let before_paren = s.split('(').next()?;
            // Then strip generic parameters if present (e.g., "foo<T>" -> "foo")
            let name = before_paren.split('<').next()?;
            Some(name.trim().to_string())
        })
    }

    /// 构建模块树的可视化
    fn build_module_tree(files: &HashMap<String, FileSummary>) -> String {
        let mut tree = String::from("src/\n");
        let mut entries: Vec<_> = files.keys().collect();
        entries.sort();

        for path in &entries {
            let summary = files.get(*path).unwrap();
            let indent = "  ".repeat(path.matches('/').count());
            let items: Vec<String> = summary
                .structs
                .iter()
                .chain(summary.enums.iter())
                .chain(summary.traits.iter())
                .take(5)
                .cloned()
                .collect();
            let items_str = if items.is_empty() {
                String::new()
            } else {
                format!(" → {}", items.join(", "))
            };
            tree.push_str(&format!(
                "{}{} ({} lines){}\n",
                indent, path, summary.lines, items_str
            ));
        }

        tree
    }

    /// 构建人类可读的架构概要
    fn build_architecture_summary(files: &HashMap<String, FileSummary>) -> String {
        let mut sections = Vec::new();

        // 按目录分组
        let mut dirs: HashMap<String, Vec<&FileSummary>> = HashMap::new();
        for (path, summary) in files {
            let dir = if path.contains('/') {
                path.rsplit_once('/').unwrap().0.to_string()
            } else {
                "root".to_string()
            };
            dirs.entry(dir).or_default().push(summary);
        }

        let mut dir_keys: Vec<_> = dirs.keys().collect();
        dir_keys.sort();

        for dir in dir_keys {
            let summaries = dirs.get(dir).unwrap();
            let mut dir_desc = format!("## {} ({})\n", dir, dir);

            for s in summaries {
                let types: Vec<String> = s
                    .structs
                    .iter()
                    .map(|n| format!("struct {}", n))
                    .chain(s.enums.iter().map(|n| format!("enum {}", n)))
                    .chain(s.traits.iter().map(|n| format!("trait {}", n)))
                    .collect();

                let fns_str: String = s
                    .functions
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");

                let doc = if s.doc_summary.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", s.doc_summary)
                };

                dir_desc.push_str(&format!(
                    "- `{}`{} ({} lines)\n  Types: {}\n  Functions: {}\n",
                    s.path,
                    doc,
                    s.lines,
                    if types.is_empty() {
                        "—".to_string()
                    } else {
                        types.join(", ")
                    },
                    if fns_str.is_empty() {
                        "—".to_string()
                    } else {
                        fns_str
                    },
                ));
            }

            sections.push(dir_desc);
        }

        sections.join("\n")
    }

    /// 加载已有的上下文（如果存在）
    pub fn load(path: &PathBuf) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_else(|| Self {
                total_files: 0,
                total_lines: 0,
                files: HashMap::new(),
                module_tree: String::new(),
                architecture_summary: String::new(),
                last_scanned: String::new(),
                total_iterations: 0,
                improvement_history: Vec::new(),
            })
    }

    /// 保存到磁盘
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 记录一次改进
    pub fn record_improvement(&mut self, file: &str, hypothesis: &str, outcome: &str) {
        let record = ImprovementRecord::new(
            file.to_string(),
            hypothesis.to_string(),
            outcome.to_string(),
        );
        self.improvement_history.push(record);
        // 只保留最近 50 条
        if self.improvement_history.len() > 50 {
            self.improvement_history = self
                .improvement_history
                .split_off(self.improvement_history.len() - 50);
        }
        self.total_iterations += 1;
    }

    /// Check if a file has been recently modified (within last N entries)
    pub fn was_recently_modified(&self, file: &str, within_last: usize) -> bool {
        self.improvement_history
            .iter()
            .rev()
            .take(within_last)
            .any(|record| record.file == file)
    }

    /// Get files modified in the last N iterations
    pub fn recent_modified_files(&self, n: usize) -> Vec<String> {
        let mut files = Vec::new();
        for record in self.improvement_history.iter().rev().take(n) {
            if !files.contains(&record.file) {
                files.push(record.file.clone());
            }
        }
        files
    }

    /// Get all improvement records for a specific file.
    ///
    /// Returns records in reverse chronological order (most recent first).
    /// This enables analysis of experiment patterns on specific files to
    /// identify what hypotheses have already been tested.
    ///
    /// # Arguments
    /// * `file` - The file path to get improvements for
    ///
    /// # Returns
    /// A vector of references to improvement records for the file
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// for record in ctx.get_file_improvements("agent/mutator.rs") {
    ///     println!("{}: {} -> {}", record.timestamp, record.hypothesis, record.outcome);
    /// }
    /// ```
    pub fn get_file_improvements(&self, file: &str) -> Vec<&ImprovementRecord> {
        self.improvement_history
            .iter()
            .rev()
            .filter(|record| record.file == file)
            .collect()
    }

    /// Get improvement records filtered by outcome type.
    ///
    /// Useful for analyzing what worked (Improved), what didn't (Regressed),
    /// or what had no effect (Neutral) to guide future experiments.
    ///
    /// # Arguments
    /// * `outcome` - The outcome filter: "Improved", "Regressed", or "Neutral"
    /// * `limit` - Maximum number of records to return (None for all)
    ///
    /// # Returns
    /// A vector of references to matching improvement records
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// // Get all successful improvements
    /// let successes = ctx.improvements_by_outcome("Improved", None);
    /// // Get last 5 regressions
    /// let regressions = ctx.improvements_by_outcome("Regressed", Some(5));
    /// ```
    pub fn improvements_by_outcome(
        &self,
        outcome: &str,
        limit: Option<usize>,
    ) -> Vec<&ImprovementRecord> {
        let filtered: Vec<&ImprovementRecord> = self
            .improvement_history
            .iter()
            .rev()
            .filter(|record| record.outcome == outcome)
            .collect();

        match limit {
            Some(n) => filtered.into_iter().take(n).collect(),
            None => filtered,
        }
    }

    /// Count improvements by outcome type for a specific file.
    ///
    /// Returns a tuple of (improved, regressed, neutral) counts for the file,
    /// enabling quick assessment of whether a file is a good target for experiments.
    ///
    /// # Arguments
    /// * `file` - The file path to analyze
    ///
    /// # Returns
    /// A tuple of (improved_count, regressed_count, neutral_count)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let (improved, regressed, neutral) = ctx.count_file_outcomes("agent/mod.rs");
    /// let success_rate = improved as f64 / (improved + regressed + neutral).max(1) as f64;
    /// ```
    pub fn count_file_outcomes(&self, file: &str) -> (u32, u32, u32) {
        let mut improved = 0u32;
        let mut regressed = 0u32;
        let mut neutral = 0u32;

        for record in &self.improvement_history {
            if record.file == file {
                match record.outcome.as_str() {
                    "Improved" => improved += 1,
                    "Regressed" => regressed += 1,
                    "Neutral" => neutral += 1,
                    _ => {}
                }
            }
        }

        (improved, regressed, neutral)
    }

    /// Clear the improvement history while preserving the scanned codebase structure.
    ///
    /// This is useful when starting a fresh research session, allowing the system
    /// to track improvements from a clean slate without re-scanning the codebase.
    ///
    /// # Example
    /// ```ignore
    /// let mut ctx = CodebaseContext::scan(project_root)?;
    /// ctx.record_improvement("test.rs", "fix", "Improved");
    /// assert_eq!(ctx.improvement_history.len(), 1);
    ///
    /// ctx.clear_history();
    /// assert!(ctx.improvement_history.is_empty());
    /// assert_eq!(ctx.total_iterations, 0);
    /// // Codebase structure (files, module_tree, etc.) remains intact
    /// assert!(ctx.total_files > 0);
    /// ```
    pub fn clear_history(&mut self) {
        self.improvement_history.clear();
        self.total_iterations = 0;
    }

    /// Get improvement records within a specific time window.
    ///
    /// Filters the improvement history to return only records that fall within
    /// the specified time range (inclusive on both ends). This enables time-based
    /// analysis of experiment patterns, complementing the count-based queries.
    ///
    /// # Arguments
    /// * `start` - The start of the time window (inclusive)
    /// * `end` - The end of the time window (inclusive)
    ///
    /// # Returns
    /// A vector of references to improvement records within the time window,
    /// in chronological order (oldest first).
    ///
    /// # Example
    /// ```ignore
    /// use chrono::{Duration, Utc};
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let now = Utc::now();
    /// let hour_ago = now - Duration::hours(1);
    /// let recent = ctx.get_improvements_in_time_window(&hour_ago, &now);
    /// println!("{} experiments in the last hour", recent.len());
    /// ```
    pub fn get_improvements_in_time_window(
        &self,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Vec<&ImprovementRecord> {
        self.improvement_history
            .iter()
            .filter(|record| &record.timestamp >= start && &record.timestamp <= end)
            .collect()
    }

    /// Calculate the success rate for a specific file based on improvement history.
    ///
    /// Returns the percentage of experiments that resulted in "Improved" outcomes
    /// for the given file, or `None` if the file has no recorded improvements.
    /// This helps prioritize files that have been receptive to past experiments.
    ///
    /// # Arguments
    /// * `file` - The file path to analyze
    ///
    /// # Returns
    /// * `Some(f64)` - Success rate as a percentage (0.0 to 100.0)
    /// * `None` - If no improvements recorded for this file
    ///
    /// # Example
    /// ```ignore
    /// let mut ctx = CodebaseContext::scan(project_root)?;
    /// ctx.record_improvement("agent/mod.rs", "refactor", "Improved");
    /// ctx.record_improvement("agent/mod.rs", "optimize", "Regressed");
    /// ctx.record_improvement("agent/mod.rs", "fix", "Improved");
    ///
    /// let rate = ctx.calculate_success_rate("agent/mod.rs");
    /// assert_eq!(rate, Some(66.66666666666666)); // 2 out of 3 improved
    ///
    /// let no_data = ctx.calculate_success_rate("other.rs");
    /// assert_eq!(no_data, None);
    /// ```
    pub fn calculate_success_rate(&self, file: &str) -> Option<f64> {
        let (improved, regressed, neutral) = self.count_file_outcomes(file);
        let total = improved + regressed + neutral;

        if total == 0 {
            return None;
        }

        Some((improved as f64 / total as f64) * 100.0)
    }

    /// Find files with the highest improvement success rates.
    ///
    /// Returns files sorted by success rate (descending), useful for identifying
    /// which files have been most receptive to improvements and may be good
    /// candidates for future experiments.
    ///
    /// # Arguments
    /// * `min_experiments` - Minimum number of experiments required to be included
    ///
    /// # Returns
    /// A vector of (file_path, success_rate, total_experiments) tuples, sorted by
    /// success rate descending, then by total experiments descending.
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let top_files = ctx.top_performing_files(2); // At least 2 experiments
    /// for (file, rate, count) in top_files {
    ///     println!("{}: {:.1}% success ({} experiments)", file, rate, count);
    /// }
    /// ```
    pub fn top_performing_files(&self, min_experiments: usize) -> Vec<(String, f64, usize)> {
        let mut file_stats: HashMap<&str, (u32, u32, u32)> = HashMap::new();

        for record in &self.improvement_history {
            let entry = file_stats.entry(&record.file).or_default();
            match record.outcome.as_str() {
                "Improved" => entry.0 += 1,
                "Regressed" => entry.1 += 1,
                "Neutral" => entry.2 += 1,
                _ => {}
            }
        }

        let mut results: Vec<(String, f64, usize)> = file_stats
            .into_iter()
            .filter_map(|(file, (improved, regressed, neutral))| {
                let total = (improved + regressed + neutral) as usize;
                if total < min_experiments {
                    return None;
                }
                let rate = (improved as f64 / total as f64) * 100.0;
                Some((file.to_string(), rate, total))
            })
            .collect();

        // Sort by success rate descending, then by total experiments descending
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.2.cmp(&a.2))
        });

        results
    }

    /// 生成给 LLM 的全局上下文 prompt
    /// target_file: 当前要改进的文件
    pub fn build_context_prompt(&self, target_file: &str) -> String {
        let mut prompt = String::new();

        // 1. 架构概要
        prompt.push_str("=== CODEBASE ARCHITECTURE ===\n");
        prompt.push_str(&format!(
            "Total: {} files, {} lines, {} iterations completed\n\n",
            self.total_files, self.total_lines, self.total_iterations
        ));
        prompt.push_str(&self.architecture_summary);
        prompt.push_str("\n");

        // 2. 目标文件的依赖信息
        if let Some(target_summary) = self.files.get(target_file) {
            // 这个文件被哪些其他文件依赖（通过 use 语句反查）
            let dependents: Vec<String> = self
                .files
                .iter()
                .filter(|(path, summary)| {
                    *path != target_file
                        && summary.uses.iter().any(|u| {
                            // 简单匹配：如果 use 语句中包含目标文件路径的相关模块
                            let module_hint = target_file
                                .strip_suffix(".rs")
                                .unwrap_or(target_file)
                                .replace('/', "::")
                                .replace("mod.rs", "");
                            u.contains(&module_hint)
                                || u.contains(&target_file.replace('/', "::").replace(".rs", ""))
                        })
                })
                .map(|(path, _)| path.clone())
                .collect();

            // 目标文件依赖了什么
            let target_uses: Vec<String> = target_summary
                .uses
                .iter()
                .filter(|u| u.contains("crate::"))
                .cloned()
                .collect();

            prompt.push_str(&format!("\n=== TARGET: {} ===\n", target_file));
            prompt.push_str(&format!(
                "Lines: {}, Structs: {:?}, Functions: {:?}\n",
                target_summary.lines,
                target_summary.structs,
                target_summary.functions.iter().take(10).collect::<Vec<_>>(),
            ));
            if !target_summary.enums.is_empty() {
                prompt.push_str(&format!(
                    "Enums: {}\n",
                    target_summary
                        .enums
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !target_summary.traits.is_empty() {
                prompt.push_str(&format!(
                    "Traits: {}\n",
                    target_summary
                        .traits
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !target_summary.impls.is_empty() {
                prompt.push_str(&format!(
                    "Impls: {}\n",
                    target_summary
                        .impls
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !target_summary.derives.is_empty() {
                // Show unique derives across all types in this file
                let unique_derives: Vec<String> = target_summary
                    .derives
                    .iter()
                    .filter(|d| !d.is_empty())
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .take(8)
                    .collect();
                if !unique_derives.is_empty() {
                    prompt.push_str(&format!("Derives: {}\n", unique_derives.join(", ")));
                }
            }
            if !target_uses.is_empty() {
                prompt.push_str(&format!("Internal deps: {}\n", target_uses.join(", ")));
            }
            if !dependents.is_empty() {
                prompt.push_str(&format!("Depended by: {}\n", dependents.join(", ")));
            }
            prompt.push_str(&format!("Doc: {}\n", target_summary.doc_summary));
        }

        // 3. 改进历史
        if !self.improvement_history.is_empty() {
            prompt.push_str("\n=== RECENT IMPROVEMENTS ===\n");
            let recent: Vec<&ImprovementRecord> =
                self.improvement_history.iter().rev().take(10).collect();
            for record in recent.into_iter().rev() {
                prompt.push_str(&record.to_display_string());
                prompt.push('\n');
            }
        }

        // 4. Recently modified files hint
        let recent_files = self.recent_modified_files(5);
        if !recent_files.is_empty() {
            prompt.push_str("\n=== RECENTLY MODIFIED FILES ===\n");
            prompt.push_str(&format!(
                "Consider other files: {}\n",
                recent_files.join(", ")
            ));
            // Show activity summary for context
            let activity = self.file_activity_summary(5);
            if !activity.is_empty() {
                prompt.push_str(&format!("Activity: {}\n", activity));
            }
        }

        // 5. Related files context (files that use this file's items)
        if let Some(target_summary) = self.files.get(target_file) {
            prompt.push_str("\n=== RELATED FILES ===\n");

            // This file depends on these modules via use statements
            let target_uses: Vec<String> = target_summary
                .uses
                .iter()
                .filter(|u| u.contains("crate::"))
                .cloned()
                .collect();

            if !target_uses.is_empty() {
                prompt.push_str(&format!("Uses: {}\n", target_uses.join(", ")));
            }

            // Include detailed content of dependent files (max 3 for context window)
            let related_content = self.get_related_files_content(target_file, 3);
            if !related_content.is_empty() {
                prompt.push_str(&related_content);
            }
        }

        prompt
    }

    /// 刷新扫描（重新读取文件）
    pub fn refresh(&mut self, project_root: &str) {
        if let Ok(mut fresh) = Self::scan(project_root) {
            // 保留历史数据
            fresh.total_iterations = self.total_iterations;
            fresh.improvement_history = self.improvement_history.clone();
            *self = fresh;
        }
    }

    /// Get a reference to the file summary for a given path, if it exists.
    ///
    /// This provides convenient access to individual file metadata without
    /// requiring callers to handle Option<&String> key lookups.
    ///
    /// # Arguments
    /// * `path` - The file path relative to src/ (e.g., "agent/mutator.rs")
    ///
    /// # Returns
    /// * `Some(&FileSummary)` if the file exists in the context
    /// * `None` if the file is not found
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// if let Some(summary) = ctx.get_file_summary("agent/mutator.rs") {
    ///     println!("File has {} lines and {} functions", summary.lines, summary.functions.len());
    /// }
    /// ```
    pub fn get_file_summary(&self, path: &str) -> Option<&FileSummary> {
        self.files.get(path)
    }

    /// Find all files that depend on the given target file.
    ///
    /// Returns a list of file paths that have `use` statements referencing the target file's module.
    /// This is useful for understanding the downstream impact of changes before making them.
    ///
    /// # Arguments
    /// * `target_file` - The file path to find dependents for (e.g., "agent/mod.rs")
    ///
    /// # Returns
    /// A vector of file paths that depend on the target file, sorted alphabetically
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let dependents = ctx.find_dependents("agent/mod.rs");
    /// // Returns ["auto_research/mod.rs", "bin/self_evolve.rs", ...]
    /// ```
    pub fn find_dependents(&self, target_file: &str) -> Vec<String> {
        let module_hint = target_file
            .strip_suffix(".rs")
            .unwrap_or(target_file)
            .replace('/', "::")
            .replace("mod.rs", "");

        self.files
            .iter()
            .filter_map(|(path, summary)| {
                if *path == target_file {
                    return None;
                }

                let has_dependency = summary.uses.iter().any(|u| {
                    u.contains(&module_hint)
                        || u.contains(&target_file.replace('/', "::").replace(".rs", ""))
                });

                if has_dependency {
                    Some(path.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all public items across the entire codebase.
    ///
    /// Returns a sorted list of all public structs, enums, traits, and functions
    /// with their file paths, useful for API discovery and LLM context injection.
    ///
    /// # Arguments
    /// * `include_functions` - Whether to include public functions in the result
    ///
    /// # Returns
    /// A vector of tuples containing (file_path, item_type, item_name)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let items = ctx.get_all_public_items(true);
    /// for (file, kind, name) in items {
    ///     println!("{} in {} is a {}", name, file, kind);
    /// }
    /// ```
    pub fn get_all_public_items(&self, include_functions: bool) -> Vec<(String, String, String)> {
        let mut items = Vec::new();

        for (path, summary) in &self.files {
            // Add structs
            for name in &summary.structs {
                items.push((path.clone(), "struct".to_string(), name.clone()));
            }

            // Add enums
            for name in &summary.enums {
                items.push((path.clone(), "enum".to_string(), name.clone()));
            }

            // Add traits
            for name in &summary.traits {
                items.push((path.clone(), "trait".to_string(), name.clone()));
            }

            // Add functions if requested
            if include_functions {
                for name in &summary.functions {
                    items.push((path.clone(), "fn".to_string(), name.clone()));
                }
            }
        }

        // Sort by file path, then by item type, then by name
        items.sort_by(|a, b| (&a.0, &a.1, &a.2).cmp(&(&b.0, &b.1, &b.2)));

        items
    }

    /// Get summaries for multiple files at once.
    ///
    /// This is a batch version of `get_file_summary()` that returns summaries for
    /// multiple file paths in a single call. Files that don't exist in the context
    /// are silently skipped.
    ///
    /// # Arguments
    /// * `paths` - Slice of file paths to retrieve summaries for
    ///
    /// # Returns
    /// A vector of references to FileSummary for existing files
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let summaries = ctx.get_file_summaries(&["agent/mod.rs", "lib.rs", "nonexistent.rs"]);
    /// // Returns summaries for agent/mod.rs and lib.rs only
    /// for summary in summaries {
    ///     println!("{}: {} lines", summary.path, summary.lines);
    /// }
    /// ```
    pub fn get_file_summaries<'a>(&'a self, paths: &[&str]) -> Vec<&'a FileSummary> {
        paths
            .iter()
            .filter_map(|path| self.files.get(*path))
            .collect()
    }

    /// Find files that have no recorded improvement experiments.
    ///
    /// These "fresh" files haven't been experimented on and may represent unexplored
    /// opportunities for improvement. Useful for identifying new research targets.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of files to return (None for all)
    ///
    /// # Returns
    /// A vector of file paths sorted by line count descending (larger files first
    /// as a heuristic for potential impact)
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let fresh_files = ctx.files_without_improvements(Some(10));
    /// println!("Unexplored files: {:?}", fresh_files);
    /// ```
    pub fn files_without_improvements(&self, limit: Option<usize>) -> Vec<String> {
        let mut unexplored: Vec<String> = self
            .files
            .keys()
            .filter(|path| {
                !self
                    .improvement_history
                    .iter()
                    .any(|record| &record.file == *path)
            })
            .cloned()
            .collect();

        // Sort by line count descending (larger files may have more impact)
        unexplored.sort_by(|a, b| {
            let a_lines = self.files.get(a).map(|f| f.lines).unwrap_or(0);
            let b_lines = self.files.get(b).map(|f| f.lines).unwrap_or(0);
            b_lines.cmp(&a_lines)
        });

        match limit {
            Some(n) => unexplored.into_iter().take(n).collect(),
            None => unexplored,
        }
    }

    /// Get formatted content summaries of files that depend on a target file.
    ///
    /// This is useful for understanding the downstream impact of changes to a file.
    /// Returns a formatted string with each dependent file's summary, limited to
    /// the top N dependents by line count (larger files first as heuristic for importance).
    ///
    /// # Arguments
    /// * `target_file` - The file path to find dependents for
    /// * `max_files` - Maximum number of dependent files to include
    ///
    /// # Returns
    /// A formatted string with dependent file summaries, or empty string if none found
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// let related = ctx.get_related_files_content("agent/mod.rs", 3);
    /// // Returns formatted summaries of files that use agent/mod.rs
    /// ```
    pub fn get_related_files_content(&self, target_file: &str, max_files: usize) -> String {
        let dependents: Vec<(&FileSummary, Vec<String>)> = self
            .files
            .iter()
            .filter_map(|(path, summary)| {
                if *path == target_file {
                    return None;
                }

                // Find which items from target file are used by this dependent
                let module_hint = target_file
                    .strip_suffix(".rs")
                    .unwrap_or(target_file)
                    .replace('/', "::")
                    .replace("mod.rs", "");

                let used_items: Vec<String> = summary
                    .uses
                    .iter()
                    .filter(|u| {
                        u.contains(&module_hint)
                            || u.contains(&target_file.replace('/', "::").replace(".rs", ""))
                    })
                    .filter_map(|u| {
                        // Extract specific imported items from use statements
                        // e.g., "use crate::agent::Agent;" -> "Agent"
                        // e.g., "use crate::agent::{Agent, MutationStrategy};" -> "Agent, MutationStrategy"
                        if let Some(stripped) = u.strip_prefix("use crate::") {
                            let rest = stripped.trim_end_matches(';');
                            // Check if this use statement references our module
                            if rest.starts_with(&module_hint)
                                || rest.contains(&format!("{}::", module_hint))
                            {
                                // Extract the item name(s)
                                if let Some(after_module) = rest.strip_prefix(&module_hint) {
                                    let after_module = after_module.trim_start_matches(':');
                                    if after_module.starts_with('{') {
                                        // Multiple items: use crate::module::{Item1, Item2}
                                        let items = after_module
                                            .trim_matches('{')
                                            .trim_matches('}')
                                            .split(',')
                                            .map(|s| s.trim().to_string())
                                            .collect::<Vec<_>>();
                                        return Some(items);
                                    } else if !after_module.is_empty() {
                                        // Single item: use crate::module::Item
                                        let item = after_module.trim().to_string();
                                        if !item.is_empty() {
                                            return Some(vec![item]);
                                        }
                                    }
                                }
                            }
                        }
                        None
                    })
                    .flatten()
                    .collect();

                if used_items.is_empty() {
                    None
                } else {
                    Some((summary, used_items))
                }
            })
            .collect();

        if dependents.is_empty() {
            return String::new();
        }

        // Sort by line count descending (larger files likely more important)
        let mut sorted_dependents = dependents;
        sorted_dependents.sort_by(|a, b| b.0.lines.cmp(&a.0.lines));

        let mut result = String::new();
        for (summary, used_items) in sorted_dependents.into_iter().take(max_files) {
            result.push_str(&format!(
                "\n--- {} ({} lines) ---\n",
                summary.path, summary.lines
            ));

            // Include doc summary if available
            if !summary.doc_summary.is_empty() {
                result.push_str(&format!("Doc: {}\n", summary.doc_summary));
            }

            // Show which items from target are used
            result.push_str(&format!("Uses from target: {}\n", used_items.join(", ")));

            // Include key types
            if !summary.structs.is_empty() {
                result.push_str(&format!(
                    "Types: {}\n",
                    summary
                        .structs
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }

            // Include derives for understanding type capabilities
            if !summary.derives.is_empty() {
                let unique_derives: Vec<String> = summary
                    .derives
                    .iter()
                    .filter(|d| !d.is_empty())
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .take(6)
                    .collect();
                if !unique_derives.is_empty() {
                    result.push_str(&format!("Derives: {}\n", unique_derives.join(", ")));
                }
            }

            // Include key functions
            if !summary.functions.is_empty() {
                result.push_str(&format!(
                    "Functions: {}\n",
                    summary
                        .functions
                        .iter()
                        .take(8)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        result
    }

    /// Returns a compact one-line summary of the codebase context.
    ///
    /// Format: `{total_files} files, {total_lines} lines, {iterations} iterations, {improvements} improvements`
    /// Useful for logging and quick status checks.
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// println!("{}", ctx.summary());
    /// // Output: "43 files, 25061 lines, 126 iterations, 10 improvements"
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "{} files, {} lines, {} iterations, {} improvements",
            self.total_files,
            self.total_lines,
            self.total_iterations,
            self.improvement_history.len()
        )
    }

    /// Returns a summary of improvement outcomes across all files.
    ///
    /// Format: `{improved} improved, {regressed} regressed, {neutral} neutral ({success_rate:.1}% success)`
    /// Useful for quick assessment of overall experiment effectiveness.
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// println!("{}", ctx.outcome_summary());
    /// // Output: "15 improved, 3 regressed, 7 neutral (60.0% success)"
    /// ```
    pub fn outcome_summary(&self) -> String {
        let mut improved = 0u32;
        let mut regressed = 0u32;
        let mut neutral = 0u32;

        for record in &self.improvement_history {
            match record.outcome.as_str() {
                "Improved" => improved += 1,
                "Regressed" => regressed += 1,
                "Neutral" => neutral += 1,
                _ => {}
            }
        }

        let total = improved + regressed + neutral;
        let success_rate = if total > 0 {
            (improved as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "{} improved, {} regressed, {} neutral ({:.1}% success)",
            improved, regressed, neutral, success_rate
        )
    }

    /// Returns a ranked summary of files by improvement activity count.
    ///
    /// This helps identify "hot" files that have been the focus of recent experiments,
    /// useful for understanding research patterns and avoiding repeated experiments
    /// on the same files.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of files to include in the summary
    ///
    /// # Returns
    /// A formatted string with file activity, sorted by count descending
    ///
    /// # Example
    /// ```ignore
    /// let ctx = CodebaseContext::scan(project_root)?;
    /// println!("{}", ctx.file_activity_summary(5));
    /// // Output: "agent/mod.rs (7), lib.rs (5), tools.rs (3)"
    /// ```
    pub fn file_activity_summary(&self, limit: usize) -> String {
        let mut file_counts: HashMap<&str, usize> = HashMap::new();

        for record in &self.improvement_history {
            *file_counts.entry(&record.file).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = file_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        sorted
            .into_iter()
            .take(limit)
            .map(|(file, count)| format!("{} ({})", file, count))
            .collect::<Vec<_>>()
            .join(", ")
    }
}
