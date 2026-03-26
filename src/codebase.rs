//! 代码库理解模块 — 给系统全局架构感知和上下文记忆
//!
//! 功能：
//! 1. 扫描所有源文件，提取模块结构、公开 API、依赖关系
//! 2. 构建架构图（哪个文件做什么、文件之间怎么调用）
//! 3. 持久化到 .hyperagent/codebase_context.json
//! 4. 每次研究迭代时注入上下文，让 LLM "看见全局"

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::{DateTime, Utc};

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
                if let Some(inner) = line.strip_prefix("#[derive(").and_then(|s| s.strip_suffix(")]")) {
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
                    let next_line = if i + 1 < lines.len() { lines[i + 1].trim() } else { "" };
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
                        stripped.split_whitespace().next().unwrap_or(stripped).to_string()
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
                let prefix = if line.starts_with("pub mod ") { "pub mod " } else { "mod " };
                if let Some(rest) = line.strip_prefix(prefix) {
                    mods.push(
                        rest.split(';')
                            .next()
                            .unwrap_or(rest)
                            .trim()
                            .to_string(),
                    );
                }
                i += 1;
                continue;
            }

            i += 1;
        }

        // 提取文件头部文档注释
        let doc_summary = content
            .lines()
            .take_while(|l| l.trim().starts_with("//!"))
            .map(|l| l.trim().trim_start_matches("//!").trim())
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(200)
            .collect();

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

        rest?.split('(').next().map(|n| n.trim().to_string())
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
            tree.push_str(&format!("{}{} ({} lines){}\n", indent, path, summary.lines, items_str));
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
                    if types.is_empty() { "—".to_string() } else { types.join(", ") },
                    if fns_str.is_empty() { "—".to_string() } else { fns_str },
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
            self.improvement_history = self.improvement_history.split_off(self.improvement_history.len() - 50);
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
                    target_summary.enums.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
                ));
            }
            if !target_summary.traits.is_empty() {
                prompt.push_str(&format!(
                    "Traits: {}\n",
                    target_summary.traits.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
                ));
            }
            if !target_summary.impls.is_empty() {
                prompt.push_str(&format!(
                    "Impls: {}\n",
                    target_summary.impls.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
                ));
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
            let recent: Vec<&ImprovementRecord> = self.improvement_history.iter().rev().take(10).collect();
            for record in recent.into_iter().rev() {
                prompt.push_str(&record.to_display_string());
                prompt.push('\n');
            }
        }

        // 4. Recently modified files hint
        let recent_files = self.recent_modified_files(5);
        if !recent_files.is_empty() {
            prompt.push_str("\n=== RECENTLY MODIFIED FILES ===\n");
            prompt.push_str(&format!("Consider other files: {}\n", recent_files.join(", ")));
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
                prompt.push_str(&format!("Depends on: {}\n", target_uses.join(", ")));
            }
            
            // Find files that depend on this file
            let dependents: Vec<String> = self
                .files
                .iter()
                .filter(|(path, summary)| {
                    *path != target_file
                        && summary.uses.iter().any(|u| {
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
            
            if !dependents.is_empty() {
                prompt.push_str(&format!("Used by: {}\n", dependents.join(", ")));
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
                            if rest.starts_with(&module_hint) || rest.contains(&format!("{}::", module_hint)) {
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
                    summary.structs.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
                ));
            }
            
            // Include key functions
            if !summary.functions.is_empty() {
                result.push_str(&format!(
                    "Functions: {}\n",
                    summary.functions.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improvement_record_creation() {
        let record = ImprovementRecord::new(
            "src/agent/mutator.rs".to_string(),
            "Add mutation strategy".to_string(),
            "Improved".to_string(),
        );
        
        assert_eq!(record.file, "src/agent/mutator.rs");
        assert_eq!(record.hypothesis, "Add mutation strategy");
        assert_eq!(record.outcome, "Improved");
    }

    #[test]
    fn test_improvement_record_display_format() {
        let record = ImprovementRecord {
            timestamp: Utc::now(),
            file: "test.rs".to_string(),
            hypothesis: "A long hypothesis that should be truncated to 80 characters when displayed in the output".to_string(),
            outcome: "Improved".to_string(),
        };
        
        let display = record.to_display_string();
        assert!(display.contains("test.rs"));
        assert!(display.contains("Improved"));
        assert!(display.contains("["));
        // Hypothesis should be truncated
        let hypothesis_part: String = record.hypothesis.chars().take(80).collect();
        assert!(display.contains(&hypothesis_part[..20]));
    }

    #[test]
    fn test_record_improvement_adds_to_history() {
        let mut ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.record_improvement("test.rs", "Add feature", "Improved");
        
        assert_eq!(ctx.improvement_history.len(), 1);
        assert_eq!(ctx.total_iterations, 1);
        assert_eq!(ctx.improvement_history[0].file, "test.rs");
    }

    #[test]
    fn test_history_limit_50_entries() {
        let mut ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        // Add 60 entries
        for i in 0..60 {
            ctx.record_improvement(&format!("file{}.rs", i), "test", "Improved");
        }
        
        assert_eq!(ctx.improvement_history.len(), 50);
        // Should keep the most recent (file10.rs through file59.rs)
        assert_eq!(ctx.improvement_history[0].file, "file10.rs");
    }

    #[test]
    fn test_was_recently_modified() {
        let mut ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.record_improvement("file_a.rs", "test", "Improved");
        ctx.record_improvement("file_b.rs", "test", "Improved");
        ctx.record_improvement("file_a.rs", "test2", "Improved");
        
        // file_a.rs was modified in last 3 entries
        assert!(ctx.was_recently_modified("file_a.rs", 3));
        // file_a.rs was modified in last 2 entries
        assert!(ctx.was_recently_modified("file_a.rs", 2));
        // file_a.rs was NOT modified in last 1 entry (last is file_a.rs again, so true)
        assert!(ctx.was_recently_modified("file_a.rs", 1));
    }

    #[test]
    fn test_recent_modified_files() {
        let mut ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.record_improvement("file_a.rs", "test", "Improved");
        ctx.record_improvement("file_b.rs", "test", "Improved");
        ctx.record_improvement("file_a.rs", "test2", "Improved");
        ctx.record_improvement("file_c.rs", "test", "Improved");
        
        let recent = ctx.recent_modified_files(3);
        // Last 3 modifications: file_c.rs, file_a.rs, file_b.rs (in reverse order)
        assert_eq!(recent.len(), 3);
        assert!(recent.contains(&"file_c.rs".to_string()));
        assert!(recent.contains(&"file_a.rs".to_string()));
        assert!(recent.contains(&"file_b.rs".to_string()));
    }

    #[test]
    fn test_build_context_prompt_includes_structured_history() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 100,
            files: HashMap::new(),
            module_tree: "src/\n  test.rs (100 lines)\n".to_string(),
            architecture_summary: "Test architecture".to_string(),
            last_scanned: "2024-01-01T00:00:00Z".to_string(),
            total_iterations: 1,
            improvement_history: vec![
                ImprovementRecord::new("test.rs".to_string(), "Add tests".to_string(), "Improved".to_string()),
            ],
        };
        
        let prompt = ctx.build_context_prompt("test.rs");
        
        assert!(prompt.contains("RECENT IMPROVEMENTS"));
        assert!(prompt.contains("test.rs"));
        assert!(prompt.contains("Add tests"));
        assert!(prompt.contains("Improved"));
    }

    #[test]
    fn test_empty_history_returns_empty() {
        let ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        assert!(!ctx.was_recently_modified("any.rs", 5));
        assert!(ctx.recent_modified_files(5).is_empty());
    }

    #[test]
    fn test_record_serialization() {
        let record = ImprovementRecord {
            timestamp: Utc::now(),
            file: "test.rs".to_string(),
            hypothesis: "Test hypothesis".to_string(),
            outcome: "Improved".to_string(),
        };
        
        // Should serialize/deserialize correctly
        let json = serde_json::to_string(&record).unwrap();
        let deserialized: ImprovementRecord = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.file, "test.rs");
        assert_eq!(deserialized.hypothesis, "Test hypothesis");
        assert_eq!(deserialized.outcome, "Improved");
    }

    #[test]
    fn test_context_serialization() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 100,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 1,
            improvement_history: vec![
                ImprovementRecord::new("test.rs".to_string(), "test".to_string(), "Improved".to_string()),
            ],
        };
        
        let json = serde_json::to_string(&ctx).unwrap();
        let deserialized: CodebaseContext = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.total_iterations, 1);
        assert_eq!(deserialized.improvement_history.len(), 1);
        assert_eq!(deserialized.improvement_history[0].file, "test.rs");
    }

    #[test]
    fn test_get_file_summary_existing_file() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 100,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        let summary = FileSummary {
            path: "test/module.rs".to_string(),
            lines: 150,
            structs: vec!["MyStruct".to_string()],
            enums: vec![],
            functions: vec!["new".to_string(), "run".to_string()],
            traits: vec![],
            impls: vec!["MyStruct".to_string()],
            uses: vec!["use crate::other;".to_string()],
            mods: vec![],
            doc_summary: "Test module".to_string(),
            derives: vec!["Debug".to_string()],
        };
        
        ctx.files.insert("test/module.rs".to_string(), summary);
        
        let result = ctx.get_file_summary("test/module.rs");
        assert!(result.is_some());
        let found = result.unwrap();
        assert_eq!(found.lines, 150);
        assert_eq!(found.structs, vec!["MyStruct"]);
        assert_eq!(found.functions.len(), 2);
    }

    #[test]
    fn test_get_file_summary_nonexistent_file() {
        let ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        let result = ctx.get_file_summary("nonexistent.rs");
        assert!(result.is_none());
    }

    #[test]
    fn test_get_file_summary_returns_reference() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 50,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        let summary = FileSummary {
            path: "lib.rs".to_string(),
            lines: 50,
            structs: vec![],
            enums: vec![],
            functions: vec!["main".to_string()],
            traits: vec![],
            impls: vec![],
            uses: vec![],
            mods: vec!["mod test".to_string()],
            doc_summary: "Main entry".to_string(),
            derives: vec![],
        };
        
        ctx.files.insert("lib.rs".to_string(), summary);
        
        // Verify we can use the reference without cloning
        if let Some(s) = ctx.get_file_summary("lib.rs") {
            assert_eq!(s.doc_summary, "Main entry");
            // Reference should point to the stored data
            assert!(std::ptr::eq(s, ctx.files.get("lib.rs").unwrap()));
        } else {
            panic!("Expected to find lib.rs");
        }
    }

    #[test]
    fn test_get_related_files_content_no_dependents() {
        let ctx = CodebaseContext {
            total_files: 0,
            total_lines: 0,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        let result = ctx.get_related_files_content("nonexistent.rs", 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_related_files_content_with_dependents() {
        let mut ctx = CodebaseContext {
            total_files: 3,
            total_lines: 300,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        // Target file
        ctx.files.insert("core/types.rs".to_string(), FileSummary {
            path: "core/types.rs".to_string(),
            lines: 100,
            structs: vec!["Config".to_string()],
            enums: vec![],
            functions: vec![],
            traits: vec![],
            impls: vec![],
            uses: vec![],
            mods: vec![],
            doc_summary: "Core type definitions".to_string(),
            derives: vec![],
        });
        
        // Dependent file 1 (larger)
        ctx.files.insert("handler/main.rs".to_string(), FileSummary {
            path: "handler/main.rs".to_string(),
            lines: 200,
            structs: vec!["Handler".to_string()],
            enums: vec![],
            functions: vec!["process".to_string(), "validate".to_string()],
            traits: vec![],
            impls: vec!["Handler".to_string()],
            uses: vec!["use crate::core::types;".to_string()],
            mods: vec![],
            doc_summary: "Request handler".to_string(),
            derives: vec![],
        });
        
        // Dependent file 2 (smaller)
        ctx.files.insert("utils/helper.rs".to_string(), FileSummary {
            path: "utils/helper.rs".to_string(),
            lines: 50,
            structs: vec![],
            enums: vec![],
            functions: vec!["format".to_string()],
            traits: vec![],
            impls: vec![],
            uses: vec!["use crate::core::types::Config;".to_string()],
            mods: vec![],
            doc_summary: "Helper utilities".to_string(),
            derives: vec![],
        });
        
        let result = ctx.get_related_files_content("core/types.rs", 5);
        
        // Should include both dependents
        assert!(result.contains("handler/main.rs"));
        assert!(result.contains("utils/helper.rs"));
        assert!(result.contains("Request handler"));
        assert!(result.contains("Helper utilities"));
        
        // Larger file should appear first (sorted by line count desc)
        let handler_pos = result.find("handler/main.rs").unwrap();
        let helper_pos = result.find("utils/helper.rs").unwrap();
        assert!(handler_pos < helper_pos, "Larger file should appear first");
    }

    #[test]
    fn test_get_related_files_content_respects_limit() {
        let mut ctx = CodebaseContext {
            total_files: 4,
            total_lines: 400,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.files.insert("core/mod.rs".to_string(), FileSummary {
            path: "core/mod.rs".to_string(),
            lines: 50,
            structs: vec![],
            enums: vec![],
            functions: vec![],
            traits: vec![],
            impls: vec![],
            uses: vec![],
            mods: vec![],
            doc_summary: "".to_string(),
            derives: vec![],
        });
        
        // Add 3 dependents
        for (name, lines) in [("a.rs", 100), ("b.rs", 200), ("c.rs", 150)] {
            ctx.files.insert(name.to_string(), FileSummary {
                path: name.to_string(),
                lines,
                structs: vec![],
                enums: vec![],
                functions: vec![],
                traits: vec![],
                impls: vec![],
                uses: vec!["use crate::core;".to_string()],
                mods: vec![],
                doc_summary: format!("File {}", name),
                derives: vec![],
            });
        }
        
        let result = ctx.get_related_files_content("core/mod.rs", 2);
        
        // Should only include 2 files (largest ones: b.rs and c.rs)
        assert!(result.contains("b.rs"));
        assert!(result.contains("c.rs"));
        assert!(!result.contains("a.rs"));
    }

    #[test]
    fn test_get_related_files_content_excludes_self() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 100,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.files.insert("self_ref.rs".to_string(), FileSummary {
            path: "self_ref.rs".to_string(),
            lines: 100,
            structs: vec![],
            enums: vec![],
            functions: vec![],
            traits: vec![],
            impls: vec![],
            uses: vec!["use crate::self_ref;".to_string()],
            mods: vec![],
            doc_summary: "Self referencing".to_string(),
            derives: vec![],
        });
        
        let result = ctx.get_related_files_content("self_ref.rs", 5);
        
        // Should not include the file itself even though it has a matching use statement
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_related_files_content_shows_used_items() {
        let mut ctx = CodebaseContext {
            total_files: 3,
            total_lines: 250,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        // Target file with exported items
        ctx.files.insert("agent/mod.rs".to_string(), FileSummary {
            path: "agent/mod.rs".to_string(),
            lines: 100,
            structs: vec!["Agent".to_string(), "MutationStrategy".to_string()],
            enums: vec![],
            functions: vec!["new".to_string(), "evolve".to_string()],
            traits: vec![],
            impls: vec!["Agent".to_string()],
            uses: vec![],
            mods: vec![],
            doc_summary: "Agent module".to_string(),
            derives: vec![],
        });
        
        // Dependent that imports multiple items
        ctx.files.insert("runtime/loop_.rs".to_string(), FileSummary {
            path: "runtime/loop_.rs".to_string(),
            lines: 100,
            structs: vec!["EvolutionLoop".to_string()],
            enums: vec![],
            functions: vec!["run".to_string()],
            traits: vec![],
            impls: vec!["EvolutionLoop".to_string()],
            uses: vec!["use crate::agent::{Agent, MutationStrategy};".to_string()],
            mods: vec![],
            doc_summary: "Evolution loop".to_string(),
            derives: vec![],
        });
        
        // Dependent that imports single item
        ctx.files.insert("bin/unified.rs".to_string(), FileSummary {
            path: "bin/unified.rs".to_string(),
            lines: 50,
            structs: vec![],
            enums: vec![],
            functions: vec!["main".to_string()],
            traits: vec![],
            impls: vec![],
            uses: vec!["use crate::agent::Agent;".to_string()],
            mods: vec![],
            doc_summary: "Unified binary".to_string(),
            derives: vec![],
        });
        
        let result = ctx.get_related_files_content("agent/mod.rs", 5);
        
        // Should show which items are used
        assert!(result.contains("Uses from target: Agent, MutationStrategy"));
        assert!(result.contains("Uses from target: Agent"));
        
        // Should still show file summaries
        assert!(result.contains("runtime/loop_.rs"));
        assert!(result.contains("bin/unified.rs"));
    }

    #[test]
    fn test_get_related_files_content_filters_non_matching_uses() {
        let mut ctx = CodebaseContext {
            total_files: 2,
            total_lines: 150,
            files: HashMap::new(),
            module_tree: String::new(),
            architecture_summary: String::new(),
            last_scanned: String::new(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.files.insert("core/types.rs".to_string(), FileSummary {
            path: "core/types.rs".to_string(),
            lines: 50,
            structs: vec!["Config".to_string()],
            enums: vec![],
            functions: vec![],
            traits: vec![],
            impls: vec![],
            uses: vec![],
            mods: vec![],
            doc_summary: "Types".to_string(),
            derives: vec![],
        });
        
        // File with unrelated imports - should not be included
        ctx.files.insert("other/mod.rs".to_string(), FileSummary {
            path: "other/mod.rs".to_string(),
            lines: 100,
            structs: vec![],
            enums: vec![],
            functions: vec![],
            traits: vec![],
            impls: vec![],
            uses: vec!["use std::collections::HashMap;".to_string(), "use serde::Serialize;".to_string()],
            mods: vec![],
            doc_summary: "Other module".to_string(),
            derives: vec![],
        });
        
        let result = ctx.get_related_files_content("core/types.rs", 5);
        
        // Should be empty since other/mod.rs doesn't use core/types.rs
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_context_prompt_includes_enums_traits_impls() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 100,
            files: HashMap::new(),
            module_tree: "src/\n  types.rs (100 lines)\n".to_string(),
            architecture_summary: "Test architecture".to_string(),
            last_scanned: "2024-01-01T00:00:00Z".to_string(),
            total_iterations: 1,
            improvement_history: Vec::new(),
        };
        
        ctx.files.insert("types.rs".to_string(), FileSummary {
            path: "types.rs".to_string(),
            lines: 100,
            structs: vec!["Config".to_string(), "Settings".to_string()],
            enums: vec!["Status".to_string(), "Mode".to_string()],
            functions: vec!["new".to_string(), "validate".to_string()],
            traits: vec!["Handler".to_string()],
            impls: vec!["Config".to_string(), "Settings".to_string()],
            uses: vec![],
            mods: vec![],
            doc_summary: "Type definitions".to_string(),
            derives: vec![],
        });
        
        let prompt = ctx.build_context_prompt("types.rs");
        
        // Should include enums
        assert!(prompt.contains("Enums: Status, Mode"));
        // Should include traits
        assert!(prompt.contains("Traits: Handler"));
        // Should include impls
        assert!(prompt.contains("Impls: Config, Settings"));
        // Should still have structs and functions
        assert!(prompt.contains("Structs:"));
        assert!(prompt.contains("Functions:"));
    }

    #[test]
    fn test_build_context_prompt_hides_empty_types() {
        let mut ctx = CodebaseContext {
            total_files: 1,
            total_lines: 50,
            files: HashMap::new(),
            module_tree: "src/\n  simple.rs (50 lines)\n".to_string(),
            architecture_summary: "Simple module".to_string(),
            last_scanned: "2024-01-01T00:00:00Z".to_string(),
            total_iterations: 0,
            improvement_history: Vec::new(),
        };
        
        ctx.files.insert("simple.rs".to_string(), FileSummary {
            path: "simple.rs".to_string(),
            lines: 50,
            structs: vec!["OnlyStruct".to_string()],
            enums: vec![],
            functions: vec!["run".to_string()],
            traits: vec![],
            impls: vec![],
            uses: vec![],
            mods: vec![],
            doc_summary: "Simple file".to_string(),
            derives: vec![],
        });
        
        let prompt = ctx.build_context_prompt("simple.rs");
        
        // Should not include empty Enums/Traits/Impls lines
        assert!(!prompt.contains("Enums:"));
        assert!(!prompt.contains("Traits:"));
        assert!(!prompt.contains("Impls:"));
        // Should still have structs and functions
        assert!(prompt.contains("Structs:"));
        assert!(prompt.contains("Functions:"));
    }
}