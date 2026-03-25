//! 代码库理解模块 — 给系统全局架构感知和上下文记忆
//!
//! 功能：
//! 1. 扫描所有源文件，提取模块结构、公开 API、依赖关系
//! 2. 构建架构图（哪个文件做什么、文件之间怎么调用）
//! 3. 持久化到 .hyperagent/codebase_context.json
//! 4. 每次研究迭代时注入上下文，让 LLM "看见全局"

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// 单个文件的代码摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSummary {
    /// 文件路径（相对 src/）
    pub path: String,
    /// 文件行数
    pub lines: usize,
    /// pub struct 名称
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
    /// 改进历史摘要（最近的 N 次）
    pub improvement_history: Vec<String>,
}

impl CodebaseContext {
    /// 扫描代码库，构建全局上下文
    pub fn scan(project_root: &str) -> Result<Self> {
        let src_dir = PathBuf::from(project_root).join("src");
        let mut files = HashMap::new();
        let mut total_lines = 0usize;

        // 递归扫描所有 .rs 文件
        if let Ok(entries) = std::fs::read_dir(&src_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "rs").unwrap_or(false) {
                    let rel = path
                        .strip_prefix(&src_dir)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();

                    if let Ok(content) = std::fs::read_to_string(&path) {
                        let summary = Self::summarize_file(&rel, &content);
                        total_lines += summary.lines;
                        files.insert(rel.clone(), summary);
                    }
                }
            }
        }

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

        for line in content.lines() {
            let line = line.trim();

            // 提取文档注释摘要（文件头部的 //!）
            if line.starts_with("//!") {
                // doc_summary 在下面统一处理
                continue;
            }

            // pub struct
            if let Some(name) = Self::extract_name(line, "pub struct") {
                structs.push(name);
            }
            // pub enum
            else if let Some(name) = Self::extract_name(line, "pub enum") {
                enums.push(name);
            }
            // pub trait
            else if let Some(name) = Self::extract_name(line, "pub trait") {
                traits.push(name);
            }
            // pub fn / pub async fn
            else if let Some(name) = Self::extract_fn_name(line) {
                functions.push(name);
            }
            // impl Xxx
            else if line.starts_with("impl ") {
                if let Some(name) = line.strip_prefix("impl ").and_then(|s| {
                    s.split_whitespace().next().map(|n| {
                        // 去掉泛型参数和 <
                        n.split('<').next().unwrap_or(n).to_string()
                    })
                }) {
                    impls.push(name);
                }
            }
            // use xxx::yyy
            else if line.starts_with("use ") {
                uses.push(line.to_string());
            }
            // mod xxx
            else if line.starts_with("pub mod ") || line.starts_with("mod ") {
                let prefix = if line.starts_with("pub mod ") { "pub mod " } else { "mod " };
                if let Some(name) = line.strip_prefix(prefix) {
                    mods.push(
                        name.split(';')
                            .next()
                            .unwrap_or(name)
                            .trim()
                            .to_string(),
                    );
                }
            }
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
        }
    }

    /// 从行中提取 `keyword Name` 形式的名称
    fn extract_name(line: &str, keyword: &str) -> Option<String> {
        if !line.starts_with(keyword) {
            return None;
        }
        let rest = line.strip_prefix(keyword)?.trim();
        rest.split(|c: char| c == '<' || c == '(' || c == ' ' || c == ';')
            .next()
            .filter(|n| !n.is_empty())
            .map(|n| n.to_string())
    }

    /// 提取 pub fn / pub async fn 名称
    fn extract_fn_name(line: &str) -> Option<String> {
        let trimmed = line.trim();
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
        let entry = format!(
            "[{}] {} — {} ({})",
            chrono::Utc::now().format("%H:%M"),
            file,
            hypothesis.chars().take(80).collect::<String>(),
            outcome,
        );
        self.improvement_history.push(entry);
        // 只保留最近 50 条
        if self.improvement_history.len() > 50 {
            self.improvement_history = self.improvement_history.split_off(self.improvement_history.len() - 50);
        }
        self.total_iterations += 1;
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
            prompt.push_str(&format!("Lines: {}, Structs: {:?}, Functions: {:?}\n",
                target_summary.lines,
                target_summary.structs,
                target_summary.functions.iter().take(10).collect::<Vec<_>>(),
            ));
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
            let recent: Vec<&String> = self.improvement_history.iter().rev().take(10).collect();
            for entry in recent.into_iter().rev() {
                prompt.push_str(entry);
                prompt.push('\n');
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
}
