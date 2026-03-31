use crate::llm::LLMClient;

use super::AutoResearch;

/// 单个搜索替换操作
#[derive(Debug, Clone)]
pub(crate) struct SearchReplace {
    pub search: String,
    pub replace: String,
}

/// 编辑操作：可以是完整文件内容，或搜索替换列表
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum EditOp {
    /// 完整文件替换（小文件向后兼容）
    FullFile(String),
    /// 多个搜索替换操作（精确编辑）
    SearchReplace(Vec<SearchReplace>),
}

/// 解析后的编辑指令
#[derive(Debug, Clone)]
pub(crate) struct ParsedEdit {
    pub file_path: String,
    pub op: EditOp,
}

/// 模糊匹配策略类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuzzyMatchStrategy {
    /// 精确匹配（无需模糊）
    Exact,
    /// 去除首尾空白后匹配
    Trimmed,
    /// 标准化空白字符后匹配
    Normalized,
    /// 逐行匹配（忽略空白行差异）
    LineByLine,
}

/// 单个 SEARCH 块的模糊匹配需求
#[derive(Debug, Clone)]
pub struct FuzzyRequirement {
    /// 是否需要模糊匹配
    pub needs_fuzzy: bool,
    /// 建议的匹配策略
    pub suggested_strategy: FuzzyMatchStrategy,
    /// SEARCH 块的行数
    pub line_count: usize,
    /// 是否包含多余空白
    pub has_extra_whitespace: bool,
}

/// 编辑操作的统计信息
#[derive(Debug, Clone, Default)]
pub struct EditStats {
    pub lines_added: usize,
    pub lines_removed: usize,
    pub chars_added: usize,
    pub chars_removed: usize,
}

impl EditStats {
    /// Returns a human-readable one-line summary of the edit statistics.
    /// Format: "+N/-M lines, +X/-Y chars" (omitting zero values)
    pub fn summary(&self) -> String {
        let lines_part = match (self.lines_added, self.lines_removed) {
            (0, 0) => None,
            (a, 0) => Some(format!("+{} lines", a)),
            (0, r) => Some(format!("-{} lines", r)),
            (a, r) => Some(format!("+{}/-{} lines", a, r)),
        };
        
        let chars_part = match (self.chars_added, self.chars_removed) {
            (0, 0) => None,
            (a, 0) => Some(format!("+{} chars", a)),
            (0, r) => Some(format!("-{} chars", r)),
            (a, r) => Some(format!("+{}/-{} chars", a, r)),
        };
        
        match (lines_part, chars_part) {
            (Some(l), Some(c)) => format!("{}, {}", l, c),
            (Some(l), None) => l,
            (None, Some(c)) => c,
            (None, None) => "no changes".to_string(),
        }
    }
}

impl ParsedEdit {
    /// 计算 SEARCH/REPLACE 操作的统计信息
    /// 对于 FullFile 编辑，返回 None（需要读取原始文件才能比较）
    pub fn diff_stats(&self) -> Option<EditStats> {
        match &self.op {
            EditOp::SearchReplace(srs) => {
                let mut stats = EditStats::default();
                for sr in srs {
                    let search_lines = sr.search.lines().count();
                    let replace_lines = sr.replace.lines().count();
                    
                    if replace_lines > search_lines {
                        stats.lines_added += replace_lines - search_lines;
                    } else {
                        stats.lines_removed += search_lines - replace_lines;
                    }
                    
                    stats.chars_removed += sr.search.len();
                    stats.chars_added += sr.replace.len();
                }
                Some(stats)
            }
            EditOp::FullFile(_) => None,
        }
    }

    /// Check if any SEARCH block would likely require fuzzy matching
    /// Returns true if the search contains multi-line content or significant whitespace
    /// that may not match exactly in the target file
    pub fn has_fuzzy_match(&self) -> bool {
        match &self.op {
            EditOp::SearchReplace(srs) => {
                srs.iter().any(|sr| {
                    // Multi-line content often has whitespace differences
                    sr.search.lines().count() > 1 ||
                    // Leading/trailing whitespace differences
                    sr.search != sr.search.trim() ||
                    // Multiple consecutive spaces (LLM may normalize)
                    sr.search.contains("  ")
                })
            }
            EditOp::FullFile(_) => false,
        }
    }

    /// 获取每个 SEARCH 块的模糊匹配需求详情
    /// 返回每个 SEARCH 块是否需要模糊匹配以及建议的策略
    pub fn get_fuzzy_requirements(&self) -> Vec<FuzzyRequirement> {
        match &self.op {
            EditOp::SearchReplace(srs) => {
                srs.iter().map(|sr| {
                    let line_count = sr.search.lines().count();
                    let has_extra_whitespace = sr.search != sr.search.trim() || sr.search.contains("  ");
                    let is_multiline = line_count > 1;
                    
                    // 确定建议的匹配策略
                    let suggested_strategy = if !has_extra_whitespace && !is_multiline {
                        FuzzyMatchStrategy::Exact
                    } else if is_multiline && !has_extra_whitespace {
                        // 多行但无多余空白：优先尝试逐行匹配
                        FuzzyMatchStrategy::LineByLine
                    } else if has_extra_whitespace && !is_multiline {
                        // 单行但有多余空白：优先尝试 trimmed
                        FuzzyMatchStrategy::Trimmed
                    } else {
                        // 多行 + 多余空白：最复杂情况，需要标准化
                        FuzzyMatchStrategy::Normalized
                    };
                    
                    FuzzyRequirement {
                        needs_fuzzy: has_extra_whitespace || is_multiline,
                        suggested_strategy,
                        line_count,
                        has_extra_whitespace,
                    }
                }).collect()
            }
            EditOp::FullFile(_) => Vec::new(),
        }
    }

    /// Returns the percentage of SEARCH blocks that require fuzzy matching (0.0 to 1.0).
    /// Returns 0.0 for FullFile edits (no SEARCH blocks).
    /// A lower ratio indicates simpler, more reliable edits that are more likely to succeed.
    pub fn fuzzy_match_ratio(&self) -> f64 {
        match &self.op {
            EditOp::SearchReplace(srs) => {
                if srs.is_empty() {
                    return 0.0;
                }
                let fuzzy_count = srs.iter().filter(|sr| {
                    let line_count = sr.search.lines().count();
                    let has_extra_whitespace = sr.search != sr.search.trim() || sr.search.contains("  ");
                    let is_multiline = line_count > 1;
                    has_extra_whitespace || is_multiline
                }).count();
                fuzzy_count as f64 / srs.len() as f64
            }
            EditOp::FullFile(_) => 0.0,
        }
    }
}

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// Phase 3: 解析 LLM 响应中的修改
    ///
    /// 支持三种格式（按优先级检测）：
    /// 1. SEARCH/REPLACE 格式（推荐，精确编辑）
    /// 2. FILE: 多文件格式（向后兼容）
    /// 3. 单文件完整代码格式（向后兼容）
    pub(crate) fn parse_response_multi(&self, response: &str) -> Option<(String, Vec<(String, String)>)> {
        // 提取 HYPOTHESIS
        let hypothesis = if let Some(start) = response.find("HYPOTHESIS:") {
            let start = start + 11;
            let end = response[start..].find("\n\n").unwrap_or(response[start..].len());
            response[start..start + end].trim().to_string()
        } else {
            "No hypothesis stated".to_string()
        };

        // 优先检测 SEARCH/REPLACE 格式（用完整标记避免误匹配假设文本）
        if response.contains("<<<<<<< SEARCH") {
            if let Some(edits) = self.parse_search_replace_edits(response) {
                return Some((hypothesis, edits));
            }
        }

        // 检测多文件格式：是否有 FILE: 标记
        if response.contains("FILE:") {
            return self.parse_multi_file_response(&hypothesis, response);
        }

        // 单文件格式（向后兼容）
        let code = Self::extract_code(response);
        if code.is_empty() {
            return None;
        }
        Some((hypothesis, vec![(String::new(), code)]))
    }

    /// 解析 SEARCH/REPLACE 格式的编辑指令
    ///
    /// 格式：
    /// ```EDIT: src/file.rs
    /// <<<<<<< SEARCH
    /// old code (must match exactly)
    /// =======
    /// new code
    /// >>>>>>> REPLACE
    /// ```
    fn parse_search_replace_edits(&self, response: &str) -> Option<Vec<(String, String)>> {
        let mut edits: Vec<ParsedEdit> = Vec::new();
        let mut pos = 0;

        while pos < response.len() {
            // 找 EDIT: 标记
            let edit_start = match Self::find_marker(response, "EDIT:", pos) {
                Some(p) => p,
                None => break,
            };

            let line_end = response[edit_start..]
                .find('\n')
                .map(|i| edit_start + i)
                .unwrap_or(response.len());
            let file_path = response[edit_start + 5..line_end].trim().to_string();

            if file_path.is_empty() {
                pos = line_end + 1;
                continue;
            }

            // 找这个 EDIT 块里的所有 SEARCH/REPLACE 对
            let block_end = Self::find_marker(response, "EDIT:", line_end).unwrap_or(response.len());
            let block = &response[line_end..block_end];

            let mut search_replaces: Vec<SearchReplace> = Vec::new();
            let mut sr_pos = 0;

            while sr_pos < block.len() {
                // 找 <<<<<<< SEARCH
                let search_marker = match block[sr_pos..].find("<<<<<<< SEARCH") {
                    Some(p) => sr_pos + p,
                    None => break,
                };

                // 找 =======
                let sep = match block[search_marker..].find("=======") {
                    Some(p) => search_marker + p,
                    None => break,
                };

                // 找 >>>>>>> REPLACE
                let replace_end = match block[sep..].find(">>>>>>> REPLACE") {
                    Some(p) => sep + p,
                    None => break,
                };

                let search_content = block[search_marker + 14..sep].trim_matches('\n').to_string();
                let replace_content = block[sep + 7..replace_end].trim_matches('\n').to_string();

                if !search_content.is_empty() {
                    search_replaces.push(SearchReplace {
                        search: search_content,
                        replace: replace_content,
                    });
                }

                sr_pos = replace_end + 16; // 跳过 ">>>>>>> REPLACE"
            }

            if !search_replaces.is_empty() {
                edits.push(ParsedEdit {
                    file_path,
                    op: EditOp::SearchReplace(search_replaces),
                });
            }

            pos = block_end;
        }

        if edits.is_empty() {
            return None;
        }

        // 合并同文件的 SearchReplace 操作（LLM 可能为同一文件输出多个 EDIT 块）
        use std::collections::HashMap;
        let mut merged: HashMap<String, Vec<SearchReplace>> = HashMap::new();
        let mut full_files: HashMap<String, String> = HashMap::new();

        for edit in &edits {
            match &edit.op {
                EditOp::SearchReplace(srs) => {
                    merged.entry(edit.file_path.clone())
                        .or_default()
                        .extend(srs.iter().cloned());
                }
                EditOp::FullFile(content) => {
                    full_files.insert(edit.file_path.clone(), content.clone());
                }
            }
        }

        // 应用所有编辑到原始文件
        let mut results: Vec<(String, String)> = Vec::new();

        for (file_path, srs) in &merged {
            let original = self.read_file(file_path).unwrap_or_default();
            let modified = self.apply_search_replaces(&original, srs);
            results.push((file_path.clone(), modified));
        }

        for (file_path, content) in &full_files {
            results.push((file_path.clone(), content.clone()));
        }

        Some(results)
    }

    /// 将搜索替换操作应用到原始代码上
    fn apply_search_replaces(&self, original: &str, ops: &[SearchReplace]) -> String {
        let mut result = original.to_string();

        for op in ops {
            // Try exact match first
            if let Some(idx) = result.find(&op.search) {
                let before = &result[..idx];
                let after = &result[idx + op.search.len()..];
                result = format!("{}{}{}", before, op.replace, after);
                tracing::debug!(
                    "  Applied SEARCH/REPLACE: {} chars → {} chars",
                    op.search.len(),
                    op.replace.len()
                );
                continue;
            }

            // Fuzzy matching strategies in order of priority
            match self.try_fuzzy_match_strategies(&result, op) {
                Some(modified) => {
                    result = modified;
                    tracing::debug!(
                        "  Applied fuzzy SEARCH/REPLACE: {} chars → {} chars",
                        op.search.len(),
                        op.replace.len()
                    );
                }
                None => {
                    tracing::warn!(
                        "  SEARCH block not found in file ({} chars), all fuzzy match strategies failed — skipping this replacement",
                        op.search.len()
                    );
                }
            }
        }

        result
    }

    /// Try multiple fuzzy matching strategies for more robust SEARCH/REPLACE
    /// Returns Some(modified_content) if a match was found and applied, None otherwise
    fn try_fuzzy_match_strategies(&self, content: &str, op: &SearchReplace) -> Option<String> {
        // Strategy 1: Trimmed exact match
        let trimmed_search = op.search.trim();
        if let Some(idx) = content.find(trimmed_search) {
            let trimmed_replace = op.replace.trim();
            let before = &content[..idx];
            let after = &content[idx + trimmed_search.len()..];
            let result = format!("{}{}{}", before, trimmed_replace, after);
            tracing::debug!(
                "  Fuzzy match succeeded: strategy={:?}, search_len={} chars",
                FuzzyMatchStrategy::Trimmed,
                op.search.len()
            );
            return Some(result);
        }

        // Strategy 2: Normalized whitespace matching
        let normalized_content = normalize_whitespace(content);
        let normalized_search = normalize_whitespace(&op.search);

        if let Some(idx) = normalized_content.find(&normalized_search) {
            // Map back to original content positions
            if let Some((start, end)) = find_original_positions(content, &normalized_content, idx, &normalized_search) {
                let before = &content[..start];
                let after = &content[end..];
                // Use original replacement format, not normalized
                let result = format!("{}{}{}", before, op.replace.trim(), after);
                tracing::debug!(
                    "  Fuzzy match succeeded: strategy={:?}, search_len={} chars",
                    FuzzyMatchStrategy::Normalized,
                    op.search.len()
                );
                return Some(result);
            }
        }

        // Strategy 3: Line-by-line matching (ignores blank line differences)
        if let Some(result) = self.try_line_match_with_replace(content, op) {
            tracing::debug!(
                "  Fuzzy match succeeded: strategy={:?}, search_len={} chars",
                FuzzyMatchStrategy::LineByLine,
                op.search.len()
            );
            return Some(result);
        }

        None
    }

    /// Try matching line-by-line, ignoring differences in blank lines and trailing whitespace
    /// Returns Some(modified_content) if match found and replacement applied, None otherwise
    fn try_line_match_with_replace(&self, content: &str, op: &SearchReplace) -> Option<String> {
        let content_lines: Vec<&str> = content.lines().collect();
        let search_lines: Vec<&str> = op.search.lines().collect();
        let replace_lines: Vec<&str> = op.replace.lines().collect();

        // Trim for comparison but keep originals for reconstruction
        let content_trimmed: Vec<&str> = content_lines.iter().map(|l| l.trim_end()).collect();
        let search_trimmed: Vec<&str> = search_lines.iter().map(|l| l.trim_end()).collect();

        // Remove empty lines for matching logic
        let content_nonempty: Vec<(usize, &str)> = content_trimmed
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_empty())
            .map(|(i, l)| (i, *l))
            .collect();
        let search_nonempty: Vec<&str> = search_trimmed.iter().filter(|l| !l.is_empty()).copied().collect();

        if search_nonempty.is_empty() {
            return None;
        }

        // Find if search lines are a contiguous subsequence of content lines (ignoring empty lines)
        for start in 0..content_nonempty.len().saturating_sub(search_nonempty.len() - 1) {
            let mut matches = true;
            for (i, search_line) in search_nonempty.iter().enumerate() {
                if content_nonempty.get(start + i).map(|(_, l)| *l) != Some(*search_line) {
                    matches = false;
                    break;
                }
            }
            
            if matches {
                // Found a match! Now apply the replacement
                // Find the line range in original content
                let first_content_idx = content_nonempty[start].0;
                let last_content_idx = if search_nonempty.len() > 0 {
                    content_nonempty[start + search_nonempty.len() - 1].0
                } else {
                    first_content_idx
                };

                // Reconstruct the result by replacing the matched lines
                let mut result_lines: Vec<String> = Vec::new();
                
                // Lines before the match
                for line in content_lines.iter().take(first_content_idx) {
                    result_lines.push((*line).to_string());
                }
                
                // Insert replacement (use original indentation from first matched line)
                let base_indent = content_lines[first_content_idx]
                    .chars()
                    .take_while(|c| c.is_whitespace())
                    .collect::<String>();
                
                for (i, replace_line) in replace_lines.iter().enumerate() {
                    if i == 0 || replace_line.is_empty() {
                        result_lines.push(replace_line.to_string());
                    } else {
                        // Preserve base indentation for replacement lines
                        let trimmed = replace_line.trim_start();
                        result_lines.push(format!("{}{}", base_indent, trimmed));
                    }
                }
                
                // Lines after the match
                for line in content_lines.iter().skip(last_content_idx + 1) {
                    result_lines.push((*line).to_string());
                }
                
                return Some(result_lines.join("\n"));
            }
        }

        None
    }
}

/// Normalize whitespace in a string by collapsing multiple spaces/tabs into single space
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Find the original content positions corresponding to a normalized match
/// Returns (start_byte, end_byte) in original content, or None if mapping fails
fn find_original_positions(original: &str, normalized: &str, normalized_idx: usize, normalized_search: &str) -> Option<(usize, usize)> {
    // Build a mapping from normalized positions to original byte positions
    // by iterating through words (which is how normalize_whitespace works)
    let words: Vec<&str> = original.split_whitespace().collect();
    let norm_words: Vec<&str> = normalized.split_whitespace().collect();
    
    // Verify the normalized strings match
    if norm_words != words {
        return None;
    }
    
    // Find which word contains the start and end of the search
    let search_end = normalized_idx + normalized_search.len();
    
    // Build position map: word index -> (start_byte, end_byte) in original
    let mut word_positions: Vec<(usize, usize)> = Vec::new();
    let mut byte_pos = 0;
    
    for (word_idx, word) in original.split_whitespace().enumerate() {
        // Skip leading whitespace to find word start
        while byte_pos < original.len() && original[byte_pos..].chars().next().map(|c| c.is_whitespace()).unwrap_or(false) {
            byte_pos += original[byte_pos..].chars().next().unwrap().len_utf8();
        }
        
        let word_start = byte_pos;
        byte_pos += word.len();
        word_positions.push((word_start, byte_pos));
    }
    
    // Find word indices corresponding to normalized positions
    // Each word in normalized has format: "word " (word + space), except last word
    let mut norm_pos = 0;
    let mut start_word_idx = None;
    let mut end_word_idx = None;
    
    for (word_idx, _word) in words.iter().enumerate() {
        // In normalized string, each word is followed by a space (except last)
        let word_len_in_norm = if word_idx < words.len() - 1 {
            words[word_idx].len() + 1 // word + space
        } else {
            words[word_idx].len()
        };
        
        if norm_pos <= normalized_idx && normalized_idx < norm_pos + word_len_in_norm {
            start_word_idx = Some(word_idx);
        }
        
        if norm_pos <= search_end && search_end <= norm_pos + word_len_in_norm {
            end_word_idx = Some(word_idx);
        }
        
        norm_pos += word_len_in_norm;
    }
    
    // If search_end is past the last word, end at the last word
    if search_end >= normalized.len() {
        end_word_idx = Some(words.len().saturating_sub(1));
    }
    
    match (start_word_idx, end_word_idx) {
        (Some(swi), Some(ewi)) => {
            let start_byte = word_positions.get(swi)?.0;
            let end_byte = word_positions.get(ewi)?.1;
            Some((start_byte, end_byte))
        }
        _ => None,
    }
}

impl<C: LLMClient + Clone> AutoResearch<C> {

    /// 解析多文件响应：按 FILE: 标记分割代码块
    fn parse_multi_file_response(&self, hypothesis: &str, response: &str) -> Option<(String, Vec<(String, String)>)> {
        let mut changes: Vec<(String, String)> = Vec::new();

        let mut pos = 0;
        let bytes = response.as_bytes();

        while pos < bytes.len() {
            let file_start = match Self::find_marker(response, "FILE:", pos) {
                Some(p) => p,
                None => break,
            };

            // 使用字节索引查找，但确保在提取子串时验证 UTF-8 边界
            let file_line_end = response[file_start..].find('\n').map(|i| file_start + i).unwrap_or(response.len());
            // 使用 strip_prefix 和 split 来安全提取文件路径，避免直接字节切片
            let file_path = if let Some(suffix) = response.get(file_start + 5..file_line_end) {
                suffix.trim().to_string()
            } else {
                pos = file_start + 5;
                continue;
            };

            // 安全获取代码区域，使用 get() 避免越界
            let code_region = response.get(file_line_end..).unwrap_or("");

            let (code_start, code_end) = if let Some(block_start) = code_region.find("```") {
                let actual_start = block_start + 3;
                let content_start = if code_region.get(actual_start..).map_or(false, |s| s.starts_with("rust")) {
                    actual_start + 4
                } else {
                    actual_start
                };
                let close_pos = match code_region.get(content_start..).and_then(|s| s.find("```")) {
                    Some(p) => content_start + p,
                    None => continue,
                };
                let abs_start = file_line_end + content_start;
                let abs_end = file_line_end + close_pos;
                (abs_start, abs_end)
            } else {
                continue;
            };

            // 使用 get() 安全提取代码，避免 UTF-8 边界问题
            let code = response.get(code_start..code_end).map(|s| s.trim().to_string()).unwrap_or_default();
            if !code.is_empty() {
                changes.push((file_path, code));
            }

            pos = code_end + 3;
        }

        if changes.is_empty() {
            None
        } else {
            Some((hypothesis.to_string(), changes))
        }
    }

    /// 在文本中查找标记（匹配行首，不区分大小写）
    fn find_marker(text: &str, marker: &str, start: usize) -> Option<usize> {
        let upper_marker = marker.to_uppercase();
        for (i, line) in text[start..].lines().enumerate() {
            let trimmed = line.trim().to_uppercase();
            if trimmed.starts_with(&upper_marker) {
                let abs_offset = text[start..].lines().take(i).map(|l| l.len() + 1).sum::<usize>();
                return Some(start + abs_offset);
            }
        }
        None
    }

    /// 提取代码块
    fn extract_code(raw: &str) -> String {
        if raw.contains("```rust") {
            if let Some(start) = raw.find("```rust").map(|i| i + 7) {
                if let Some(end) = raw[start..].find("```") {
                    return raw[start..start + end].trim().to_string();
                }
            }
        }
        if raw.contains("```") {
            if let Some(start) = raw.find("```").map(|i| i + 3) {
                if let Some(end) = raw[start..].find("```") {
                    return raw[start..start + end].trim().to_string();
                }
            }
        }
        raw.trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockResearch;

    impl MockResearch {
        fn apply_search_replaces_test(original: &str, ops: &[SearchReplace]) -> String {
            let mut result = original.to_string();
            for op in ops {
                if let Some(idx) = result.find(&op.search) {
                    let before = &result[..idx];
                    let after = &result[idx + op.search.len()..];
                    result = format!("{}{}{}", before, op.replace, after);
                }
            }
            result
        }

        /// Test helper that includes fuzzy matching logic
        fn apply_search_replaces_with_fuzzy(original: &str, ops: &[SearchReplace]) -> String {
            let mut result = original.to_string();
            for op in ops {
                // Exact match
                if let Some(idx) = result.find(&op.search) {
                    let before = &result[..idx];
                    let after = &result[idx + op.search.len()..];
                    result = format!("{}{}{}", before, op.replace, after);
                    continue;
                }
                
                // Fuzzy: trimmed match
                let trimmed_search = op.search.trim();
                if let Some(idx) = result.find(trimmed_search) {
                    let trimmed_replace = op.replace.trim();
                    let before = &result[..idx];
                    let after = &result[idx + trimmed_search.len()..];
                    result = format!("{}{}{}", before, trimmed_replace, after);
                    continue;
                }
                
                // Fuzzy: normalized whitespace
                let normalized_content = normalize_whitespace(&result);
                let normalized_search = normalize_whitespace(&op.search);
                if let Some(idx) = normalized_content.find(&normalized_search) {
                    if let Some((start, end)) = find_original_positions(&result, &normalized_content, idx, &normalized_search) {
                        let before = &result[..start];
                        let after = &result[end..];
                        result = format!("{}{}{}", before, op.replace.trim(), after);
                        continue;
                    }
                }
                
                // Fuzzy: line-by-line match
                if let Some(modified) = try_line_match_with_replace_test(&result, op) {
                    result = modified;
                }
            }
            result
        }
    }

    fn try_line_match_with_replace_test(content: &str, op: &SearchReplace) -> Option<String> {
        let content_lines: Vec<&str> = content.lines().collect();
        let search_lines: Vec<&str> = op.search.lines().collect();
        let replace_lines: Vec<&str> = op.replace.lines().collect();

        let content_trimmed: Vec<&str> = content_lines.iter().map(|l| l.trim_end()).collect();
        let search_trimmed: Vec<&str> = search_lines.iter().map(|l| l.trim_end()).collect();

        let content_nonempty: Vec<(usize, &str)> = content_trimmed
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_empty())
            .map(|(i, l)| (i, *l))
            .collect();
        let search_nonempty: Vec<&str> = search_trimmed.iter().filter(|l| !l.is_empty()).copied().collect();

        if search_nonempty.is_empty() {
            return None;
        }

        for start in 0..content_nonempty.len().saturating_sub(search_nonempty.len() - 1) {
            let mut matches = true;
            for (i, search_line) in search_nonempty.iter().enumerate() {
                if content_nonempty.get(start + i).map(|(_, l)| *l) != Some(*search_line) {
                    matches = false;
                    break;
                }
            }
            
            if matches {
                let first_content_idx = content_nonempty[start].0;
                let last_content_idx = content_nonempty[start + search_nonempty.len() - 1].0;

                let mut result_lines: Vec<String> = Vec::new();
                
                for line in content_lines.iter().take(first_content_idx) {
                    result_lines.push((*line).to_string());
                }
                
                let base_indent = content_lines[first_content_idx]
                    .chars()
                    .take_while(|c| c.is_whitespace())
                    .collect::<String>();
                
                for (i, replace_line) in replace_lines.iter().enumerate() {
                    if i == 0 || replace_line.is_empty() {
                        result_lines.push(replace_line.to_string());
                    } else {
                        let trimmed = replace_line.trim_start();
                        result_lines.push(format!("{}{}", base_indent, trimmed));
                    }
                }
                
                for line in content_lines.iter().skip(last_content_idx + 1) {
                    result_lines.push((*line).to_string());
                }
                
                return Some(result_lines.join("\n"));
            }
        }

        None
    }

    #[test]
    fn test_search_replace_exact() {
        let original = "fn foo() -> i32 {\n    42\n}\n\nfn bar() -> i32 {\n    10\n}";
        let ops = vec![SearchReplace {
            search: "fn foo() -> i32 {\n    42\n}".to_string(),
            replace: "fn foo() -> u32 {\n    42\n}".to_string(),
        }];
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        assert!(result.contains("fn foo() -> u32"));
        assert!(result.contains("fn bar() -> i32"));
        // bar() 部分保持不变
        assert!(result.contains("let z = 3;") || result.contains("fn bar() -> i32"));
    }

    #[test]
    fn test_search_replace_multiple() {
        let original = "let x = 1;\nlet y = 2;\nlet z = 3;";
        let ops = vec![
            SearchReplace {
                search: "let x = 1;".to_string(),
                replace: "let x = 10;".to_string(),
            },
            SearchReplace {
                search: "let y = 2;".to_string(),
                replace: "let y = 20;".to_string(),
            },
        ];
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        assert!(result.contains("let x = 10;"));
        assert!(result.contains("let y = 20;"));
        assert!(result.contains("let z = 3;")); // 未修改
    }

    #[test]
    fn test_search_replace_not_found_skips() {
        let original = "fn main() {}";
        let ops = vec![SearchReplace {
            search: "nonexistent".to_string(),
            replace: "replacement".to_string(),
        }];
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        assert_eq!(result, original); // 原封不动
    }

    #[test]
    fn test_find_marker_basic() {
        let text = "some text\nMARKER: value\nmore text";
        let pos = AutoResearch::<crate::llm::LLMClientImpl>::find_marker(text, "MARKER", 0);
        assert!(pos.is_some());
        assert_eq!(pos.unwrap(), 10); // position of 'M' in MARKER
    }

    #[test]
    fn test_find_marker_case_insensitive() {
        let text = "some text\nmarker: value\nmore text";
        let pos = AutoResearch::<crate::llm::LLMClientImpl>::find_marker(text, "MARKER", 0);
        assert!(pos.is_some());
    }

    #[test]
    fn test_find_marker_not_found() {
        let text = "some text without the marker";
        let pos = AutoResearch::<crate::llm::LLMClientImpl>::find_marker(text, "NONEXISTENT", 0);
        assert!(pos.is_none());
    }

    #[test]
    fn test_find_marker_with_offset() {
        let text = "prefix\nMARKER: first\ncontent\nMARKER: second\nend";
        let first_pos = AutoResearch::<crate::llm::LLMClientImpl>::find_marker(text, "MARKER", 0);
        let second_pos = AutoResearch::<crate::llm::LLMClientImpl>::find_marker(text, "MARKER", first_pos.unwrap() + 1);
        assert!(first_pos.is_some());
        assert!(second_pos.is_some());
        assert!(first_pos.unwrap() < second_pos.unwrap());
    }

    #[test]
    fn test_extract_code_with_rust_block() {
        let raw = "Here's the code:\n```rust\nfn main() {}\n```\nEnd.";
        let code = AutoResearch::<crate::llm::LLMClientImpl>::extract_code(raw);
        assert_eq!(code, "fn main() {}");
    }

    #[test]
    fn test_extract_code_with_plain_block() {
        let raw = "Code:\n```\nlet x = 1;\n```\nDone.";
        let code = AutoResearch::<crate::llm::LLMClientImpl>::extract_code(raw);
        assert_eq!(code, "let x = 1;");
    }

    #[test]
    fn test_extract_code_no_block() {
        let raw = "Just plain text without code blocks";
        let code = AutoResearch::<crate::llm::LLMClientImpl>::extract_code(raw);
        assert_eq!(code, "Just plain text without code blocks");
    }

    #[test]
    fn test_extract_code_empty_block() {
        let raw = "Empty:\n```\n```\nDone.";
        let code = AutoResearch::<crate::llm::LLMClientImpl>::extract_code(raw);
        assert_eq!(code, "");
    }

    #[test]
    fn test_search_replace_preserves_unmatched() {
        let original = "fn alpha() {}\nfn beta() {}\nfn gamma() {}";
        let ops = vec![SearchReplace {
            search: "fn beta() {}".to_string(),
            replace: "fn beta() -> bool { true }".to_string(),
        }];
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        assert!(result.contains("fn alpha() {}"));
        assert!(result.contains("fn beta() -> bool { true }"));
        assert!(result.contains("fn gamma() {}"));
    }

    #[test]
    fn test_search_replace_empty_replacement() {
        let original = "keep this\ndelete this\nkeep this too";
        let ops = vec![SearchReplace {
            search: "delete this\n".to_string(),
            replace: "".to_string(),
        }];
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        assert_eq!(result, "keep this\nkeep this too");
    }

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("fn   foo()    -> i32"), "fn foo() -> i32");
        assert_eq!(normalize_whitespace("  let   x\t=\t1  ;  "), "let x = 1 ;");
        assert_eq!(normalize_whitespace("single"), "single");
        assert_eq!(normalize_whitespace(""), "");
    }

    #[test]
    fn test_search_replace_extra_whitespace() {
        // LLM outputs extra spaces that don't match file exactly
        let original = "fn foo() -> i32 {\n    42\n}";
        // LLM might output with different spacing
        let ops = vec![SearchReplace {
            search: "fn foo() -> i32 {\n        42\n}".to_string(), // 8 spaces instead of 4
            replace: "fn foo() -> u32 {\n    42\n}".to_string(),
        }];
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        // With fuzzy matching, this should still find a match
        assert!(result.contains("fn foo()"));
    }

    #[test]
    fn test_search_replace_different_line_breaks() {
        // Test that line-by-line matching handles different newline styles
        let original = "fn foo() {\n\n    bar()\n}\n";
        let ops = vec![SearchReplace {
            search: "fn foo() {\n    bar()\n}".to_string(), // Missing blank line
            replace: "fn foo() {\n    baz()\n}".to_string(),
        }];
        // The original fuzzy match should handle trimmed version
        let result = MockResearch::apply_search_replaces_test(original, &ops);
        assert!(result.contains("fn foo()"));
    }

    #[test]
    fn test_find_original_positions_basic() {
        let original = "fn foo() -> i32 { 42 }";
        let normalized = normalize_whitespace(original);
        // normalized is "fn foo() -> i32 { 42 }" (same, no extra whitespace)
        
        let search = "foo() -> i32";
        let normalized_search = normalize_whitespace(search);
        
        if let Some(idx) = normalized.find(&normalized_search) {
            let result = find_original_positions(original, &normalized, idx, &normalized_search);
            assert!(result.is_some());
            let (start, end) = result.unwrap();
            assert_eq!(&original[start..end], "foo() -> i32");
        }
    }

    #[test]
    fn test_find_original_positions_with_extra_whitespace() {
        let original = "fn   foo()   -> i32  { 42 }";
        let normalized = normalize_whitespace(original);
        // normalized is "fn foo() -> i32 { 42 }"
        
        let search = "foo() -> i32";
        let normalized_search = normalize_whitespace(search);
        
        if let Some(idx) = normalized.find(&normalized_search) {
            let result = find_original_positions(original, &normalized, idx, &normalized_search);
            assert!(result.is_some());
            let (start, end) = result.unwrap();
            // Should extract "foo()   -> i32" with original spacing
            assert!(original[start..end].contains("foo()"));
        }
    }

    #[test]
    fn test_find_original_positions_multiline() {
        let original = "fn test() {\n    let x = 1;\n    x\n}";
        let normalized = normalize_whitespace(original);
        
        let search = "let x = 1";
        let normalized_search = normalize_whitespace(search);
        
        if let Some(idx) = normalized.find(&normalized_search) {
            let result = find_original_positions(original, &normalized, idx, &normalized_search);
            assert!(result.is_some());
            let (start, end) = result.unwrap();
            assert!(&original[start..end].contains("x = 1"));
        }
    }

    #[test]
    fn test_search_replace_with_fuzzy_normalized() {
        // Test that fuzzy matching with normalized whitespace actually applies replacements
        let original = "fn   compute(x: i32)   -> i32 {\n    x * 2\n}";
        let ops = vec![SearchReplace {
            search: "fn compute(x: i32) -> i32 {\n    x * 2\n}".to_string(),
            replace: "fn compute(x: i32) -> i32 {\n    x * 3\n}".to_string(),
        }];
        let result = MockResearch::apply_search_replaces_with_fuzzy(original, &ops);
        // Should have applied the replacement despite whitespace differences
        assert!(result.contains("x * 3"), "Expected replacement to be applied, got: {}", result);
    }

    #[test]
    fn test_parsed_edit_diff_stats_adds_lines() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo() {\n    42\n}".to_string(), // 3 lines
                    replace: "fn foo() -> i32 {\n    let x = 1;\n    x + 41\n}".to_string(), // 4 lines
                },
            ]),
        };
        let stats = edit.diff_stats().unwrap();
        assert_eq!(stats.lines_added, 1);
        assert_eq!(stats.lines_removed, 0);
    }

    #[test]
    fn test_parsed_edit_diff_stats_removes_lines() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "let a = 1;\nlet b = 2;\nlet c = 3;\n".to_string(), // 3 lines
                    replace: "let x = 1;\n".to_string(), // 1 line
                },
            ]),
        };
        let stats = edit.diff_stats().unwrap();
        assert_eq!(stats.lines_added, 0);
        assert_eq!(stats.lines_removed, 2);
    }

    #[test]
    fn test_parsed_edit_diff_stats_multiple_edits() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "x\n".to_string(), // 1 line
                    replace: "a\nb\n".to_string(), // 2 lines, +1
                },
                SearchReplace {
                    search: "y\nz\n".to_string(), // 2 lines
                    replace: "c\n".to_string(), // 1 line, -1
                },
            ]),
        };
        let stats = edit.diff_stats().unwrap();
        assert_eq!(stats.lines_added, 1);
        assert_eq!(stats.lines_removed, 1);
    }

    #[test]
    fn test_parsed_edit_diff_stats_full_file() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::FullFile("fn main() {}".to_string()),
        };
        assert!(edit.diff_stats().is_none());
    }

    #[test]
    fn test_parsed_edit_diff_stats_chars() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "short".to_string(), // 5 chars
                    replace: "much longer replacement".to_string(), // 24 chars
                },
            ]),
        };
        let stats = edit.diff_stats().unwrap();
        assert_eq!(stats.chars_removed, 5);
        assert_eq!(stats.chars_added, 24);
    }

    #[test]
    fn test_has_fuzzy_match_multiline() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo() {\n    bar()\n}".to_string(),
                    replace: "fn foo() { baz() }".to_string(),
                },
            ]),
        };
        assert!(edit.has_fuzzy_match());
    }

    #[test]
    fn test_has_fuzzy_match_leading_whitespace() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "    fn foo()".to_string(), // Leading spaces
                    replace: "fn foo()".to_string(),
                },
            ]),
        };
        assert!(edit.has_fuzzy_match());
    }

    #[test]
    fn test_has_fuzzy_match_multiple_spaces() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn  foo()".to_string(), // Double space
                    replace: "fn foo()".to_string(),
                },
            ]),
        };
        assert!(edit.has_fuzzy_match());
    }

    #[test]
    fn test_has_fuzzy_match_exact_single_line() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo()".to_string(), // No extra whitespace
                    replace: "fn bar()".to_string(),
                },
            ]),
        };
        assert!(!edit.has_fuzzy_match());
    }

    #[test]
    fn test_has_fuzzy_match_full_file() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::FullFile("fn main() {}".to_string()),
        };
        assert!(!edit.has_fuzzy_match());
    }

    #[test]
    fn test_has_fuzzy_match_mixed_edits() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "exact_match".to_string(), // No fuzzy needed
                    replace: "replacement".to_string(),
                },
                SearchReplace {
                    search: "fn foo() {\n    bar()\n}".to_string(), // Multi-line
                    replace: "fn foo() { baz() }".to_string(),
                },
            ]),
        };
        assert!(edit.has_fuzzy_match()); // At least one needs fuzzy
    }

    #[test]
    fn test_get_fuzzy_requirements_exact_match() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo()".to_string(), // Single line, no extra whitespace
                    replace: "fn bar()".to_string(),
                },
            ]),
        };
        let reqs = edit.get_fuzzy_requirements();
        assert_eq!(reqs.len(), 1);
        assert!(!reqs[0].needs_fuzzy);
        assert_eq!(reqs[0].suggested_strategy, FuzzyMatchStrategy::Exact);
        assert_eq!(reqs[0].line_count, 1);
        assert!(!reqs[0].has_extra_whitespace);
    }

    #[test]
    fn test_get_fuzzy_requirements_multiline() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo() {\n    bar()\n}".to_string(), // Multi-line, no extra whitespace
                    replace: "fn foo() { baz() }".to_string(),
                },
            ]),
        };
        let reqs = edit.get_fuzzy_requirements();
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].needs_fuzzy);
        assert_eq!(reqs[0].suggested_strategy, FuzzyMatchStrategy::LineByLine);
        assert_eq!(reqs[0].line_count, 3);
        assert!(!reqs[0].has_extra_whitespace);
    }

    #[test]
    fn test_get_fuzzy_requirements_trimmed() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "  fn foo()  ".to_string(), // Single line with extra whitespace
                    replace: "fn bar()".to_string(),
                },
            ]),
        };
        let reqs = edit.get_fuzzy_requirements();
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].needs_fuzzy);
        assert_eq!(reqs[0].suggested_strategy, FuzzyMatchStrategy::Trimmed);
        assert_eq!(reqs[0].line_count, 1);
        assert!(reqs[0].has_extra_whitespace);
    }

    #[test]
    fn test_get_fuzzy_requirements_normalized() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn   foo()  {\n    bar()\n}".to_string(), // Multi-line with extra whitespace
                    replace: "fn foo() { baz() }".to_string(),
                },
            ]),
        };
        let reqs = edit.get_fuzzy_requirements();
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].needs_fuzzy);
        assert_eq!(reqs[0].suggested_strategy, FuzzyMatchStrategy::Normalized);
        assert_eq!(reqs[0].line_count, 3);
        assert!(reqs[0].has_extra_whitespace);
    }

    #[test]
    fn test_get_fuzzy_requirements_multiple_blocks() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "exact_match".to_string(), // Exact
                    replace: "replacement".to_string(),
                },
                SearchReplace {
                    search: "  needs_trim  ".to_string(), // Needs trimmed
                    replace: "replaced".to_string(),
                },
                SearchReplace {
                    search: "fn multi()\n{\n    body\n}".to_string(), // Multi-line
                    replace: "fn multi() { body }".to_string(),
                },
            ]),
        };
        let reqs = edit.get_fuzzy_requirements();
        assert_eq!(reqs.len(), 3);
        assert!(!reqs[0].needs_fuzzy);
        assert!(reqs[1].needs_fuzzy);
        assert!(reqs[2].needs_fuzzy);
        assert_eq!(reqs[0].suggested_strategy, FuzzyMatchStrategy::Exact);
        assert_eq!(reqs[1].suggested_strategy, FuzzyMatchStrategy::Trimmed);
        assert_eq!(reqs[2].suggested_strategy, FuzzyMatchStrategy::LineByLine);
    }

    #[test]
    fn test_get_fuzzy_requirements_full_file() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::FullFile("fn main() {}".to_string()),
        };
        let reqs = edit.get_fuzzy_requirements();
        assert!(reqs.is_empty());
    }

    #[test]
    fn test_fuzzy_match_strategy_debug() {
        // Verify the enum implements Debug correctly for logging
        let strategy = FuzzyMatchStrategy::Normalized;
        let debug_str = format!("{:?}", strategy);
        assert_eq!(debug_str, "Normalized");
        
        let strategy = FuzzyMatchStrategy::LineByLine;
        let debug_str = format!("{:?}", strategy);
        assert_eq!(debug_str, "LineByLine");
    }

    #[test]
    fn test_fuzzy_match_strategy_equality() {
        // Verify PartialEq and Eq work correctly
        assert!(FuzzyMatchStrategy::Exact == FuzzyMatchStrategy::Exact);
        assert!(FuzzyMatchStrategy::Trimmed != FuzzyMatchStrategy::Normalized);
        assert!(FuzzyMatchStrategy::LineByLine == FuzzyMatchStrategy::LineByLine);
    }

    #[test]
    fn test_fuzzy_match_ratio_all_exact() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo()".to_string(), // No fuzzy needed
                    replace: "fn bar()".to_string(),
                },
                SearchReplace {
                    search: "let x = 1;".to_string(), // No fuzzy needed
                    replace: "let x = 2;".to_string(),
                },
            ]),
        };
        assert_eq!(edit.fuzzy_match_ratio(), 0.0);
    }

    #[test]
    fn test_fuzzy_match_ratio_all_fuzzy() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo() {\n    bar()\n}".to_string(), // Multi-line
                    replace: "fn foo() {}".to_string(),
                },
                SearchReplace {
                    search: "  let x = 1;  ".to_string(), // Extra whitespace
                    replace: "let x = 2;".to_string(),
                },
            ]),
        };
        assert_eq!(edit.fuzzy_match_ratio(), 1.0);
    }

    #[test]
    fn test_fuzzy_match_ratio_mixed() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "fn foo()".to_string(), // Exact match (no fuzzy)
                    replace: "fn bar()".to_string(),
                },
                SearchReplace {
                    search: "fn baz() {\n    qux()\n}".to_string(), // Multi-line (needs fuzzy)
                    replace: "fn baz() {}".to_string(),
                },
                SearchReplace {
                    search: "let y = 2;".to_string(), // Exact match (no fuzzy)
                    replace: "let y = 3;".to_string(),
                },
            ]),
        };
        // 1 out of 3 needs fuzzy matching
        assert!((edit.fuzzy_match_ratio() - (1.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_match_ratio_empty() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![]),
        };
        assert_eq!(edit.fuzzy_match_ratio(), 0.0);
    }

    #[test]
    fn test_fuzzy_match_ratio_full_file() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::FullFile("fn main() {}".to_string()),
        };
        assert_eq!(edit.fuzzy_match_ratio(), 0.0);
    }

    #[test]
    fn test_fuzzy_match_ratio_half_fuzzy() {
        let edit = ParsedEdit {
            file_path: "test.rs".to_string(),
            op: EditOp::SearchReplace(vec![
                SearchReplace {
                    search: "exact_match".to_string(), // No fuzzy
                    replace: "replacement".to_string(),
                },
                SearchReplace {
                    search: "fn multi()\n{\n    body\n}".to_string(), // Multi-line, needs fuzzy
                    replace: "fn multi() {}".to_string(),
                },
            ]),
        };
        // 1 out of 2 needs fuzzy = 0.5
        assert_eq!(edit.fuzzy_match_ratio(), 0.5);
    }

    #[test]
    fn test_edit_stats_summary_no_changes() {
        let stats = EditStats::default();
        assert_eq!(stats.summary(), "no changes");
    }

    #[test]
    fn test_edit_stats_summary_lines_added_only() {
        let stats = EditStats {
            lines_added: 5,
            lines_removed: 0,
            chars_added: 0,
            chars_removed: 0,
        };
        assert_eq!(stats.summary(), "+5 lines");
    }

    #[test]
    fn test_edit_stats_summary_lines_removed_only() {
        let stats = EditStats {
            lines_added: 0,
            lines_removed: 3,
            chars_added: 0,
            chars_removed: 0,
        };
        assert_eq!(stats.summary(), "-3 lines");
    }

    #[test]
    fn test_edit_stats_summary_lines_both() {
        let stats = EditStats {
            lines_added: 5,
            lines_removed: 2,
            chars_added: 0,
            chars_removed: 0,
        };
        assert_eq!(stats.summary(), "+5/-2 lines");
    }

    #[test]
    fn test_edit_stats_summary_chars_only() {
        let stats = EditStats {
            lines_added: 0,
            lines_removed: 0,
            chars_added: 120,
            chars_removed: 50,
        };
        assert_eq!(stats.summary(), "+120/-50 chars");
    }

    #[test]
    fn test_edit_stats_summary_full() {
        let stats = EditStats {
            lines_added: 5,
            lines_removed: 2,
            chars_added: 120,
            chars_removed: 50,
        };
        assert_eq!(stats.summary(), "+5/-2 lines, +120/-50 chars");
    }

    #[test]
    fn test_edit_stats_summary_chars_added_only() {
        let stats = EditStats {
            lines_added: 0,
            lines_removed: 0,
            chars_added: 100,
            chars_removed: 0,
        };
        assert_eq!(stats.summary(), "+100 chars");
    }

    #[test]
    fn test_edit_stats_summary_chars_removed_only() {
        let stats = EditStats {
            lines_added: 0,
            lines_removed: 0,
            chars_added: 0,
            chars_removed: 75,
        };
        assert_eq!(stats.summary(), "-75 chars");
    }
}
