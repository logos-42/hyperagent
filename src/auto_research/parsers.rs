use crate::llm::LLMClient;

use super::AutoResearch;

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// Phase 3: 解析 LLM 响应中的多文件修改
    ///
    /// 支持两种格式：
    /// 1. 单文件格式（向后兼容）: HYPOTHESIS: ... \n\n ```rust ... ```
    /// 2. 多文件格式: HYPOTHESIS: ... \n\n FILE: path \n ```rust ... ``` \n FILE: path \n ```rust ... ```
    pub(crate) fn parse_response_multi(&self, response: &str) -> Option<(String, Vec<(String, String)>)> {
        // 提取 HYPOTHESIS
        let hypothesis = if let Some(start) = response.find("HYPOTHESIS:") {
            let start = start + 11;
            let end = response[start..].find("\n\n").unwrap_or(response[start..].len());
            response[start..start + end].trim().to_string()
        } else {
            "No hypothesis stated".to_string()
        };

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

    /// 解析多文件响应：按 FILE: 标记分割代码块
    fn parse_multi_file_response(&self, hypothesis: &str, response: &str) -> Option<(String, Vec<(String, String)>)> {
        let mut changes: Vec<(String, String)> = Vec::new();

        // 找所有 FILE: 标记和对应的代码块
        let mut pos = 0;
        let bytes = response.as_bytes();

        while pos < bytes.len() {
            // 找下一个 FILE: 标记
            let file_start = match Self::find_marker(response, "FILE:", pos) {
                Some(p) => p,
                None => break,
            };

            let file_line_end = response[file_start..].find('\n').map(|i| file_start + i).unwrap_or(response.len());
            let file_path = response[file_start + 5..file_line_end].trim().to_string();

            // 跳到 FILE: 行之后找代码块
            let code_region = &response[file_line_end..];

            // 找 ```rust ... ``` 或 ``` ... ```
            let (code_start, code_end) = if let Some(block_start) = code_region.find("```") {
                let actual_start = block_start + 3;
                // 跳过语言标识（如 "rust"）
                let content_start = if code_region[actual_start..].starts_with("rust") {
                    actual_start + 4
                } else {
                    actual_start
                };
                // 找闭合的 ```
                let close_pos = match code_region[content_start..].find("```") {
                    Some(p) => content_start + p,
                    None => continue,
                };
                let abs_start = file_line_end + content_start;
                let abs_end = file_line_end + close_pos;
                (abs_start, abs_end)
            } else {
                continue;
            };

            let code = response[code_start..code_end].trim().to_string();
            if !code.is_empty() {
                changes.push((file_path, code));
            }

            pos = code_end + 3; // 跳过闭合的 ```
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
                // 计算在原文本中的绝对位置
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
