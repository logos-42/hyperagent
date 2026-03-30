use crate::llm::LLMClient;

use super::{AutoResearch, Experiment};

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// Truncate a string to a maximum number of characters, appending "..." if truncated.
    fn truncate_output(s: &str, max_chars: usize) -> String {
        if s.len() <= max_chars {
            s.to_string()
        } else {
            format!("{}...", &s[..max_chars])
        }
    }

    /// Format test results as a "passed/total" string
    fn format_tests(passed: u32, total: u32) -> String {
        format!("{}/{}", passed, total)
    }

    /// Format a test transition from before to after
    fn format_test_transition(before: (u32, u32), after: (u32, u32)) -> String {
        format!(
            "{} → {}",
            Self::format_tests(before.0, before.1),
            Self::format_tests(after.0, after.1)
        )
    }

    /// Format a single experiment for display in the research prompt
    fn format_experiment_summary(experiment: &Experiment) -> String {
        format!(
            "---\nExp {}: {}\nHypothesis: {}\nOutcome: {:?}\nReflection: {}\nTests: {}",
            experiment.iteration,
            experiment.file,
            experiment.hypothesis,
            experiment.outcome,
            experiment.reflection,
            Self::format_test_transition(experiment.tests_before, experiment.tests_after),
        )
    }

    /// 构建研究 prompt：注入全局架构上下文 + Web 搜索上下文 + 相关文件（Phase 4）
    pub(crate) fn build_research_prompt(&self, file: &str, code: &str, history: &[Experiment], web_context: Option<&str>) -> String {
        let recent_history = history.iter().rev().take(5)
            .map(Self::format_experiment_summary)
            .collect::<Vec<_>>()
            .join("\n");

        let codebase_context = self.codebase.build_context_prompt(file);

        // Phase 4: 加载相关文件的签名（被依赖和依赖的文件）
        let related_context = self.build_related_files_context(file);

        // Phase 2: 测试生成指令
        let test_gen_instruction = "\n\
             7. Write NEW TESTS for your improvement inside `#[cfg(test)] mod tests { ... }`.\n\
                - Test the specific behavior you improved\n\
                - Use descriptive test names\n\
                - If the file already has tests, ADD new ones (don't remove existing)\n";

        // 编辑模式说明：优先使用 SEARCH/REPLACE（精确编辑），小文件可回退到 FILE 格式
        let edit_mode_instruction = "
             === EDIT FORMAT (IMPORTANT) ===
             Use SEARCH/REPLACE blocks for PRECISE editing. This avoids rewriting entire files and losing code.\n\n\
             For the PRIMARY file:\n\
             EDIT: {file}\n\
             <<<<<<< SEARCH\n\
             <exact existing code to find — must match character-by-character>\n\
             =======\n\
             <replacement code>\n\
             >>>>>>> REPLACE\n\n\
             You can include MULTIPLE SEARCH/REPLACE blocks in one EDIT section for the same file.\n\
             For OTHER files that also need changes, add separate EDIT blocks:\n\n\
             EDIT: <other/file.rs>\n\
             <<<<<<< SEARCH\n\
             <exact existing code>\n\
             =======\n\
             <replacement code>\n\
             >>>>>>> REPLACE\n\n\
             CRITICAL RULES for SEARCH/REPLACE:\n\
             - The SEARCH block MUST match the original code EXACTLY (same whitespace, same newlines)\n\
             - Include ENOUGH context (3-5 lines) so the search block is unique in the file\n\
             - Do NOT try to match the entire file — only include the lines you want to change\n\
             - You can add code (make REPLACE longer than SEARCH) or remove code (make REPLACE shorter)\n\
             - If you need to add new functions, put the SEARCH block around the insertion point\n\
               (e.g., include the closing brace of the previous function)\n";

        format!(
            "You are an AI researcher improving your own codebase. This is a self-research loop.\n\n\
             {codebase_context}\n\n\
             {related_context}\n\
             {web_section}\n\
             === PRIMARY FILE: src/{file} ===\n\
             {code}\n\n\
             === PAST EXPERIMENTS ===\n\
             {history}\n\n\
             === YOUR TASK ===\n\
             1. Understand the architecture above — how this file fits into the system.\n\
             2. Read the code carefully, including the related files context.\n\
             3. Identify ONE specific, concrete improvement.\n\
             4. Consider cross-file dependencies — don't break other modules.\n\
             5. Output in this EXACT format:\n\n\
             HYPOTHESIS: <one sentence describing what you'll improve and why>\n\n\
             {edit_mode_section}\n\
             {test_gen_section}\n\
             Rules:\n\
             - Do NOT change public API signatures (function names, trait methods, struct fields)\n\
             - Do NOT add new external dependencies\n\
             - Do NOT break imports used by other files\n\
             - Focus on one improvement per iteration (but may span multiple files)\n\
             - Use SEARCH/REPLACE format — do NOT output complete files\n\
             - The SEARCH block must match the original code EXACTLY\n\
             - Include enough surrounding context (3-5 lines) for unique matching\n\
             - Include any new tests in the #[cfg(test)] module\n\
             - When modifying multiple files, ensure cross-file consistency",
            file = file,
            code = code,
            history = if recent_history.is_empty() { "(no experiments yet)".to_string() } else { recent_history },
            web_section = web_context.map(|ctx| format!("{}\n", ctx)).unwrap_or_default(),
            related_context = related_context,
            edit_mode_section = edit_mode_instruction,
            test_gen_section = test_gen_instruction,
        )
    }

    /// Phase 4: 构建相关文件上下文 — 被依赖和依赖的文件签名
    pub(crate) fn build_related_files_context(&self, target_file: &str) -> String {
        let mut context = String::new();

        if let Some(target_summary) = self.codebase.files.get(target_file) {
            // 找到依赖目标文件的文件（使用目标文件中导出的东西）
            let dependents: Vec<String> = self
                .codebase
                .files
                .iter()
                .filter(|(path, summary)| {
                    *path != target_file
                        && summary.uses.iter().any(|u| {
                            let module_hint = target_file
                                .strip_suffix(".rs")
                                .unwrap_or(target_file)
                                .replace('/', "::");
                            u.contains(&module_hint)
                                || u.contains(
                                    &target_file.replace('/', "::").replace(".rs", "")
                                        .replace("mod.rs", "")
                                )
                        })
                })
                .map(|(path, _)| path.clone())
                .collect();

            // 找到目标文件依赖的文件（目标文件 use crate::xxx 的文件）
            let dependencies: Vec<String> = target_summary
                .uses
                .iter()
                .filter(|u| u.contains("crate::"))
                .filter_map(|u| {
                    let binding = u.replace("crate::", "");
                    let module = binding.split("::").next()?;
                    let candidates: Vec<String> = self
                        .codebase
                        .files
                        .keys()
                        .filter(|p| {
                            p.replace('/', "::")
                                .replace(".rs", "")
                                .replace("mod.rs", "")
                                .contains(module)
                        })
                        .cloned()
                        .collect();
                    candidates.first().cloned()
                })
                .collect();

            let related_files: Vec<String> = dependents
                .into_iter()
                .chain(dependencies.into_iter())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            if !related_files.is_empty() {
                context.push_str("=== RELATED FILES ===\n");
                for rel_file in related_files {
                    if let Some(summary) = self.codebase.files.get(&rel_file) {
                        let types: Vec<String> = summary
                            .structs
                            .iter()
                            .chain(summary.enums.iter())
                            .chain(summary.traits.iter())
                            .take(8)
                            .cloned()
                            .collect();
                        let fns_str: Vec<String> = summary
                            .functions
                            .iter()
                            .take(10)
                            .cloned()
                            .collect();

                        context.push_str(&format!(
                            "\n--- {} ({} lines) ---\n",
                            rel_file, summary.lines
                        ));
                        if !types.is_empty() {
                            context.push_str(&format!("Types: {}\n", types.join(", ")));
                        }
                        if !fns_str.is_empty() {
                            context.push_str(&format!("Functions: {}\n", fns_str.join(", ")));
                        }
                    }
                }
                context.push('\n');
            }
        }

        context
    }

    /// 构建反思 prompt
    pub(crate) fn build_reflection_prompt(
        &self,
        file: &str,
        hypothesis: &str,
        tests_before: (u32, u32),
        tests_after: (u32, u32),
        compile_ok: bool,
        test_output: &str,
    ) -> String {
        format!(
            "You are an AI researcher reflecting on an experiment.\n\n\
             File: src/{file}\n\
             Hypothesis: {hypothesis}\n\
             Before: {before_passed}/{before_total} tests passing\n\
             After: {after_passed}/{after_total} tests passing\n\
             Compiled: {compile_ok}\n\n\
             Test output (last 500 chars):\n\
             {output}\n\n\
             Write a 1-2 sentence reflection:\n\
             - What worked or didn't\n\
             - What to try next\n\
             - Keep it specific and actionable\n\n\
             REFLECTION:",
            file = file,
            hypothesis = hypothesis,
            before_passed = tests_before.0,
            before_total = tests_before.1,
            after_passed = tests_after.0,
            after_total = tests_after.1,
            compile_ok = compile_ok,
            output = Self::truncate_output(test_output, 500),
        )
    }
}
