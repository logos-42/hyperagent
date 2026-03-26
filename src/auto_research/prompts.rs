use crate::llm::LLMClient;

use super::{AutoResearch, Experiment};

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// 构建研究 prompt：注入全局架构上下文 + Web 搜索上下文 + 相关文件（Phase 4）
    pub(crate) fn build_research_prompt(&self, file: &str, code: &str, history: &[Experiment], web_context: Option<&str>) -> String {
        let recent_history = history.iter().rev().take(5).map(|e| {
            format!(
                "---\nExp {}: {}\nHypothesis: {}\nOutcome: {:?}\nReflection: {}\nTests: {}/{} → {}/{}",
                e.iteration, e.file, e.hypothesis, e.outcome, e.reflection,
                e.tests_before.0, e.tests_before.1, e.tests_after.0, e.tests_after.1,
            )
        }).collect::<Vec<_>>().join("\n");

        let codebase_context = self.codebase.build_context_prompt(file);

        // Phase 4: 加载相关文件的签名（被依赖和依赖的文件）
        let related_context = self.build_related_files_context(file);

        // Phase 2: 测试生成指令
        let test_gen_instruction = "\n\
             7. Write NEW TESTS for your improvement inside `#[cfg(test)] mod tests { ... }`.\n\
                - Test the specific behavior you improved\n\
                - Use descriptive test names\n\
                - If the file already has tests, ADD new ones (don't remove existing)\n";

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
             FILE: {file}\n\
             ```rust\n\
             <complete improved primary file, including any new #[cfg(test)] tests>\n\
             ```\n\n\
             If your improvement requires changes to OTHER files (e.g., updating a trait in another module,\n\
             or modifying a type used by the primary file), add additional FILE blocks:\n\n\
             FILE: <other/file.rs>\n\
             ```rust\n\
             <complete improved secondary file>\n\
             ```\n\n\
             {test_gen_section}\n\
             Rules:\n\
             - Do NOT change public API signatures (function names, trait methods, struct fields)\n\
             - Do NOT add new external dependencies\n\
             - Do NOT break imports used by other files\n\
             - Focus on one improvement per iteration (but may span multiple files)\n\
             - Output COMPLETE files, not diffs\n\
             - Include any new tests in the #[cfg(test)] module\n\
             - When modifying multiple files, ensure cross-file consistency",
            file = file,
            code = code,
            history = if recent_history.is_empty() { "(no experiments yet)".to_string() } else { recent_history },
            web_section = web_context.map(|ctx| format!("{}\n", ctx)).unwrap_or_default(),
            related_context = related_context,
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
            output = test_output.chars().take(500).collect::<String>(),
        )
    }
}
