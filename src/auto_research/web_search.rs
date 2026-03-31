use crate::llm::LLMClient;
use crate::web::{WebFetchTool, WebSearchResult, FetchOutput, build_web_context_prompt};
use futures::future::join_all;

use super::AutoResearch;

impl<C: LLMClient + Clone> AutoResearch<C> {
    /// 使用 LLM 生成搜索查询，执行 Web 搜索，并返回上下文字符串
    pub(crate) async fn gather_web_context(&self, file: &str, code: &str) -> Option<String> {
        if !self.config.enable_web {
            return None;
        }

        // 1. LLM 生成搜索查询
        let search_prompt = format!(
            "You are researching improvements for a Rust file. Given the code below, \
             suggest 2-3 web search queries to find best practices, idioms, or solutions.\n\
             Be specific. Output ONLY the queries, one per line. No numbering.\n\n\
             File: src/{file}\n\
             Code (first 2000 chars):\n{code_snippet}",
            file = file,
            code_snippet = code.chars().take(2000).collect::<String>(),
        );

        let search_response = match self.client.complete(&search_prompt).await {
            Ok(r) => r.content,
            Err(e) => {
                tracing::warn!("  Web: failed to generate search queries: {}", e);
                return None;
            }
        };

        let queries: Vec<String> = search_response
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && l.len() < 200)
            .take(3)
            .collect();

        if queries.is_empty() {
            return None;
        }

        tracing::info!("  Web: searching for: {:?}", queries);

        // 2. 并发搜索
        let mut all_results: Vec<WebSearchResult> = Vec::new();
        for query in &queries {
            match self.web_client.search(query, self.config.web_search_limit).await {
                Ok(results) => {
                    tracing::info!("  Web: {} results for '{}'", results.len(), query);
                    all_results.extend(results);
                }
                Err(e) => tracing::warn!("  Web: search failed for '{}': {}", query, e),
            }
        }

        if all_results.is_empty() {
            return None;
        }

        // 3. 抓取前 N 个页面
        let fetch_tool = WebFetchTool::new();
        let fetch_urls: Vec<String> = all_results
            .iter()
            .take(self.config.web_fetch_limit)
            .map(|r| r.url.clone())
            .collect();

        if !fetch_urls.is_empty() {
            tracing::info!("  Web: fetching {} pages concurrently...", fetch_urls.len());
            let fetch_futures: Vec<_> = fetch_urls
                .iter()
                .map(|url| fetch_tool.fetch(url))
                .collect();
            
            let fetch_results = join_all(fetch_futures).await;
            
            let pages: Vec<FetchOutput> = fetch_results
                .into_iter()
                .zip(fetch_urls.iter())
                .filter_map(|(result, url)| {
                    match result {
                        Ok(page) => {
                            tracing::info!("  Web: fetched {} ({} chars)", url, page.text_length);
                            Some(page)
                        }
                        Err(e) => {
                            tracing::warn!("  Web: failed to fetch {}: {}", url, e);
                            None
                        }
                    }
                })
                .collect();

            let context = build_web_context_prompt(&all_results, &pages);
            tracing::info!("  Web: gathered {} chars of context", context.len());
            return Some(context);
        }

        Some(build_web_context_prompt(&all_results, &[]))
    }
}
