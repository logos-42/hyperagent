//! Web browsing and search tools for the auto research agent.
//!
//! Implements rig `Tool` trait so the LLM agent can directly call:
//!   - `web_search` — search the internet via DuckDuckGo
//!   - `web_fetch`  — fetch and extract text from a URL
//!
//! These tools integrate with `AutoResearch` to give the agent internet access.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Context, Result};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared HTTP client
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HttpClient {
    inner: reqwest::Client,
}

impl HttpClient {
    fn new() -> Self {
        let inner = reqwest::Client::builder()
            .user_agent("Hyperagent/0.1 (AI self-research agent; +https://github.com)")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self { inner }
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

fn default_http_client() -> Arc<HttpClient> {
    Arc::new(HttpClient::new())
}

// ---------------------------------------------------------------------------
// rig Tool: web_search
// ---------------------------------------------------------------------------

/// Arguments for the web_search tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct SearchArgs {
    /// Search query string
    pub query: String,
    /// Maximum number of results to return (default: 5, max: 10)
    #[serde(default = "default_max_results")]
    pub max_results: usize,
}

fn default_max_results() -> usize {
    5
}

/// Output of the web_search tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct SearchOutput {
    pub results: Vec<WebSearchResult>,
    pub query: String,
}

/// A single search result.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Error type for web tools.
#[derive(Debug, thiserror::Error)]
pub enum WebToolError {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Timeout: {0}")]
    Timeout(String),
}

impl From<anyhow::Error> for WebToolError {
    fn from(e: anyhow::Error) -> Self {
        WebToolError::Http(e.to_string())
    }
}

/// Web search tool (rig Tool trait).
///
/// Searches DuckDuckGo HTML endpoint — no API key required.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct WebSearchTool {
    #[serde(skip, default = "default_http_client")]
    client: Arc<HttpClient>,
}

impl WebSearchTool {
    pub fn new() -> Self {
        Self {
            client: Arc::new(HttpClient::new()),
        }
    }

    /// Direct search (without rig Tool machinery).
    pub async fn search(&self, query: &str, max_results: usize) -> Result<Vec<WebSearchResult>> {
        let max = max_results.min(10).max(1);
        let encoded = percent_encode(query);
        let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

        let resp = self
            .client
            .inner
            .get(&url)
            .send()
            .await
            .with_context(|| "Search request failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("Search HTTP {}", resp.status());
        }

        let html = resp.text().await?;
        Ok(parse_ddg_results(&html, max))
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebSearchTool {
    const NAME: &'static str = "web_search";

    type Error = WebToolError;
    type Args = SearchArgs;
    type Output = SearchOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the internet for information. Returns a list of search results \
                          with titles, URLs, and snippets. Use this to find best practices, \
                          documentation, solutions to problems, or any information needed for \
                          improving code.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (1-10, default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }),
        };
        async move { def }
    }

    fn call(&self, args: Self::Args) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let client = self.client.clone();
        let query = args.query.clone();
        let max = args.max_results;
        async move {
            let results = client_search(&client, &query, max).await?;
            Ok(SearchOutput {
                results,
                query,
            })
        }
    }
}

/// Standalone search function using a shared client.
async fn client_search(client: &HttpClient, query: &str, max_results: usize) -> Result<Vec<WebSearchResult>, WebToolError> {
    let max = max_results.min(10).max(1);
    let encoded = percent_encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

    let resp = client
        .inner
        .get(&url)
        .send()
        .await
        .map_err(|e| WebToolError::Http(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(WebToolError::Http(format!("HTTP {}", resp.status())));
    }

    let html = resp.text().await.map_err(|e| WebToolError::Http(e.to_string()))?;
    Ok(parse_ddg_results(&html, max))
}

// ---------------------------------------------------------------------------
// rig Tool: web_fetch
// ---------------------------------------------------------------------------

/// Arguments for the web_fetch tool.
#[derive(Deserialize, Serialize, Debug)]
pub struct FetchArgs {
    /// URL to fetch
    pub url: String,
}

/// Output of the web_fetch tool.
#[derive(Serialize, Deserialize, Debug)]
pub struct FetchOutput {
    pub url: String,
    pub title: String,
    pub text: String,
    /// Length of extracted text in characters
    pub text_length: usize,
}

/// Web fetch tool (rig Tool trait).
///
/// Fetches a URL and extracts readable text from the HTML.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct WebFetchTool {
    #[serde(skip, default = "default_http_client")]
    client: Arc<HttpClient>,
}

impl WebFetchTool {
    pub fn new() -> Self {
        Self {
            client: Arc::new(HttpClient::new()),
        }
    }

    /// Direct fetch (without rig Tool machinery).
    pub async fn fetch(&self, url: &str) -> Result<FetchOutput> {
        let resp = self
            .client
            .inner
            .get(url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch {}", url))?;

        if !resp.status().is_success() {
            anyhow::bail!("HTTP {} for {}", resp.status(), url);
        }

        let html = resp.text().await?;
        let title = extract_title(&html);
        let text = html_to_text(&html);
        let text_length = text.len();

        Ok(FetchOutput {
            url: url.to_string(),
            title,
            text,
            text_length,
        })
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WebFetchTool {
    const NAME: &'static str = "web_fetch";

    type Error = WebToolError;
    type Args = FetchArgs;
    type Output = FetchOutput;

    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send {
        let def = ToolDefinition {
            name: "web_fetch".to_string(),
            description: "Fetch a web page and extract its readable text content. \
                          Returns the page title and main body text (scripts, styles, \
                          navigation are stripped). Use this to read documentation, \
                          blog posts, or any web content in detail.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must start with http:// or https://)"
                    }
                },
                "required": ["url"]
            }),
        };
        async move { def }
    }

    fn call(&self, args: Self::Args) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        let client = self.client.clone();
        let url = args.url;
        async move {
            let resp = client
                .inner
                .get(&url)
                .send()
                .await
                .map_err(|e| WebToolError::Http(e.to_string()))?;

            if !resp.status().is_success() {
                return Err(WebToolError::Http(format!("HTTP {} for {}", resp.status(), url)));
            }

            let html = resp.text().await.map_err(|e| WebToolError::Http(e.to_string()))?;
            let title = extract_title(&html);
            let text = html_to_text(&html);

            Ok(FetchOutput {
                url: url.clone(),
                title,
                text_length: text.len(),
                text,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// HTML parsing (lightweight, no scraper dependency)
// ---------------------------------------------------------------------------

/// Simple HTML tag stripper — removes all tags and extracts text.
/// Does not need a full HTML parser crate.
fn html_to_text(html: &str) -> String {
    let mut result = String::with_capacity(html.len() / 3);
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let mut tag_buf = String::new();

    for ch in html.chars() {
        match ch {
            '<' => {
                in_tag = true;
                tag_buf.clear();
                tag_buf.push(ch);
            }
            '>' if in_tag => {
                in_tag = false;
                tag_buf.push(ch);
                let tag_lower = tag_buf.to_lowercase();

                // Track script/style blocks
                if tag_lower.starts_with("<script") {
                    in_script = true;
                } else if tag_lower.starts_with("</script") {
                    in_script = false;
                } else if tag_lower.starts_with("<style") {
                    in_style = true;
                } else if tag_lower.starts_with("</style") {
                    in_style = false;
                }

                // Add paragraph breaks for block elements
                if tag_lower.starts_with("<p")
                    || tag_lower.starts_with("<div")
                    || tag_lower.starts_with("<h1")
                    || tag_lower.starts_with("<h2")
                    || tag_lower.starts_with("<h3")
                    || tag_lower.starts_with("<h4")
                    || tag_lower.starts_with("<br")
                    || tag_lower.starts_with("<li")
                    || tag_lower.starts_with("<tr")
                {
                    result.push('\n');
                }
            }
            _ if in_tag => {
                tag_buf.push(ch);
            }
            _ if in_script || in_style => {
                // Skip content inside script/style
            }
            '&' => {
                // Decode common HTML entities
                result.push('&');
            }
            '\n' | '\r' | '\t' => {
                result.push(' ');
            }
            _ => {
                result.push(ch);
            }
        }
    }

    // Clean up whitespace
    let text: String = result
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    // Truncate very long pages
    if text.len() > 8000 {
        format!("{}...", &text[..8000])
    } else {
        text
    }
}

/// Extract <title> from HTML.
fn extract_title(html: &str) -> String {
    let lower = html.to_lowercase();
    if let Some(start) = lower.find("<title>") {
        let start = start + 7;
        if let Some(end) = lower[start..].find("</title>") {
            return html[start..start + end].trim().to_string();
        }
    }
    String::new()
}

/// Parse DuckDuckGo HTML search results.
fn parse_ddg_results(html: &str, max_results: usize) -> Vec<WebSearchResult> {
    let mut results = Vec::new();
    let lower = html.to_lowercase();

    // DuckDuckGo HTML results: each result is in a <a class="result__a"> with href
    // and snippets in <a class="result__snippet">
    let mut pos = 0;
    while results.len() < max_results {
        // Find next result link
        let link_start = match lower[pos..].find("class=\"result__a\"") {
            Some(i) => pos + i,
            None => break,
        };

        // Find href within this element
        let href_start = match lower[link_start..].find("href=") {
            Some(i) => link_start + i + 6,
            None => break,
        };

        // Extract URL (between quotes after href=)
        let raw_url = extract_quoted(&html[href_start..]);
        // DDG returns redirect URLs like "//duckduckgo.com/l/?uddg=...", extract the real URL
        let url = resolve_ddg_redirect(&raw_url);

        // Find the link text (between > and <)
        let text_start = match lower[href_start..].find('>') {
            Some(i) => href_start + i + 1,
            None => break,
        };
        let title_end = match lower[text_start..].find('<') {
            Some(i) => text_start + i,
            None => break,
        };
        let title = html[text_start..title_end].trim().to_string();

        // Find snippet (class="result__snippet")
        let snippet_area = &lower[title_end..];
        let snippet_start = match snippet_area.find("class=\"result__snippet\"") {
            Some(i) => title_end + i,
            None => {
                pos = title_end;
                continue;
            }
        };

        let snip_text_start = match lower[snippet_start..].find('>') {
            Some(i) => snippet_start + i + 1,
            None => break,
        };
        let snip_text_end = match lower[snip_text_start..].find('<') {
            Some(i) => snip_text_start + i,
            None => break,
        };
        let snippet = html[snip_text_start..snip_text_end]
            .trim()
            .to_string();

        if !title.is_empty() && !url.is_empty() {
            results.push(WebSearchResult {
                title,
                url,
                snippet,
            });
        }

        pos = snip_text_end;
    }

    results
}

/// Extract a quoted string starting from a position (handles both " and ').
fn extract_quoted(s: &str) -> String {
    let s = s.trim_start();
    if s.starts_with('"') {
        if let Some(end) = s[1..].find('"') {
            return s[1..end + 1].to_string();
        }
    } else if s.starts_with('\'') {
        if let Some(end) = s[1..].find('\'') {
            return s[1..end + 1].to_string();
        }
    }
    // No quotes — take until whitespace or >
    s.chars()
        .take_while(|c| !c.is_whitespace() && *c != '>')
        .collect()
}

/// Resolve DuckDuckGo redirect URL to the actual destination URL.
///
/// DDG HTML search returns URLs like:
///   //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...
/// We extract the `uddg` parameter and URL-decode it.
fn resolve_ddg_redirect(url: &str) -> String {
    // Check if it's a DDG redirect URL
    if !url.contains("uddg=") {
        return url.to_string();
    }

    // Find uddg= parameter
    if let Some(uddg_start) = url.find("uddg=") {
        let rest = &url[uddg_start + 5..];
        // Take until next & or end
        let encoded = match rest.find('&') {
            Some(end) => &rest[..end],
            None => rest,
        };
        // URL-decode percent-encoded characters
        return url_decode(encoded);
    }

    url.to_string()
}

/// Minimal URL percent-decoding (no external crate).
fn url_decode(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            // Take next 2 hex chars
            let hex: String = chars.by_ref().take(2).collect();
            if hex.len() == 2 {
                if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                    out.push(byte as char);
                    continue;
                }
            }
            out.push('%');
            out.push_str(&hex);
        } else if c == '+' {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// URL percent-encoding (minimal, no external crate).
fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

/// Build a prompt section with web context to inject into research prompts.
pub fn build_web_context_prompt(results: &[WebSearchResult], pages: &[FetchOutput]) -> String {
    if results.is_empty() && pages.is_empty() {
        return String::new();
    }

    let mut sections = Vec::new();

    if !results.is_empty() {
        let items: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("[{}] {} — {}\n    {}", i + 1, r.title, r.url, r.snippet))
            .collect();
        sections.push(format!("=== WEB SEARCH RESULTS ===\n{}\n", items.join("\n")));
    }

    if !pages.is_empty() {
        let items: Vec<String> = pages
            .iter()
            .map(|p| {
                let text = if p.text.len() > 2000 {
                    format!("{}...", &p.text[..2000])
                } else {
                    p.text.clone()
                };
                format!("[{}] {}\n{}", p.url, p.title, text)
            })
            .collect();
        sections.push(format!("=== FETCHED WEB PAGES ===\n{}\n", items.join("\n---\n")));
    }

    sections.join("\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_text_basic() {
        let html = r#"<html><head><title>Test Page</title></head><body><h1>Hello</h1><p>World</p></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_html_to_text_strips_script() {
        let html = r#"<html><body><script>alert('xss')</script><p>Visible</p></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Visible"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn test_html_to_text_strips_style() {
        let html = r#"<html><body><style>.x{color:red}</style><p>Content</p></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn test_extract_title() {
        let html = r#"<html><head><title>My Title</title></head><body></body></html>"#;
        assert_eq!(extract_title(html), "My Title");
    }

    #[test]
    fn test_extract_title_missing() {
        let html = r#"<html><body>No title here</body></html>"#;
        assert_eq!(extract_title(html), "");
    }

    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("hello world"), "hello+world");
        assert_eq!(percent_encode("a=b&c=d"), "a%3Db%26c%3Dd");
        assert_eq!(percent_encode("Rust async"), "Rust+async");
    }

    #[test]
    fn test_parse_ddg_results_empty() {
        let results = parse_ddg_results("<html><body></body></html>", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_build_web_context_prompt_empty() {
        let prompt = build_web_context_prompt(&[], &[]);
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_build_web_context_prompt_with_results() {
        let results = vec![WebSearchResult {
            title: "Rust async".to_string(),
            url: "https://example.com/rust".to_string(),
            snippet: "Async programming in Rust".to_string(),
        }];
        let prompt = build_web_context_prompt(&results, &[]);
        assert!(prompt.contains("WEB SEARCH RESULTS"));
        assert!(prompt.contains("Rust async"));
    }

    #[test]
    fn test_extract_quoted_double() {
        assert_eq!(extract_quoted("\"hello world\" rest"), "hello world");
    }

    #[test]
    fn test_extract_quoted_single() {
        assert_eq!(extract_quoted("'hello' rest"), "hello");
    }

    #[test]
    fn test_resolve_ddg_redirect() {
        // Already a real URL — pass through
        assert_eq!(
            resolve_ddg_redirect("https://example.com/path"),
            "https://example.com/path"
        );
        // DDG redirect URL — extract uddg
        let ddg = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fstela2502.github.io%2Fmdbook_simulted_annealing_in_rust%2F&rut=abc123";
        assert_eq!(
            resolve_ddg_redirect(ddg),
            "https://stela2502.github.io/mdbook_simulted_annealing_in_rust/"
        );
    }

    #[test]
    fn test_url_decode() {
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("https%3A%2F%2Fexample.com"), "https://example.com");
    }

    #[test]
    fn test_search_args_default() {
        let args = SearchArgs {
            query: "test".to_string(),
            max_results: 0,
        };
        // serde default should handle this, but test the function
        assert_eq!(default_max_results(), 5);
    }

    #[test]
    fn test_search_output_serialization() {
        let output = SearchOutput {
            query: "test".to_string(),
            results: vec![WebSearchResult {
                title: "Test".to_string(),
                url: "https://example.com".to_string(),
                snippet: "A test page".to_string(),
            }],
        };
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("Test"));
    }

    #[tokio::test]
    async fn test_web_search_tool_trait() {
        let tool = WebSearchTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "web_search");
        assert!(!def.description.is_empty());
    }

    #[tokio::test]
    async fn test_web_fetch_tool_trait() {
        let tool = WebFetchTool::new();
        let def = tool.definition(String::new()).await;
        assert_eq!(def.name, "web_fetch");
        assert!(!def.description.is_empty());
    }

    #[tokio::test]
    async fn test_web_fetch_example() {
        let tool = WebFetchTool::new();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tool.fetch("https://example.com"),
        )
        .await;

        if let Ok(Ok(page)) = result {
            assert!(!page.url.is_empty());
            assert!(page.text.contains("Example Domain") || page.title.contains("Example"));
        }
    }
}
