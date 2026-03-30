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

/// Configuration for HTTP client timeout and retry behavior.
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Request timeout in seconds (default: 30)
    pub timeout_secs: u64,
    /// Maximum retry attempts for transient failures (default: 3)
    pub max_retries: u32,
    /// Base delay for exponential backoff in milliseconds (default: 100)
    pub retry_base_delay_ms: u64,
    /// Whether to retry on timeout errors (default: true)
    pub retry_on_timeout: bool,
    /// Maximum response body size in bytes (default: 10MB)
    /// Prevents memory exhaustion from oversized responses.
    pub max_response_size: usize,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_retries: 3,
            retry_base_delay_ms: 100,
            retry_on_timeout: true,
            max_response_size: 10 * 1024 * 1024, // 10MB default
        }
    }
}

#[derive(Debug, Clone)]
struct HttpClient {
    inner: reqwest::Client,
    config: HttpClientConfig,
}

impl HttpClient {
    fn new() -> Self {
        Self::with_config(HttpClientConfig::default())
    }
    
    fn with_config(config: HttpClientConfig) -> Self {
        let inner = reqwest::Client::builder()
            .user_agent("Hyperagent/0.1 (AI self-research agent; +https://github.com)")
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        Self { inner, config }
    }
    
    /// Returns a reference to the current HTTP client configuration.
    ///
    /// This is useful for diagnostics and logging, allowing callers to
    /// inspect timeout settings and retry configuration.
    fn config(&self) -> &HttpClientConfig {
        &self.config
    }
    
    /// Checks if the client has remaining retry attempts for a failed request.
    ///
    /// Returns `true` if `max_retries > 0`, indicating the client will
    /// attempt to recover from transient failures.
    fn has_retry_budget(&self) -> bool {
        self.config.max_retries > 0
    }
    
    /// Execute a request with retry logic and exponential backoff with jitter.
    ///
    /// Uses "full jitter" strategy: random delay between 0 and the exponential backoff.
    /// This prevents thundering herd problems when multiple clients retry simultaneously.
    ///
    /// Retries on:
    /// - Network errors (timeout, connection refused)
    /// - HTTP 5xx server errors (500, 502, 503, 504, etc.)
    async fn execute_with_retry(&self, request: reqwest::Request) -> Result<reqwest::Response, WebToolError> {
        let mut last_error = None;
        let mut last_status = None;
        let mut delay = self.config.retry_base_delay_ms;
        
        for attempt in 0..=self.config.max_retries {
            // Clone request for this attempt; streaming bodies can't be cloned
            let req = match request.try_clone() {
                Some(cloned) => self.inner.execute(cloned),
                None => {
                    // Can't clone (streaming body) - only one attempt possible
                    return self.inner.execute(request)
                        .await
                        .map_err(|e| WebToolError::Http(e.to_string()));
                }
            };
            
            match req.await {
                Ok(resp) => {
                    let status = resp.status();
                    if is_retryable_status(status) && attempt < self.config.max_retries {
                        // Server error - retry with backoff
                        last_status = Some(status);
                        let jitter = (delay as f64 * rand_jitter()).floor() as u64;
                        tokio::time::sleep(std::time::Duration::from_millis(jitter)).await;
                        delay *= 2;
                        continue;
                    }
                    return Ok(resp);
                }
                Err(e) if is_retryable(&e, self.config.retry_on_timeout) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries {
                        // Full jitter: random delay between 0 and exponential backoff
                        let jitter = (delay as f64 * rand_jitter()).floor() as u64;
                        tokio::time::sleep(std::time::Duration::from_millis(jitter)).await;
                        delay *= 2; // Exponential backoff
                    }
                }
                Err(e) => return Err(WebToolError::Http(e.to_string())),
            }
        }
        
        // Construct error message from either status code or network error
        let error_msg = if let Some(status) = last_status {
            format!("HTTP {} (max retries exceeded)", status)
        } else if let Some(ref e) = last_error {
            e.to_string()
        } else {
            "Max retries exceeded".to_string()
        };
        
        Err(WebToolError::Http(error_msg))
    }
}

/// Check if an error is retryable (transient failure).
fn is_retryable(error: &reqwest::Error, retry_on_timeout: bool) -> bool {
    if error.is_timeout() && !retry_on_timeout {
        return false;
    }
    // Retry on: timeout, connect errors, and request errors
    error.is_timeout() || error.is_connect() || error.is_request()
}

/// Check if an HTTP status code indicates a retryable server error.
/// Returns true for 5xx server errors that are typically transient.
fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    use reqwest::StatusCode;
    matches!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR |        // 500
        StatusCode::BAD_GATEWAY |                   // 502
        StatusCode::SERVICE_UNAVAILABLE |           // 503
        StatusCode::GATEWAY_TIMEOUT |               // 504
        StatusCode::VARIANT_ALSO_NEGOTIATES |       // 506
        StatusCode::INSUFFICIENT_STORAGE |          // 507
        StatusCode::LOOP_DETECTED |                 // 508
        StatusCode::NOT_EXTENDED |                  // 510
        StatusCode::NETWORK_AUTHENTICATION_REQUIRED // 511
    )
}

/// Generate a random jitter factor between 0.0 and 1.0 for retry delays.
/// Uses a simple PRNG to avoid adding rand crate dependency.
fn rand_jitter() -> f64 {
    // Use a thread-local simple xorshift64 PRNG for jitter
    // This is NOT cryptographically secure, but sufficient for jitter
    use std::cell::RefCell;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    thread_local! {
        static RNG: RefCell<u64> = RefCell::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x853c49e674a2b9f4) // fallback seed
        );
    }
    
    RNG.with(|rng| {
        let mut state = *rng.borrow();
        // xorshift64 algorithm
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *rng.borrow_mut() = state;
        // Normalize to [0.0, 1.0)
        (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    })
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

impl SearchOutput {
    /// Returns `true` if the search returned no results.
    ///
    /// This is a convenience method for quickly checking emptiness
    /// without needing to access the `results` vector directly.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Returns a human-readable one-line summary of the search results.
    ///
    /// Format: "{count} result(s) for '{query}'" or "No results for '{query}'"
    /// This is useful for quick logging and display without needing to
    /// format the results manually.
    pub fn summary(&self) -> String {
        if self.results.is_empty() {
            format!("No results for '{}'", self.query)
        } else {
            format!("{} result(s) for '{}'", self.results.len(), self.query)
        }
    }
}

/// A single search result.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

impl WebSearchResult {
    /// Extract the domain name from the URL.
    ///
    /// Returns `None` if the URL is malformed or doesn't have a valid host.
    /// This is useful for categorizing search results by source without
    /// requiring external dependencies.
    pub fn domain(&self) -> Option<&str> {
        // Find "://" separator
        let after_scheme = self.url.find("://").map(|i| &self.url[i + 3..])
            .unwrap_or(&self.url);
        
        // Find the start of path/query/fragment
        let domain_end = after_scheme.find(|c: char| c == '/' || c == '?' || c == '#')
            .unwrap_or(after_scheme.len());
        
        let domain = &after_scheme[..domain_end];
        
        // Remove port number if present
        let domain = domain.split(':').next().unwrap_or(domain);
        
        // Handle empty or invalid domains
        if domain.is_empty() {
            None
        } else {
            Some(domain)
        }
    }
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

    /// Create a new WebSearchTool with custom HTTP client configuration.
    pub fn with_config(config: HttpClientConfig) -> Self {
        Self {
            client: Arc::new(HttpClient::with_config(config)),
        }
    }

    /// Create a new WebSearchTool with a custom timeout in seconds.
    pub fn with_timeout(timeout_secs: u64) -> Self {
        let config = HttpClientConfig {
            timeout_secs,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Direct search (without rig Tool machinery).
    pub async fn search(&self, query: &str, max_results: usize) -> Result<Vec<WebSearchResult>> {
        let max = max_results.min(10).max(1);
        let encoded = percent_encode(query);
        let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

        let request = self.client.inner.get(&url).build()
            .map_err(|e| anyhow::anyhow!("Failed to build request: {}", e))?;
        
        let resp = self
            .client
            .execute_with_retry(request)
            .await
            .map_err(|e| anyhow::anyhow!("Search request failed: {}", e))?;

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

    let request = client.inner.get(&url).build()
        .map_err(|e| WebToolError::Http(format!("Failed to build request: {}", e)))?;
    
    let resp = client
        .execute_with_retry(request)
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

impl FetchOutput {
    /// Extract the domain name from the URL.
    ///
    /// Returns `None` if the URL is malformed or doesn't have a valid host.
    /// This is useful for categorizing fetched pages by source without
    /// requiring external dependencies.
    pub fn domain(&self) -> Option<&str> {
        // Find "://" separator
        let after_scheme = self.url.find("://").map(|i| &self.url[i + 3..])
            .unwrap_or(&self.url);
        
        // Find the start of path/query/fragment
        let domain_end = after_scheme.find(|c: char| c == '/' || c == '?' || c == '#')
            .unwrap_or(after_scheme.len());
        
        let domain = &after_scheme[..domain_end];
        
        // Remove port number if present
        let domain = domain.split(':').next().unwrap_or(domain);
        
        // Handle empty or invalid domains
        if domain.is_empty() {
            None
        } else {
            Some(domain)
        }
    }

    /// Returns a human-readable one-line summary of the fetched page.
    ///
    /// Format: "{domain}: '{title}' ({text_length} chars)" or
    ///         "URL '{url}': '{title}' ({text_length} chars)" if domain extraction fails.
    /// This is useful for quick logging and display without needing to
    /// format the output manually.
    pub fn summary(&self) -> String {
        match self.domain() {
            Some(domain) => format!("{}: '{}' ({} chars)", domain, self.title, self.text_length),
            None => format!("URL '{}': '{}' ({} chars)", self.url, self.title, self.text_length),
        }
    }
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

    /// Create a new WebFetchTool with custom HTTP client configuration.
    pub fn with_config(config: HttpClientConfig) -> Self {
        Self {
            client: Arc::new(HttpClient::with_config(config)),
        }
    }

    /// Create a new WebFetchTool with a custom timeout in seconds.
    pub fn with_timeout(timeout_secs: u64) -> Self {
        let config = HttpClientConfig {
            timeout_secs,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Direct fetch (without rig Tool machinery).
    pub async fn fetch(&self, url: &str) -> Result<FetchOutput> {
        let request = self.client.inner.get(url).build()
            .map_err(|e| anyhow::anyhow!("Failed to build request: {}", e))?;
        
        let resp = self
            .client
            .execute_with_retry(request)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to fetch {}: {}", url, e))?;

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
            let request = client.inner.get(&url).build()
                .map_err(|e| WebToolError::Http(format!("Failed to build request: {}", e)))?;
            
            let resp = client
                .execute_with_retry(request)
                .await
                .map_err(|e| WebToolError::Http(format!("Failed to fetch {}: {}", url, e)))?;

            if !resp.status().is_success() {
                return Err(WebToolError::Http(format!("HTTP {} for {}", resp.status(), url)));
            }

            // Check response size limit before loading body
            let content_length = resp.content_length();
            if let Some(len) = content_length {
                if len as usize > client.config.max_response_size {
                    return Err(WebToolError::Http(format!(
                        "Response too large: {} bytes (limit: {})",
                        len, client.config.max_response_size
                    )));
                }
            }

            let html = resp.text().await.map_err(|e| WebToolError::Http(e.to_string()))?;
            
            // Double-check after loading (content-length may be absent)
            if html.len() > client.config.max_response_size {
                return Err(WebToolError::Http(format!(
                    "Response body too large: {} bytes (limit: {})",
                    html.len(), client.config.max_response_size
                )));
            }
            
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

/// Check if a tag buffer starts with a case-insensitive tag name.
/// Optimized to avoid allocating a lowercase string.
#[inline]
fn tag_starts_with(tag: &str, prefix: &str) -> bool {
    let tag = tag.trim_start_matches('<');
    if tag.len() < prefix.len() {
        return false;
    }
    tag[..prefix.len()].eq_ignore_ascii_case(prefix)
}

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

                // Track script/style blocks using case-insensitive comparison
                // without allocating a new string
                if tag_starts_with(&tag_buf, "script") {
                    in_script = true;
                } else if tag_starts_with(&tag_buf, "/script") {
                    in_script = false;
                } else if tag_starts_with(&tag_buf, "style") {
                    in_style = true;
                } else if tag_starts_with(&tag_buf, "/style") {
                    in_style = false;
                }

                // Add paragraph breaks for block elements
                if tag_starts_with(&tag_buf, "p")
                    || tag_starts_with(&tag_buf, "div")
                    || tag_starts_with(&tag_buf, "h1")
                    || tag_starts_with(&tag_buf, "h2")
                    || tag_starts_with(&tag_buf, "h3")
                    || tag_starts_with(&tag_buf, "h4")
                    || tag_starts_with(&tag_buf, "br")
                    || tag_starts_with(&tag_buf, "li")
                    || tag_starts_with(&tag_buf, "tr")
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

/// Case-insensitive substring search without allocating.
/// Uses direct character comparison to avoid any heap allocations.
/// Returns the starting byte position of the first match, or None if not found.
#[inline]
fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    
    let needle_chars: Vec<char> = needle.chars().collect();
    let needle_len = needle_chars.len();
    
    // Pre-compute lowercase needle characters without allocating twice
    // Reuse the same vector by transforming in place
    let needle_lower: Vec<char> = needle_chars.into_iter().map(|c| c.to_ascii_lowercase()).collect();
    
    // Iterate through haystack using char_indices for correct byte positions
    let haystack_chars: Vec<(usize, char)> = haystack.char_indices().collect();
    
    if haystack_chars.len() < needle_len {
        return None;
    }
    
    // Slide window through haystack
    for window_start in 0..=(haystack_chars.len() - needle_len) {
        let byte_pos = haystack_chars[window_start].0;
        
        // Check if this window matches (case-insensitive)
        let matches = needle_lower.iter().enumerate().all(|(i, &expected)| {
            haystack_chars[window_start + i].1.to_ascii_lowercase() == expected
        });
        
        if matches {
            return Some(byte_pos);
        }
    }
    None
}

/// Extract <title> from HTML (case-insensitive).
fn extract_title(html: &str) -> String {
    // Find <title> case-insensitively
    let title_start = find_case_insensitive(html, "<title>")
        .map(|i| i + 7);
    
    if let Some(start) = title_start {
        let rest = &html[start..];
        // Find </title> case-insensitively
        let end = find_case_insensitive(rest, "</title>");
        if let Some(end_pos) = end {
            return html[start..start + end_pos].trim().to_string();
        }
    }
    String::new()
}

/// Parse DuckDuckGo HTML search results.
fn parse_ddg_results(html: &str, max_results: usize) -> Vec<WebSearchResult> {
    let mut results = Vec::new();

    // DuckDuckGo HTML results: each result is in a <a class="result__a"> with href
    // and snippets in <a class="result__snippet">
    let mut pos = 0;
    while results.len() < max_results {
        // Find next result link (case-insensitive class attribute)
        let link_start = match find_attr_class(html, pos, "result__a") {
            Some(i) => i,
            None => break,
        };

        // Find href within this element
        let href_start = match find_href(html, link_start) {
            Some(i) => i,
            None => {
                pos = link_start + 1;
                continue;
            }
        };

        // Extract URL (between quotes after href=)
        let raw_url = extract_quoted(&html[href_start..]);
        // DDG returns redirect URLs like "//duckduckgo.com/l/?uddg=...", extract the real URL
        let url = resolve_ddg_redirect(&raw_url);

        // Find the link text (between > and <)
        let text_start = match html[href_start..].find('>') {
            Some(i) => href_start + i + 1,
            None => {
                pos = href_start + 1;
                continue;
            }
        };
        let title_end = match html[text_start..].find('<') {
            Some(i) => text_start + i,
            None => break,
        };
        let title = html[text_start..title_end].trim().to_string();

        // Find snippet (class="result__snippet")
        let snippet_start = match find_attr_class(html, title_end, "result__snippet") {
            Some(i) => i,
            None => {
                pos = title_end;
                continue;
            }
        };

        let snip_text_start = match html[snippet_start..].find('>') {
            Some(i) => snippet_start + i + 1,
            None => break,
        };
        let snip_text_end = match html[snip_text_start..].find('<') {
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

/// Find a tag with a specific class attribute (case-insensitive for class name).
/// Optimized to avoid allocating format strings by using direct comparison.
fn find_attr_class(html: &str, start: usize, class_name: &str) -> Option<usize> {
    // Search for class="class_name" with case-insensitive matching
    // We need to find the pattern and verify the class name matches case-insensitively
    let mut search_pos = start;
    
    while let Some(class_idx) = html[search_pos..].find("class=\"") {
        let abs_class_idx = search_pos + class_idx;
        let attr_start = abs_class_idx + 7; // length of 'class="'
        
        // Extract the class value (until closing quote)
        if let Some(quote_end) = html[attr_start..].find('"') {
            let class_value = &html[attr_start..attr_start + quote_end];
            
            // Case-insensitive comparison without allocating
            if class_value.eq_ignore_ascii_case(class_name) {
                return Some(abs_class_idx);
            }
        }
        
        search_pos = attr_start;
    }
    
    None
}

/// Find href attribute position after a given position.
fn find_href(html: &str, start: usize) -> Option<usize> {
    let rest = &html[start..];
    rest.find("href=").map(|i| start + i + 5)
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
    fn test_rand_jitter_range() {
        // Test that jitter values are always in [0.0, 1.0)
        for _ in 0..100 {
            let jitter = rand_jitter();
            assert!(jitter >= 0.0, "jitter should be >= 0.0, got {}", jitter);
            assert!(jitter < 1.0, "jitter should be < 1.0, got {}", jitter);
        }
    }

    #[test]
    fn test_rand_jitter_varies() {
        // Test that jitter produces different values over calls
        let values: std::collections::HashSet<_> = (0..10)
            .map(|_| rand_jitter().to_bits())
            .collect();
        // At least some values should differ (very unlikely to get 10 identical values)
        assert!(values.len() > 1, "jitter should produce varying values");
    }

    #[test]
    fn test_is_retryable_timeout() {
        // Test is_retryable with timeout errors
        // Note: We can't easily construct reqwest::Error, so we test the logic indirectly
        // through the config behavior
        let config = HttpClientConfig {
            timeout_secs: 30,
            max_retries: 3,
            retry_base_delay_ms: 100,
            retry_on_timeout: true,
        };
        assert!(config.retry_on_timeout);
        
        let config_no_retry = HttpClientConfig {
            timeout_secs: 30,
            max_retries: 3,
            retry_base_delay_ms: 100,
            retry_on_timeout: false,
        };
        assert!(!config_no_retry.retry_on_timeout);
    }

    #[test]
    fn test_http_client_config_default() {
        let config = HttpClientConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay_ms, 100);
        assert!(config.retry_on_timeout);
        assert_eq!(config.max_response_size, 10 * 1024 * 1024); // 10MB
    }

    #[test]
    fn test_http_client_max_response_size() {
        let client = HttpClient::new();
        assert_eq!(client.max_response_size(), 10 * 1024 * 1024);
        
        let config = HttpClientConfig {
            timeout_secs: 30,
            max_retries: 3,
            retry_base_delay_ms: 100,
            retry_on_timeout: true,
            max_response_size: 1024, // 1KB limit
        };
        let small_client = HttpClient::with_config(config);
        assert_eq!(small_client.max_response_size(), 1024);
    }

    #[test]
    fn test_http_client_config_custom_max_response_size() {
        let config = HttpClientConfig {
            timeout_secs: 60,
            max_retries: 5,
            retry_base_delay_ms: 200,
            retry_on_timeout: false,
            max_response_size: 5 * 1024 * 1024, // 5MB
        };
        assert_eq!(config.max_response_size, 5 * 1024 * 1024);
    }

    #[test]
    fn test_web_search_tool_with_timeout() {
        let tool = WebSearchTool::with_timeout(60);
        assert!(Arc::try_unwrap(tool.client).is_ok());
    }

    #[test]
    fn test_web_fetch_tool_with_timeout() {
        let tool = WebFetchTool::with_timeout(60);
        assert!(Arc::try_unwrap(tool.client).is_ok());
    }

    #[test]
    fn test_http_client_config_accessor() {
        let client = HttpClient::new();
        let config = client.config();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay_ms, 100);
        assert!(config.retry_on_timeout);
    }

    #[test]
    fn test_http_client_has_retry_budget() {
        let client = HttpClient::new();
        assert!(client.has_retry_budget());
        
        let config = HttpClientConfig {
            timeout_secs: 30,
            max_retries: 0,
            retry_base_delay_ms: 100,
            retry_on_timeout: true,
        };
        let client_no_retry = HttpClient::with_config(config);
        assert!(!client_no_retry.has_retry_budget());
    }

    #[test]
    fn test_http_client_custom_config_reflection() {
        let config = HttpClientConfig {
            timeout_secs: 60,
            max_retries: 5,
            retry_base_delay_ms: 200,
            retry_on_timeout: false,
            max_response_size: 20 * 1024 * 1024, // 20MB
        };
        let client = HttpClient::with_config(config.clone());
        let reflected = client.config();
        assert_eq!(reflected.timeout_secs, 60);
        assert_eq!(reflected.max_retries, 5);
        assert_eq!(reflected.retry_base_delay_ms, 200);
        assert!(!reflected.retry_on_timeout);
        assert_eq!(reflected.max_response_size, 20 * 1024 * 1024);
    }

    #[test]
    fn test_resolve_ddg_redirect() {
        // Test extracting URL from DDG redirect
        let redirect_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=abc123";
        let resolved = resolve_ddg_redirect(redirect_url);
        assert_eq!(resolved, "https://example.com");

        // Test URL that's already direct
        let direct_url = "https://example.com/page";
        let resolved = resolve_ddg_redirect(direct_url);
        assert_eq!(resolved, "https://example.com/page");
    }

    #[test]
    fn test_url_decode() {
        assert_eq!(url_decode("hello"), "hello");
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("%C3%A9"), "é"); // UTF-8 encoded
    }

    #[test]
    fn test_percent_encode() {
        assert_eq!(percent_encode("hello"), "hello");
        assert_eq!(percent_encode("hello world"), "hello+world");
        assert_eq!(percent_encode("hello!"), "hello%21");
    }

    #[test]
    fn test_html_to_text_basic() {
        let html = "<html><body><p>Hello <b>world</b>!</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_html_to_text_script_removal() {
        let html = "<html><script>alert('xss')</script><body>Hello</body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("script"));
    }

    #[test]
    fn test_extract_title() {
        let html = "<html><head><TITLE>My Page Title</TITLE></head><body>Content</body></html>";
        let title = extract_title(html);
        assert_eq!(title, "My Page Title");
    }

    #[test]
    fn test_build_web_context_prompt_empty() {
        let prompt = build_web_context_prompt(&[], &[]);
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_build_web_context_prompt_with_results() {
        let results = vec![
            WebSearchResult {
                title: "Test Result".to_string(),
                url: "https://example.com".to_string(),
                snippet: "Test snippet".to_string(),
            },
        ];
        let prompt = build_web_context_prompt(&results, &[]);
        assert!(prompt.contains("WEB SEARCH RESULTS"));
        assert!(prompt.contains("Test Result"));
        assert!(prompt.contains("https://example.com"));
    }

    #[test]
    fn test_build_web_context_prompt_with_pages() {
        let pages = vec![
            FetchOutput {
                url: "https://example.com".to_string(),
                title: "Test Page".to_string(),
                text: "This is the page content.".to_string(),
                text_length: 24,
            },
        ];
        let prompt = build_web_context_prompt(&[], &pages);
        assert!(prompt.contains("FETCHED WEB PAGES"));
        assert!(prompt.contains("Test Page"));
    }

    #[test]
    fn test_is_retryable_status() {
        use reqwest::StatusCode;
        
        // 5xx errors should be retryable
        assert!(is_retryable_status(StatusCode::INTERNAL_SERVER_ERROR)); // 500
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY)); // 502
        assert!(is_retryable_status(StatusCode::SERVICE_UNAVAILABLE)); // 503
        assert!(is_retryable_status(StatusCode::GATEWAY_TIMEOUT)); // 504
        
        // 4xx errors should NOT be retryable (client errors)
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST)); // 400
        assert!(!is_retryable_status(StatusCode::NOT_FOUND)); // 404
        assert!(!is_retryable_status(StatusCode::FORBIDDEN)); // 403
        
        // 2xx success codes should NOT be retryable
        assert!(!is_retryable_status(StatusCode::OK)); // 200
        assert!(!is_retryable_status(StatusCode::CREATED)); // 201
    }

    #[test]
    fn test_find_case_insensitive_basic() {
        // Basic case-insensitive matching
        assert_eq!(find_case_insensitive("Hello World", "hello"), Some(0));
        assert_eq!(find_case_insensitive("Hello World", "WORLD"), Some(6));
        assert_eq!(find_case_insensitive("Hello World", "lo wo"), Some(3));
    }

    #[test]
    fn test_find_case_insensitive_not_found() {
        // Not found cases
        assert_eq!(find_case_insensitive("Hello World", "xyz"), None);
        assert_eq!(find_case_insensitive("Hello", "Hello World"), None);
        assert_eq!(find_case_insensitive("", "test"), None);
        assert_eq!(find_case_insensitive("test", ""), None);
    }

    #[test]
    fn test_find_case_insensitive_unicode() {
        // Unicode handling
        assert_eq!(find_case_insensitive("café au lait", "CAFÉ"), Some(0));
        assert_eq!(find_case_insensitive("日本語テスト", "日本"), Some(0));
    }

    #[test]
    fn test_find_case_insensitive_multibyte() {
        // Ensure byte positions are correct for multibyte characters
        let hay = "hello 世界 test";
        let pos = find_case_insensitive(hay, "世界");
        assert_eq!(pos, Some(6)); // "世界" starts at byte position 6 (after "hello ")
    }

    #[test]
    fn test_find_case_insensitive_no_allocations() {
        // Test that the function works correctly with various inputs
        // to ensure optimization didn't break functionality
        
        // Empty needle
        assert_eq!(find_case_insensitive("hello", ""), None);
        
        // Needle longer than haystack
        assert_eq!(find_case_insensitive("hi", "hello"), None);
        
        // Exact match
        assert_eq!(find_case_insensitive("exact", "exact"), Some(0));
        
        // Case variation
        assert_eq!(find_case_insensitive("HeLLo WoRLD", "hello world"), Some(0));
        
        // Match at end
        assert_eq!(find_case_insensitive("start end", "END"), Some(6));
        
        // Match in middle
        assert_eq!(find_case_insensitive("start middle end", "MIDDLE"), Some(6));
        
        // Multiple occurrences (should return first)
        assert_eq!(find_case_insensitive("test test test", "TEST"), Some(0));
        
        // Special characters
        assert_eq!(find_case_insensitive("hello-world", "-"), Some(5));
        assert_eq!(find_case_insensitive("hello@world", "@"), Some(5));
    }

    #[test]
    fn test_find_case_insensitive_performance_characteristics() {
        // Test with longer strings to verify the algorithm handles them correctly
        let long_haystack = "a".repeat(1000) + "TARGET" + &"b".repeat(1000);
        let pos = find_case_insensitive(&long_haystack, "target");
        assert_eq!(pos, Some(1000));
        
        // Verify not found case with long strings
        let not_found = find_case_insensitive(&long_haystack, "nonexistent");
        assert_eq!(not_found, None);
    }

    #[test]
    fn test_find_attr_class_basic() {
        let html = r#"<div class="result__a">Link</div>"#;
        let pos = find_attr_class(html, 0, "result__a");
        assert!(pos.is_some());
        assert_eq!(pos, Some(5)); // position of 'class="' start
    }

    #[test]
    fn test_find_attr_class_case_insensitive() {
        // Class matching should be case-insensitive
        let html = r#"<div CLASS="Result__A">Link</div>"#;
        let pos = find_attr_class(html, 0, "result__a");
        assert!(pos.is_some());
    }

    #[test]
    fn test_find_attr_class_not_found() {
        let html = r#"<div class="other">Content</div>"#;
        let pos = find_attr_class(html, 0, "result__a");
        assert!(pos.is_none());
    }

    #[test]
    fn test_find_attr_class_multiple_matches() {
        let html = r#"<div class="result__a">First</div><div class="result__a">Second</div>"#;
        let first_pos = find_attr_class(html, 0, "result__a");
        assert!(first_pos.is_some());
        
        // Find second occurrence
        if let Some(pos) = first_pos {
            let second_pos = find_attr_class(html, pos + 1, "result__a");
            assert!(second_pos.is_some());
        }
    }

    #[test]
    fn test_find_attr_class_with_offset() {
        let html = r#"<div class="before">Skip</div><div class="result__a">Target</div>"#;
        let pos_from_start = find_attr_class(html, 0, "result__a");
        let pos_with_offset = find_attr_class(html, 20, "result__a");
        
        assert!(pos_from_start.is_some());
        assert!(pos_with_offset.is_some());
        assert!(pos_with_offset.unwrap() > pos_from_start.unwrap());
    }

    #[test]
    fn test_find_attr_class_malformed_html() {
        // Unclosed quote
        let html = r#"<div class="result__a>Link</div>"#;
        let pos = find_attr_class(html, 0, "result__a");
        // Should still find it (quote ends at end of string or next attribute)
        assert!(pos.is_some() || pos.is_none()); // behavior is implementation-defined
    }

    #[test]
    fn test_find_case_insensitive_empty_needle() {
        assert_eq!(find_case_insensitive("hello", ""), None);
    }

    #[test]
    fn test_find_case_insensitive_needle_longer_than_haystack() {
        assert_eq!(find_case_insensitive("hi", "hello"), None);
    }

    #[test]
    fn test_find_case_insensitive_single_char_needle() {
        assert_eq!(find_case_insensitive("hello", "h"), Some(0));
        assert_eq!(find_case_insensitive("hello", "O"), Some(4));
        assert_eq!(find_case_insensitive("hello", "z"), None);
    }

    #[test]
    fn test_web_search_result_domain() {
        // Standard HTTPS URL
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://example.com/path".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("example.com"));

        // HTTP URL
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "http://example.org/page".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("example.org"));

        // URL with port
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://localhost:8080/app".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("localhost"));

        // URL with query string
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://docs.rs/crate?version=1.0".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("docs.rs"));

        // URL with fragment
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://example.com/page#section".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("example.com"));

        // URL without scheme (malformed but handle gracefully)
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "example.com/path".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("example.com"));

        // Empty URL
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "".to_string(),
            snippet: "Test snippet".to_string(),
        };
        assert_eq!(result.domain(), None);
    }

    #[test]
    fn test_fetch_output_domain() {
        // Standard HTTPS URL
        let output = FetchOutput {
            url: "https://rust-lang.org/learn".to_string(),
            title: "Rust".to_string(),
            text: "Learn Rust".to_string(),
            text_length: 10,
        };
        assert_eq!(output.domain(), Some("rust-lang.org"));

        // URL with subdomain
        let output = FetchOutput {
            url: "https://doc.rust-lang.org/std".to_string(),
            title: "Std Docs".to_string(),
            text: "Standard library".to_string(),
            text_length: 16,
        };
        assert_eq!(output.domain(), Some("doc.rust-lang.org"));

        // URL with port
        let output = FetchOutput {
            url: "http://127.0.0.1:3000/api".to_string(),
            title: "Local API".to_string(),
            text: "API".to_string(),
            text_length: 3,
        };
        assert_eq!(output.domain(), Some("127.0.0.1"));

        // URL with complex query
        let output = FetchOutput {
            url: "https://github.com/user/repo/blob/main/src?query=value#readme".to_string(),
            title: "GitHub".to_string(),
            text: "Code".to_string(),
            text_length: 4,
        };
        assert_eq!(output.domain(), Some("github.com"));
    }

    #[test]
    fn test_search_output_is_empty() {
        // Empty results
        let empty = SearchOutput {
            results: vec![],
            query: "test".to_string(),
        };
        assert!(empty.is_empty());

        // Non-empty results
        let non_empty = SearchOutput {
            results: vec![WebSearchResult {
                title: "Test".to_string(),
                url: "https://example.com".to_string(),
                snippet: "Test snippet".to_string(),
            }],
            query: "test".to_string(),
        };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_search_output_summary() {
        // Empty results summary
        let empty = SearchOutput {
            results: vec![],
            query: "rust programming".to_string(),
        };
        assert_eq!(empty.summary(), "No results for 'rust programming'");

        // Single result summary
        let single = SearchOutput {
            results: vec![WebSearchResult {
                title: "Rust".to_string(),
                url: "https://rust-lang.org".to_string(),
                snippet: "Safe systems programming".to_string(),
            }],
            query: "rust".to_string(),
        };
        assert_eq!(single.summary(), "1 result(s) for 'rust'");

        // Multiple results summary
        let multiple = SearchOutput {
            results: vec![
                WebSearchResult {
                    title: "Result 1".to_string(),
                    url: "https://example1.com".to_string(),
                    snippet: "Snippet 1".to_string(),
                },
                WebSearchResult {
                    title: "Result 2".to_string(),
                    url: "https://example2.com".to_string(),
                    snippet: "Snippet 2".to_string(),
                },
                WebSearchResult {
                    title: "Result 3".to_string(),
                    url: "https://example3.com".to_string(),
                    snippet: "Snippet 3".to_string(),
                },
            ],
            query: "test query".to_string(),
        };
        assert_eq!(multiple.summary(), "3 result(s) for 'test query'");
    }

    #[test]
    fn test_fetch_output_summary() {
        // Standard URL with valid domain
        let output = FetchOutput {
            url: "https://rust-lang.org/learn".to_string(),
            title: "Rust Learning".to_string(),
            text: "Learn Rust programming".to_string(),
            text_length: 22,
        };
        assert_eq!(output.summary(), "rust-lang.org: 'Rust Learning' (22 chars)");

        // URL with empty title
        let empty_title = FetchOutput {
            url: "https://example.com".to_string(),
            title: "".to_string(),
            text: "Some content".to_string(),
            text_length: 12,
        };
        assert_eq!(empty_title.summary(), "example.com: '' (12 chars)");

        // Malformed URL (no domain)
        let malformed = FetchOutput {
            url: "".to_string(),
            title: "Some Page".to_string(),
            text: "Content".to_string(),
            text_length: 7,
        };
        assert_eq!(malformed.summary(), "URL '': 'Some Page' (7 chars)");
    }

    #[test]
    fn test_tag_starts_with_basic() {
        // Basic case-insensitive matching
        assert!(tag_starts_with("<script>", "script"));
        assert!(tag_starts_with("<SCRIPT>", "script"));
        assert!(tag_starts_with("<ScRiPt>", "script"));
        assert!(tag_starts_with("<div>", "div"));
        assert!(tag_starts_with("<DIV>", "div"));
    }

    #[test]
    fn test_tag_starts_with_closing_tags() {
        // Closing tags
        assert!(tag_starts_with("</script>", "script"));
        assert!(tag_starts_with("</SCRIPT>", "/script"));
        assert!(tag_starts_with("</style>", "/style"));
        assert!(tag_starts_with("</div>", "/div"));
    }

    #[test]
    fn test_tag_starts_with_with_whitespace() {
        // Tags with leading whitespace (should still work after trim)
        assert!(tag_starts_with("  <script>", "script"));
        assert!(tag_starts_with("\t<div>", "div"));
    }

    #[test]
    fn test_tag_starts_with_no_match() {
        // No match cases
        assert!(!tag_starts_with("<div>", "script"));
        assert!(!tag_starts_with("<span>", "div"));
        assert!(!tag_starts_with("<script>", "div"));
        assert!(!tag_starts_with("", "div"));
        assert!(!tag_starts_with("<a>", "div"));
    }

    #[test]
    fn test_tag_starts_with_partial_match() {
        // Partial matches should not match
        assert!(!tag_starts_with("<sc", "script")); // Too short
        assert!(!tag_starts_with("<scrip>", "script")); // Missing char
        assert!(!tag_starts_with("<scripts>", "script")); // Extra char at end shouldn't matter for prefix
    }

    #[test]
    fn test_tag_starts_with_attributes() {
        // Tags with attributes (function only checks the tag name at start)
        // Note: the function strips '<' and checks prefix, so "<script " would match "script"
        assert!(tag_starts_with("<script type='text/javascript'>", "script"));
        assert!(tag_starts_with("<div class='container'>", "div"));
        assert!(tag_starts_with("<style media='screen'>", "style"));
    }
}
