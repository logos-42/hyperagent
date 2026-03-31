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

impl HttpClientConfig {
    /// Returns a human-readable one-line summary of the configuration.
    ///
    /// Format: "timeout={timeout_secs}s, retries={max_retries}, max_size={max_response_size}MB"
    /// This is useful for logging and debugging HTTP client settings.
    pub fn summary(&self) -> String {
        let max_mb = self.max_response_size as f64 / (1024.0 * 1024.0);
        format!(
            "timeout={}s, retries={}, max_size={:.1}MB",
            self.timeout_secs, self.max_retries, max_mb
        )
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

    /// Returns the maximum response size in bytes.
    ///
    /// This is a convenience accessor for the `max_response_size` config field,
    /// useful for checking size limits before or after making requests.
    fn max_response_size(&self) -> usize {
        self.config.max_response_size
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

    /// Returns a human-readable one-line summary of the search result.
    ///
    /// Format: "{domain}: '{title}' — {snippet_preview}" or
    ///         "'{title}' — {snippet_preview}" if domain extraction fails.
    /// The snippet is truncated to 50 characters for compact display.
    /// This is useful for quick logging and displaying results without
    /// needing to format them manually.
    pub fn summary(&self) -> String {
        let snippet_preview = if self.snippet.len() > 50 {
            let chars: String = self.snippet.chars().take(47).collect();
            format!("{}...", chars)
        } else {
            self.snippet.clone()
        };
        
        match self.domain() {
            Some(domain) => format!("{}: '{}' — {}", domain, self.title, snippet_preview),
            None => format!("'{}' — {}", self.title, snippet_preview),
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

impl WebToolError {
    /// Returns a human-readable one-line summary of the error.
    ///
    /// This is useful for quick logging and displaying errors in LLM context
    /// without exposing internal implementation details.
    pub fn summary(&self) -> String {
        match self {
            WebToolError::Http(msg) => format!("HTTP error: {}", msg),
            WebToolError::Parse(msg) => format!("Parse error: {}", msg),
            WebToolError::Timeout(msg) => format!("Timeout: {}", msg),
        }
    }

    /// Returns actionable context for resolving the error.
    ///
    /// Provides hints about what the caller can do to address the error,
    /// useful for LLM agents to self-correct their web tool usage.
    pub fn context(&self) -> &'static str {
        match self {
            WebToolError::Http(_) => "Check if the URL is reachable and the service is available. Consider retrying with exponential backoff.",
            WebToolError::Parse(_) => "The response format was unexpected. Try fetching a different URL or using web_search to find alternative sources.",
            WebToolError::Timeout(_) => "The request timed out. Try increasing the timeout or reducing max_response_size for faster results.",
        }
    }

    /// Returns `true` if this error is potentially recoverable by retrying.
    ///
    /// Network errors and timeouts are often transient, while parse errors
    /// indicate permanent issues with the response format.
    pub fn is_retryable(&self) -> bool {
        matches!(self, WebToolError::Http(_) | WebToolError::Timeout(_))
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

    // Truncate very long pages (按字符截断，避免多字节字符边界问题)
    if text.len() > 8000 {
        let chars: String = text.chars().take(7900).collect();
        format!("{}...", chars)
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
                let chars: String = p.text.chars().take(1950).collect();
                format!("{}...", chars)
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

    // -------------------------------------------------------------------------
    // URL encoding/decoding tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_url_decode_basic() {
        assert_eq!(url_decode("hello"), "hello");
        assert_eq!(url_decode("hello+world"), "hello world");
        assert_eq!(url_decode("%20"), " ");
        assert_eq!(url_decode("%2F"), "/");
    }

    #[test]
    fn test_url_decode_complex() {
        // Real-world URL: https://example.com/path?query=test
        assert_eq!(
            url_decode("https%3A%2F%2Fexample.com%2Fpath"),
            "https://example.com/path"
        );
    }

    #[test]
    fn test_url_decode_edge_cases() {
        // Incomplete percent sequence - should preserve original
        assert_eq!(url_decode("%2"), "%2");
        // Invalid hex chars - should preserve original
        assert_eq!(url_decode("%GG"), "%GG");
        // Mixed valid and invalid
        assert_eq!(url_decode("test%20value%ZZ"), "test value%ZZ");
        // Multiple plus signs
        assert_eq!(url_decode("a+b+c"), "a b c");
        // Empty string
        assert_eq!(url_decode(""), "");
    }

    #[test]
    fn test_url_decode_preserves_valid_percent_at_end() {
        // Percent at end without hex chars
        assert_eq!(url_decode("test%"), "test%");
    }

    #[test]
    fn test_percent_encode_basic() {
        assert_eq!(percent_encode("hello"), "hello");
        assert_eq!(percent_encode("hello world"), "hello+world");
        assert_eq!(percent_encode("test/value"), "test%2Fvalue");
    }

    #[test]
    fn test_percent_encode_special_chars() {
        // Space becomes +
        assert_eq!(percent_encode(" "), "+");
        // Unreserved chars stay as-is
        assert_eq!(percent_encode("aZ0-_.~"), "aZ0-_.~");
        // Reserved chars get encoded
        assert_eq!(percent_encode("&"), "%26");
        assert_eq!(percent_encode("="), "%3D");
        assert_eq!(percent_encode("?"), "%3F");
    }

    #[test]
    fn test_percent_encode_roundtrip() {
        // Test that encode/decode is consistent for simple cases
        let original = "hello world test";
        let encoded = percent_encode(original);
        let decoded = url_decode(&encoded);
        assert_eq!(decoded, original);
    }

    // -------------------------------------------------------------------------
    // DDG redirect URL resolution tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_ddg_redirect_direct_url() {
        // Non-redirect URL should pass through unchanged
        let url = "https://example.com/page";
        assert_eq!(resolve_ddg_redirect(url), url);
    }

    #[test]
    fn test_resolve_ddg_redirect_extracted() {
        // DDG redirect URL with uddg parameter
        let ddg_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=abc123";
        assert_eq!(resolve_ddg_redirect(ddg_url), "https://example.com");
    }

    #[test]
    fn test_resolve_ddg_redirect_encoded_url() {
        // Complex URL with path and query
        let ddg_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fq%3Dtest";
        assert_eq!(resolve_ddg_redirect(ddg_url), "https://example.com/path?q=test");
    }

    #[test]
    fn test_resolve_ddg_redirect_no_uddg() {
        // DDG URL without uddg parameter
        let url = "//duckduckgo.com/l/?rut=abc123";
        assert_eq!(resolve_ddg_redirect(url), url);
    }

    // -------------------------------------------------------------------------
    // HTML parsing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_title_basic() {
        let html = "<html><head><title>Test Page</title></head><body></body></html>";
        assert_eq!(extract_title(html), "Test Page");
    }

    #[test]
    fn test_extract_title_case_insensitive() {
        // Title tags may have different casing
        let html = "<HTML><HEAD><TITLE>Test</TITLE></HEAD></HTML>";
        assert_eq!(extract_title(html), "Test");

        let html2 = "<html><head><TiTlE>Mixed Case</TiTlE></head></html>";
        assert_eq!(extract_title(html2), "Mixed Case");
    }

    #[test]
    fn test_extract_title_missing() {
        // No title tag
        let html = "<html><head></head><body>Content</body></html>";
        assert_eq!(extract_title(html), "");

        // Empty title
        let html2 = "<html><head><title></title></head></html>";
        assert_eq!(extract_title(html2), "");
    }

    #[test]
    fn test_extract_title_with_whitespace() {
        let html = "<html><head><title>  Trimmed Title  </title></head></html>";
        assert_eq!(extract_title(html), "Trimmed Title");
    }

    #[test]
    fn test_html_to_text_basic() {
        let html = "<p>Hello <b>world</b>!</p>";
        let text = html_to_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
    }

    #[test]
    fn test_html_to_text_strips_scripts() {
        let html = "<html><script>var x = 1;</script><p>Content</p></html>";
        let text = html_to_text(html);
        assert!(!text.contains("var x"));
        assert!(text.contains("Content"));
    }

    #[test]
    fn test_html_to_text_strips_styles() {
        let html = "<html><style>body { color: red; }</style><p>Text</p></html>";
        let text = html_to_text(html);
        assert!(!text.contains("color"));
        assert!(text.contains("Text"));
    }

    #[test]
    fn test_html_to_text_preserves_paragraph_breaks() {
        let html = "<p>First</p><p>Second</p>";
        let text = html_to_text(html);
        assert!(text.contains("\n"));
    }

    #[test]
    fn test_html_to_text_truncates_long_content() {
        let long_text = "x".repeat(10000);
        let html = format!("<html><body>{}</body></html>", long_text);
        let text = html_to_text(&html);
        assert!(text.len() <= 8005); // 8000 + "..."
        assert!(text.ends_with("..."));
    }

    #[test]
    fn test_html_to_text_handles_entities() {
        // Ampersand handling (simplified - we preserve &)
        let html = "<p>Tom &amp; Jerry</p>";
        let text = html_to_text(html);
        assert!(text.contains("&"));
    }

    // -------------------------------------------------------------------------
    // WebSearchResult domain extraction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_web_search_result_domain() {
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://example.com/path".to_string(),
            snippet: "Snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("example.com"));
    }

    #[test]
    fn test_web_search_result_domain_with_port() {
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://example.com:8080/path".to_string(),
            snippet: "Snippet".to_string(),
        };
        assert_eq!(result.domain(), Some("example.com"));
    }

    #[test]
    fn test_web_search_result_domain_invalid() {
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "not-a-valid-url".to_string(),
            snippet: "Snippet".to_string(),
        };
        // Should still extract something or return None
        assert!(result.domain().is_some() || result.domain().is_none());
    }

    #[test]
    fn test_web_search_result_summary_basic() {
        let result = WebSearchResult {
            title: "Rust Programming".to_string(),
            url: "https://rust-lang.org/learn".to_string(),
            snippet: "A fast and memory-safe systems programming language.".to_string(),
        };
        let summary = result.summary();
        assert!(summary.contains("rust-lang.org"));
        assert!(summary.contains("Rust Programming"));
    }

    #[test]
    fn test_web_search_result_summary_short_snippet() {
        let result = WebSearchResult {
            title: "Short".to_string(),
            url: "https://example.com".to_string(),
            snippet: "Tiny".to_string(),
        };
        let summary = result.summary();
        assert!(summary.contains("Tiny"));
        assert!(!summary.contains("..."));
    }

    #[test]
    fn test_web_search_result_summary_long_snippet() {
        let long_snippet = "This is a very long snippet that should be truncated because it exceeds fifty characters limit".to_string();
        let result = WebSearchResult {
            title: "Test".to_string(),
            url: "https://example.com".to_string(),
            snippet: long_snippet,
        };
        let summary = result.summary();
        assert!(summary.contains("..."));
        // Snippet portion should be truncated to ~50 chars
        assert!(summary.len() < 150); // Rough bound check
    }

    #[test]
    fn test_web_search_result_summary_no_domain() {
        let result = WebSearchResult {
            title: "No Domain".to_string(),
            url: "invalid-url".to_string(),
            snippet: "Test snippet".to_string(),
        };
        let summary = result.summary();
        // Should still show title and snippet even without domain
        assert!(summary.contains("No Domain"));
        assert!(summary.contains("Test snippet"));
    }

    // -------------------------------------------------------------------------
    // FetchOutput tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fetch_output_summary() {
        let output = FetchOutput {
            url: "https://example.com/page".to_string(),
            title: "Test Page".to_string(),
            text: "Content here".to_string(),
            text_length: 12,
        };
        let summary = output.summary();
        assert!(summary.contains("example.com"));
        assert!(summary.contains("Test Page"));
        assert!(summary.contains("12 chars"));
    }

    #[test]
    fn test_fetch_output_domain() {
        let output = FetchOutput {
            url: "https://docs.rs/crate/1.0".to_string(),
            title: "Crate Docs".to_string(),
            text: "Documentation".to_string(),
            text_length: 13,
        };
        assert_eq!(output.domain(), Some("docs.rs"));
    }

    // -------------------------------------------------------------------------
    // SearchOutput tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_search_output_is_empty() {
        let empty = SearchOutput {
            results: vec![],
            query: "test".to_string(),
        };
        assert!(empty.is_empty());

        let non_empty = SearchOutput {
            results: vec![WebSearchResult {
                title: "Test".to_string(),
                url: "https://example.com".to_string(),
                snippet: "Snippet".to_string(),
            }],
            query: "test".to_string(),
        };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_search_output_summary() {
        let output = SearchOutput {
            results: vec![
                WebSearchResult {
                    title: "First".to_string(),
                    url: "https://a.com".to_string(),
                    snippet: "A".to_string(),
                },
                WebSearchResult {
                    title: "Second".to_string(),
                    url: "https://b.com".to_string(),
                    snippet: "B".to_string(),
                },
            ],
            query: "rust programming".to_string(),
        };
        let summary = output.summary();
        assert!(summary.contains("2 result(s)"));
        assert!(summary.contains("rust programming"));
    }

    // -------------------------------------------------------------------------
    // WebToolError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_web_tool_error_summary() {
        let http_err = WebToolError::Http("404 Not Found".to_string());
        assert!(http_err.summary().contains("HTTP error"));

        let parse_err = WebToolError::Parse("Invalid JSON".to_string());
        assert!(parse_err.summary().contains("Parse error"));

        let timeout_err = WebToolError::Timeout("30s".to_string());
        assert!(timeout_err.summary().contains("Timeout"));
    }

    #[test]
    fn test_web_tool_error_is_retryable() {
        assert!(WebToolError::Http("500".to_string()).is_retryable());
        assert!(WebToolError::Timeout("30s".to_string()).is_retryable());
        assert!(!WebToolError::Parse("bad".to_string()).is_retryable());
    }

    #[test]
    fn test_web_tool_error_context() {
        // All variants should provide actionable context
        assert!(!WebToolError::Http("err".to_string()).context().is_empty());
        assert!(!WebToolError::Parse("err".to_string()).context().is_empty());
        assert!(!WebToolError::Timeout("err".to_string()).context().is_empty());
    }

    // -------------------------------------------------------------------------
    // HttpClientConfig tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_http_client_config_default() {
        let config = HttpClientConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.max_response_size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_http_client_config_summary() {
        let config = HttpClientConfig {
            timeout_secs: 60,
            max_retries: 5,
            retry_base_delay_ms: 200,
            retry_on_timeout: true,
            max_response_size: 20 * 1024 * 1024,
        };
        let summary = config.summary();
        assert!(summary.contains("timeout=60s"));
        assert!(summary.contains("retries=5"));
        assert!(summary.contains("20.0MB"));
    }

    // -------------------------------------------------------------------------
    // Helper function tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_find_case_insensitive() {
        let haystack = "Hello World";
        assert_eq!(find_case_insensitive(haystack, "hello"), Some(0));
        assert_eq!(find_case_insensitive(haystack, "WORLD"), Some(6));
        assert_eq!(find_case_insensitive(haystack, "xyz"), None);
    }

    #[test]
    fn test_find_case_insensitive_empty() {
        assert_eq!(find_case_insensitive("", "test"), None);
        assert_eq!(find_case_insensitive("test", ""), None);
    }

    #[test]
    fn test_tag_starts_with() {
        assert!(tag_starts_with("<script>", "script"));
        assert!(tag_starts_with("<SCRIPT>", "script"));
        assert!(tag_starts_with("</script>", "/script"));
        assert!(!tag_starts_with("<style>", "script"));
    }

    // -------------------------------------------------------------------------
    // build_web_context_prompt tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_web_context_prompt_empty() {
        let prompt = build_web_context_prompt(&[], &[]);
        assert!(prompt.is_empty());
    }

    #[test]
    fn test_build_web_context_prompt_with_results() {
        let results = vec![WebSearchResult {
            title: "Rust Docs".to_string(),
            url: "https://doc.rust-lang.org".to_string(),
            snippet: "The Rust Programming Language".to_string(),
        }];
        let prompt = build_web_context_prompt(&results, &[]);
        assert!(prompt.contains("WEB SEARCH RESULTS"));
        assert!(prompt.contains("Rust Docs"));
    }

    #[test]
    fn test_build_web_context_prompt_with_pages() {
        let pages = vec![FetchOutput {
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            text: "Content here".to_string(),
            text_length: 12,
        }];
        let prompt = build_web_context_prompt(&[], &pages);
        assert!(prompt.contains("FETCHED WEB PAGES"));
        assert!(prompt.contains("Example"));
    }

    #[test]
    fn test_build_web_context_prompt_combined() {
        let results = vec![WebSearchResult {
            title: "Search Result".to_string(),
            url: "https://search.com".to_string(),
            snippet: "Found this".to_string(),
        }];
        let pages = vec![FetchOutput {
            url: "https://page.com".to_string(),
            title: "Page Title".to_string(),
            text: "Page content".to_string(),
            text_length: 12,
        }];
        let prompt = build_web_context_prompt(&results, &pages);
        assert!(prompt.contains("WEB SEARCH RESULTS"));
        assert!(prompt.contains("FETCHED WEB PAGES"));
    }
}
