//! Environment module for managing agent execution contexts
//! 
//! This module provides a standardized folder structure for agent execution,
//! including configuration, memory, logs, and state management.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Base directory for the environment
    pub base_dir: PathBuf,
    /// Environment name
    pub name: String,
    /// Maximum number of iterations to keep
    pub max_iterations: usize,
    /// Enable auto-cleanup of old iterations
    pub auto_cleanup: bool,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        let base_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".hyperagent");
        
        Self {
            base_dir,
            name: "default".to_string(),
            max_iterations: 100,
            auto_cleanup: true,
        }
    }
}

impl EnvironmentConfig {
    /// Create a new environment config with custom base directory
    pub fn new(base_dir: &str, name: &str) -> Self {
        Self {
            base_dir: PathBuf::from(base_dir),
            name: name.to_string(),
            max_iterations: 100,
            auto_cleanup: true,
        }
    }

    /// Create from environment variable or default
    pub fn from_env() -> Self {
        let base_dir = std::env::var("HYPERAGENT_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .map(|h| h.join(".hyperagent"))
                    .unwrap_or_else(|| PathBuf::from(".hyperagent"))
            });

        let name = std::env::var("HYPERAGENT_ENV")
            .unwrap_or_else(|_| "default".to_string());

        Self {
            base_dir,
            name,
            max_iterations: 100,
            auto_cleanup: true,
        }
    }
}

/// Environment folder structure
/// 
/// ```text
/// {base_dir}/
/// ├── config/
/// │   ├── environment.json      # Environment configuration
/// │   ├── llm.json              # LLM configuration
/// │   └── agent.json            # Agent configuration
/// ├── sessions/
/// │   └── {session_id}/
/// │       ├── session.json      # Session metadata
/// │       ├── iterations/
/// │       │   └── {iteration_id}/
/// │       │       ├── state.json      # Iteration state
/// │       │       ├── messages/       # Conversation history
/// │       │       │   ├── user.json
/// │       │       │   └── assistant.json
/// │       │       ├── artifacts/      # Generated files
/// │       │       └── logs/
/// │       │           └── execution.log
/// │       └── memory/
/// │           ├── short_term.json     # Short-term memory
/// │           └── long_term.json      # Long-term memory
/// ├── memory/
/// │   ├── archive/              # Archived memories
/// │   └── cache/                # Cached responses
/// └── logs/
///     └── {date}/
///         └── execution.log
/// ```
#[derive(Debug, Clone)]
pub struct Environment {
    config: EnvironmentConfig,
    session_id: Option<String>,
    iteration_id: Option<String>,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub iteration_count: usize,
    pub status: SessionStatus,
    pub llm_provider: String,
    pub llm_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Paused,
    Completed,
    Failed,
}

/// Iteration state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationState {
    pub id: String,
    pub session_id: String,
    pub iteration_number: usize,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: IterationStatus,
    pub prompt: String,
    pub response: Option<String>,
    pub error: Option<String>,
    pub metrics: IterationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IterationMetrics {
    pub tokens_used: i32,
    pub execution_time_ms: u64,
    pub success: bool,
    pub retry_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IterationStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Timeout,
}

impl Environment {
    /// Create a new environment
    pub fn new(config: EnvironmentConfig) -> Result<Self> {
        let env = Self {
            config,
            session_id: None,
            iteration_id: None,
        };
        env.initialize()?;
        Ok(env)
    }

    /// Initialize the environment folder structure
    pub fn initialize(&self) -> Result<()> {
        let dirs = [
            self.config_dir(),
            self.sessions_dir(),
            self.memory_dir(),
            self.memory_archive_dir(),
            self.memory_cache_dir(),
            self.logs_dir(),
        ];

        for dir in dirs {
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create directory: {:?}", dir))?;
        }

        // Write default environment config if not exists
        let env_config_path = self.config_dir().join("environment.json");
        if !env_config_path.exists() {
            let config_json = serde_json::to_string_pretty(&self.config)?;
            fs::write(&env_config_path, config_json)?;
        }

        Ok(())
    }

    /// Get the base directory
    pub fn base_dir(&self) -> &Path {
        &self.config.base_dir
    }

    /// Get the config directory
    pub fn config_dir(&self) -> PathBuf {
        self.config.base_dir.join("config")
    }

    /// Get the sessions directory
    pub fn sessions_dir(&self) -> PathBuf {
        self.config.base_dir.join("sessions")
    }

    /// Get the memory directory
    pub fn memory_dir(&self) -> PathBuf {
        self.config.base_dir.join("memory")
    }

    /// Get the memory archive directory
    pub fn memory_archive_dir(&self) -> PathBuf {
        self.memory_dir().join("archive")
    }

    /// Get the memory cache directory
    pub fn memory_cache_dir(&self) -> PathBuf {
        self.memory_dir().join("cache")
    }

    /// Get the logs directory
    pub fn logs_dir(&self) -> PathBuf {
        self.config.base_dir.join("logs")
    }

    /// Create a new session
    pub fn create_session(&mut self, name: &str, llm_provider: &str, llm_model: &str) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let session_dir = self.sessions_dir().join(&session_id);
        
        fs::create_dir_all(&session_dir)?;
        fs::create_dir_all(session_dir.join("iterations"))?;
        fs::create_dir_all(session_dir.join("memory"))?;

        let now = Utc::now();
        let session_meta = SessionMeta {
            id: session_id.clone(),
            name: name.to_string(),
            created_at: now,
            updated_at: now,
            iteration_count: 0,
            status: SessionStatus::Active,
            llm_provider: llm_provider.to_string(),
            llm_model: llm_model.to_string(),
        };

        let session_path = session_dir.join("session.json");
        let session_json = serde_json::to_string_pretty(&session_meta)?;
        fs::write(&session_path, session_json)?;

        self.session_id = Some(session_id.clone());
        
        Ok(session_id)
    }

    /// Load an existing session
    pub fn load_session(&mut self, session_id: &str) -> Result<SessionMeta> {
        let session_dir = self.sessions_dir().join(session_id);
        if !session_dir.exists() {
            return Err(anyhow!("Session not found: {}", session_id));
        }

        let session_path = session_dir.join("session.json");
        let session_json = fs::read_to_string(&session_path)?;
        let session_meta: SessionMeta = serde_json::from_str(&session_json)?;

        self.session_id = Some(session_id.to_string());
        
        Ok(session_meta)
    }

    /// Get the current session directory
    pub fn current_session_dir(&self) -> Result<PathBuf> {
        self.session_id
            .as_ref()
            .map(|id| self.sessions_dir().join(id))
            .ok_or_else(|| anyhow!("No active session"))
    }

    /// Start a new iteration
    pub fn start_iteration(&mut self, prompt: &str) -> Result<String> {
        let session_dir = self.current_session_dir()?;
        let iteration_id = Uuid::new_v4().to_string();
        let iteration_dir = session_dir.join("iterations").join(&iteration_id);

        fs::create_dir_all(&iteration_dir)?;
        fs::create_dir_all(iteration_dir.join("messages"))?;
        fs::create_dir_all(iteration_dir.join("artifacts"))?;
        fs::create_dir_all(iteration_dir.join("logs"))?;

        // Get current iteration number
        let iterations_dir = session_dir.join("iterations");
        let iteration_number = fs::read_dir(&iterations_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .count();

        let now = Utc::now();
        let iteration_state = IterationState {
            id: iteration_id.clone(),
            session_id: self.session_id.clone().unwrap(),
            iteration_number,
            created_at: now,
            completed_at: None,
            status: IterationStatus::Running,
            prompt: prompt.to_string(),
            response: None,
            error: None,
            metrics: IterationMetrics::default(),
        };

        let state_path = iteration_dir.join("state.json");
        let state_json = serde_json::to_string_pretty(&iteration_state)?;
        fs::write(&state_path, state_json)?;

        // Write user message
        let user_msg_path = iteration_dir.join("messages").join("user.json");
        fs::write(&user_msg_path, prompt)?;

        self.iteration_id = Some(iteration_id.clone());

        // Update session iteration count
        self.update_session_iteration_count()?;

        Ok(iteration_id)
    }

    /// Complete the current iteration
    pub fn complete_iteration(&mut self, response: &str, metrics: IterationMetrics) -> Result<()> {
        let iteration_dir = self.current_iteration_dir()?;

        // Update state
        let state_path = iteration_dir.join("state.json");
        let mut state: IterationState = {
            let state_json = fs::read_to_string(&state_path)?;
            serde_json::from_str(&state_json)?
        };

        state.completed_at = Some(Utc::now());
        state.status = if metrics.success {
            IterationStatus::Completed
        } else {
            IterationStatus::Failed
        };
        state.response = Some(response.to_string());
        state.metrics = metrics;

        let state_json = serde_json::to_string_pretty(&state)?;
        fs::write(&state_path, state_json)?;

        // Write assistant message
        let assistant_msg_path = iteration_dir.join("messages").join("assistant.json");
        fs::write(&assistant_msg_path, response)?;

        Ok(())
    }

    /// Fail the current iteration
    pub fn fail_iteration(&mut self, error: &str) -> Result<()> {
        let iteration_dir = self.current_iteration_dir()?;

        // Update state
        let state_path = iteration_dir.join("state.json");
        let mut state: IterationState = {
            let state_json = fs::read_to_string(&state_path)?;
            serde_json::from_str(&state_json)?
        };

        state.completed_at = Some(Utc::now());
        state.status = IterationStatus::Failed;
        state.error = Some(error.to_string());

        let state_json = serde_json::to_string_pretty(&state)?;
        fs::write(&state_path, state_json)?;

        Ok(())
    }

    /// Get the current iteration directory
    pub fn current_iteration_dir(&self) -> Result<PathBuf> {
        let session_dir = self.current_session_dir()?;
        self.iteration_id
            .as_ref()
            .map(|id| session_dir.join("iterations").join(id))
            .ok_or_else(|| anyhow!("No active iteration"))
    }

    /// Get iteration artifacts directory
    pub fn artifacts_dir(&self) -> Result<PathBuf> {
        Ok(self.current_iteration_dir()?.join("artifacts"))
    }

    /// Get iteration logs directory
    pub fn iteration_logs_dir(&self) -> Result<PathBuf> {
        Ok(self.current_iteration_dir()?.join("logs"))
    }

    /// Save an artifact
    pub fn save_artifact(&self, name: &str, content: &[u8]) -> Result<PathBuf> {
        let artifacts_dir = self.artifacts_dir()?;
        let artifact_path = artifacts_dir.join(name);
        fs::write(&artifact_path, content)?;
        Ok(artifact_path)
    }

    /// Save a text artifact
    pub fn save_artifact_text(&self, name: &str, content: &str) -> Result<PathBuf> {
        let artifacts_dir = self.artifacts_dir()?;
        let artifact_path = artifacts_dir.join(name);
        fs::write(&artifact_path, content)?;
        Ok(artifact_path)
    }

    /// Write iteration log
    pub fn write_iteration_log(&self, message: &str) -> Result<()> {
        let logs_dir = self.iteration_logs_dir()?;
        let log_path = logs_dir.join("execution.log");
        
        let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let log_entry = format!("[{}] {}\n", timestamp, message);
        
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?
            .write_all(log_entry.as_bytes())?;

        Ok(())
    }

    /// Update session iteration count
    fn update_session_iteration_count(&self) -> Result<()> {
        let session_dir = self.current_session_dir()?;
        let session_path = session_dir.join("session.json");
        
        let mut session: SessionMeta = {
            let session_json = fs::read_to_string(&session_path)?;
            serde_json::from_str(&session_json)?
        };

        session.iteration_count += 1;
        session.updated_at = Utc::now();

        let session_json = serde_json::to_string_pretty(&session)?;
        fs::write(&session_path, session_json)?;

        Ok(())
    }

    /// List all sessions
    pub fn list_sessions(&self) -> Result<Vec<SessionMeta>> {
        let sessions_dir = self.sessions_dir();
        let mut sessions = Vec::new();

        if !sessions_dir.exists() {
            return Ok(sessions);
        }

        for entry in fs::read_dir(&sessions_dir)? {
            let entry = entry?;
            let session_dir = entry.path();
            if !session_dir.is_dir() {
                continue;
            }

            let session_path = session_dir.join("session.json");
            if session_path.exists() {
                let session_json = fs::read_to_string(&session_path)?;
                let session: SessionMeta = serde_json::from_str(&session_json)?;
                sessions.push(session);
            }
        }

        Ok(sessions)
    }

    /// Get session history
    pub fn get_session_history(&self, session_id: &str) -> Result<Vec<IterationState>> {
        let session_dir = self.sessions_dir().join(session_id);
        if !session_dir.exists() {
            return Err(anyhow!("Session not found: {}", session_id));
        }

        let iterations_dir = session_dir.join("iterations");
        let mut iterations = Vec::new();

        if !iterations_dir.exists() {
            return Ok(iterations);
        }

        for entry in fs::read_dir(&iterations_dir)? {
            let entry = entry?;
            let iteration_dir = entry.path();
            if !iteration_dir.is_dir() {
                continue;
            }

            let state_path = iteration_dir.join("state.json");
            if state_path.exists() {
                let state_json = fs::read_to_string(&state_path)?;
                let state: IterationState = serde_json::from_str(&state_json)?;
                iterations.push(state);
            }
        }

        // Sort by iteration number
        iterations.sort_by_key(|i| i.iteration_number);

        Ok(iterations)
    }

    /// Cleanup old iterations
    pub fn cleanup_old_iterations(&self) -> Result<usize> {
        let session_dir = self.current_session_dir()?;
        let iterations_dir = session_dir.join("iterations");

        if !iterations_dir.exists() {
            return Ok(0);
        }

        let mut iterations: Vec<_> = fs::read_dir(&iterations_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                let state_path = e.path().join("state.json");
                if state_path.exists() {
                    fs::read_to_string(&state_path)
                        .ok()
                        .and_then(|s| serde_json::from_str::<IterationState>(&s).ok())
                        .map(|s| (e.path(), s))
                } else {
                    None
                }
            })
            .collect();

        // Sort by creation date (oldest first)
        iterations.sort_by_key(|(_, s)| s.created_at);

        let to_remove = iterations.len().saturating_sub(self.config.max_iterations);
        let mut removed = 0;

        for (path, _) in iterations.into_iter().take(to_remove) {
            if let Err(e) = fs::remove_dir_all(&path) {
                eprintln!("Failed to remove iteration {:?}: {}", path, e);
            } else {
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Save to short-term memory
    pub fn save_short_term_memory(&self, key: &str, value: &str) -> Result<()> {
        let session_dir = self.current_session_dir()?;
        let memory_path = session_dir.join("memory").join("short_term.json");

        let mut memory: std::collections::HashMap<String, String> = if memory_path.exists() {
            let json = fs::read_to_string(&memory_path)?;
            serde_json::from_str(&json).unwrap_or_default()
        } else {
            std::collections::HashMap::new()
        };

        memory.insert(key.to_string(), value.to_string());

        let json = serde_json::to_string_pretty(&memory)?;
        fs::write(&memory_path, json)?;

        Ok(())
    }

    /// Load from short-term memory
    pub fn load_short_term_memory(&self, key: &str) -> Result<Option<String>> {
        let session_dir = self.current_session_dir()?;
        let memory_path = session_dir.join("memory").join("short_term.json");

        if !memory_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&memory_path)?;
        let memory: std::collections::HashMap<String, String> = serde_json::from_str(&json)?;

        Ok(memory.get(key).cloned())
    }

    /// Archive session
    pub fn archive_session(&self, session_id: &str) -> Result<()> {
        let session_dir = self.sessions_dir().join(session_id);
        let archive_dir = self.memory_archive_dir().join(session_id);

        if !session_dir.exists() {
            return Err(anyhow!("Session not found: {}", session_id));
        }

        fs::rename(&session_dir, &archive_dir)?;

        Ok(())
    }

    /// Get environment info
    pub fn info(&self) -> EnvironmentInfo {
        EnvironmentInfo {
            base_dir: self.config.base_dir.clone(),
            name: self.config.name.clone(),
            session_id: self.session_id.clone(),
            iteration_id: self.iteration_id.clone(),
            max_iterations: self.config.max_iterations,
        }
    }
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub base_dir: PathBuf,
    pub name: String,
    pub session_id: Option<String>,
    pub iteration_id: Option<String>,
    pub max_iterations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_env() -> (Environment, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let config = EnvironmentConfig {
            base_dir: temp_dir.path().to_path_buf(),
            name: "test".to_string(),
            max_iterations: 10,
            auto_cleanup: true,
        };
        let env = Environment::new(config).unwrap();
        (env, temp_dir)
    }

    #[test]
    fn test_environment_initialization() {
        let (env, _temp) = create_test_env();
        
        assert!(env.config_dir().exists());
        assert!(env.sessions_dir().exists());
        assert!(env.memory_dir().exists());
        assert!(env.logs_dir().exists());
    }

    #[test]
    fn test_session_creation() {
        let (mut env, _temp) = create_test_env();
        
        let session_id = env.create_session("Test Session", "openai", "gpt-4o").unwrap();
        assert!(!session_id.is_empty());
        assert!(env.sessions_dir().join(&session_id).exists());
    }

    #[test]
    fn test_iteration_lifecycle() {
        let (mut env, _temp) = create_test_env();
        
        env.create_session("Test", "openai", "gpt-4o").unwrap();
        let _iteration_id = env.start_iteration("Test prompt").unwrap();
        
        let metrics = IterationMetrics {
            tokens_used: 100,
            execution_time_ms: 500,
            success: true,
            retry_count: 0,
        };
        
        env.complete_iteration("Test response", metrics).unwrap();
        
        let iterations = env.get_session_history(env.session_id.as_ref().unwrap()).unwrap();
        assert_eq!(iterations.len(), 1);
        assert_eq!(iterations[0].status, IterationStatus::Completed);
    }
}
