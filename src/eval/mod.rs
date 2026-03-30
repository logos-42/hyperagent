//! Evaluation subsystem for multi-dimensional code and agent assessment.
//!
//! This module provides comprehensive evaluation capabilities beyond simple
//! test pass/fail, including:
//!
//! - **Metrics**: Multi-dimensional iteration metrics (test coverage, code
//!   complexity, binary size, warnings) with weighted comparison
//! - **Benchmark**: Task-based performance evaluation across categories
//!   (code generation, algorithms, etc.)
//! - **Evaluator**: LLM-based and rule-based evaluation with ensemble support
//!
//! # Example
//!
//! ```ignore
//! use hyperagent::eval::{IterationMetrics, MultiEvalResult, MetricDirection};
//!
//! let metrics = IterationMetrics::from_code("./src");
//! let result = MultiEvalResult::from_metrics(&[metrics]);
//! ```

pub mod evaluator;
pub mod benchmark;
pub mod metrics;

pub use evaluator::{Evaluator, EvaluationResult, Score};
pub use benchmark::{Benchmark, BenchmarkTask, BenchmarkResult};
pub use metrics::{IterationMetrics, MultiEvalResult, MetricDirection};
