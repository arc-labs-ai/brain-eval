//! Scoring layer — judges, metric shapes, and aggregation.
//!
//! - [`judge`]          — heuristic answer judge (always available).
//! - [`judge_prompt`]   — pinned judge prompts + version/temperature.
//! - [`llm_judge`]       — LLM-as-judge (behind the `live-llm` feature).
//! - [`metrics`]         — aggregate `EvalMetrics` + `compute_full_metrics`.
//! - [`context_recall`]  — answer-supporting context recall (headline).
//! - [`retrieval`]       — substring Recall@K, NDCG@K (deprecated diagnostic).
//! - [`latency`]         — `LatencyStats` + percentile computation.

pub mod context_recall;
pub mod judge;
pub mod judge_prompt;
pub mod latency;
#[cfg(feature = "live-llm")]
pub mod llm_judge;
pub mod metrics;
pub mod retrieval;

pub use context_recall::{compute_context_recall, ContextRecallStats};
pub use judge::judge_answer_heuristic;
pub use latency::{compute_latency_stats, LatencyStats};
pub use metrics::{
    compute_full_metrics, DimensionMetrics, EvalMetrics, TokenStats, WriteQualityMetrics,
};
pub use retrieval::{ndcg_at_k, recall_at_k, RetrievalStats};
