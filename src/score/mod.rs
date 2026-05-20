//! Scoring layer — judges, metric shapes, and aggregation.
//!
//! - [`judge`]     — heuristic answer judge; LLM judge slots in later.
//! - [`metrics`]   — aggregate `EvalMetrics` + `compute_full_metrics`.
//! - [`retrieval`] — `RetrievalStats`, Recall@K, NDCG@K.
//! - [`latency`]   — `LatencyStats` + percentile computation.

pub mod judge;
pub mod latency;
pub mod metrics;
pub mod retrieval;

pub use judge::judge_answer_heuristic;
pub use latency::{compute_latency_stats, LatencyStats};
pub use metrics::{
    compute_full_metrics, DimensionMetrics, EvalMetrics, TokenStats, WriteQualityMetrics,
};
pub use retrieval::{ndcg_at_k, recall_at_k, RetrievalStats};
