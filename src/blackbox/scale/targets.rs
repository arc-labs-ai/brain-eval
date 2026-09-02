//! Warm-shard performance targets the substrate commits to.
//!
//! These are the numbers a single warm shard must hold on reference
//! hardware (16 cores / 64 GB / NVMe / Linux) on the default text-input
//! path (CPU embedding). They're the gate the acceptance scale-run checks
//! against. Two notes on the values used here:
//!
//! - Latency assumes the CPU-embedding ENCODE path (the GPU and
//!   pre-supplied-vector paths have lower targets, measured separately).
//! - RECALL latency is the fused-retrieval read path. The eval harness
//!   runs the reranker disabled (embed-only deploy), so this is the
//!   RRF-only read budget — which is what the read target is set against.
//!
//! Beyond ENCODE/RECALL, the per-verb latency budgets for the typed-graph
//! write verbs, the storage-only vector-direct write, and the entity
//! resolver tiers come straight from the acceptance gate
//! (`spec/19_benchmarks/06_complete_acceptance.md` §Performance
//! acceptance → Latency). Where the acceptance gate specifies only a P50
//! bound (the two resolver-tier rows), the P99 budget is left ungated
//! (`f64::INFINITY`) rather than inventing a number the spec does not
//! give.

/// A latency budget for one verb, in milliseconds.
#[derive(Debug, Clone, Copy)]
pub struct VerbTarget {
    /// Median (p50) latency budget, milliseconds.
    pub p50_ms: f64,
    /// 99th-percentile latency budget, milliseconds. `f64::INFINITY` means
    /// the spec gives no P99 bound for this verb (only P50 is gated).
    pub p99_ms: f64,
}

impl VerbTarget {
    /// True iff a measured `(p50, p99)` pair meets this budget. A P99
    /// budget of `f64::INFINITY` (spec gives no P99 bound) always passes on
    /// the P99 axis, so only the P50 bound gates.
    #[must_use]
    pub fn pass(&self, measured_p50_ms: f64, measured_p99_ms: f64) -> bool {
        measured_p50_ms <= self.p50_ms && measured_p99_ms <= self.p99_ms
    }

    /// Whether this verb carries a finite (gated) P99 budget.
    #[must_use]
    pub fn p99_gated(&self) -> bool {
        self.p99_ms.is_finite()
    }
}

/// The full target set a scale run checks against.
#[derive(Debug, Clone, Copy)]
pub struct Targets {
    /// ENCODE (text, CPU embedding) latency budget.
    pub encode: VerbTarget,
    /// ENCODE_VECTOR_DIRECT (pre-supplied vector, no embedding) latency
    /// budget — the storage-only write path.
    pub encode_vector_direct: VerbTarget,
    /// RECALL (fused retrieval) latency budget.
    pub recall: VerbTarget,
    /// STATEMENT_CREATE latency budget (typed-graph statement write).
    pub statement_create: VerbTarget,
    /// RELATION_CREATE latency budget (typed-graph relation write).
    pub relation_create: VerbTarget,
    /// Entity resolution tiers 1–2 (exact + trigram fuzzy) latency budget.
    /// The acceptance gate bounds P50 only.
    pub entity_resolve_tier12: VerbTarget,
    /// Entity resolution tier 3 (embedding HNSW) latency budget. The
    /// acceptance gate bounds P50 only.
    pub entity_resolve_tier3: VerbTarget,
    /// Sustained ENCODE throughput floor, ops/second per shard.
    pub encode_ops_per_sec: f64,
    /// Sustained RECALL/QUERY throughput floor, ops/second per shard.
    pub recall_ops_per_sec: f64,
}

impl Default for Targets {
    fn default() -> Self {
        Self {
            // ENCODE text/CPU: p50 ≤ 12 ms, p99 ≤ 25 ms.
            encode: VerbTarget {
                p50_ms: 12.0,
                p99_ms: 25.0,
            },
            // ENCODE_VECTOR_DIRECT (storage-only): p50 ≤ 1 ms, p99 ≤ 5 ms.
            encode_vector_direct: VerbTarget {
                p50_ms: 1.0,
                p99_ms: 5.0,
            },
            // RECALL retrieval: p50 ≤ 10 ms, p99 ≤ 50 ms.
            recall: VerbTarget {
                p50_ms: 10.0,
                p99_ms: 50.0,
            },
            // STATEMENT_CREATE: p50 ≤ 1 ms, p99 ≤ 5 ms.
            statement_create: VerbTarget {
                p50_ms: 1.0,
                p99_ms: 5.0,
            },
            // RELATION_CREATE: p50 ≤ 1 ms, p99 ≤ 5 ms.
            relation_create: VerbTarget {
                p50_ms: 1.0,
                p99_ms: 5.0,
            },
            // Entity resolution (tiers 1–2, exact+fuzzy): p50 ≤ 1 ms. The
            // acceptance gate gives no P99 bound.
            entity_resolve_tier12: VerbTarget {
                p50_ms: 1.0,
                p99_ms: f64::INFINITY,
            },
            // Entity resolution (tier 3, embedding HNSW): p50 ≤ 10 ms. The
            // acceptance gate gives no P99 bound.
            entity_resolve_tier3: VerbTarget {
                p50_ms: 10.0,
                p99_ms: f64::INFINITY,
            },
            // ENCODE ≥ 100/s, QUERY ≥ 1000/s per shard.
            encode_ops_per_sec: 100.0,
            recall_ops_per_sec: 1_000.0,
        }
    }
}

impl Targets {
    /// Every per-verb latency budget, labeled, in a stable order. Lets a
    /// report enumerate the full spec target set (including the vector-direct
    /// and tier-3 budgets, which a dev-box probe does not itself drive).
    #[must_use]
    pub fn latency_manifest(&self) -> Vec<(&'static str, VerbTarget)> {
        vec![
            ("encode", self.encode),
            ("encode_vector_direct", self.encode_vector_direct),
            ("recall", self.recall),
            ("statement_create", self.statement_create),
            ("relation_create", self.relation_create),
            ("entity_resolve_tier12", self.entity_resolve_tier12),
            ("entity_resolve_tier3", self.entity_resolve_tier3),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_gates_both_p50_and_p99_when_finite() {
        // STATEMENT_CREATE budget: p50 ≤ 1 ms, p99 ≤ 5 ms.
        let t = Targets::default().statement_create;
        // At the floor (both axes exactly at budget) → passes.
        assert!(t.pass(1.0, 5.0));
        // p50 above budget → fails.
        assert!(!t.pass(1.01, 5.0));
        // p99 above budget → fails.
        assert!(!t.pass(1.0, 5.01));
        // Comfortably under → passes.
        assert!(t.pass(0.5, 3.0));
    }

    #[test]
    fn pass_ignores_p99_when_ungated() {
        // Tier 1–2 resolver budget bounds P50 only (p99 = INFINITY).
        let t = Targets::default().entity_resolve_tier12;
        assert!(!t.p99_gated());
        // At the P50 floor with an arbitrarily large P99 → still passes.
        assert!(t.pass(1.0, 9_999.0));
        // Over the P50 floor → fails regardless of P99.
        assert!(!t.pass(1.5, 0.1));
    }

    #[test]
    fn spec_default_values() {
        let d = Targets::default();
        // Typed-graph writes: 1 ms / 5 ms (acceptance gate §Latency).
        assert_eq!(d.statement_create.p50_ms, 1.0);
        assert_eq!(d.statement_create.p99_ms, 5.0);
        assert_eq!(d.relation_create.p50_ms, 1.0);
        assert_eq!(d.relation_create.p99_ms, 5.0);
        // Storage-only vector-direct write: 1 ms / 5 ms.
        assert_eq!(d.encode_vector_direct.p50_ms, 1.0);
        assert_eq!(d.encode_vector_direct.p99_ms, 5.0);
        // Resolver tiers: P50 only.
        assert_eq!(d.entity_resolve_tier12.p50_ms, 1.0);
        assert!(d.entity_resolve_tier12.p99_ms.is_infinite());
        assert_eq!(d.entity_resolve_tier3.p50_ms, 10.0);
        assert!(d.entity_resolve_tier3.p99_ms.is_infinite());
    }

    #[test]
    fn latency_manifest_covers_every_verb() {
        let labels: Vec<&str> = Targets::default()
            .latency_manifest()
            .into_iter()
            .map(|(l, _)| l)
            .collect();
        assert_eq!(labels.len(), 7);
        for expected in [
            "encode",
            "encode_vector_direct",
            "recall",
            "statement_create",
            "relation_create",
            "entity_resolve_tier12",
            "entity_resolve_tier3",
        ] {
            assert!(labels.contains(&expected), "missing {expected}");
        }
    }
}
