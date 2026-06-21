//! Plain-text reporter — ASCII summary for terminal viewing.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::Reporter;
use crate::report::shape::BenchmarkReport;
use crate::score::metrics::DimensionMetrics;

/// Fixed-width ASCII summary.
pub struct TextReporter;

impl Reporter for TextReporter {
    fn write(&self, report: &BenchmarkReport, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        writeln!(f, "Brain eval report")?;
        writeln!(f, "=================")?;
        writeln!(f)?;
        writeln!(
            f,
            "benchmark   : {} ({})",
            report.meta.benchmark_name, report.meta.benchmark_id
        )?;
        writeln!(f, "url         : {}", report.meta.benchmark_url)?;
        writeln!(f, "judge       : {}", report.meta.judge_type)?;
        if report.meta.judge_heuristic_fallbacks > 0 {
            writeln!(
                f,
                "  WARNING   : {} question(s) fell back to the HEURISTIC judge \
                 (failed/unparseable LLM call) — accuracy is NOT fully LLM-judged",
                report.meta.judge_heuristic_fallbacks,
            )?;
        }
        writeln!(
            f,
            "judge prompt: {} (sha256 {}, temp {})",
            report.meta.judge_prompt_version,
            &report.meta.judge_prompt_sha256[..report.meta.judge_prompt_sha256.len().min(12)],
            report.meta.judge_temperature,
        )?;
        writeln!(f, "instances   : {}", report.meta.instance_count)?;
        writeln!(f, "brain       : {}", report.meta.brain_version)?;
        writeln!(f)?;

        let m = &report.metrics;
        writeln!(f, "Summary")?;
        writeln!(f, "-------")?;
        writeln!(
            f,
            "accuracy            : {:.4} ({}/{} correct, {} partial, {} incorrect)",
            m.accuracy, m.correct, m.total_questions, m.partial, m.incorrect
        )?;
        writeln!(
            f,
            "ingestion errors    : {}    retrieval errors : {}",
            m.ingestion_errors, m.retrieval_errors
        )?;
        writeln!(f)?;
        writeln!(f, "Latency (ms)")?;
        writeln!(
            f,
            "  write  : p50 {}  p95 {}  mean {}",
            m.latency.write_p50_ms, m.latency.write_p95_ms, m.latency.write_mean_ms
        )?;
        writeln!(
            f,
            "  read   : p50 {}  p95 {}  mean {}",
            m.latency.read_p50_ms, m.latency.read_p95_ms, m.latency.read_mean_ms
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Tokens   : write_avg {:.1}  read_avg {:.1}  total {}",
            m.tokens.write_avg, m.tokens.read_avg, m.tokens.grand_total
        )?;
        writeln!(f)?;

        writeln!(f, "Retrieval quality")?;
        // Headline: answer-supporting context recall (semantic, LLM-judged).
        match &m.context_recall {
            Some(c) => writeln!(
                f,
                "  Context recall (headline) : {:.4}  ({}/{} answer-supporting)",
                c.supported_rate, c.n_supported, c.n_judged
            )?,
            None => writeln!(
                f,
                "  Context recall (headline) : n/a  (no LLM judge; run with --features live-llm)"
            )?,
        }
        // Deprecated diagnostic: substring recall@k. Rewards lexical
        // overlap (Kamalloo 2023 / NoLiMa); kept only to contrast.
        if let Some(r) = &m.retrieval {
            writeln!(
                f,
                "  Substring recall@k (DEPRECATED diagnostic — Kamalloo 2023 / NoLiMa):"
            )?;
            writeln!(f, "    Recall@1  : {:.4}", r.recall_at_1)?;
            writeln!(
                f,
                "    Recall@5  : {:.4}    Recall@10 : {:.4}",
                r.recall_at_5, r.recall_at_10
            )?;
            writeln!(
                f,
                "    NDCG@5    : {:.4}    NDCG@10   : {:.4}",
                r.ndcg_at_5, r.ndcg_at_10
            )?;
        }
        writeln!(f)?;

        if !m.per_dimension.is_empty() {
            writeln!(f, "Per-dimension")?;
            let mut entries: Vec<(&String, &DimensionMetrics)> = m.per_dimension.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (tag, d) in entries {
                writeln!(
                    f,
                    "  {:<18} {:.4}  ({}/{})",
                    tag, d.accuracy, d.correct, d.count
                )?;
            }
            writeln!(f)?;
        }

        if !report.competitors.is_empty() {
            writeln!(f, "Competitor comparison")?;
            writeln!(f, "---------------------")?;
            for c in &report.competitors {
                writeln!(f, "  {:<18} {:.4}    {}", c.system, c.accuracy, c.source)?;
                if !c.note.is_empty() {
                    writeln!(f, "                       note: {}", c.note)?;
                }
            }
        }

        f.flush()?;
        Ok(())
    }
}
