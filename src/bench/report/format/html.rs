//! HTML reporter — a single self-contained page for a benchmark run.
//!
//! Writes one `.html` file with all CSS inlined and no external requests, so
//! it opens straight from disk (or a CI artifact store) with no network. It
//! renders the same [`BenchmarkReport`] the JSON/text reporters do: the
//! headline metrics as stat cards, retrieval quality, per-dimension and
//! competitor tables, and a scrollable per-question ledger with a colour-coded
//! verdict on every row. Every dynamic string is HTML-escaped.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::Reporter;
use crate::bench::core::outcome::{AnswerKind, Verdict};
use crate::bench::report::shape::BenchmarkReport;
use crate::bench::score::metrics::DimensionMetrics;

/// Self-contained single-page HTML summary.
pub struct HtmlReporter;

impl Reporter for HtmlReporter {
    fn write(&self, report: &BenchmarkReport, path: &Path) -> std::io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(render(report).as_bytes())?;
        f.flush()?;
        Ok(())
    }
}

/// HTML-escape a string for safe interpolation into element text / attributes.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// CSS class + label for a verdict (drives the row/chip colour).
fn verdict_class(v: Verdict) -> (&'static str, &'static str) {
    match v {
        Verdict::Correct => ("v-correct", "correct"),
        Verdict::Partial => ("v-partial", "partial"),
        Verdict::Incorrect => ("v-incorrect", "incorrect"),
        // Its own chip, not "incorrect". A retrieval-only run grades nothing,
        // and a report that renders that as a wall of failures is worse than
        // one that says plainly that no answer was judged.
        Verdict::NotJudged => ("v-notjudged", "not judged"),
    }
}

fn answer_kind_label(k: AnswerKind) -> &'static str {
    match k {
        AnswerKind::Single => "single",
        AnswerKind::Many => "many",
        AnswerKind::None => "none",
        AnswerKind::Error => "error",
    }
}

/// A percentage bar cell (0.0..=1.0) — a slim inline meter behind the number.
fn pct_bar(value: f64) -> String {
    let pct = (value.clamp(0.0, 1.0) * 100.0).round() as u32;
    format!(
        "<div class=\"bar\"><div class=\"bar-fill\" style=\"width:{pct}%\"></div>\
         <span class=\"bar-label\">{value:.4}</span></div>"
    )
}

#[allow(clippy::too_many_lines)]
fn render(report: &BenchmarkReport) -> String {
    let m = &report.metrics;
    let meta = &report.meta;
    let mut h = String::with_capacity(16 * 1024);

    // --- head + style -----------------------------------------------------
    h.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    h.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    let _ = write!(
        h,
        "<title>Brain eval — {}</title>",
        esc(&meta.benchmark_name)
    );
    h.push_str(STYLE);
    h.push_str("</head><body>");

    // --- header -----------------------------------------------------------
    h.push_str("<header class=\"hdr\"><div class=\"hdr-in\">");
    let _ = write!(
        h,
        "<div class=\"eyebrow\">Brain evaluation</div><h1>{}</h1>",
        esc(&meta.benchmark_name)
    );
    h.push_str("<div class=\"meta-grid\">");
    meta_item(&mut h, "benchmark id", &esc(&meta.benchmark_id));
    if meta.benchmark_url.is_empty() {
        meta_item(&mut h, "source", "—");
    } else {
        meta_item(
            &mut h,
            "source",
            &format!(
                "<a href=\"{u}\" rel=\"noreferrer noopener\">{u}</a>",
                u = esc(&meta.benchmark_url)
            ),
        );
    }
    meta_item(&mut h, "judge", &esc(&meta.judge_type));
    meta_item(
        &mut h,
        "judge prompt",
        &esc(&format!(
            "{} ({}…, temp {})",
            meta.judge_prompt_version,
            &meta.judge_prompt_sha256[..meta.judge_prompt_sha256.len().min(12)],
            meta.judge_temperature
        )),
    );
    meta_item(&mut h, "instances", &meta.instance_count.to_string());
    meta_item(&mut h, "brain version", &esc(&meta.brain_version));
    meta_item(
        &mut h,
        "run (unix ns)",
        &meta.run_started_unix_nanos.to_string(),
    );
    h.push_str("</div></div></header>");

    h.push_str("<main class=\"wrap\">");

    // --- heuristic-fallback warning --------------------------------------
    if meta.judge_heuristic_fallbacks > 0 {
        let _ = write!(
            h,
            "<div class=\"warn\">⚠ {} question(s) fell back to the <b>heuristic</b> judge \
             (the LLM call failed or was unparseable) — accuracy is <b>not</b> fully LLM-judged.</div>",
            meta.judge_heuristic_fallbacks
        );
    }

    // --- headline stat cards ---------------------------------------------
    h.push_str("<section class=\"cards\">");
    stat_card(
        &mut h,
        "accuracy",
        &format!("{:.1}%", m.accuracy * 100.0),
        &format!(
            "{}/{} correct · {} partial · {} incorrect",
            m.correct, m.total_questions, m.partial, m.incorrect
        ),
        "accent",
    );
    match &m.context_recall {
        Some(c) => stat_card(
            &mut h,
            "context recall",
            &format!("{:.1}%", c.supported_rate * 100.0),
            &format!(
                "{}/{} answer-supporting (LLM-judged)",
                c.n_supported, c.n_judged
            ),
            "ok",
        ),
        None => stat_card(
            &mut h,
            "context recall",
            "n/a",
            "no LLM judge (run with --features live-llm)",
            "muted",
        ),
    }
    let err_tone = if m.ingestion_errors + m.retrieval_errors == 0 {
        "ok"
    } else {
        "bad"
    };
    stat_card(
        &mut h,
        "errors",
        &(m.ingestion_errors + m.retrieval_errors).to_string(),
        &format!(
            "{} ingestion · {} retrieval",
            m.ingestion_errors, m.retrieval_errors
        ),
        err_tone,
    );
    stat_card(
        &mut h,
        "read latency p50",
        &format!("{} ms", m.latency.read_p50_ms),
        &format!(
            "p95 {} ms · write p50 {} ms",
            m.latency.read_p95_ms, m.latency.write_p50_ms
        ),
        "muted",
    );
    h.push_str("</section>");

    // --- latency + tokens -------------------------------------------------
    h.push_str("<section class=\"panel\"><h2>Latency &amp; tokens</h2><div class=\"tbl-wrap\"><table class=\"kv\">");
    h.push_str("<thead><tr><th></th><th>p50</th><th>p95</th><th>mean</th></tr></thead><tbody>");
    let _ = write!(
        h,
        "<tr><th>write (ms)</th><td>{}</td><td>{}</td><td>{}</td></tr>",
        m.latency.write_p50_ms, m.latency.write_p95_ms, m.latency.write_mean_ms
    );
    let _ = write!(
        h,
        "<tr><th>read (ms)</th><td>{}</td><td>{}</td><td>{}</td></tr>",
        m.latency.read_p50_ms, m.latency.read_p95_ms, m.latency.read_mean_ms
    );
    h.push_str("</tbody></table>");
    let _ = write!(
        h,
        "<p class=\"note\">tokens — write avg {:.1} · read avg {:.1} · total {}</p>",
        m.tokens.write_avg, m.tokens.read_avg, m.tokens.grand_total
    );
    h.push_str("</div></section>");

    // --- retrieval quality ------------------------------------------------
    h.push_str("<section class=\"panel\"><h2>Retrieval quality</h2>");
    if let Some(r) = &m.retrieval {
        h.push_str("<div class=\"tbl-wrap\"><table class=\"kv\"><thead><tr>\
            <th>substring recall@k <span class=\"dep\">(deprecated diagnostic — Kamalloo 2023 / NoLiMa)</span></th>\
            <th>value</th></tr></thead><tbody>");
        for (label, v) in [
            ("recall@1", r.recall_at_1),
            ("recall@5", r.recall_at_5),
            ("recall@10", r.recall_at_10),
            ("ndcg@5", r.ndcg_at_5),
            ("ndcg@10", r.ndcg_at_10),
        ] {
            let _ = write!(h, "<tr><th>{label}</th><td>{}</td></tr>", pct_bar(v));
        }
        h.push_str("</tbody></table></div>");
    } else {
        h.push_str("<p class=\"note\">no substring-recall diagnostic recorded.</p>");
    }
    h.push_str("</section>");

    // --- per-dimension ----------------------------------------------------
    if !m.per_dimension.is_empty() {
        h.push_str("<section class=\"panel\"><h2>Per-dimension accuracy</h2><div class=\"tbl-wrap\"><table><thead><tr><th>dimension</th><th>accuracy</th><th>n</th></tr></thead><tbody>");
        let mut entries: Vec<(&String, &DimensionMetrics)> = m.per_dimension.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (tag, d) in entries {
            let _ = write!(
                h,
                "<tr><th>{}</th><td>{}</td><td class=\"num\">{}/{}</td></tr>",
                esc(tag),
                pct_bar(d.accuracy),
                d.correct,
                d.count
            );
        }
        h.push_str("</tbody></table></div></section>");
    }

    // --- competitor comparison -------------------------------------------
    if !report.competitors.is_empty() {
        h.push_str("<section class=\"panel\"><h2>Competitor comparison</h2><div class=\"tbl-wrap\"><table><thead><tr><th>system</th><th>accuracy</th><th>source</th><th>note</th></tr></thead><tbody>");
        // Brain's own row first, highlighted.
        let _ = write!(
            h,
            "<tr class=\"self-row\"><th>Brain (this run)</th><td>{}</td><td>—</td><td>—</td></tr>",
            pct_bar(m.accuracy)
        );
        for c in &report.competitors {
            let _ = write!(
                h,
                "<tr><th>{}</th><td>{}</td><td>{}</td><td class=\"note-cell\">{}</td></tr>",
                esc(&c.system),
                pct_bar(c.accuracy),
                esc(&c.source),
                esc(&c.note)
            );
        }
        h.push_str("</tbody></table></div></section>");
    }

    // --- per-question ledger ---------------------------------------------
    if !report.per_question.is_empty() {
        let _ = write!(
            h,
            "<section class=\"panel\"><h2>Per-question ledger <span class=\"count\">({})</span></h2>",
            report.per_question.len()
        );
        h.push_str(
            "<div class=\"tbl-wrap\"><table class=\"ledger\"><thead><tr>\
            <th>#</th><th>type</th><th>verdict</th><th>shape</th><th>question</th>\
            <th>ground truth</th><th>system answer</th><th>judge reasoning</th><th>ret</th>\
            </tr></thead><tbody>",
        );
        for (i, q) in report.per_question.iter().enumerate() {
            let (vclass, vlabel) = verdict_class(q.verdict);
            let _ = write!(
                h,
                "<tr class=\"{vclass}\"><td class=\"num\">{}</td><td>{}</td>\
                 <td><span class=\"chip {vclass}\">{vlabel}</span></td>\
                 <td>{}</td><td class=\"q\">{}</td><td class=\"gt\">{}</td>\
                 <td class=\"ans\">{}</td><td class=\"rz\">{}</td><td class=\"num\">{}</td></tr>",
                i + 1,
                esc(&format!("{:?}", q.question_type)),
                answer_kind_label(q.answer_kind),
                esc(&q.question),
                esc(&q.ground_truth),
                esc(&q.system_answer),
                esc(&q.judge_reasoning),
                q.memories_retrieved,
            );
        }
        h.push_str("</tbody></table></div></section>");
    }

    h.push_str("<footer class=\"ftr\">Generated by brain-eval · self-contained, no external requests.</footer>");
    h.push_str("</main></body></html>");
    h
}

fn meta_item(h: &mut String, label: &str, value_html: &str) {
    let _ = write!(
        h,
        "<div class=\"meta-item\"><span class=\"ml\">{}</span><span class=\"mv\">{}</span></div>",
        esc(label),
        value_html
    );
}

fn stat_card(h: &mut String, label: &str, value: &str, sub: &str, tone: &str) {
    let _ = write!(
        h,
        "<div class=\"card {tone}\"><div class=\"card-label\">{}</div>\
         <div class=\"card-value\">{}</div><div class=\"card-sub\">{}</div></div>",
        esc(label),
        esc(value),
        esc(sub)
    );
}

/// All CSS, inlined. Dark-first with a light `prefers-color-scheme` override;
/// tables scroll inside their own container so the page body never scrolls
/// sideways.
const STYLE: &str = r#"<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel-2:#1d212b; --line:#2a2f3a;
  --fg:#e6e9ef; --muted:#9aa4b2; --accent:#6ea8fe; --accent-dim:#2a3550;
  --ok:#4ade80; --warn:#fbbf24; --bad:#f87171;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:light){
  :root{
    --bg:#f6f7f9; --panel:#ffffff; --panel-2:#f0f2f5; --line:#e2e6ec;
    --fg:#1a1d24; --muted:#5c6675; --accent:#2563eb; --accent-dim:#dbe6ff;
    --ok:#16a34a; --warn:#b45309; --bad:#dc2626;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--accent)}
.hdr{background:linear-gradient(180deg,var(--panel-2),var(--panel));border-bottom:1px solid var(--line)}
.hdr-in{max-width:1180px;margin:0 auto;padding:28px 24px 22px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{margin:6px 0 18px;font-size:26px;font-weight:700;letter-spacing:-.01em}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px 26px}
.meta-item{display:flex;flex-direction:column;gap:2px;min-width:0}
.ml{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.mv{font-family:var(--mono);font-size:12.5px;word-break:break-word}
.wrap{max-width:1180px;margin:0 auto;padding:24px}
.warn{background:color-mix(in srgb,var(--warn) 16%,transparent);border:1px solid var(--warn);
  color:var(--fg);padding:12px 14px;border-radius:10px;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;position:relative;overflow:hidden}
.card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--muted)}
.card.accent::before{background:var(--accent)} .card.ok::before{background:var(--ok)}
.card.bad::before{background:var(--bad)} .card.muted::before{background:var(--muted)}
.card-label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.card-value{font-size:30px;font-weight:700;margin:6px 0 4px;font-variant-numeric:tabular-nums}
.card.accent .card-value{color:var(--accent)} .card.ok .card-value{color:var(--ok)} .card.bad .card-value{color:var(--bad)}
.card-sub{font-size:12px;color:var(--muted)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:20px}
.panel h2{margin:0 0 14px;font-size:15px;font-weight:650;letter-spacing:.01em}
.count,.dep{color:var(--muted);font-weight:400;font-size:12px}
.dep{font-style:italic}
.note{color:var(--muted);font-size:12.5px;margin:10px 0 0}
.tbl-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  position:sticky;top:0;background:var(--panel);white-space:nowrap}
tbody th{font-weight:600}
.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.kv td{font-variant-numeric:tabular-nums}
.self-row{background:var(--accent-dim)} .self-row th{color:var(--accent)}
.note-cell{color:var(--muted);max-width:360px}
.bar{position:relative;background:var(--panel-2);border-radius:6px;height:20px;min-width:120px;overflow:hidden}
.bar-fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent);opacity:.55}
.bar-label{position:relative;padding:0 8px;font-family:var(--mono);font-size:12px;line-height:20px}
.chip{display:inline-block;padding:1px 9px;border-radius:999px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.chip.v-correct{background:color-mix(in srgb,var(--ok) 22%,transparent);color:var(--ok)}
.chip.v-partial{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn)}
.chip.v-incorrect{background:color-mix(in srgb,var(--bad) 22%,transparent);color:var(--bad)}
/* Not graded is not a failure: muted, so a retrieval-only run does not read
   as a wall of red. */
.chip.v-notjudged{background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--muted)}
.ledger td.q,.ledger td.gt,.ledger td.ans,.ledger td.rz{max-width:280px;white-space:pre-wrap;word-break:break-word}
.ledger td.rz{color:var(--muted);font-size:12px}
.ledger tr.v-correct td:first-child{box-shadow:inset 3px 0 var(--ok)}
.ledger tr.v-partial td:first-child{box-shadow:inset 3px 0 var(--warn)}
.ledger tr.v-incorrect td:first-child{box-shadow:inset 3px 0 var(--bad)}
.ftr{max-width:1180px;margin:0 auto;padding:8px 24px 40px;color:var(--muted);font-size:12px}
</style>"#;
