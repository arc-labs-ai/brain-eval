//! `rank_probe` — diagnostic for the "rank-depth" retrieval question.
//!
//! Ingests LoCoMo sample 0 (conv-26) TWICE into two fresh agents on the
//! same server:
//!   - `raw`      — turn text is `"<speaker>: <text>"` (no date prefix)
//!   - `prefixed` — turn text is `"[<date>] <speaker>: <text>"` (what the
//!                  eval actually sends today)
//!
//! Then for a handful of known-failing questions it RECALLs a deep window
//! and reports the rank at which the answer-bearing turn first appears in
//! each ingest. If `prefixed` ranks the answer turn materially deeper than
//! `raw`, the `[date] speaker:` boilerplate is diluting the embedding /
//! BM25 signal — a self-inflicted rank-depth cause with a clean fix.
//!
//! Run against a server (rerank-OFF isolates retrieval+RRF ranking):
//!   BRAIN_EVAL_ENDPOINT=127.0.0.1:38120 \
//!   cargo run --bin rank_probe -- ~/brain-datasets/locomo/locomo10.json

use std::net::SocketAddr;
use std::process::ExitCode;

use brain_eval::run::harness::BrainEvalHarness;
use brain_eval::core::instance::TurnRecord;

/// Deep window: we want to see the answer turn even when it's buried.
const PROBE_TOP_K: u32 = 100;

/// (question cue, distinctive lowercase substring of the answer turn).
const PROBES: &[(&str, &str)] = &[
    ("When did Melanie paint a sunrise?", "lake sunrise"),
    ("What did Caroline research?", "adoption"),
    ("When did Melanie sign up for a pottery class?", "signed up for a pottery"),
    ("What fields would Caroline be likely to pursue in her education?", "psychology"),
    ("When is Melanie planning on going camping?", "camping"),
];

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let path = match args.get(1) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: rank_probe <locomo10.json>");
            return ExitCode::from(2);
        }
    };
    let endpoint: SocketAddr = std::env::var("BRAIN_EVAL_ENDPOINT")
        .unwrap_or_else(|_| "127.0.0.1:38120".to_owned())
        .parse()
        .expect("BRAIN_EVAL_ENDPOINT must be host:port");

    let bytes = std::fs::read(&path).expect("read dataset");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse dataset");
    let sample = &json[0];
    let conv = &sample["conversation"];

    let (raw_turns, prefixed_turns) = build_turns(conv);
    println!(
        "conv-26: {} raw turns, {} prefixed turns",
        raw_turns.len(),
        prefixed_turns.len()
    );

    println!("ingesting RAW ...");
    let raw = ingest(endpoint, &raw_turns).await;
    println!("ingesting PREFIXED ...");
    let prefixed = ingest(endpoint, &prefixed_turns).await;

    println!("\n{:<60} | raw rank | prefixed rank", "question");
    println!("{}", "-".repeat(90));
    for (cue, needle) in PROBES {
        let r = answer_rank(&raw, cue, needle).await;
        let p = answer_rank(&prefixed, cue, needle).await;
        println!(
            "{:<60} | {:>8} | {:>13}",
            truncate(cue, 58),
            fmt_rank(r),
            fmt_rank(p)
        );
    }

    let _ = raw.close().await;
    let _ = prefixed.close().await;
    ExitCode::SUCCESS
}

/// Build (raw, prefixed) turn lists from the heterogeneous conversation map.
fn build_turns(conv: &serde_json::Value) -> (Vec<TurnRecord>, Vec<TurnRecord>) {
    let obj = conv.as_object().expect("conversation is a map");
    // Collect session date_times first.
    let mut raw = Vec::new();
    let mut prefixed = Vec::new();
    // Deterministic order: session_1, session_2, ...
    let mut labels: Vec<&String> = obj
        .keys()
        .filter(|k| is_session_label(k))
        .collect();
    labels.sort_by_key(|k| session_num(k));
    for label in labels {
        let date = obj
            .get(&format!("{label}_date_time"))
            .and_then(|v| v.as_str());
        let arr = match obj.get(label).and_then(|v| v.as_array()) {
            Some(a) => a,
            None => continue,
        };
        for t in arr {
            let speaker = t.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let text = t.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.trim().is_empty() {
                continue;
            }
            let when = t.get("date_time").and_then(|v| v.as_str()).or(date);
            raw.push(TurnRecord {
                role: "user".to_owned(),
                content: format!("{speaker}: {text}"),
            });
            let pfx = match when {
                Some(d) => format!("[{d}] {speaker}: {text}"),
                None => format!("{speaker}: {text}"),
            };
            prefixed.push(TurnRecord {
                role: "user".to_owned(),
                content: pfx,
            });
        }
    }
    (raw, prefixed)
}

fn is_session_label(k: &str) -> bool {
    k.strip_prefix("session_")
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

fn session_num(k: &str) -> u32 {
    k.strip_prefix("session_")
        .and_then(|r| r.parse().ok())
        .unwrap_or(u32::MAX)
}

async fn ingest(endpoint: SocketAddr, turns: &[TurnRecord]) -> BrainEvalHarness {
    let h = BrainEvalHarness::connect(endpoint)
        .await
        .expect("connect harness");
    let out = h.ingest(turns).await.expect("ingest");
    println!(
        "  stored={} attempted={} deduped={}",
        out.stored_ids.len(),
        out.attempted,
        out.deduplicated
    );
    h
}

/// First rank (1-based) at which a hit's text contains `needle`.
async fn answer_rank(h: &BrainEvalHarness, cue: &str, needle: &str) -> Option<usize> {
    let out = h.recall(cue, PROBE_TOP_K).await.ok()?;
    let needle = needle.to_ascii_lowercase();
    out.hits
        .iter()
        .position(|m| m.text.to_ascii_lowercase().contains(&needle))
        .map(|i| i + 1)
}

fn fmt_rank(r: Option<usize>) -> String {
    match r {
        Some(n) => n.to_string(),
        None => format!("MISS(>{PROBE_TOP_K})"),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        format!("{}…", &s[..max])
    }
}
