//! `topk_probe` — does the requested `top_k` change the HEAD of the
//! ranking, or only truncate it?
//!
//! Ingests LoCoMo sample 0 (conv-26, prefixed format) into ONE fresh
//! agent, then issues the SAME query at several `top_k` values and dumps
//! the first 5 returned texts for each. If the top-5 differs across
//! `top_k`, the limit is feeding back into candidate selection / ranking
//! (a read-path bug), not just truncating a stable list.
//!
//!   BRAIN_EVAL_ENDPOINT=127.0.0.1:38120 \
//!   cargo run --bin topk_probe -- ~/brain-datasets/locomo/locomo10.json

use std::net::SocketAddr;
use std::process::ExitCode;

use brain_eval::core::instance::TurnRecord;
use brain_eval::run::harness::BrainEvalHarness;

const QUERIES: &[&str] = &[
    // Cased (triggers Title-Case entity heuristic → EntityAnchored).
    "What did Caroline research?",
    // Lowercased control (no Title-Case token → Paraphrase routing).
    "what did caroline research?",
    "When did Melanie paint a sunrise?",
    "when did melanie paint a sunrise?",
];
const TOPKS: &[u32] = &[5];

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: topk_probe <locomo10.json>");
        return ExitCode::from(2);
    };
    let endpoint: SocketAddr = std::env::var("BRAIN_EVAL_ENDPOINT")
        .unwrap_or_else(|_| "127.0.0.1:38120".to_owned())
        .parse()
        .expect("endpoint");

    let bytes = std::fs::read(path).expect("read dataset");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let turns = build_prefixed(&json[0]["conversation"]);
    println!("ingesting {} prefixed turns into one fresh agent ...", turns.len());

    let h = BrainEvalHarness::connect(endpoint).await.expect("connect");
    let out = h.ingest(&turns).await.expect("ingest");
    println!("  stored={} deduped={}\n", out.stored_ids.len(), out.deduplicated);

    for q in QUERIES {
        println!("################ QUERY: {q}");
        for &k in TOPKS {
            let hits = h.recall(q, k).await.expect("recall").hits;
            let unique: std::collections::HashSet<_> =
                hits.iter().map(|m| format!("{:?}", m.memory_id)).collect();
            println!(
                "  --- top_k={k} (returned {}, {} unique ids) — first 6:",
                hits.len(),
                unique.len()
            );
            for (i, m) in hits.iter().take(6).enumerate() {
                println!(
                    "    [{}] id={:?} retr={:?}\n        {}",
                    i + 1,
                    m.memory_id,
                    m.contributing_retrievers,
                    truncate(&m.text, 70)
                );
            }
        }
        println!();
    }

    let _ = h.close().await;
    ExitCode::SUCCESS
}

fn build_prefixed(conv: &serde_json::Value) -> Vec<TurnRecord> {
    let obj = conv.as_object().expect("map");
    let mut labels: Vec<&String> = obj.keys().filter(|k| is_session(k)).collect();
    labels.sort_by_key(|k| num(k));
    let mut out = Vec::new();
    for label in labels {
        let date = obj.get(&format!("{label}_date_time")).and_then(|v| v.as_str());
        let Some(arr) = obj.get(label).and_then(|v| v.as_array()) else {
            continue;
        };
        for t in arr {
            let sp = t.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
            let tx = t.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if tx.trim().is_empty() {
                continue;
            }
            let when = t.get("date_time").and_then(|v| v.as_str()).or(date);
            let content = match when {
                Some(d) => format!("[{d}] {sp}: {tx}"),
                None => format!("{sp}: {tx}"),
            };
            out.push(TurnRecord {
                role: "user".to_owned(),
                content,
            });
        }
    }
    out
}

fn is_session(k: &str) -> bool {
    k.strip_prefix("session_")
        .is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
}
fn num(k: &str) -> u32 {
    k.strip_prefix("session_").and_then(|r| r.parse().ok()).unwrap_or(u32::MAX)
}
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        format!("{}…", &s[..max])
    }
}
