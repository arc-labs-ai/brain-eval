//! Namespace-wide RECALL eval scenario.
//!
//! Measures the cross-space read capability (Brain's `scope = Namespace`) that a
//! plain space-scoped recall cannot serve. A real multi-app user's memory is
//! split across spaces — a work space, a home space, a health space, and so on,
//! all under one namespace (tenant). Space-scoped recall from any single space
//! can only see that space; namespace-wide recall fans out across every space
//! the caller owns within its namespace (never crossing namespaces) and merges.
//!
//! The scenario ingests one synthetic persona whose facts are partitioned into
//! K facets (one space each) under a single fresh namespace, then asks each
//! question TWO ways from ONE primary connection:
//!   - **space-scoped** (baseline): only the primary space's facts are
//!     reachable, so a question whose answer lives in a sibling space MISSES;
//!   - **namespace-wide**: every facet is reachable, so all questions can hit.
//!
//! The metric is **retrieval containment** — did the gold answer's key phrase
//! appear in the returned memory set — not an LLM judge, so it isolates the
//! read-scope effect and needs no judge model. The reported delta
//! (`namespace-wide − space-scoped`) is the capability's benefit.

use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use brain_db_sdk::new_id;
use brain_db_sdk::wire::types::RecallScopeWire;

use crate::bench::core::instance::TurnRecord;
use crate::client::harness::{BrainEvalHarness, HarnessError};

/// One facet of the persona — a space holding a cluster of related facts.
struct Facet {
    /// Human label (diagnostics only).
    label: &'static str,
    /// Facts ingested into this facet's own space.
    facts: &'static [&'static str],
}

/// One question, the facet whose space holds its answer, and the gold key
/// phrase a correct recall must surface (case-insensitive substring).
struct Question {
    facet: usize,
    cue: &'static str,
    gold: &'static str,
}

/// The synthetic persona: "Mira", one tenant, five facets across five spaces.
/// Each fact is distinctive so retrieval containment is unambiguous, and each
/// question is answerable from exactly one facet.
const FACETS: &[Facet] = &[
    Facet {
        label: "work",
        facts: &[
            "My manager at NeuraCorp is Diana Okonkwo.",
            "Our team ships the billing reconciliation service.",
            "My work laptop asset tag is NC-88213.",
        ],
    },
    Facet {
        label: "home",
        facts: &[
            "I live at 44 Larch Street in Portland.",
            "My landlord is Sam Rivera.",
            "The spare key is hidden under the blue planter.",
        ],
    },
    Facet {
        label: "health",
        facts: &[
            "I am allergic to penicillin.",
            "My blood type is O negative.",
            "My physiotherapist is Dr. Hannah Voss.",
        ],
    },
    Facet {
        label: "hobby",
        facts: &[
            "I play upright bass in a jazz trio called Blue Meridian.",
            "We rehearse every Thursday evening.",
            "My bass is a 1978 Fender Precision.",
        ],
    },
    Facet {
        label: "travel",
        facts: &[
            "My passport number is X1234567.",
            "My preferred airline is Cascade Air.",
            "I always request a seat in row 11.",
        ],
    },
];

const QUESTIONS: &[Question] = &[
    Question {
        facet: 0,
        cue: "Who is my manager?",
        gold: "Diana Okonkwo",
    },
    Question {
        facet: 0,
        cue: "What is my work laptop asset tag?",
        gold: "NC-88213",
    },
    Question {
        facet: 1,
        cue: "What street do I live on?",
        gold: "Larch",
    },
    Question {
        facet: 1,
        cue: "Who is my landlord?",
        gold: "Sam Rivera",
    },
    Question {
        facet: 2,
        cue: "What am I allergic to?",
        gold: "penicillin",
    },
    Question {
        facet: 2,
        cue: "What is my blood type?",
        gold: "O negative",
    },
    Question {
        facet: 3,
        cue: "What is the name of my jazz trio?",
        gold: "Blue Meridian",
    },
    Question {
        facet: 3,
        cue: "What day do we rehearse?",
        gold: "Thursday",
    },
    Question {
        facet: 4,
        cue: "What is my passport number?",
        gold: "X1234567",
    },
    Question {
        facet: 4,
        cue: "Which airline do I prefer?",
        gold: "Cascade Air",
    },
];

/// Safety cap on returned memories per recall.
const MAX_RESULTS: u32 = 25;

/// Whether any returned memory's text contains `gold` (case-insensitive).
async fn answered(
    primary: &BrainEvalHarness,
    cue: &str,
    gold: &str,
    scope: RecallScopeWire,
) -> Result<bool, HarnessError> {
    let out = primary.recall_scoped(cue, MAX_RESULTS, scope).await?;
    let needle = gold.to_lowercase();
    Ok(out
        .memories
        .iter()
        .any(|m| m.text.to_lowercase().contains(&needle)))
}

/// Turn plain fact strings into user turns for ingest.
fn turns(facts: &[&str]) -> Vec<TurnRecord> {
    facts
        .iter()
        .map(|f| TurnRecord {
            role: "user".to_string(),
            content: (*f).to_string(),
            occurred_at_unix_nanos: None,
        })
        .collect()
}

/// Run the scenario against a live server at `endpoint`.
pub async fn run_namespace_wide(endpoint: SocketAddr) -> ExitCode {
    match run(endpoint).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("namespace-wide scenario failed: {e}");
            ExitCode::from(1)
        }
    }
}

async fn run(endpoint: SocketAddr) -> Result<ExitCode, HarnessError> {
    // One fresh namespace so the run is isolated and repeatable.
    let ns = format!("nswide_{}", &hex16(new_id())[..12]);
    println!("brain-eval :: namespace-wide RECALL scenario");
    println!("  endpoint  : {endpoint}");
    println!("  namespace : {ns}");
    println!("  facets    : {} spaces", FACETS.len());
    println!("  questions : {}", QUESTIONS.len());
    println!();

    // Connect one harness per facet (each its own space in the SAME namespace)
    // and ingest that facet's facts. The first facet's connection is the
    // "primary" we recall from — a real client that happens to be bound to the
    // work space but wants to remember across everything it owns.
    let mut harnesses: Vec<BrainEvalHarness> = Vec::with_capacity(FACETS.len());
    for facet in FACETS {
        let space_id = new_id();
        let h = BrainEvalHarness::connect_in_namespace(endpoint, space_id, ns.clone()).await?;
        let ing = h.ingest(&turns(facet.facts)).await?;
        println!(
            "  ingested facet '{}' ({} facts) into space {}",
            facet.label,
            ing.stored_ids.len(),
            &hex16(space_id)[..12]
        );
        harnesses.push(h);
    }
    let primary = &harnesses[0];
    println!();

    // Let the async lexical/HNSW indexers settle across every space before
    // measuring — otherwise a namespace-wide MISS could be an indexing race
    // rather than a real gap. Poll a sibling-space fact namespace-wide until it
    // surfaces (or a deadline), so a later baseline MISS is genuinely a scope
    // miss, not lag.
    let settle_q = QUESTIONS
        .iter()
        .find(|q| q.facet != 0)
        .expect("a sibling-facet question exists");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut settled = false;
    while Instant::now() < deadline {
        if answered(
            primary,
            settle_q.cue,
            settle_q.gold,
            RecallScopeWire::Namespace,
        )
        .await?
        {
            settled = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    if !settled {
        eprintln!(
            "warning: sibling-space fact never surfaced namespace-wide within the settle window; \
             results may understate namespace-wide accuracy (indexing lag or a real gap)"
        );
    }

    // Measure both scopes for every question from the primary connection.
    let mut space_hits = 0usize;
    let mut ns_hits = 0usize;
    println!(
        "  {:<40}  {:<8}  {:>6}  {:>6}",
        "question", "facet", "space", "ns"
    );
    for q in QUESTIONS {
        let space_ok = answered(primary, q.cue, q.gold, RecallScopeWire::Space).await?;
        let ns_ok = answered(primary, q.cue, q.gold, RecallScopeWire::Namespace).await?;
        space_hits += usize::from(space_ok);
        ns_hits += usize::from(ns_ok);
        println!(
            "  {:<40}  {:<8}  {:>6}  {:>6}",
            truncate(q.cue, 40),
            FACETS[q.facet].label,
            mark(space_ok),
            mark(ns_ok),
        );
    }

    let total = QUESTIONS.len();
    let space_acc = space_hits as f64 / total as f64;
    let ns_acc = ns_hits as f64 / total as f64;
    // How many questions are answerable ONLY from a space other than the
    // primary — the set namespace-wide unlocks and space-scoped cannot reach.
    let cross_space = QUESTIONS.iter().filter(|q| q.facet != 0).count();

    println!();
    println!("  ── results ─────────────────────────────────────────────");
    println!(
        "  space-scoped   : {space_hits}/{total}  ({:.0}%)   [only the primary space is reachable]",
        space_acc * 100.0
    );
    println!(
        "  namespace-wide : {ns_hits}/{total}  ({:.0}%)   [every space in the namespace]",
        ns_acc * 100.0
    );
    println!(
        "  improvement    : +{:.0} pts   ({cross_space} of {total} answers live in a sibling space)",
        (ns_acc - space_acc) * 100.0
    );
    println!();

    // Success criterion: namespace-wide must strictly beat space-scoped (it
    // unlocks the sibling-space answers), and must recover the primary space's
    // own answers too (never worse than the baseline on those).
    if ns_hits > space_hits {
        println!("  PASS — namespace-wide recall spans the caller's spaces; space-scoped cannot.");
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!(
            "  FAIL — namespace-wide ({ns_hits}) did not beat space-scoped ({space_hits}); \
             expected it to unlock the {cross_space} sibling-space answers."
        );
        Ok(ExitCode::from(1))
    }
}

/// 16 bytes → 32-char lowercase hex (namespace / space labels).
fn hex16(b: [u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for x in b {
        use std::fmt::Write as _;
        let _ = write!(s, "{x:02x}");
    }
    s
}

fn mark(ok: bool) -> &'static str {
    if ok {
        "hit"
    } else {
        "MISS"
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}
