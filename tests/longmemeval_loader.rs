//! Golden parser test — runs the [`LongMemEvalS`] loader against a
//! small fixture file shipped in `tests/fixtures/`. Doesn't need a
//! live server, doesn't download anything, runs in milliseconds.
//!
//! What it proves:
//!
//! 1. The loader resolves `longmemeval/longmemeval_s.json` under the
//!    given datasets dir.
//! 2. Every documented `question_type` tag maps to the right
//!    [`QuestionType`] variant.
//! 3. Sessions + turns survive the round-trip with structure intact.
//! 4. Abstention rows (empty answer) parse without error.

use std::path::PathBuf;

use brain_eval::core::benchmark::Benchmark;
use brain_eval::core::instance::QuestionType;
use brain_eval::datasets::longmemeval::LongMemEvalS;

/// Stage the smoke fixture under a temp `BRAIN_EVAL_DATASETS_DIR`-shaped
/// directory and return the path. Each test gets its own temp dir.
fn stage_smoke_fixture() -> tempdir::TempDir {
    let dir = tempdir::TempDir::new("brain-eval-lme").expect("temp dir");
    let nested = dir.path().join("longmemeval");
    std::fs::create_dir_all(&nested).expect("mkdir");

    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("longmemeval_smoke.json");
    let dst = nested.join("longmemeval_s.json");
    std::fs::copy(&src, &dst).expect("copy fixture");
    dir
}

#[test]
fn loads_smoke_fixture_into_five_instances() {
    let dir = stage_smoke_fixture();
    let instances = LongMemEvalS
        .load(dir.path())
        .expect("smoke fixture must parse");
    assert_eq!(instances.len(), 5, "fixture has 5 rows");
}

#[test]
fn question_type_tags_map_correctly() {
    let dir = stage_smoke_fixture();
    let instances = LongMemEvalS.load(dir.path()).expect("parse");

    // Lookup by question_id; we don't rely on order beyond the fixture
    // being well-formed.
    let by_id: std::collections::HashMap<&str, QuestionType> = instances
        .iter()
        .map(|i| (i.question_id.as_str(), i.question_type))
        .collect();

    assert_eq!(by_id["smoke-001"], QuestionType::SingleHop);
    assert_eq!(by_id["smoke-002"], QuestionType::Preference);
    assert_eq!(by_id["smoke-003"], QuestionType::MultiHop);
    assert_eq!(by_id["smoke-004"], QuestionType::KnowledgeUpdate);
    assert_eq!(by_id["smoke-005"], QuestionType::Abstention);
}

#[test]
fn sessions_and_turns_survive_round_trip() {
    let dir = stage_smoke_fixture();
    let instances = LongMemEvalS.load(dir.path()).expect("parse");

    let multi = instances
        .iter()
        .find(|i| i.question_id == "smoke-003")
        .expect("smoke-003 present");
    assert_eq!(multi.sessions.len(), 2);
    // The real LongMemEval release ships sessions as bare turn lists with
    // no inline ids, so the loader assigns positional ids.
    assert_eq!(multi.sessions[0].session_id, "session-0");
    assert_eq!(multi.sessions[1].session_id, "session-1");

    let first_turn = &multi.sessions[1].turns[0];
    assert_eq!(first_turn.role, "user");
    assert!(first_turn.content.contains("auth-rewrite"));
}

#[test]
fn abstention_rows_carry_empty_ground_truth() {
    let dir = stage_smoke_fixture();
    let instances = LongMemEvalS.load(dir.path()).expect("parse");

    let abstain = instances
        .iter()
        .find(|i| i.question_id == "smoke-005")
        .expect("smoke-005 present");
    assert!(abstain.answer.is_empty());
    assert_eq!(abstain.question_type, QuestionType::Abstention);
}

#[test]
fn requires_synthesis_returns_true() {
    // LongMemEval expects free-form answers — the loader signals that
    // the runner should use the LLM synthesizer (once wired). The
    // heuristic synthesizer still works as a fallback today.
    assert!(LongMemEvalS.requires_synthesis());
}

// ---------------------------------------------------------------------------
// In-tree tempdir helper.
//
// brain-eval is a standalone crate; we don't pull in the `tempfile` /
// `tempdir` workspace dependency just for one test. This shim writes
// to `std::env::temp_dir()` with a randomized suffix and cleans up
// on drop.
// ---------------------------------------------------------------------------

mod tempdir {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new(prefix: &str) -> std::io::Result<Self> {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let suffix = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("{prefix}-{nanos}-{suffix}"));
            std::fs::create_dir_all(&path)?;
            Ok(Self { path })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
