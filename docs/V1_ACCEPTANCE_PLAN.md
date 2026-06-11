# brain-eval — v1.0 acceptance build-out plan

**Status:** DRAFT — awaiting approval. Authored 2026-06-11 from the production-readiness audit.

## Why this exists

brain-eval today is a solid **cognitive-primitive + smoke** harness: it exercises
ENCODE / RECALL / FORGET / TXN with real assertions, a known-answer recall@K gate,
a regression baseline, and a docker-boot/external `ServerHandle`. Nothing is an
assert-nothing scaffold.

But the v1.0 release is gated on `brain/spec/19_benchmarks/06_complete_acceptance.md`,
and **the bulk of that document — the entire typed-graph functional surface — has
zero coverage here.** Running the current suite green certifies only the
cognitive-primitive slice. This plan closes the gap so that "acceptance green"
actually means "met the spec's v1.0 bar."

## Coverage gap (from the audit)

| Spec §19/06 group | Today | Gap |
|---|---|---|
| Schemaless primitives (encode/recall/forget) | ✅ real | — |
| Schema ops (upload/get/validate/version/breaking-change) | ❌ none | SDK has the verbs; eval never calls them |
| Entity ops (create/resolve tiers/merge/unmerge/rename/tombstone) | ❌ none | unused |
| Statement ops (Fact/Pref/Event, supersession, contradiction, retract, history) | ❌ none | unused |
| Relation ops (create, cardinality, symmetric, traverse depth 1–3/>5, cycles) | ❌ none | unused |
| Extraction (pattern/classifier/LLM tiers, idempotency) | ❌ none | server booted with classifier+LLM **disabled** |
| Query (free-text, entity-anchored, EXPLAIN, TRACE, streaming-cancel) | ❌ none | only legacy recall path |
| Provenance/versioning (evidence, FORGET cascade, confidence agg, stale-flag) | ❌ none | — |
| Perf latency (10 verbs) | 🟡 2/10 | only ENCODE + RECALL |
| Perf throughput (concurrent, group-commit) | ❌ wrong | measured **sequentially** (1/mean-latency), not 100-client sustained |
| Storage efficiency (1M+500K ≈ 10GB, tantivy reopen, entity-HNSW 5ms, cache evict) | ❌ none | no disk-footprint measurement |
| Correctness/durability invariants | 🟡 1/7 | only clean-restart WAL replay; no kill-during-op, no idempotency, no tombstone-grace |
| Soak 48h (leak/drift/disk) | 🟡 harness only | only error-count + recall-floor; no RSS/latency-trend/disk assertions |
| CI live tier | ❌ never runs | `live` job self-skips when `brain:latest` absent; no runner/image build |

## Plan — phased, each phase independently shippable

### Phase E1 — Enable the real server profile for acceptance
- `src/run/server.rs`: stop forcing `CLASSIFIER__ENABLED=false` / `LLM__ENABLED=false`
  in the acceptance/scenario boot. Add an `AcceptanceProfile` (smoke vs full) that
  selects tiers; full enables classifier (model bind-mount) and, when an LLM key is
  present, the LLM tier. Keep a `--no-llm` escape for keyless CI.
- Acceptance: lets every downstream phase observe extraction output. Blocks E5/E6.

### Phase E2 — Typed-graph functional acceptance suite (the big one)
New module `src/system/typed_graph/` with one scenario file per group, each
asserting against the SDK verbs (`brain-db-sdk` already exposes them):
- `schema.rs` — upload valid/invalid, get, validate, version bump, breaking-change rejection.
- `entity.rs` — create, resolve (exact / alias / fuzzy / embedding tiers), merge, unmerge, rename, tombstone, list-by-type.
- `statement.rs` — Fact/Preference/Event create, supersession chain, contradiction detection, retract, history.
- `relation.rs` — create, cardinality enforcement, symmetric auto-insert, traverse depth 1/2/3, depth>5 rejection, cycle handling.
- `query.rs` — free-text QUERY, entity-anchored QUERY, EXPLAIN (contribution breakdown), TRACE, streaming + CANCEL_STREAM.
- `provenance.rs` — evidence list on statements, FORGET cascade re-derive/tombstone, confidence aggregation, stale-flag after narrowing schema.
- Each scenario returns a structured pass/fail mapped to a §19/06 criterion id.
- Wire all into `acceptance.rs::run_acceptance` behind the `full` profile.

### Phase E3 — Real concurrent throughput
- `src/scale/throughput.rs` (replace the sequential `throughput()`): N concurrent
  clients (default 100) over the connection pool, sustained for a fixed window,
  measuring achieved ops/s under group-commit. Per-verb: ENCODE, STATEMENT_CREATE,
  RELATION_CREATE, QUERY, ENTITY_RESOLVE.
- Add the missing latency verbs to `scale/mod.rs` so all 10 spec verbs are measured
  (p50/p99), plus ENCODE_VECTOR_DIRECT.
- Gate against `scale/targets.rs` thresholds (informational off reference-HW, hard on ref-HW).

### Phase E4 — Durability / kill-during-operation chaos
New `src/system/chaos.rs`:
- Kill the server (SIGKILL the container, not graceful stop) at random points during
  a write stream; on restart assert no acked write is lost and no torn/partial row
  is served (WAL-before-ack).
- Crash-before-fsync simulation where the harness can induce it (or document the
  in-crate test that must cover it if black-box can't).
- Kill during WAL replay → restart again → assert recovery idempotency.
- Run the kill-point loop ~N times (configurable; spec suggests ~1000 on ref-HW).

### Phase E5 — Invariant coverage gaps
- `src/system/idempotency.rs`: resend a request with the same RequestId, assert
  same MemoryId + single durable write (invariant #5). Also assert different-params
  + same RequestId → Conflict.
- Tombstone-grace: FORGET (soft) → assert reclamation does **not** occur before the
  grace window; hard FORGET → assert immediate zeroing/NotFound.
- Slot-version: capture a MemoryId, force a version bump, assert stale id → NotFound.

### Phase E6 — Soak hardening + 1M-scale config
- `src/soak.rs`: add RSS sampling (leak), latency-percentile trend (drift), and disk
  /WAL growth assertions to `SoakReport::healthy()`. Keep the smoke default; add a
  real long-run config.
- `src/acceptance.rs`: add a non-smoke `AcceptanceConfig::full_scale` at the spec's
  1M-memory + 500K-statement scale, with the storage-efficiency assertion
  (≈10GB footprint, tantivy reopen, entity-HNSW p99 ≤ 5ms).

### Phase E7 — CI live tier that actually runs
- `.github/workflows/ci.yml`: add an image-build step (or a self-hosted runner with
  `brain:latest`) so the `live` job runs the acceptance + scenario + soak-smoke gates
  on every push to main (not just `workflow_dispatch`).
- Persist + compare the regression baseline (`no_regression`) in CI so perf/recall
  regressions actually fail the build.

## Verification
- Each phase: `cargo test` + `cargo clippy -D warnings` in brain-eval's own CI.
- Full acceptance run is Linux + docker (servers are Linux-only); validate against a
  locally-built `brain:latest`.
- This work lives entirely in brain-eval — **no cross-repo CI/just targets added to
  the brain repo** (per the repo-separation rule).

## Sequencing / priority
E1 → E2 (largest, unblocks most §06 criteria) → E5 (cheap, high-value invariants)
→ E3 → E4 → E6 → E7. E2 and E5 are the two that most change "is this really v1.0."
