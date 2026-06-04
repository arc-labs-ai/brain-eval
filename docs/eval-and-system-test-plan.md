# brain-eval — Evaluation, Scale-Run & System-Test Plan

**Status:** DRAFT — awaiting owner approval before any implementation.
**Owns:** all black-box evaluation of Brain — quality, performance-at-scale,
durability/soak, and in-depth system/scenario tests — plus the v1.0
acceptance scale-run (closes brain task **#85**).

Authoritative targets: **`brain-db` `spec/19_benchmarks/`** (combined gate
`06_complete_acceptance.md`, perf `02_performance_targets.md`, recall
`03_recall_quality.md`, methodology `04`). brain-eval implements; the spec
defines. (brain-eval's README still cites the old `spec/16_...` path — fix
in Phase 0.)

---

## 0. Why this exists (separation of concerns)

| Repo | Owns | Does NOT own |
|---|---|---|
| **brain-db** (the DB) | unit tests, **in-crate criterion micro-benches** (code-coupled regression detectors, no server), fast in-process integration tests, CI correctness gates | black-box evals, scale-runs, soak, quality scoring |
| **brain-eval** (this repo) | everything black-box: drives a **real running brain-server over the wire (SDK)** — quality evals, perf/scale, durability/soak, system scenarios, the acceptance scale-run, its **own** CI | the substrate source; never edits brain-db code |

**Hard rule** (carried from prior guidance): no cross-repo CI / `just` / path
deps *from* brain-db *into* brain-eval. Dependency direction is one-way:
brain-eval → brain-db (eval depends on the thing it evaluates). Eval
drift-guards live here.

---

## 1. The boundary — what moves, what stays

| Thing | Home | Rationale |
|---|---|---|
| In-crate criterion micro-benches (`relation_traverse`, `lexical_retrieve`, `frame_codec`, `crc32c`, `schema_ops`, …) | **stay in brain-db** | bench internal Rust APIs directly, no server; CI regression detectors belong with the code. (DECISION D1 — see §8) |
| `nightly-perf.yml` (runs those micro-benches) | **stays in brain-db** | per-commit micro-regression guard on x86 |
| `acceptance.sh` CI gates 1–4, 6, 9, 10 (`cargo test`, fuzz, docs) | **stay in brain-db** | they're workspace tests |
| `acceptance.sh` gate 5 (perf **at scale**), gate 7 (soak), full chaos, recall-quality | **move to brain-eval** | need a running server + reference hardware + datasets |
| Quality evals (smoke / dmr / longmemeval / locomo) | **already brain-eval** | ✓ |
| Scale-run (100K/1M ingest + latency/throughput/recall/storage) | **build in brain-eval** | the missing #85 harness |
| In-depth system/scenario tests (multi-verb, multi-agent, txn, subscribe, restart-recovery, backfill, schema on/off, chaos) | **build in brain-eval** | black-box, via SDK |
| 48h soak | **build in brain-eval** | operator-tier |

Net: very little *moves* (mostly the perf-at-scale / soak *responsibility*
shifts out of `acceptance.sh`). The bulk is *new* black-box harness built in
brain-eval. brain-db stays lean; its `acceptance.sh` keeps only the
CI-runnable tiers and points operators at brain-eval for the scale-run.

---

## 2. brain-eval architecture — 3 pillars

Existing layout (`core/ run/ score/ report/ datasets/`) stays. Add three
capability areas:

### Pillar A — Quality (extend what exists)
- LLM-as-judge wired through the SDK (`live-llm`) — required for
  cross-comparable LongMemEval / LoCoMo numbers.
- **Recall@K correctness** (`score/recall_exhaustive.rs`): HNSW result vs
  brute-force exhaustive top-K → recall@1/10/100. Targets (spec §19/03):
  recall@10 ≥ 0.95, @1 ≥ 0.97, @100 ≥ 0.90, **at 1M @ default M=16/ef=64**.
- Exact-match + near-duplicate + post-FORGET recall checks (spec §19/03 §11/12/17).
- **Quality-regression gate**: fail if recall drops >1% absolute vs the
  stored baseline (spec §19/03 §18).

### Pillar B — Performance & Scale (new: `src/scale/`, `src/perf/`)
- **Load generator**: ingest N memories (+ statements / entities / relations)
  with realistic, structured data. Scales: 10K (CI), 100K (nightly), 1M
  (acceptance).
- **Latency probes** per verb → p50/p99, checked against spec §19/06:
  ENCODE (text/CPU) p50 ≤ 12 / p99 ≤ 25 ms; RECALL/QUERY p50 ≤ 10 / p99 ≤ 50 ms;
  entity-anchored 2-hop p50 ≤ 15 / p99 ≤ 100 ms; STATEMENT/RELATION_CREATE p50 ≤ 1 ms.
- **Throughput**: sustained ops/s per shard → ENCODE ≥ 100/s, QUERY ≥ 1K/s,
  STATEMENT_CREATE ≥ 10K/s.
- **Storage footprint**: 1M mem + 500K stmt + 10K ent + 5K rel ≤ ~10 GB.
- Emits a `ScaleReport` with per-target pass/fail; exits nonzero on any miss.

### Pillar C — System & Durability (new: `src/system/`, `tests/system/`)
Black-box scenario suites driving a real server through the SDK:
- **Multi-verb flows**: encode → extract → recall → plan → reason → forget
  cascade; verify each stage's contract end-to-end.
- **Multi-agent isolation**: BLAKE3(agent) shard routing; cross-agent leakage = fail.
- **Txn read-your-writes**: in-txn recall sees buffered writes; commit/abort semantics.
- **Subscribe change-feed**: events land for encode/forget/typed-graph ops.
- **Restart-recovery**: kill the server mid-load, restart, verify WAL replay +
  index rebuild + **no data loss** (spec §19/06 operational).
- **Backfill**: 1M backfill completes bounded + resumable on interrupt.
- **Schema on/off transitions** (spec §19/06).
- **Chaos**: kill-during-write, bit-flip surface (drives the server; complements
  brain-db's in-crate chaos tests).
- **48h soak**: steady mixed workload; watch RSS, queue depths, recall drift,
  `brain_hnsw_recall_estimate`.

---

## 3. The server harness (key new infra) — `src/run/server.rs`

A `ServerHandle` that gives every suite a real server to talk to:
- **Local / CI mode**: `docker run brain:<tag>` (the production image) with the
  bind-mounted models + `seccomp=unconfined`, expose the data plane, wait
  `/healthz`, hand back an endpoint. Mirrors brain-db's `just serve-local`.
- **External mode**: `BRAIN_EVAL_ENDPOINT` points at an already-running server
  (the reference-hardware path) — harness skips boot, just connects.
- Lifecycle: start → wait healthy → run suite → scrape `:9091/metrics` (assert
  `brain_*` taxonomy consistency) → stop. Restart-recovery suites drive
  start/kill/restart explicitly.

---

## 4. spec §19 → suite mapping (coverage matrix)

| §19/06 acceptance area | Pillar / suite |
|---|---|
| Functional: schema / entity / statement / relation / extraction / query / provenance | C — system scenarios (one suite per op family) |
| Latency (P50/P99) | B — latency probes |
| Throughput | B — throughput |
| Storage (1M ≤ 10GB, HNSW top-K < 5ms @100K entities) | B — storage + entity-HNSW probe |
| Operational (shutdown, restart, backfill, metrics, audit) | C — durability |
| Schema on/off transitions | C — transitions |
| Recall quality (recall@K) | A — recall_exhaustive |
| Docs acceptance | stays in brain-db (link-check) |

Every §19/06 checkbox maps to exactly one brain-eval suite (full matrix
fleshed out in Phase 1).

---

## 5. The v1.0 acceptance scale-run (#85) — the deliverable

One command on reference hardware (16c / 64 GB / NVMe / Linux 6.6+):

```
brain-eval acceptance --scale 1m [--endpoint host:port | --boot-image brain:tag]
```

Orchestrates: boot/connect → ingest 1M → latency + throughput + recall@K +
storage → operational (restart, backfill) → write `acceptance-report.{json,txt}`
with per-gate pass/fail vs spec thresholds → exit nonzero on any miss. This
report is the artifact that closes #85. Soak is a separate `--soak 48h` run.

---

## 6. brain-eval CI (its own — never in brain-db)

- **PR**: unit + parser/golden tests + `smoke` eval against an ephemeral
  dockerized server. Fast.
- **Nightly**: full quality (longmemeval/locomo, LLM judge) + **100K** scale +
  latency gate, on an x86 runner.
- **Release / manual**: full **1M** scale-run + 48h soak on reference hardware.

---

## 7. Phasing (incremental, each independently shippable)

- **Phase 0 — wiring & boundary.** Fix `Cargo.toml` deps (see §8 D4): point the
  SDK dep at the real SDK repo/crate; reduce brain-core/brain-protocol direct
  deps toward SDK-only. Fix README `spec/16→spec/19`. Write the boundary note
  into brain-db `acceptance.sh` (point operators here for scale/soak).
- **Phase 1 — server harness** (`ServerHandle`, docker-boot + external mode) +
  the §4 coverage matrix.
- **Phase 2 — Pillar B** perf/scale (latency + throughput + storage; 100K first,
  1M behind a flag) with spec-threshold gates.
- **Phase 3 — Pillar C** system scenarios + restart-recovery.
- **Phase 4 — Pillar A** recall@K exhaustive + LLM judge + quality-regression gate.
- **Phase 5 — acceptance orchestrator** (`acceptance --scale 1m`) + reference-hw
  runbook + brain-eval CI.
- **Phase 6 — soak** (48h) harness.

Each phase: research → plan note → implement → verify (smoke against a
dockerized server) → commit in brain-eval.

---

## 8. Decisions — RESOLVED (owner, this session)

- **D1 = keep in-crate micro-benches in brain-db.** No benches move.
- **D2 = dockerized `brain:<tag>` image** for the eval server harness.
- **D3 = full plan in order**, Phase 0 → 6.
- **D4 = SDK-only deps** — fix the stale SDK dep; drop direct brain-core/
  brain-protocol where the SDK re-exports suffice.

### Original options (for the record)

- **D1 — micro-benches**: keep in-crate criterion benches in brain-db
  (recommended) vs relocate all benches to brain-eval. *Recommend keep* — they
  test internal APIs and are CI regression detectors; relocating forces
  brain-eval to depend on every internal crate + replicate fixtures.
- **D2 — server boot for eval**: dockerized `brain:<tag>` image (recommended) vs
  release binary on Linux vs external-endpoint-only. *Recommend docker image* —
  matches `just serve-local`, works the same in CI and locally.
- **D3 — first focus**: build the full plan in order, or jump to the scale-run
  (#85) + system tests first (Pillars B+C) and backfill quality (A) after.
- **D4 — dependency coupling**: depend on **SDK only** (recommended; fix the
  stale `brain-sdk-rust = brain-db-io/brain-db-io` dep → the real
  `brain-db-sdk` crate, drop direct brain-core/brain-protocol where the SDK
  re-exports suffice) vs keep direct substrate deps.
