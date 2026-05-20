# brain-eval

Evaluation, benchmarking, and report generation for [Brain](https://github.com/brain-db-io/brain) — the cognitive substrate for AI agents.

A client-side rig that talks to a running `brain-server` over the wire (via `brain-sdk-rust`), drives a benchmark dataset through the cognitive ops loop (ENCODE → RECALL), judges answers against ground truth, and produces a `BenchmarkReport` in JSON + plain-text form.

## What this crate is

| Layer | Purpose |
|---|---|
| `Benchmark` trait | One impl per dataset. `DmrBenchmark` ships first; LongMemEval / LoCoMo / BEAM are follow-ups. |
| `BrainEvalHarness` | Wraps `brain_sdk_rust::Client` with `ingest()` + `recall()` helpers. One harness = one `AgentId` = full data-level isolation from other questions. |
| `EvalRunner` | Loops over instances, drives the harness, calls the judge, aggregates metrics, writes reports. |
| Heuristic judge | Substring + token-overlap scoring against ground truth. Honest for fact benchmarks; a directional signal for multi-hop. |
| `live-llm` feature | Stub for LLM-as-judge — wires in once `brain-sdk-rust` grows the surface (follow-up). |

## Status

| Component | State |
|---|---|
| Crate scaffold + folder layout | ✅ |
| `Benchmark` trait + `EvalInstance` | ✅ |
| Metrics: Recall@K, NDCG@K, latency, tokens, write quality | ✅ |
| Heuristic judge | ✅ |
| Answer synthesis (top-K concatenation) | ✅ |
| `BrainEvalHarness` (remote SDK) | ✅ |
| `EvalRunner` | ✅ |
| Reporters: JSON + text | ✅ |
| `ScriptedLlm` mock | ✅ |
| Deterministic fixtures | ✅ |
| `DmrBenchmark` loader | ✅ |
| `LongMemEvalS` loader (500 questions, ICLR 2025) | ✅ |
| `LocomoBenchmark` loader (~1540 questions, ACL 2024) | ✅ |
| LongMemEval-S + LoCoMo competitor baselines | ✅ |
| Unit tests (32) | ✅ |
| `tests/basic_e2e.rs` integration test | ✅ (gated; needs running server) |
| BEAM loader (1M–10M scale) | 🟡 follow-up |
| LLM judge (`live-llm`) wired through SDK | 🟡 follow-up |
| HTML reporter | 🟡 follow-up |
| Criterion benches (`hybrid_vs_substrate`, etc.) | 🟡 follow-up |
| `scripts/download_datasets.sh` | 🟡 follow-up |
| `run-report.sh` driver | 🟡 follow-up |
| Spec §16/02 latency gate | 🟡 follow-up |

## Repo layout expected

This crate has path dependencies on three crates in the upstream `brain` source tree. The on-disk layout the manifest expects:

```
~/Desktop/
├── brain/                         # the substrate source tree (separate repo)
│   └── crates/
│       ├── brain-core/
│       ├── brain-protocol/
│       └── brain-sdk-rust/
└── brain-db-io/
    └── brain-eval/                # ← this repo
```

If the upstream `brain` repo lives somewhere else on your machine, edit the three `path = "../../brain/..."` entries in `Cargo.toml` or set up a symlink at `../../brain` pointing at your checkout. Once `brain-db-io/brain` is published publicly, the Cargo.toml comments document the swap to git deps + `[patch]` override.

## Quick start

### Build

```bash
cd brain-eval
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

### Run a benchmark against a live server

```bash
# 1. Start a brain-server (Linux only — see "Limitations" below):
cargo run --bin brain-server --manifest-path ../../brain/Cargo.toml

# 2. Point the eval at your datasets directory:
export BRAIN_EVAL_DATASETS_DIR=/path/to/your/datasets

# 3. Drive the integration test (one ingest + recall round-trip):
BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 \
  cargo test --test basic_e2e -- --ignored --nocapture
```

---

## Running LongMemEval-S end to end

This is the fully wired path — one dataset, runnable today. Start to finish:

### Step 1 — start a brain-server (Linux host)

```bash
# In one terminal — default dev endpoint is 127.0.0.1:9090
cargo run --release --bin brain-server \
  --manifest-path ../../brain/Cargo.toml
```

Keep this running. The eval connects to it over the wire (same path a real client would use).

### Step 2 — download the dataset

```bash
# In a second terminal
export BRAIN_EVAL_DATASETS_DIR="$HOME/.brain-eval-datasets"
./scripts/download_longmemeval.sh
```

Result: `$BRAIN_EVAL_DATASETS_DIR/longmemeval/longmemeval_s.json` (~200 MB). The script is a no-op if the file is already present; delete it to force-redownload.

If the URL has moved, set `BRAIN_EVAL_LONGMEMEVAL_URL` to override.

### Step 3 — run a smoke pass (10 questions)

```bash
export BRAIN_EVAL_ENDPOINT=127.0.0.1:9090
export BRAIN_EVAL_MAX_QUESTIONS=10        # smoke cap; remove to run the full 500
export BRAIN_EVAL_TOP_K=10

cargo run --release --example run_longmemeval
```

Expected stdout shape:

```
brain-eval :: LongMemEval-S
  endpoint    : 127.0.0.1:9090
  max_q       : 10
  top_k       : 10
  output_dir  : target/eval-reports

=== LongMemEval-S — heuristic judge ===
instances          : 10
accuracy           : 0.3500
  correct/partial/incorrect : 2/3/5
ingestion errors   : 0    retrieval errors : 0
write p50/p95 (ms) : 80/120     read p50/p95 (ms) : 4/8
Recall@5 / @10     : 0.4000 / 0.5000

Reports written under: (filename stem: longmemeval-s-<unix_nanos>)
Tip: numbers from a heuristic judge are directional. Wire the LLM
     judge before quoting them in a comparison.
```

### Step 4 — inspect the report

```bash
# JSON sidecar (full per-question detail)
ls target/eval-reports/longmemeval-s-*.json | tail -1 | xargs cat | jq '.meta, .metrics'

# Plain-text summary
ls target/eval-reports/longmemeval-s-*.txt | tail -1 | xargs cat
```

### Step 5 — full run

When you're confident the smoke run is healthy, drop the cap and re-run:

```bash
unset BRAIN_EVAL_MAX_QUESTIONS
cargo run --release --example run_longmemeval
```

A full 500-question pass with the heuristic judge takes 5–15 minutes depending on hardware (dominated by ingest time — each question carries its own haystack).

### What the numbers mean today

- `meta.judge_type = "heuristic"` — substring + token-overlap scoring. Cross-comparable for `single-session-user` and `single-session-preference` questions; **directional only** for `multi-session`, `temporal-reasoning`, `knowledge-update`, `abstention`. Wire the LLM judge before quoting these numbers against [Wu et al.'s published Table 4](https://arxiv.org/abs/2410.10813) or competitor baselines.
- `metrics.tokens.*` are all `0` — Brain's wire doesn't surface per-request token counts to the SDK yet. Honest zeros, not invented numbers.
- `metrics.write_quality.*` reflects the substrate's fingerprint dedupe behaviour (driven by `EncodeBuilder::deduplicate(true)` in the harness).

### Sanity-check without a live server

Two paths, depending on how much of the pipeline you want to exercise:

**Parser-only golden test** — runs against the 5-row smoke fixture, asserts the loader produces the right `EvalInstance` shape per question type:

```bash
cargo test --test longmemeval_loader
```

~10 ms.

**Full-pipeline validator with mocked recall** — runs parser → synthesize → judge → metrics → reporters end-to-end against the smoke fixture, substituting a deterministic mock RECALL response for the wire roundtrip. Produces a real `BenchmarkReport` (JSON + text) under `target/eval-reports/`:

```bash
cargo run --example validate_pipeline
```

Output is a 5-instance benchmark report — same shape and code paths a Linux run would produce, just with mocked retrieval. Useful on macOS / non-Linux dev hosts where `brain-server` exits as a stub (Glommio + io_uring require Linux).

Sample headline numbers on the smoke fixture:

```
instances         : 5
accuracy          : 0.8000  (4 correct, 0 partial, 1 incorrect)
write p50/p95 ms  : 59/59
read p50/p95 ms   : 7/7
Recall@5 / @10    : 1.0000 / 1.0000
NDCG@5 / @10      : 0.8155 / 0.8155
Per-dimension     : single_hop 1.0, preference 1.0, multi_hop 1.0,
                    knowledge_update 1.0, abstention 0.0
```

The abstention miss is genuine: the heuristic synthesizer can't refuse on its own — even with no relevant hit it returns content. That's an honest demonstration of why the LLM-judge follow-up matters for abstention-heavy benchmarks.

---

A full DMR / LoCoMo run follows the same shape; the dataset-specific `_competitor_baselines()` functions in `src/report/baselines.rs` are the substitution point for those benchmarks.

## Architecture

Six top-level folders, each answering one question — pipeline order from top to bottom:

```
crates/brain-eval/
├── src/
│   ├── lib.rs                     — module declarations only
│   │
│   ├── core/                      — what types does eval revolve around?
│   │   ├── benchmark.rs           — Benchmark trait + EvalError
│   │   ├── instance.rs            — EvalInstance, Session, TurnRecord, QuestionType
│   │   └── outcome.rs             — QuestionResult, JudgeResult, Verdict
│   │
│   ├── run/                       — how does a run happen?
│   │   ├── config.rs              — RunConfig + env vars + ReporterKind
│   │   ├── harness.rs             — BrainEvalHarness (wraps brain-sdk-rust)
│   │   ├── synthesize.rs          — top-K → candidate answer
│   │   └── runner.rs              — EvalRunner orchestration
│   │
│   ├── score/                     — how do we score answers?
│   │   ├── judge.rs               — heuristic judge (LLM judge follow-up)
│   │   ├── metrics.rs             — EvalMetrics, compute_full_metrics
│   │   ├── retrieval.rs           — RetrievalStats, Recall@K, NDCG@K
│   │   └── latency.rs             — LatencyStats, percentile helpers
│   │
│   ├── report/                    — what does the output look like?
│   │   ├── shape.rs               — BenchmarkReport, BenchmarkMeta, CompetitorRow
│   │   ├── baselines.rs           — *_competitor_baselines() per benchmark
│   │   └── format/                — output writers
│   │       ├── json.rs
│   │       └── text.rs            — (html.rs lands here later)
│   │
│   ├── datasets/                  — which benchmarks can we load?
│   │   ├── dmr.rs                 — DMR (MemGPT 2023)
│   │   ├── longmemeval.rs         — LongMemEval-S (ICLR 2025)
│   │   └── locomo.rs              — LoCoMo (ACL 2024)
│   │
│   └── testing/                   — mocks + fixtures for tests
│       ├── scripted_llm.rs
│       └── fixtures.rs
│
└── tests/basic_e2e.rs             — tier-1 round-trip (ignored by default)
```

Pipeline reads top-to-bottom: define types in `core/`, run with `run/`, score with `score/`, present with `report/`. `datasets/` and `testing/` are the supporting cast.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BRAIN_EVAL_ENDPOINT` | `127.0.0.1:7878` | Brain-server endpoint. |
| `BRAIN_EVAL_DATASETS_DIR` | _(unset)_ | Root containing `dmr/dmr.jsonl`, etc. |
| `BRAIN_EVAL_MAX_QUESTIONS` | _(unset = all)_ | Smoke-mode cap. |
| `BRAIN_EVAL_TOP_K` | `10` | `top_k` passed to RECALL. |
| `BRAIN_EVAL_OUTPUT_DIR` | `target/eval-reports` | Where reports land. |
| `BRAIN_EVAL_FORMATS` | `json,text` | Comma list of `json` / `text`. |

## Dataset layout

Every loader expects a file under `$BRAIN_EVAL_DATASETS_DIR/<dataset>/`.

### DMR — `dmr/dmr.jsonl`

One JSON object per line:

```jsonc
{
  "id": "dmr-001",
  "question": "What is the user's favourite colour?",
  "answer": "blue",
  "conversation_id": "conv-007",
  "sessions": [
    {
      "session_id": "s1",
      "turns": [
        {"role": "user",      "content": "My favourite colour is blue."},
        {"role": "assistant", "content": "Got it."}
      ]
    }
  ]
}
```

### LongMemEval-S — `longmemeval/longmemeval_s.json`

JSON array (or JSONL — loader auto-detects). Each row matches the LongMemEval release format with `haystack_sessions` and a `question_type` tag:

```jsonc
{
  "question_id": "lme-001",
  "question_type": "multi-session",
  "question": "When did the user move to Berlin?",
  "answer": "March 2024",
  "haystack_sessions": [
    {
      "session_id": "s-1",
      "turns": [{"role": "user", "content": "I'm planning a move."}]
    }
  ]
}
```

Tag mapping: `single-session-user|single-session-assistant` → `SingleHop`, `single-session-preference` → `Preference`, `multi-session` → `MultiHop`, `temporal-reasoning` → `Temporal`, `knowledge-update` → `KnowledgeUpdate`, `abstention` → `Abstention`.

### LoCoMo — `locomo/locomo10.json`

JSON array. Each sample carries one conversation + many QA pairs sharing it. The loader expands each sample into N `EvalInstance`s with a common `conversation_id` so the runner ingests the conversation once.

```jsonc
[
  {
    "sample_id": "sample-0",
    "conversation": {
      "session_1": [
        {"speaker": "Alice", "text": "Hi Bob.", "date_time": "2024-01-01"},
        {"speaker": "Bob",   "text": "Hey."}
      ]
    },
    "qa": [
      {"question": "Who greeted whom?", "answer": "Alice greeted Bob.", "category": 1}
    ]
  }
]
```

Category mapping: `1 → SingleHop`, `2 → MultiHop`, `3 → Temporal`, `4 → Other`, `5 → Adversarial`. **Adversarial (category 5) is included in the denominator** per the standard protocol — see `datasets/locomo.rs` for the methodology footnote.

## Design notes

- **Per-question agent isolation.** Each `EvalRunner` group spins up a fresh `BrainEvalHarness` with a fresh `AgentId`. Brain routes by `BLAKE3(agent_uuid) mod shard_count`, so two harnesses with different agent ids touch independent slices of substrate state. This is the natural isolation primitive for benchmark questions — no "scope" or "namespace" wrapper needed.
- **Honest numbers.** The heuristic judge is the only judge wired today, and we say so in `BenchmarkMeta.judge_type = "heuristic"`. Numbers published with the heuristic judge are a directional signal; cross-comparable LongMemEval numbers need the LLM judge (follow-up).
- **No invented features.** Brain's wire today doesn't surface per-request token counts to the SDK, so `tokens_write` and `tokens_read` are zeros until the wire grows the fields. Better an honest zero than an invented number.
- **Single-shard scope.** Each connection is pinned to one shard at AUTH (spec §12/02). Multi-shard agents are v2; this eval crate runs the v1 single-shard contract.

## Follow-ups

In rough priority order:

1. **LLM judge wired through `brain-sdk-rust`.** Brain serves LLM extractor calls internally; expose a `judge_with_llm` op so eval can use the same provider without bringing in a separate Anthropic/OpenAI SDK. **This is the next high-leverage piece — both LongMemEval and LoCoMo `requires_synthesis() == true`, so without it published numbers are heuristic-only and not cross-comparable.**
2. **Token accounting on the wire.** Add `prompt_tokens` / `completion_tokens` to `EncodeResponse` and `RecallResponseFrame` (or carry them on a side-channel telemetry frame). Wire VERSION bump.
3. **BEAM loader.** 1M–10M scale benchmark; pairs with criterion runs.
4. **HTML reporter.** Self-contained dark/light-mode report with Chart.js latency/accuracy plots.
5. **`scripts/download_datasets.sh`** + **`run-report.sh`** + per-benchmark cargo aliases.
6. **Spec §16/02 latency gate.** Make CI fail when p99 RECALL > target.
7. **Hybrid-vs-substrate criterion bench.** Side-by-side latency + accuracy comparison; the Brain-unique eval axis.

Authoritative source: `spec/16_benchmarks_acceptance/` — read before adding any new benchmark dimension.
