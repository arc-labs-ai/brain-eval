# brain-eval

Evaluation, benchmarking, and report generation for [Brain](https://github.com/arc-labs-ai/brain-db) — the cognitive substrate for AI agents.

A client-side rig that talks to a running `brain-server` over the wire (via `brain-db-sdk`), drives a benchmark dataset through the cognitive ops loop (ENCODE → RECALL), judges answers against ground truth, and produces a `BenchmarkReport` in JSON + plain-text form.

## What this crate is

| Layer | Purpose |
|---|---|
| `Benchmark` trait | One impl per dataset: `SmokeBenchmark`, `DmrBenchmark`, `LongMemEvalS`, `LocomoBenchmark`, plus the compiled-in `LexicalStressBenchmark` / `ParaphraseStressBenchmark` / `SupersessionStressBenchmark` (no-download anti-overfit sets). BEAM (1M–10M scale) is a follow-up. |
| `BrainEvalHarness` | Wraps `brain_db_sdk::BrainClient` with `ingest()` + `recall()` helpers. One harness = one `SpaceId` = full data-level isolation from other questions. Auth is mandatory server-side, so the harness mints its own data-plane key via `KeyMinter` before connecting. |
| `EvalRunner` | Loops over instances, drives the harness, calls the judge, aggregates metrics, writes reports. |
| Heuristic judge | Substring + token-overlap scoring against ground truth. Honest for fact benchmarks; a directional signal for multi-hop. Always available, feature-independent. |
| `live-llm` feature | A real LLM-as-judge + answer-synthesis path (`src/llm.rs`, `src/bench/score/llm_judge.rs`, `src/bench/score/judge_prompt.rs`) — not a stub. Off by default (real API calls cost money); falls back to the heuristic judge when disabled or no key is configured. |
| `scale` / `system` / `acceptance` / `soak` (under `blackbox/`) | The black-box suites beyond per-question benchmarking: load/latency probes, behavioural scenarios (isolation, durability, chaos, typed-graph), the v1.0 acceptance gate that composes them, and a sustained-workload soak harness. See [Architecture](#architecture). |

## Status

Broad strokes — see [Architecture](#architecture) for the full module map.

| Component | State |
|---|---|
| Core pipeline (`client`, `bench::core`, `bench::runner`, `bench::score`, `bench::report`, `bench::datasets`) | ✅ |
| CLI binary (`brain-eval <benchmark>`, plus `acceptance` / `soak` / `summary`) | ✅ |
| Reporters: JSON + text | ✅ |
| `SmokeBenchmark`, `DmrBenchmark`, `LongMemEvalS`, `LocomoBenchmark` loaders | ✅ |
| Compiled-in stress sets: lexical / paraphrase / supersession | ✅ |
| Heuristic judge | ✅ |
| LLM judge + answer synthesis (`live-llm` feature) | ✅ (feature-gated; needs a real API key) |
| Mandatory auth (`KeyMinter`, admin-plane key minting) | ✅ |
| `blackbox::scale` — latency + throughput load generation | ✅ |
| `blackbox::system` — behavioural + durability + chaos + typed-graph scenarios | ✅ |
| `blackbox::acceptance` — v1.0 gate composing `scale` + `system` + `client` | ✅ |
| `blackbox::soak` — sustained workload + drift sampling | ✅ |
| Tracing subscriber (installed in the binary; `RUST_LOG`-controlled) | ✅ |
| BEAM loader (1M–10M scale) | 🟡 follow-up |
| HTML reporter | 🟡 follow-up |
| Criterion benches | 🟡 follow-up |
| Spec §16/02 CI latency gate | 🟡 follow-up |

## Repo layout expected

brain-eval depends on **the SDK only** (`brain-db-sdk`) — it is a black-box
client that talks to a running `brain-server` over the wire, never the
substrate's internal crates. It declares `brain-db-sdk = "0.2"` and resolves
it from crates.io, so a bare clone of this repo builds on its own — no
sibling checkout and no `[patch.crates-io]` block required.

To develop against an unreleased SDK change, add a patch locally:

```toml
[patch.crates-io]
brain-db-sdk = { path = "../brain-sdk/rust" }
```

```
arc-labs-ai/
├── brain-sdk/                     # the SDK (separate repo)
│   └── rust/                      # the brain-db-sdk crate
└── brain-eval/                    # ← this repo
```

A patch only applies if its version *satisfies* the requirement above — if
the checkout's version does not (e.g. a `0.3.0` checkout against `"0.2"`),
Cargo silently ignores the patch, warns `patch ... was not used in the crate
graph`, and builds the registry copy instead. Keep both in step, and revert
the patch before committing.

## Authentication (mandatory)

Brain requires a per-connection API key — there is no anonymous connect,
for any run, on any judge. The eval **mints** a data-plane key for
`(namespace, space)` through the server's admin HTTP listener, then
connects with it. Every command in [Quick start](#quick-start) and
[Running LongMemEval-S end to end](#running-longmemeval-s-end-to-end)
below assumes this is set up.

- **Docker-boot runs** (the `*_live` tests via `ServerHandle::start_docker`,
  and the sibling `brain-db` repo's `just docker-serve` path) configure this
  automatically: the harness sets the server's admin secret + admin
  address and mints against the loopback admin port. Nothing to do.
- **External-server runs** (you boot `brain-server` yourself and point
  `--endpoint` / `BRAIN_EVAL_ENDPOINT` at it — the common case for the
  walkthroughs below) must set:
  - `BRAIN_EVAL_ADMIN_SECRET` — **required.** Must equal the server's own
    `[admin] token` / `BRAIN__ADMIN__TOKEN` — brain-server refuses to even
    *start* without that token configured. Without a matching
    `BRAIN_EVAL_ADMIN_SECRET`, the eval fails fast at startup
    ("preflight: cannot mint data-plane keys …") instead of silently
    recording every question as failed.
  - `BRAIN_EVAL_ADMIN_PORT` — admin listener port on the endpoint host
    (default `9092`, matching the server's default `admin_addr`).
  - `BRAIN_EVAL_NAMESPACE` — tenant the run writes under (default `eval`).

  On the *same host* as the server (both walkthroughs below), the
  server's default loopback-only admin bind is already reachable from the
  eval process, so only `BRAIN_EVAL_ADMIN_SECRET` is actually required.
  Rebinding the admin listener to a routable interface
  (`BRAIN__SERVER__ADMIN_ADDR=0.0.0.0:9092`) is only needed when the eval
  runs *outside* the server's network namespace — e.g. driving a
  dockerized server from the host.

## Judges

- **Heuristic** (default, no key required): substring + token-overlap
  scoring. Honest for fact-style answers (DMR); a directional signal only
  for free-form multi-hop answers (LongMemEval, LoCoMo).
- **LLM-as-judge** (`--features live-llm` + a provider key): a real model
  grades each answer correct / partial / incorrect — required for
  cross-comparable LongMemEval / LoCoMo numbers. Uses the **single shared
  credential** `BRAIN__LLM__API_KEY` (the same variable Brain itself
  uses, so one key drives both the server's write-time LLM and the
  eval's judge). Model resolution: `BRAIN_EVAL_JUDGE_MODEL`
  (judge-specific override) → `BRAIN__LLM__MODEL` → `gpt-4o-mini`; the
  **provider is derived from the model id** (`gpt-*` ⇒ OpenAI, `claude-*`
  ⇒ Anthropic — no separate `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). A
  transient API failure on one question falls back to the heuristic
  judge for that question only — the run never aborts.
  `BenchmarkMeta.judge_type` in the report records which judge actually
  ran (`llm:openai:<model>@<version>` / `llm:anthropic:<model>@<version>`
  vs `heuristic`), so a run that partially fell back is never misreported
  as fully LLM-judged.
- The answer **synthesizer** (composing a candidate answer from retrieved
  memories) has the same live/heuristic split, via `BRAIN_EVAL_SYNTH_MODEL`
  and the same `live-llm` feature + `BRAIN__LLM__API_KEY` gate — see
  [`bench::synthesize`](src/bench/synthesize.rs).

Without `--features live-llm` (or without a key), every command below
still runs — just on the heuristic judge. Fine as a free smoke check; not
for headline LongMemEval/LoCoMo accuracy.

## Retrieval-only runs

Judge noise can be wider than the thing you are measuring. Three runs over
identically ingested data scored **0.628 / 0.648 / 0.673** — a 4.5-point spread
with the inputs held constant, which is wider than most retrieval changes worth
making. Tuning retrieval against a number that moves that much on its own tells
you very little.

`BRAIN_EVAL_RETRIEVAL_ONLY=1` takes the LLM out of the loop: no answer
synthesis, no answer judge, no support judge. Three LLM calls per question
become zero, so a run is free, fast, and deterministic. **Retrieval itself is
untouched** — same RECALL, same ranking, same `top_k`.

```bash
BRAIN_EVAL_RETRIEVAL_ONLY=1 cargo run --bin brain-eval -- bench --dataset dmr
```

What changes in the report:

- Every verdict is `NotJudged`, and **accuracy is scored over graded questions
  only**. A retrieval-only run therefore reports accuracy over zero questions
  rather than 0.0-out-of-N — counting ungraded questions in the denominator
  would make a flawless retrieval run read as total failure.
- The retrieval metrics carry the signal: **recall@k**, **nDCG@k**, and
  **MRR** (mean reciprocal rank).

MRR earns its place alongside recall@k because it is sensitive to *where* in
the ranking the first relevant memory lands, not merely whether it appeared in
the top k. Two runs with identical recall@5 can differ sharply in MRR — one
putting the right memory first, the other fifth — and that difference is
exactly what reranking and beam-width changes move.

MRR is reported everywhere the other retrieval stats are: the JSON report
(`RetrievalStats.mrr`), the text and HTML reports, and the CLI summary line. A
test reads the fields out of `RetrievalStats` and fails if either file reporter
stops showing one, which is how MRR came to be missing from all three in the
first place.

## Quick start

### Build

```bash
cd brain-eval
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
```

### Run a benchmark against a live server

`brain-server` is Linux-only (Glommio + `io_uring`); run it on a Linux host
or inside the sibling `brain-db` repo's dev container (`just docker-up`).

```bash
# 1. Start a brain-server. The admin token is mandatory — brain-server
#    refuses to start without one (see "Authentication" above).
BRAIN__ADMIN__TOKEN=eval-admin-secret \
  cargo run --bin brain-server --manifest-path ../../brain/Cargo.toml

# 2. Point the eval at your datasets directory and the matching admin secret:
export BRAIN_EVAL_DATASETS_DIR=/path/to/your/datasets
export BRAIN_EVAL_ADMIN_SECRET=eval-admin-secret   # must match BRAIN__ADMIN__TOKEN

# 3. Drive the integration test (one ingest + recall round-trip):
BRAIN_EVAL_ENDPOINT=127.0.0.1:9090 \
  cargo test --test basic_e2e -- --ignored --nocapture
```

---

## Running LongMemEval-S end to end

This is the fully wired path — one dataset, runnable today. Start to finish:

### Step 1 — start a brain-server (Linux host)

```bash
# In one terminal — default dev endpoint is 127.0.0.1:9090. The admin
# token is mandatory; brain-server refuses to start without one (see
# "Authentication" above).
BRAIN__ADMIN__TOKEN=eval-admin-secret \
  cargo run --release --bin brain-server \
  --manifest-path ../../brain/Cargo.toml
```

Keep this running. The eval connects to it over the wire (same path a real client would use).

### Step 2 — fetch the dataset

```bash
# In a second terminal
export BRAIN_EVAL_DATASETS_DIR="$HOME/.brain-eval-datasets"
./scripts/fetch-datasets.sh
```

Fetches LoCoMo and LongMemEval-S automatically (re-running is a no-op for
files already present); DMR is MemGPT's "augmented MSC" set, not published
in our normalized shape, so it always needs a one-time manual conversion —
the script prints the source and exact target path
(`$BRAIN_EVAL_DATASETS_DIR/dmr/dmr.jsonl`). LoCoMo and LongMemEval-S are
the stronger benchmarks to start with regardless.

### Step 3 — run a smoke pass (10 questions)

```bash
export BRAIN_EVAL_ENDPOINT=127.0.0.1:9090
export BRAIN_EVAL_ADMIN_SECRET=eval-admin-secret   # must match BRAIN__ADMIN__TOKEN
export BRAIN_EVAL_MAX_QUESTIONS=10        # smoke cap; remove to run the full 500
export BRAIN_EVAL_TOP_K=10

cargo run --release --bin brain-eval -- longmemeval-s
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
cargo run --release --bin brain-eval -- longmemeval-s
```

A full 500-question pass with the heuristic judge takes 5–15 minutes depending on hardware (dominated by ingest time — each question carries its own haystack).

### Step 6 — switch on the LLM judge (optional, costs money)

The heuristic judge above is a directional signal only for LongMemEval's
free-form question types (see [Judges](#judges)). For cross-comparable
numbers, rebuild with `--features live-llm` and set the one shared
credential:

```bash
export BRAIN__LLM__API_KEY=sk-...   # one key; provider derived from the model id
cargo run --release --features live-llm --bin brain-eval -- \
    longmemeval-s --endpoint 127.0.0.1:9090
```

Everything else — `BRAIN_EVAL_ADMIN_SECRET`, `BRAIN_EVAL_MAX_QUESTIONS`,
`BRAIN_EVAL_TOP_K` — carries over unchanged from the steps above. The
printed report's `judge_type` line now reads `llm:openai:<model>@<version>`
(or `llm:anthropic:...`) instead of `heuristic` — that's the tell that the
numbers are the trustworthy ones. `BRAIN_EVAL_JUDGE_MODEL` overrides the
judge's model specifically (else falls back to `BRAIN__LLM__MODEL`, else
`gpt-4o-mini`); note the LLM judge bills one API call per question, so
`BRAIN_EVAL_MAX_QUESTIONS` is worth keeping set until you're ready for a
full paid run.

### What the numbers mean today

- `meta.judge_type` records which judge actually scored the run (see [Judges](#judges)). On the default heuristic judge, numbers are cross-comparable for `single-session-user` and `single-session-preference` questions but **directional only** for `multi-session`, `temporal-reasoning`, `knowledge-update`, `abstention`. Use the LLM judge (Step 6) before quoting numbers against [Wu et al.'s published Table 4](https://arxiv.org/abs/2410.10813) or the competitor baselines.
- `metrics.tokens.*` are all `0` — Brain's wire doesn't surface per-request token counts to the SDK yet. Honest zeros, not invented numbers.
- `metrics.write_quality.*` reflects the substrate's fingerprint dedupe behaviour (driven by `EncodeBuilder::deduplicate(true)` in the harness).

### Zero-config smoke run (Recall@1 canary)

The fastest real signal — a compiled-in 18-memory corpus + 12 questions, no dataset download. Needs only a running `brain-server` and `BRAIN_EVAL_ADMIN_SECRET` set as in Step 1 above:

```bash
cargo run --bin brain-eval -- smoke --endpoint 127.0.0.1:9090
```

Each question's gold answer is a substring unique to exactly one memory, so Recall@1 == 1.0 iff every question's single best hit is the intended memory. Use it as the inner-loop check after substrate changes.

### Sanity-check without a live server

Everything except retrieval is exercised by the unit + parser tests — no server, no datasets, no mocks:

```bash
cargo test
```

This covers the dataset loaders (against checked-in fixtures), the judge, metrics, retrieval math, and the reporters. The retrieval roundtrip is genuinely real-only: every other test file under `tests/` (`basic_e2e`, `runner_e2e`, `scale_smoke`, `server_boot`, `durability`, `membership_invariants`, `system_scenarios`, and the six `*_live` suites) is `#[ignore]`d and requires a live server, opted into via the `live` nextest profile (`cargo nextest run --profile live --run-ignored ignored-only`) or `BRAIN_EVAL_ENDPOINT` + `--ignored` for an individual file.

---

A full DMR / LoCoMo run follows the same shape as the smoke run; the dataset-specific `_competitor_baselines()` functions in `src/bench/report/baselines.rs` are the substitution point for those benchmarks.

## Architecture

One shared adapter layer (`client/`), and two independent surfaces built on
top of it (`bench/`, `blackbox/`):

```
brain-eval/
├── src/
│   ├── lib.rs                     — module declarations + crate-level docs
│   ├── bin/brain-eval.rs          — CLI entrypoint: per-dataset benchmarks
│   │                                 plus the acceptance / soak / summary
│   │                                 system commands
│   │
│   ├── client/                    — shared adapter layer: connects to and
│   │   │                             drives a brain-server. Used by both
│   │   │                             surfaces below.
│   │   ├── harness.rs             — BrainEvalHarness (wraps brain-db-sdk)
│   │   ├── auth.rs                — KeyMinter: mints data-plane keys via the admin plane (auth is mandatory)
│   │   ├── metrics.rs             — Prometheus /metrics scraper (extraction-drain barrier)
│   │   └── server.rs              — ServerHandle: docker-boot or attach-to-external brain-server
│   │
│   ├── bench/                     — surface (A): per-question dataset benchmarking
│   │   ├── config.rs              — RunConfig + env vars + ReporterKind
│   │   ├── space_manifest.rs      — persist/reload the per-conversation SpaceId manifest
│   │   ├── synthesize.rs          — top-K → candidate answer
│   │   ├── runner.rs              — EvalRunner orchestration
│   │   │
│   │   ├── core/                  — what types does eval revolve around?
│   │   │   ├── benchmark.rs       — Benchmark trait + EvalError
│   │   │   ├── instance.rs        — EvalInstance, Session, TurnRecord, QuestionType
│   │   │   └── outcome.rs         — QuestionResult, JudgeResult, Verdict
│   │   │
│   │   ├── score/                 — how do we score answers?
│   │   │   ├── judge.rs           — heuristic judge (always available)
│   │   │   ├── llm_judge.rs       — LLM-as-judge (feature `live-llm`)
│   │   │   ├── judge_prompt.rs    — the judge prompt template + its pinned sha256
│   │   │   ├── context_recall.rs  — LLM-judged "did retrieval surface the supporting context" metric
│   │   │   ├── metrics.rs         — EvalMetrics, compute_full_metrics
│   │   │   ├── retrieval.rs       — RetrievalStats, Recall@1/5/10 (deprecated substring diagnostic)
│   │   │   └── latency.rs         — LatencyStats, percentile helpers
│   │   │
│   │   ├── report/                — what does the output look like?
│   │   │   ├── shape.rs           — BenchmarkReport, BenchmarkMeta, CompetitorRow
│   │   │   ├── baselines.rs       — *_competitor_baselines() per benchmark
│   │   │   └── format/            — output writers
│   │   │       ├── json.rs
│   │   │       └── text.rs        — (an HTML writer is a follow-up)
│   │   │
│   │   └── datasets/              — which benchmarks can we load?
│   │       ├── smoke.rs           — compiled-in Aurora corpus (zero download)
│   │       ├── dmr.rs             — DMR (MemGPT 2023)
│   │       ├── longmemeval.rs     — LongMemEval-S (ICLR 2025)
│   │       ├── locomo.rs          — LoCoMo (ACL 2024)
│   │       ├── lexical_stress.rs  — compiled-in no-overlap set (proves semantic, not substring, retrieval)
│   │       ├── paraphrase_stress.rs — compiled-in generated no-overlap triples + distractors
│   │       └── supersession_stress.rs — compiled-in generated OLD→NEW updates
│   │
│   ├── blackbox/                  — surface (B): whole-server black-box suites
│   │   ├── acceptance.rs          — composes scale + system + client into the v1.0 gate
│   │   ├── soak.rs                — sustained mixed workload + drift sampling
│   │   │
│   │   ├── scale/                 — latency + sustained-throughput load generation
│   │   │   ├── recall.rs
│   │   │   ├── throughput.rs
│   │   │   └── targets.rs         — the warm-shard p50/p99 targets each run is checked against
│   │   │
│   │   └── system/                — black-box behavioural + durability scenarios
│   │       ├── invariants.rs      — idempotency / tombstone-grace / slot-version gates
│   │       ├── durability.rs
│   │       ├── chaos.rs
│   │       ├── storage.rs
│   │       └── typed_graph/       — entity / statement / relation / query / schema / extraction scenarios
│   │
│   └── llm.rs (feature `live-llm`) — shared LLM HTTP client for judge + synthesis
│
└── tests/
    ├── longmemeval_loader.rs — the one test file with no #[ignore]d tests;
    │                           pure parser/golden coverage, runs on a plain `cargo test`
    └── everything else (basic_e2e.rs, runner_e2e.rs, scale_smoke.rs,
        server_boot.rs, durability.rs, membership_invariants.rs,
        system_scenarios.rs, chaos_live.rs, invariants_live.rs,
        regression_live.rs, soak_live.rs, storage_live.rs,
        throughput_live.rs, typed_graph_live.rs) is `#[ignore]`d — each
        boots or attaches to a real server. Selected via the `live`
        nextest profile (`--run-ignored ignored-only`), never a plain
        `cargo test`.
```

`client/` is the shared adapter layer both surfaces sit on: connect, authenticate, scrape metrics, manage the server process. `bench/` drives one dataset through the cognitive ops loop per question — types in `bench/core/`, orchestration in `bench/runner.rs`, scoring in `bench/score/`, presentation in `bench/report/`, dataset loaders in `bench/datasets/`. `blackbox/` is the parallel track — whole-server suites (`scale/`, `system/`, `soak.rs`) rather than per-question scoring — composed together by `blackbox/acceptance.rs` into the v1.0 gate. The library surface is production code only — test fixtures live in the tests that use them.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BRAIN_EVAL_ENDPOINT` | `127.0.0.1:9090` | Brain-server endpoint. |
| `BRAIN_EVAL_DATASETS_DIR` | _(unset)_ | Root containing `dmr/dmr.jsonl`, etc. |
| `BRAIN_EVAL_MAX_QUESTIONS` | _(unset = all)_ | Smoke-mode cap. |
| `BRAIN_EVAL_TOP_K` | `50` | `top_k` (max_results) passed to RECALL. |
| `BRAIN_EVAL_RETRIEVAL_ONLY` | _(unset = off)_ | Truthy (`1`/`true`) skips answer synthesis and both judges — zero LLM calls per question. Verdicts come back `NotJudged`, accuracy is scored over graded questions only, and the retrieval metrics (recall@k, nDCG@k, MRR) carry the signal. See [Retrieval-only runs](#retrieval-only-runs). |
| `BRAIN_EVAL_OUTPUT_DIR` | `target/eval-reports` | Where reports land. |
| `BRAIN_EVAL_FORMATS` | `json,text` | Comma list of `json` / `text`. |
| `BRAIN_EVAL_QUESTION_TYPES` | _(unset = all)_ | Comma list restricting the run to specific question-type tags (e.g. `single_hop`). |
| `BRAIN_EVAL_PERSIST_SPACES` | _(unset)_ | Write the per-conversation `space_id` manifest here during an ingesting run. |
| `BRAIN_EVAL_READ_ONLY_SPACES` | _(unset)_ | Skip ingest; reconnect to the spaces in this manifest and only retrieve/judge. |
| `BRAIN_EVAL_ADMIN_SECRET` | _(unset)_ | Admin bearer secret — **required** to mint data-plane keys against an **external** server; must equal the server's own `BRAIN__ADMIN__TOKEN` (the docker-boot harness sets this itself). See [Authentication](#authentication-mandatory). |
| `BRAIN_EVAL_ADMIN_PORT` | `9092` | Admin listener port on the endpoint host. |
| `BRAIN_EVAL_NAMESPACE` | `eval` | Tenant namespace minted keys bind to. |
| `BRAIN_EVAL_NO_DRAIN` | _(unset = drain)_ | Truthy (`1`/`true`/`yes`/`on`) skips waiting for the async write-time extraction pipeline (entity/statement/relation + statement embedding) to plateau before reading — reads against the live, still-settling graph instead. |
| `BRAIN_EVAL_DRAIN_ADMIN_PORT` | `9091` | Metrics port polled for the extraction-drain plateau check above (the server's public `/metrics`, not the admin plane). |
| `BRAIN_EVAL_FULL_TRACE` | _(unset = on)_ | Default ON. Captures per-read stage detail into `*.run.json` (each question's `QUERY_TRACE` plan trace + the extracted graph per retrieved memory) and per-write detail (the extracted graph per stored memory). Adds a few server round-trips per question and per stored memory. Set falsey (`0`/`false`/`no`/`off`) to skip the heavy captures on a large run. |
| `BRAIN_EVAL_RESTART_RECOVERY` | _(unset)_ | `1` adds the (slower, docker-booting) restart-recovery gate to `brain-eval acceptance`. |
| `BRAIN_EVAL_SOAK_SECS` | `5` | Soak run duration for `brain-eval soak`. |
| `BRAIN_EVAL_JUDGE_MODEL` | _(unset)_ | `live-llm` only: override the judge's model (else `BRAIN__LLM__MODEL`, else `gpt-4o-mini`). See [Judges](#judges). |
| `BRAIN_EVAL_SYNTH_MODEL` | _(unset)_ | `live-llm` only: override the answer synthesizer's model, same fallback chain. |
| `BRAIN__LLM__API_KEY` | _(unset)_ | `live-llm` only: the single shared LLM provider credential — the same variable `brain-server` itself reads. No key ⇒ falls back to the heuristic judge/synthesizer. |
| `BRAIN__LLM__MODEL` | _(unset)_ | `live-llm` only: shared model fallback when `BRAIN_EVAL_JUDGE_MODEL`/`BRAIN_EVAL_SYNTH_MODEL` aren't set. Provider (OpenAI vs Anthropic) is derived from the model id. |
| `RUST_LOG` | `info` | Tracing verbosity (installed by the binary at startup). |

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

- **Per-question space isolation.** Each `EvalRunner` group spins up a fresh `BrainEvalHarness` with a fresh `SpaceId`. Brain routes by `BLAKE3(space_uuid) mod shard_count`, so two harnesses with different space ids touch independent slices of substrate state. This is the natural isolation primitive for benchmark questions — no "scope" or "namespace" wrapper needed.
- **Honest numbers.** `BenchmarkMeta.judge_type` records which judge actually scored a run (see [Judges](#judges)), so a run that partially fell back from the LLM judge to the heuristic is never misreported as fully LLM-judged.
- **No invented features.** Brain's wire today doesn't surface per-request token counts to the SDK, so `tokens_write` and `tokens_read` are zeros until the wire grows the fields. Better an honest zero than an invented number.
- **Single-shard scope.** Each connection is pinned to one shard at AUTH (spec §12/02). Multi-shard spaces are v2; this eval crate runs the v1 single-shard contract.

## Follow-ups

In rough priority order:

1. **Token accounting on the wire.** Add `prompt_tokens` / `completion_tokens` to `EncodeResponse` and `RecallResponseFrame` (or carry them on a side-channel telemetry frame). Wire VERSION bump — cross-repo work in `brain-db`, not this crate.
2. **BEAM loader.** 1M–10M scale benchmark; pairs with criterion runs.
3. **HTML reporter.** Self-contained dark/light-mode report with Chart.js latency/accuracy plots (`src/bench/report/format/` currently ships `json.rs` + `text.rs` only).
4. **`run-report.sh`** + per-benchmark cargo aliases. (`scripts/fetch-datasets.sh` already covers dataset downloading — DMR/LongMemEval/LoCoMo — superseding the old `download_datasets.sh` follow-up.)
5. **Spec §16/02 CI latency gate.** Make CI fail when p99 RECALL > target — the `live` job runs the acceptance/scenario/soak gates today but doesn't yet assert this specific target.
6. **Hybrid-vs-substrate criterion bench.** Side-by-side latency + accuracy comparison; the Brain-unique eval axis.

Authoritative source: `spec/19_benchmarks/ (in the brain-db repo)` — read before adding any new benchmark dimension.
