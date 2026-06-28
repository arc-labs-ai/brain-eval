# Running the dataset evals (LLM-as-judge)

The dataset benchmarks (LongMemEval-S, LoCoMo, DMR) measure end-to-end memory
quality: ingest a corpus → RECALL → synthesize an answer → **judge** it against
ground truth → score, and compare to the competitor baselines in
`report/baselines.rs`. This is separate from the acceptance/correctness gates
(`acceptance`, the `*_live` tests).

## Judges

- **Heuristic** (default, no key): substring + token-overlap. Honest for
  fact-style answers (DMR), too blunt for free-form multi-hop (LongMemEval,
  LoCoMo).
- **LLM-as-judge** (`--features live-llm` + an API key): a real model grades
  each answer correct / partial / incorrect. Required for honest LongMemEval /
  LoCoMo numbers. The judge uses the **single shared credential**
  `BRAIN__LLM__API_KEY` — the same variable Brain itself uses, so one key drives
  both the server's write-time LLM and the eval's judge. The model is
  `BRAIN_EVAL_JUDGE_MODEL` (judge-specific override) → `BRAIN__LLM__MODEL` →
  `gpt-4o-mini`, and the **provider is derived from the model id** (a `gpt-*`
  id ⇒ OpenAI, a `claude-*` id ⇒ Anthropic) — there is no separate
  `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` auto-detect any more. A transient API
  failure on one question falls back to the heuristic for that question (the run
  never aborts). The report header records which judge actually ran
  (`llm:openai:<model>` / `llm:anthropic:<model>` vs `heuristic`).

## Authentication (mandatory)

Brain requires a per-connection API key — there is no anonymous connect. The
eval **mints** a data-plane key for `(namespace, agent)` through the server's
admin HTTP listener, then connects with it.

- **Docker-boot runs** (the `*_live` tests via `ServerHandle::start_docker`, and
  `just up`'s docker path) configure this automatically: the harness sets the
  server's admin secret + admin address and mints against the loopback admin
  port. Nothing to do.
- **External-server runs** (you booted the server yourself and point
  `--endpoint`/`BRAIN_EVAL_ENDPOINT` at it) must be told how to reach + gate the
  admin plane:
  - `BRAIN_EVAL_ADMIN_SECRET` — **required.** Must equal the server's
    `BRAIN__ADMIN__TOKEN`. Without it the run fails fast at startup
    ("preflight: cannot mint data-plane keys …") rather than silently recording
    every question as failed.
  - `BRAIN_EVAL_ADMIN_PORT` — admin listener port on the endpoint host
    (default `9092`). The server's admin listener must be reachable there, i.e.
    boot it with `BRAIN__SERVER__ADMIN_ADDR=0.0.0.0:9092` (the default is
    loopback-only).
  - `BRAIN_EVAL_NAMESPACE` — tenant the run writes under (default `eval`).

## One-time setup

```bash
# 1. Datasets. LoCoMo and LongMemEval-S (the `longmemeval-cleaned` release)
#    auto-download. DMR needs a one-time conversion (see below); the script
#    prints the source + exact target path for anything it can't fetch.
export BRAIN_EVAL_DATASETS_DIR=~/brain-datasets
scripts/fetch-datasets.sh
```

DMR (`dmr`) is the MemGPT "augmented MSC" set and isn't published in our
normalized shape, so it needs a one-time conversion to
`$BRAIN_EVAL_DATASETS_DIR/dmr/dmr.jsonl` — one JSON object per line:
`{id, question, answer, conversation_id, sessions:[{session_id, turns:[{role, content}]}]}`.
LoCoMo and LongMemEval-S are the stronger benchmarks to start with.

## Run

```bash
# 2. Boot the full-stack server (reranker + extractors on) with the admin
#    plane reachable + gated so the eval can mint a key.
BRAIN__SERVER__ADMIN_ADDR=0.0.0.0:9092 BRAIN__ADMIN__TOKEN=eval-admin-secret \
    just up 38100                   # 127.0.0.1:38100 (data), :9092 (admin)

# 3. Run a benchmark with the LLM judge.
export BRAIN_EVAL_DATASETS_DIR=~/brain-datasets
export BRAIN__LLM__API_KEY=sk-...           # one key; provider derived from model
export BRAIN_EVAL_ADMIN_SECRET=eval-admin-secret   # must match BRAIN__ADMIN__TOKEN
cargo run --release --features live-llm --bin brain-eval -- \
    longmemeval-s --endpoint 127.0.0.1:38100
#   benchmarks: smoke | dmr | longmemeval-s | locomo
```

Useful env knobs (all optional):

- `BRAIN_EVAL_MAX_QUESTIONS=N` — cap questions (cost control; LLM judge bills
  one call per question).
- `BRAIN_EVAL_TOP_K`, `BRAIN_EVAL_OUTPUT_DIR`, `BRAIN_EVAL_FORMATS`.
- `BRAIN_EVAL_JUDGE_MODEL` — override the judge model (else `BRAIN__LLM__MODEL`,
  else `gpt-4o-mini`).

Without `--features live-llm` (or without a key) the same commands run on the
heuristic judge — fine as a free smoke, not for headline LongMemEval/LoCoMo
accuracy.

## Validation

The report prints accuracy (correct/partial/incorrect), recall@1/5/10, and the
competitor table for the benchmark. Trustworthy accuracy on the free-form
benchmarks requires the LLM judge.
