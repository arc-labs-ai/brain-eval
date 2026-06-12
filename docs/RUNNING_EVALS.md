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
  LoCoMo numbers. Provider is auto-detected — `ANTHROPIC_API_KEY` (Claude,
  default) or `OPENAI_API_KEY`; override the model with
  `BRAIN_EVAL_JUDGE_MODEL`. A transient API failure on one question falls back
  to the heuristic for that question (the run never aborts). The report header
  records which judge actually ran (`llm:anthropic:<model>` vs `heuristic`).

## One-time setup

```bash
# 1. Datasets (LoCoMo auto-downloads; LongMemEval-S / DMR may need a manual
#    step — the script prints the source + exact target path).
export BRAIN_EVAL_DATASETS_DIR=~/brain-datasets
scripts/fetch-datasets.sh
```

## Run

```bash
# 2. Boot the full-stack server (reranker + extractors on).
just up 38100                       # 127.0.0.1:38100

# 3. Run a benchmark with the LLM judge.
export BRAIN_EVAL_DATASETS_DIR=~/brain-datasets
export ANTHROPIC_API_KEY=sk-ant-...     # or OPENAI_API_KEY=sk-...
cargo run --release --features live-llm --bin brain-eval -- \
    longmemeval-s --endpoint 127.0.0.1:38100
#   benchmarks: smoke | dmr | longmemeval-s | locomo
```

Useful env knobs (all optional):

- `BRAIN_EVAL_MAX_QUESTIONS=N` — cap questions (cost control; LLM judge bills
  one call per question).
- `BRAIN_EVAL_TOP_K`, `BRAIN_EVAL_OUTPUT_DIR`, `BRAIN_EVAL_FORMATS`.
- `BRAIN_EVAL_JUDGE_MODEL` — override the judge model.

Without `--features live-llm` (or without a key) the same commands run on the
heuristic judge — fine as a free smoke, not for headline LongMemEval/LoCoMo
accuracy.

## Validation

The report prints accuracy (correct/partial/incorrect), recall@1/5/10, and the
competitor table for the benchmark. Trustworthy accuracy on the free-form
benchmarks requires the LLM judge.
