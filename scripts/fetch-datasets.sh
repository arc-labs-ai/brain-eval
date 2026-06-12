#!/usr/bin/env bash
#
# Fetch the eval datasets into $BRAIN_EVAL_DATASETS_DIR (default ./datasets),
# laid out exactly where the loaders expect them:
#
#   $DIR/locomo/locomo10.json
#   $DIR/longmemeval/longmemeval_s.json
#   $DIR/dmr/dmr.jsonl
#
# LoCoMo is fetched automatically from its stable raw GitHub URL. LongMemEval
# and DMR are released through gated / large hosts (HuggingFace, Google Drive),
# so this script can't always pull them non-interactively — when it can't, it
# prints the canonical source and the exact target path and moves on, rather
# than failing the whole run. Re-run any time; existing files are left alone.
#
# Usage:
#   BRAIN_EVAL_DATASETS_DIR=~/brain-datasets scripts/fetch-datasets.sh
set -euo pipefail

DIR="${BRAIN_EVAL_DATASETS_DIR:-$(pwd)/datasets}"
echo "datasets dir: $DIR"
mkdir -p "$DIR/locomo" "$DIR/longmemeval" "$DIR/dmr"

missing=0

# fetch <url> <dest>  — skip if present; curl with fail-on-error + redirects.
fetch() {
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "  ✓ already present: $dest"
    return 0
  fi
  echo "  ↓ $url"
  if curl -fsSL "$url" -o "$dest.tmp"; then
    mv "$dest.tmp" "$dest"
    echo "  ✓ wrote $dest"
  else
    rm -f "$dest.tmp"
    return 1
  fi
}

manual() {
  local name="$1" source="$2" dest="$3"
  echo "  ⚠ $name needs a manual download (gated/large host)."
  echo "      source: $source"
  echo "      save to: $dest"
  missing=1
}

echo "== LoCoMo =="
fetch \
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" \
  "$DIR/locomo/locomo10.json" \
  || manual "LoCoMo" "https://github.com/snap-research/locomo" "$DIR/locomo/locomo10.json"

echo "== LongMemEval-S =="
# Official release is the `longmemeval-cleaned` HF dataset; its
# `longmemeval_s_cleaned.json` already matches our loader's schema
# (question_id / question_type / question / answer / haystack_sessions).
# Large file (~hundreds of MB) — the download takes a moment.
fetch \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
  "$DIR/longmemeval/longmemeval_s.json" \
  || manual "LongMemEval-S" \
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (file: longmemeval_s_cleaned.json)" \
            "$DIR/longmemeval/longmemeval_s.json"

echo "== DMR =="
# DMR is the MemGPT "augmented MSC" set (memgpt.ai / the Letta repo). It is
# NOT published in our normalized shape, so it needs a one-time conversion to
# dmr.jsonl: one JSON object per line with fields
#   {id, question, answer, conversation_id,
#    sessions:[{session_id, turns:[{role, content}]}]}
# Obtain the raw MemGPT MSC/DMR data, then map each conversation+question
# onto that shape. (LoCoMo + LongMemEval-S are the stronger benchmarks to
# start with; DMR is single-hop fact retrieval.)
manual "DMR" "https://github.com/letta-ai/letta (MemGPT MSC/DMR data) — convert to the normalized dmr.jsonl shape above" \
       "$DIR/dmr/dmr.jsonl"

echo
if [[ "$missing" -eq 0 ]]; then
  echo "all datasets present under $DIR"
else
  echo "some datasets need a manual download (see ⚠ above). Set"
  echo "BRAIN_EVAL_DATASETS_DIR=$DIR and re-run a benchmark once they're in place."
fi
