#!/usr/bin/env bash
#
# download_longmemeval.sh — fetch the LongMemEval-S dataset into
# $BRAIN_EVAL_DATASETS_DIR/longmemeval/.
#
# Usage:
#   BRAIN_EVAL_DATASETS_DIR=/path/to/datasets ./scripts/download_longmemeval.sh
#
# What it does:
#   1. Ensures BRAIN_EVAL_DATASETS_DIR is set.
#   2. Creates $BRAIN_EVAL_DATASETS_DIR/longmemeval/ if needed.
#   3. Fetches longmemeval_s.json from the public release.
#   4. Sanity-checks the file is non-empty + parses as JSON.
#
# The dataset URL points at the upstream public release. If the URL
# 404s, the LongMemEval authors moved it — check
# https://github.com/xiaowu0162/LongMemEval for the current location.

set -euo pipefail

DATASET_URL="${BRAIN_EVAL_LONGMEMEVAL_URL:-https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json}"
DATASETS_DIR="${BRAIN_EVAL_DATASETS_DIR:-}"

if [[ -z "$DATASETS_DIR" ]]; then
  echo "error: BRAIN_EVAL_DATASETS_DIR is not set" >&2
  echo "  Pick a directory and re-run:" >&2
  echo "  BRAIN_EVAL_DATASETS_DIR=\$HOME/.brain-eval-datasets $0" >&2
  exit 1
fi

LME_DIR="$DATASETS_DIR/longmemeval"
TARGET_FILE="$LME_DIR/longmemeval_s.json"

mkdir -p "$LME_DIR"

if [[ -s "$TARGET_FILE" ]]; then
  echo "longmemeval_s.json already exists at $TARGET_FILE"
  echo "  size: $(du -h "$TARGET_FILE" | awk '{print $1}')"
  echo "  delete it and re-run to force-redownload."
  exit 0
fi

echo "fetching $DATASET_URL"
echo "  → $TARGET_FILE"

if command -v curl >/dev/null 2>&1; then
  curl -fL -o "$TARGET_FILE" "$DATASET_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TARGET_FILE" "$DATASET_URL"
else
  echo "error: neither curl nor wget is available" >&2
  exit 1
fi

if [[ ! -s "$TARGET_FILE" ]]; then
  echo "error: download produced an empty file at $TARGET_FILE" >&2
  rm -f "$TARGET_FILE"
  exit 1
fi

# Light parse-validity check: read the first few bytes and confirm we
# got JSON (array or object), not an HTML error page.
FIRST_CHAR=$(head -c 1 "$TARGET_FILE")
if [[ "$FIRST_CHAR" != "[" && "$FIRST_CHAR" != "{" ]]; then
  echo "error: downloaded file does not look like JSON (first byte: '$FIRST_CHAR')" >&2
  echo "  This usually means the URL returned an HTML error page." >&2
  echo "  Inspect: head $TARGET_FILE" >&2
  exit 1
fi

SIZE=$(du -h "$TARGET_FILE" | awk '{print $1}')
echo "ok: $TARGET_FILE ($SIZE)"
echo
echo "Next:"
echo "  cargo run --release --example run_longmemeval \\"
echo "    --manifest-path crates/brain-eval/Cargo.toml"
