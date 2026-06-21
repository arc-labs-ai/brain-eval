#!/usr/bin/env python3
"""Set-level membership recall — measures the DATABASE directly, decoupled from
the answer synthesizer and the LLM judge: for each question, did the returned
memory set CONTAIN the question's gold-evidence memory?

Offline: reads the dataset gold + the newest brain-eval partial.jsonl. No server,
no re-ingest. Run from the brain-eval repo root:
    python3 scripts/membership_recall.py [datasets_dir]
"""
import json, os, glob, re, sys

DSDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/brain-datasets")
REPORTS = "target/eval-reports"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def newest(prefix):
    fs = sorted(glob.glob(f"{REPORTS}/{prefix}*.partial.jsonl"), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def load_partial(path):
    rows = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            rows[norm(r.get("question", ""))] = r
        except Exception:
            pass
    return rows


def evidence_present(gold_chunks, retrieved):
    """gold_chunks: list of distinctive substrings of gold evidence turns.
    retrieved: list of returned memory texts. Lenient: a chunk (>=24 chars of a
    gold turn) is a substring of any returned memory."""
    blob = " || ".join(norm(m) for m in retrieved)
    return any(c and c in blob for c in gold_chunks)


def chunk(text):
    n = norm(text)
    return n[:60] if len(n) >= 24 else None  # distinctive head of the turn


def locomo():
    d = json.load(open(f"{DSDIR}/locomo/locomo10.json"))
    p = newest("locomo-")
    if not p:
        return None
    rows = load_partial(p)
    sample = d[0]  # conv-26 is sample 0
    conv = sample["conversation"]
    dia = {}
    for k, v in conv.items():
        if isinstance(v, list):
            for t in v:
                if isinstance(t, dict) and "dia_id" in t:
                    dia[t["dia_id"]] = t.get("text", "")
    hit = tot = miss_examples = 0
    misses = []
    for qa in sample["qa"]:
        q = norm(qa.get("question", ""))
        if q not in rows:
            continue
        ev = qa.get("evidence") or []
        chunks = [chunk(dia.get(e, "")) for e in ev if dia.get(e)]
        chunks = [c for c in chunks if c]
        if not chunks:
            continue
        tot += 1
        rc = rows[q].get("retrieved_memory_contents") or []
        if evidence_present(chunks, rc):
            hit += 1
        else:
            misses.append(qa.get("question", "")[:50])
    return ("LoCoMo (deterministic: evidence dia_ids)", hit, tot, os.path.basename(p), misses)


def longmemeval():
    d = json.load(open(f"{DSDIR}/longmemeval/longmemeval_s.json"))
    p = newest("longmemeval-s-")
    if not p:
        return None
    rows = load_partial(p)
    hit = tot = 0
    misses = []
    for q in d:
        qn = norm(q.get("question", ""))
        if qn not in rows:
            continue
        ans = norm(q.get("answer", ""))
        if not ans:
            continue
        tot += 1
        rc = rows[qn].get("retrieved_memory_contents") or []
        blob = " || ".join(norm(m) for m in rc)
        if ans in blob:
            hit += 1
        else:
            misses.append(q.get("question", "")[:50])
    return ("LongMemEval-S (approx: gold-answer string in set)", hit, tot, os.path.basename(p), misses)


def main():
    for fn in (locomo, longmemeval):
        r = fn()
        if not r:
            continue
        label, hit, tot, src, misses = r
        rate = hit / tot if tot else 0.0
        print(f"\n=== {label} ===")
        print(f"  source: {src}")
        print(f"  membership-recall: {hit}/{tot} = {rate:.3f}")
        if misses:
            print(f"  not-in-set ({len(misses)}): " + "; ".join(misses[:8]))
    print("\n(lexical-stress: gold answer is by design absent from the memory text,")
    print(" so substring membership N/A — rely on LLM-judged ctx-recall for it.)")


if __name__ == "__main__":
    main()
