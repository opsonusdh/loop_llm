#!/usr/bin/env python3
"""Corpus scanner using the exact DocumentClassifier used by corpus auditing.

Read-only. It does not delete, quarantine, or modify files.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
for p in (ROOT / "scripts", ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from filters import DocumentClassifier, infer_source as _infer_source

SKIP_NAMES = {
    "manifest.jsonl", "_state.json", "state.json", "audit.jsonl",
    "github_test_audit.jsonl", "suspicious_corpus_report.json",
}
SOURCE_DIRS = {"github", "books", "book", "gutenberg", "standard_ebooks",
               "wikipedia", "w3schools", "custom"}

def infer_source(path: Path) -> str:
    return _infer_source(str(path))

def iter_text_files(root: Path):
    for p in sorted(root.rglob("*.txt")):
        if p.is_file() and p.name not in SKIP_NAMES:
            yield p

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--min-score", type=int, default=1,
                    help="Classifier reject categories are always scored above zero.")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    root = args.input.resolve()
    if not root.is_dir():
        ap.error(f"Input directory does not exist: {root}")

    clf = DocumentClassifier()
    findings = []
    scanned = 0
    for path in iter_text_files(root):
        scanned += 1
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        source = infer_source(path)
        decision, reason, stats = clf.classify(text, path=str(path), source=source)
        if decision.value.startswith("REJECT_"):
            score = 10
            findings.append({
                "file": str(path.relative_to(root)),
                "source": source,
                "decision": decision.value,
                "reason": reason,
                "score": score,
                "bytes": len(raw),
                "lines": len(text.splitlines()),
                "stats": stats,
            })

    findings.sort(key=lambda r: (-r["score"], -r["bytes"], r["file"].lower()))
    print("="*96)
    print("SUSPICIOUS CORPUS SCAN (same classifier as audit/scrapers)")
    print("="*96)
    print(f"Directory:    {root}")
    print(f"Files scanned: {scanned:,}")
    print(f"Flagged:       {len(findings):,}")
    print()
    counts = Counter(x["decision"] for x in findings)
    for k, v in counts.most_common():
        print(f"{k:26s} {v:7,d}")
    print()
    for i, x in enumerate(findings[:args.top], 1):
        print(f"[{i:3}] {x['decision']:24s} {x['file']}")
        print(f"      source={x['source'] or 'unknown'} reason={x['reason']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report: {args.json}")
    print("="*96)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
