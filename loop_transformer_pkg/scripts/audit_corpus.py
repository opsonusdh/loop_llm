#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

from filters import DocumentClassifier, Decision


def iter_text_files(root: Path) -> Iterable[Path]:
    yield from sorted(p for p in root.rglob("*.txt") if p.is_file())


def infer_source(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "github" in parts:
        return "github"
    if parts & {"books", "book", "gutenberg", "standard_ebooks", "wikisource"}:
        return "books"
    for source in ("wikipedia", "w3schools", "custom"):
        if source in parts:
            return source
    return ""


def audit_file(path: Path, classifier: DocumentClassifier) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    source = infer_source(path)
    decision, reason, stats = classifier.classify(text, path=str(path), source=source)
    exact_hash = hashlib.sha256(raw).hexdigest()
    return {
        "file": str(path),
        "source": source,
        "bytes": len(raw),
        "sha256": exact_hash,
        "decision": decision.value,
        "category": decision.value,
        "reason": reason,
        "score": 10 if decision.value.startswith("REJECT_") else 0,
        "suspicious": decision.value.startswith("REJECT_"),
        "stats": stats,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit an LM corpus using the same quality gate used by the scrapers.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("corpus_audit.jsonl"))
    p.add_argument("--show", type=int, default=100)
    args = p.parse_args()

    files = list(iter_text_files(args.input))
    classifier = DocumentClassifier()
    results = [audit_file(f, classifier) for f in files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    bad = [r for r in results if r["suspicious"]]
    counts = Counter(r["category"] for r in bad)
    print("="*80)
    print("CORPUS AUDIT")
    print("="*80)
    print(f"Total files:      {len(results):,}")
    print(f"Rejected:         {len(bad):,}")
    print(f"Accepted:         {len(results)-len(bad):,}")
    print("\nReject reasons:")
    for k,v in counts.most_common():
        print(f"  {k:24s} {v:6,d}")
    print("\nHighest-risk files:")
    for r in bad[:args.show]:
        print(f"{r['category']:24s} {r['file']} :: {r['reason']}")
    print(f"\nReport: {args.output}")

if __name__ == "__main__":
    main()
