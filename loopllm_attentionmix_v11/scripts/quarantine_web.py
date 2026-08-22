#!/usr/bin/env python3
"""
Non-destructively quarantine suspicious NON-BOOK web-scraped files.

This expects an audit JSONL produced by audit_corpus.py. It COPIES suspicious
files and leaves the original corpus untouched.

Use it for GitHub, Wikipedia, W3Schools, and custom web data.

Example:
    python quarantine_web.py \
        --input data/raw \
        --audit data/web_audit.jsonl \
        --quarantine data/quarantine/web \
        --min-score 5

Optional:
    --source github
    --source wikipedia
    --source w3schools
    --source custom
    --category REJECT_WEB_ARTIFACT
    --category REJECT_TABULAR
    --dry-run

(--category matches the "category" field audit_corpus.py writes, i.e. one
of the Decision enum names in filters.py, e.g. REJECT_WEB_ARTIFACT,
REJECT_GENERATED, REJECT_TABULAR, REJECT_LOG. Run without --category first
to see which ones are actually present in your audit file.)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
for _p in (ROOT / "scripts", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from filters import infer_source as _infer_source

BOOK_MARKERS = {
    "books",
    "book",
    "gutenberg",
    "standard_ebooks",
    "wikisource",
}


def load_audit(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("suspicious"):
                rows.append(row)
    return rows


def infer_source(path: Path) -> str:
    return _infer_source(str(path))


def looks_like_book(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return bool(parts & BOOK_MARKERS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Root directory containing scraped web data.")
    p.add_argument("--audit", type=Path, required=True,
                   help="Audit JSONL produced by audit_corpus.py.")
    p.add_argument("--quarantine", type=Path, required=True,
                   help="Destination for copied suspicious files.")
    p.add_argument("--source", action="append", default=[],
                   choices=["github", "wikipedia", "w3schools", "custom"],
                   help="Only quarantine this source. Repeatable. Omit to include all sources.")
    p.add_argument("--min-score", type=int, default=5)
    p.add_argument("--category", action="append", default=[],
                   help="Only quarantine this category. Repeatable.")
    p.add_argument("--limit", type=int, default=0,
                   help="Maximum files to quarantine. 0 = unlimited.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows = load_audit(args.audit)

    selected = []
    for row in rows:
        if row.get("score", 0) < args.min_score:
            continue

        src = Path(row["file"])

        # Never quarantine book files with this script.
        if looks_like_book(src):
            continue

        source = infer_source(src)
        if args.source and source not in args.source:
            continue

        category = str(row.get("category", ""))
        if args.category and category not in args.category:
            continue

        selected.append(row)

    if args.limit > 0:
        selected = selected[:args.limit]

    print(f"Suspicious web files selected: {len(selected)}")

    copied = 0
    missing = 0

    for row in selected:
        src = Path(row["file"])
        try:
            rel = src.relative_to(args.input)
        except ValueError:
            rel = Path(src.name)

        dst = args.quarantine / rel

        print(f"{'[DRY] ' if args.dry_run else ''}{row.get('category')}: {src}")

        if not src.exists():
            missing += 1
            continue

        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print("\nDone.")
    print(f"Copied:  {copied}")
    print(f"Missing: {missing}")
    print("Original files were NOT deleted.")
    print(f"Quarantine: {args.quarantine}")

if __name__ == "__main__":
    main()
