#!/usr/bin/env python3
"""
Temporary corpus scanner for suspicious web/JSON/metadata noise.

Scans a directory recursively and reports files containing suspicious patterns
such as package metadata, schema-like JSON, GitHub/npm metadata, URLs, and
other web-scraping artifacts.

This script DOES NOT modify or delete anything.

Examples:
    python scripts/scan_suspicious_corpus.py --input data/corpus
    python scripts/scan_suspicious_corpus.py --input data/corpus --top 100
    python scripts/scan_suspicious_corpus.py --input data/corpus --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# Patterns are intentionally broad for discovery, not automatic deletion.
PATTERNS: dict[str, re.Pattern[str]] = {
    "json_metadata_field": re.compile(
        r'(?m)^\s*["\']?(description|source|repository|homepage|license|'
        r'version|maintainers|keywords|peerDependencies|devDependencies|'
        r'dist-tags|_id|_rev|directories|author|bugs|engines|scripts)'
        r'["\']?\s*[:=]'
    ),
    "schema_type_field": re.compile(
        r'(?i)["\']?type["\']?\s*:\s*["\']?(string|number|integer|object|array|boolean)'
    ),
    "github_url": re.compile(
        r'https?://(?:www\.)?github\.com/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+'
    ),
    "npm_registry": re.compile(
        r'https?://(?:registry\.)?npmjs\.org/'
    ),
    "github_metadata": re.compile(
        r'(?i)\b(github|gitlab|bitbucket)\b.*\b(repository|source|homepage|issues?)\b|'
        r'\b(repository|homepage|source)\b.*\b(github|gitlab|bitbucket)\b'
    ),
    "markdown_metadata": re.compile(
        r'(?m)(?:^\s*["\']?(description|source|type|url|title|author)["\']?\s*:\s*'
        r'|\b(markdown|front[- ]matter|yaml front matter)\b)',
        re.IGNORECASE,
    ),
    "json_object_density": re.compile(
        r'(?s)^\s*\{.*\}\s*$'
    ),
    "npm_package_markers": re.compile(
        r'(?i)\b(dist-tags|peerDependencies|devDependencies|maintainers|_npmVersion|'
        r'_nodeVersion|_npmUser|_npmOperationalInternal|tarball|shasum|integrity)\b'
    ),
    "url_density": re.compile(
        r'https?://|www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    ),
    "serialized_object_markers": re.compile(
        r'(?i)\b(?:_id|_rev|_resolved|_integrity|dist-tags|dependencies|devDependencies)\b'
    ),
}


@dataclass
class Finding:
    path: str
    bytes: int
    lines: int
    score: int
    matched_patterns: list[str]
    sample_lines: list[str]


def iter_text_files(root: Path) -> Iterable[Path]:
    skip_names = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    text_exts = {
        ".txt", ".md", ".markdown", ".json", ".jsonl", ".js", ".ts", ".tsx",
        ".jsx", ".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".go",
        ".rs", ".php", ".html", ".htm", ".css", ".scss", ".sql", ".xml",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash",
        ".rst", ".tex",
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_names for part in path.parts):
            continue
        if path.suffix.lower() not in text_exts and path.suffix.lower() != "":
            continue
        yield path


def scan_file(path: Path, root: Path) -> Finding | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    matched: list[str] = []
    snippets: list[str] = []

    lines = text.splitlines()
    line_hits: Counter[str] = Counter()

    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            matched.append(name)
            for i, line in enumerate(lines):
                if pattern.search(line):
                    line_hits[name] += 1
                    if len(snippets) < 8:
                        snippets.append(f"{i + 1}: {line[:240]}")

    # Extra heuristics for obvious metadata blobs.
    lower = text.lower()
    score = len(matched)

    if '"description"' in lower and '"type"' in lower and '"source"' in lower:
        score += 3
        if "json_metadata_bundle" not in matched:
            matched.append("json_metadata_bundle")

    if lower.count('"') > 100 and len(lines) > 5:
        score += 1
        matched.append("quote_heavy")

    # Strong signal: many metadata-style fields in a relatively short file.
    metadata_hits = sum(
        line_hits[k]
        for k in (
            "json_metadata_field",
            "schema_type_field",
            "npm_package_markers",
            "serialized_object_markers",
        )
    )
    if metadata_hits >= 8:
        score += 4
        matched.append("metadata_field_density")

    if not matched:
        return None

    matched = list(dict.fromkeys(matched))
    return Finding(
        path=str(path.relative_to(root)),
        bytes=path.stat().st_size,
        lines=len(lines),
        score=score,
        matched_patterns=matched,
        sample_lines=snippets,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    root = args.input.resolve()
    if not root.exists():
        parser.error(f"Input directory does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    findings: list[Finding] = []
    scanned = 0

    for path in iter_text_files(root):
        scanned += 1
        finding = scan_file(path, root)
        if finding and finding.score >= args.min_score:
            findings.append(finding)

    findings.sort(
        key=lambda x: (-x.score, -len(x.matched_patterns), -x.bytes, x.path.lower())
    )

    print("=" * 88)
    print("SUSPICIOUS CORPUS SCAN")
    print("=" * 88)
    print(f"Directory:       {root}")
    print(f"Files scanned:    {scanned:,}")
    print(f"Files flagged:    {len(findings):,}")
    print(f"Minimum score:    {args.min_score}")
    print()
    print("This is a DISCOVERY report. Nothing was modified or deleted.")
    print()

    if not findings:
        print("No files met the threshold.")
    else:
        for rank, item in enumerate(findings[: args.top], 1):
            print(f"[{rank:>3}] score={item.score:<2}  {item.path}")
            print(f"      size={item.bytes:,} bytes  lines={item.lines:,}")
            print(f"      patterns={', '.join(item.matched_patterns)}")
            if item.sample_lines:
                print("      samples:")
                for sample in item.sample_lines[:4]:
                    print(f"        {sample}")
            print()

        if len(findings) > args.top:
            print(f"... {len(findings) - args.top:,} more findings not shown.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(x) for x in findings], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON report saved to: {args.json}")

    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
