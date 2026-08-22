#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path


def collect_files(root: Path) -> list[Path]:
    """Find all UTF-8 text files recursively."""
    return sorted(
        p for p in root.rglob("*.txt")
        if p.is_file()
    )


def random_chunk(
    text: str,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
) -> str:
    """Return a random character chunk from a document."""
    text = text.strip()

    if not text:
        return ""

    if len(text) <= min_chars:
        return text

    chunk_size = rng.randint(
        min_chars,
        min(max_chars, len(text)),
    )

    start = rng.randint(0, len(text) - chunk_size)
    chunk = text[start:start + chunk_size]

    # Try to avoid cutting immediately in the middle of a line.
    first_newline = chunk.find("\n")
    if first_newline != -1 and first_newline < 100:
        chunk = chunk[first_newline + 1 :]

    return chunk.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show random text chunks from a corpus."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Corpus directory.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of random chunks to display.",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum chunk size.",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="Maximum chunk size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )

    parser.add_argument(
        "--full-file",
        action="store_true",
        help="Show the selected entire file instead of a random chunk.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Corpus does not exist: {args.input}")

    files = collect_files(args.input)

    if not files:
        raise SystemExit(f"No .txt files found under: {args.input}")

    rng = random.Random(args.seed)

    print(f"Found {len(files):,} documents")
    print(f"Showing {args.samples} random samples")
    print("=" * 100)

    chosen = rng.sample(
        files,
        k=min(args.samples, len(files)),
    )

    for i, path in enumerate(chosen, 1):
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            print(f"\n[{i}] ERROR reading {path}: {exc}")
            continue

        print("\n" + "=" * 100)
        print(f"[SAMPLE {i}/{len(chosen)}]")
        print(f"FILE: {path}")
        print(f"CHARS: {len(text):,}")
        print("-" * 100)

        if args.full_file:
            sample = text.strip()
        else:
            sample = random_chunk(
                text,
                rng,
                args.min_chars,
                args.max_chars,
            )

        print(sample)

    print("\n" + "=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()