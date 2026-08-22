#!/usr/bin/env python3
"""Reference data-prep script: turns raw UTF-8 text into a packed
token-id binary file that train.py's data loader reads via numpy.memmap.

--input accepts either a single text file OR a directory of text files
(e.g. scrape_github.py's output, or any other directory of documents) --
directories are walked, sorted for determinism, and concatenated with a
document-boundary separator between files so unrelated documents don't
silently blend into one another.

This deliberately does NOT hard-depend on any tokenizer library. Default
is byte-level encoding (each byte 0-255 is one token) -- zero
dependencies, always works, good enough to smoke-test the training
pipeline end to end. For real training, bring your own tokenizer:

    python scripts/prepare_data.py --input corpus.txt --output data/train.bin \\
        --tokenizer mytokenizer:encode --vocab-size 32000

where mytokenizer.py defines `def encode(text: str) -> list[int]: ...`
(a thin wrapper around HuggingFace tokenizers, tiktoken, sentencepiece,
a custom BPE, whatever you already have).

Content filtering: if filters.py (shared with scan_suspicious_corpus.py /
audit_corpus.py / quarantine_web.py) is importable, every file is also run
through its DocumentClassifier here, and anything that isn't a KEEP_*
decision is left out of train.bin -- this is a last content-quality gate
at the point the corpus actually gets built, independent of whether the
separate audit -> quarantine -> fix_quarantines cleanup workflow was run
to completion. Pass --no-filter for the old unfiltered behavior.

Usage
-----
    python scripts/prepare_data.py --input corpus.txt --output data/train.bin
    python scripts/prepare_data.py --input data/github_raw --output data/train.bin
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Never treat a scraper's own bookkeeping files as training text if
# --input points directly at e.g. scrape_github.py's output directory.
EXCLUDED_FILENAMES = {"manifest.jsonl", "_state.json"}
DOCUMENT_SEPARATOR = "\n\n<|endoftext|>\n\n"

# filters.py (shared with scan_suspicious_corpus.py / audit_corpus.py /
# quarantine_web.py) is a SOFT dependency: if it's importable, its
# DocumentClassifier is applied as a last content-quality gate right here
# at build time, so train.bin is clean even if the separate
# audit -> quarantine -> fix_quarantines workflow was never run (or wasn't
# finished -- fix_quarantines.py's interactive mode doesn't scale past a
# few dozen files, so in practice that workflow alone tends to leave
# flagged files sitting in the corpus rather than actually removing them).
# If filters.py isn't present, prepare_data.py still works exactly as
# before -- this never becomes a hard dependency.
_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path(__file__).resolve().parent
for _p in (_ROOT / "scripts", _ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from filters import DocumentClassifier, infer_source
except ImportError:
    DocumentClassifier = None  # type: ignore[assignment,misc]
    infer_source = None  # type: ignore[assignment]


def load_tokenizer(spec: Optional[str]) -> Callable[[str], List[int]]:
    if spec is None:
        log.info(
            "No --tokenizer given; using byte-level encoding (token ids 0-255). "
            "Fine for smoke-testing the pipeline, not for real training quality "
            "or vocab efficiency -- pass --tokenizer module:function for that."
        )
        return lambda text: list(text.encode("utf-8"))
    if ":" not in spec:
        raise ValueError(f"--tokenizer must be 'module:function', got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    # Only the SCRIPT's own directory (scripts/) is on sys.path by default
    # when invoked as `python3 scripts/prepare_data.py` -- the current
    # working directory is not, which would silently break a --tokenizer
    # module sitting anywhere else (e.g. the repo root). Add cwd so a
    # tokenizer file can live wherever's convenient to run commands from.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def read_input(path: Path, classifier: Optional["DocumentClassifier"] = None) -> str:
    """Read --input as either a single file or a directory of files.

    Directory case: files are sorted by name (deterministic across runs
    on the same data) and concatenated with DOCUMENT_SEPARATOR between
    them, so the model can learn to recognize document boundaries rather
    than having unrelated files silently run together. Unreadable files
    (bad encoding, permission errors) are logged and skipped rather than
    aborting the whole run -- with thousands of scraped files, a handful
    of bad ones is normal, not fatal.

    If a classifier is given, each file is also run through it and
    anything that isn't a KEEP_* decision (raw package-registry dumps,
    link-list pages, logs, generated/minified blobs, ...) is skipped
    before it ever reaches the tokenizer.
    """
    if path.is_file():
        return path.read_text(encoding="utf-8")

    if not path.is_dir():
        raise FileNotFoundError(f"--input path does not exist: {path}")

    files = sorted(
        f for f in path.rglob("*")
        if f.is_file() and f.name not in EXCLUDED_FILENAMES
    )
    if not files:
        raise ValueError(f"--input directory {path} contains no usable files")

    log.info(f"Reading {len(files):,} files from directory {path}")
    chunks = []
    skipped_unreadable = 0
    skipped_filtered = 0
    filtered_reasons: Counter = Counter()
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            skipped_unreadable += 1
            log.debug(f"Skipping unreadable file {f}: {e}")
            continue
        if classifier is not None:
            source = infer_source(str(f))
            decision, reason, _ = classifier.classify(text, path=str(f), source=source)
            if decision.value.startswith("REJECT_"):
                skipped_filtered += 1
                filtered_reasons[decision.value] += 1
                log.debug(f"Filtered {f}: {decision.value} ({reason})")
                continue
        chunks.append(text)
    if skipped_unreadable:
        log.info(f"Skipped {skipped_unreadable:,}/{len(files):,} unreadable files")
    if skipped_filtered:
        log.info(f"Filtered {skipped_filtered:,}/{len(files):,} files as low-quality (see reasons below):")
        for k, v in filtered_reasons.most_common():
            log.info(f"  {k:24s} {v:6,d}")
    log.info(f"Keeping {len(chunks):,}/{len(files):,} files")
    if not chunks:
        raise ValueError(f"--input directory {path}: every file was filtered or unreadable, nothing left to write")
    return DOCUMENT_SEPARATOR.join(chunks)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="a UTF-8 text file, OR a directory of text files (walked and concatenated)")
    p.add_argument("--output", required=True, type=Path, help="output .bin path for the train split")
    p.add_argument("--tokenizer", default=None, help="'module:function' returning list[int]; default: byte-level")
    p.add_argument("--val-fraction", type=float, default=0.0005, help="fraction of tokens held out as val")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"],
                    help="uint16 covers vocab_size up to 65536; use uint32 for larger vocabs")
    p.add_argument("--no-filter", action="store_true",
                    help="skip the filters.py content-quality gate (if present) and use every file as-is, "
                         "same as before this flag existed")
    args = p.parse_args()

    classifier = None
    if not args.no_filter:
        if DocumentClassifier is not None:
            classifier = DocumentClassifier()
            log.info("Content filter: ON (filters.DocumentClassifier)")
        else:
            log.info("Content filter: OFF (filters.py not found next to this script -- pass --no-filter to silence this)")

    encode = load_tokenizer(args.tokenizer)
    text = read_input(args.input, classifier=classifier)
    log.info(f"Read {len(text):,} characters from {args.input}")

    ids = encode(text)
    log.info(f"Encoded to {len(ids):,} tokens")
    max_id = max(ids) if ids else 0
    limit = 65536 if args.dtype == "uint16" else 2**32
    if max_id >= limit:
        raise ValueError(
            f"Max token id {max_id} doesn't fit in {args.dtype} (limit {limit}). "
            f"Use --dtype uint32, or check your tokenizer's vocab size."
        )

    arr = np.array(ids, dtype=args.dtype)
    n_val = int(len(arr) * args.val_fraction)
    train_arr = arr[:-n_val] if n_val > 0 else arr
    val_arr = arr[-n_val:] if n_val > 0 else arr[:0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_arr.tofile(args.output)
    log.info(f"Wrote {len(train_arr):,} train tokens to {args.output}")

    if n_val > 0:
        val_path = args.output.with_name(args.output.stem + ".val" + args.output.suffix)
        val_arr.tofile(val_path)
        log.info(f"Wrote {len(val_arr):,} val tokens to {val_path}")


if __name__ == "__main__":
    main()
