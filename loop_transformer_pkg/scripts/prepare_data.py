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

Usage
-----
    python scripts/prepare_data.py --input corpus.txt --output data/train.bin
    python scripts/prepare_data.py --input data/github_raw --output data/train.bin
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import json
import hashlib
import random
from pathlib import Path
from typing import Callable, List, Optional, Dict, Tuple

from filters import DocumentClassifier, Decision

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Never treat a scraper's own bookkeeping files as training text if
# --input points directly at e.g. scrape_github.py's output directory.
EXCLUDED_FILENAMES = {"manifest.jsonl", "_state.json"}
DOCUMENT_SEPARATOR = "\n\n<|endoftext|>\n\n"


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


def get_document_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"--input path does not exist: {path}")
    
    files = sorted(
        f for f in path.rglob("*")
        if f.is_file() and f.name not in EXCLUDED_FILENAMES
    )
    if not files:
        raise ValueError(f"--input directory {path} contains no usable files")
    return files

def compute_hash(text: str) -> str:
    # Normalized hash to prevent trivial duplicates from leaking
    norm_text = "".join(text.split())
    return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

def split_documents(files: List[Path], val_fraction: float, seed: int) -> Tuple[List[Path], List[Path]]:
    if len(files) <= 1:
        log.warning("Only one document found. Document-level validation split is impossible. All data will be used for training.")
        return files, []

    # Read files to compute hashes and group duplicates
    groups: Dict[str, List[Path]] = {}
    unreadable = 0
    
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            h = compute_hash(text)
            if h not in groups:
                groups[h] = []
            groups[h].append(f)
        except (UnicodeDecodeError, PermissionError, OSError) as e:
            unreadable += 1
            log.debug(f"Skipping unreadable file {f}: {e}")
            
    if unreadable:
        log.info(f"Skipped {unreadable:,} unreadable files")

    unique_hashes = sorted(list(groups.keys()))
    
    rng = random.Random(seed)
    rng.shuffle(unique_hashes)
    
    n_val_groups = max(1, int(len(unique_hashes) * val_fraction)) if val_fraction > 0 else 0
    if n_val_groups >= len(unique_hashes):
        n_val_groups = len(unique_hashes) - 1 # keep at least one for train
        
    val_hashes = set(unique_hashes[:n_val_groups])
    
    train_files = []
    val_files = []
    
    for h in unique_hashes:
        if h in val_hashes:
            val_files.extend(groups[h])
        else:
            train_files.extend(groups[h])
            
    # Sort files within splits for determinism
    train_files.sort()
    val_files.sort()
    
    return train_files, val_files

def process_split(files: List[Path], encode: Callable[[str], List[int]], dtype: str, output_path: Path,
                  classifier: DocumentClassifier, skip_rejected: bool = True) -> Tuple[int, int]:
    if not files:
        return 0, 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = 65536 if dtype == "uint16" else 2**32
    np_dtype = np.uint16 if dtype == "uint16" else np.uint32
    total_tokens = 0
    skipped = 0
    sep_ids = encode(DOCUMENT_SEPARATOR)
    with open(output_path, "wb") as f_out:
        for i, f in enumerate(files):
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                skipped += 1
                log.warning("Skipping unreadable document %s: %s", f, e)
                continue
            source = ""
            parts = {p.lower() for p in f.parts}
            if "github" in parts: source = "github"
            elif parts & {"books", "book", "gutenberg", "standard_ebooks", "wikisource"}: source = "books"
            elif "wikipedia" in parts: source = "wikipedia"
            elif "w3schools" in parts: source = "w3schools"
            decision, reason, _ = classifier.classify(text, path=str(f), source=source)
            if skip_rejected and decision.value.startswith("REJECT_"):
                skipped += 1
                log.warning("Skipping %s: %s (%s)", f, reason, decision.value)
                continue
            ids = encode(text)
            if i > 0 and ids:
                ids = sep_ids + ids
            if ids:
                max_id = max(ids)
                if max_id >= limit:
                    raise ValueError(f"Max token id {max_id} doesn't fit in {dtype} (limit {limit}).")
                f_out.write(np.asarray(ids, dtype=np_dtype).tobytes())
                total_tokens += len(ids)
    return total_tokens, skipped

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                    help="a UTF-8 text file, OR a directory of text files (walked and concatenated)")
    p.add_argument("--output", required=True, type=Path, help="output .bin path for the train split")
    p.add_argument("--tokenizer", default=None, help="'module:function' returning list[int]; default: byte-level")
    p.add_argument("--val-fraction", type=float, default=0.0005,
                    help="fraction of DOCUMENTS held out as validation (document-level split)")
    p.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"],
                    help="uint16 covers vocab_size up to 65536; use uint32 for larger vocabs")
    p.add_argument("--seed", type=int, default=0,
                    help="random seed for deterministic document-level train/val split")
    p.add_argument("--allow-rejected", action="store_true", help="Disable the shared junk-data gate (not recommended).")
    args = p.parse_args()

    encode = load_tokenizer(args.tokenizer)
    classifier = DocumentClassifier()

    all_files = get_document_files(args.input)
    log.info(f"Discovered {len(all_files):,} documents from {args.input}")

    # Apply the same quality gate before document-level splitting so rejected
    # junk cannot accidentally occupy the validation set.
    if args.allow_rejected:
        clean_files = all_files
        prefiltered = 0
    else:
        clean_files = []
        prefiltered = 0
        for f in all_files:
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                prefiltered += 1
                log.warning("Excluding unreadable document %s: %s", f, e)
                continue
            parts = {p.lower() for p in f.parts}
            source = ""
            if "github" in parts: source = "github"
            elif parts & {"books", "book", "gutenberg", "standard_ebooks", "wikisource"}: source = "books"
            elif "wikipedia" in parts: source = "wikipedia"
            elif "w3schools" in parts: source = "w3schools"
            decision, reason, _ = classifier.classify(text, path=str(f), source=source)
            if decision.value.startswith("REJECT_"):
                prefiltered += 1
                log.info("Quality gate: excluded %s: %s (%s)", f, reason, decision.value)
                continue
            clean_files.append(f)
        all_files = clean_files
        log.info(f"Quality gate excluded {prefiltered:,} documents before split")

    train_files, val_files = split_documents(all_files, args.val_fraction, args.seed)
    log.info(f"Split: {len(train_files):,} train documents, {len(val_files):,} validation documents")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("Tokenizing and writing train split...")
    train_tokens, train_skipped = process_split(train_files, encode, args.dtype, args.output, classifier, not args.allow_rejected)
    log.info(f"Wrote {train_tokens:,} train tokens to {args.output}")

    val_path = args.output.with_name(args.output.stem + ".val" + args.output.suffix)
    if val_files:
        log.info("Tokenizing and writing validation split...")
        val_tokens, val_skipped = process_split(val_files, encode, args.dtype, val_path, classifier, not args.allow_rejected)
        log.info(f"Wrote {val_tokens:,} validation tokens to {val_path}")
    else:
        val_tokens = 0
        val_skipped = 0
        log.info("No validation documents; validation file not written.")

    total_tokens = train_tokens + val_tokens
    val_pct = 100.0 * val_tokens / total_tokens if total_tokens > 0 else 0.0
    log.info(
        f"Summary: {len(all_files):,} docs total | "
        f"train={len(train_files):,} docs / {train_tokens:,} tokens | "
        f"val={len(val_files):,} docs / {val_tokens:,} tokens ({val_pct:.2f}%)"
    )

    split_report = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_document_count": len(train_files),
        "val_document_count": len(val_files),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_skipped_by_quality_gate": train_skipped,
        "val_skipped_by_quality_gate": val_skipped,
        "prefiltered_by_quality_gate": prefiltered,
        "train_documents": [str(f) for f in train_files],
        "val_documents": [str(f) for f in val_files],
    }
    report_path = args.output.with_name(args.output.stem + ".split.json")
    report_path.write_text(json.dumps(split_report, indent=2))
    log.info(f"Split report written to {report_path}")


if __name__ == "__main__":
    main()
