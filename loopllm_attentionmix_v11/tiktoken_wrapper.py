"""Tiktoken wrapper for scripts/prepare_data.py's --tokenizer interface.

Usage:
    python scripts/prepare_data.py --input corpus.txt --output data/train.bin \
        --tokenizer tiktoken_wrapper:encode_p50k --dtype uint32

Note --dtype uint32, not the default uint16. p50k_base's own vocab
(50,281) actually FITS in uint16 (ceiling 65,536) -- but cl100k_base
(~100k) and o200k_base (~200k) do not, by a wide margin regardless of
their exact last digits. uint32 is recommended here as a single safe
default that works for any of the three without needing to remember
which ones fit and which don't.

Place this file somewhere importable (e.g. the repo root, or anywhere on
PYTHONPATH) -- prepare_data.py imports it by module name via importlib,
the same as any other --tokenizer target.
"""

from __future__ import annotations

from typing import List

import tiktoken

_encoders = {}  # cache: encoding.encode() re-compiles its regex on first
                 # call, so reuse one Encoding instance across calls
                 # rather than re-fetching per invocation.


def _get(name: str) -> "tiktoken.Encoding":
    if name not in _encoders:
        _encoders[name] = tiktoken.get_encoding(name)
    return _encoders[name]


def encode_p50k(text: str) -> List[int]:
    """p50k_base -- GPT-3/Codex era, vocab_size=50281. The minimum vocab
    size mentioned for this project; use --dtype uint32 in prepare_data.py."""
    return _get("p50k_base").encode(text, allowed_special="all")


def encode_cl100k(text: str) -> List[int]:
    """cl100k_base -- GPT-3.5/GPT-4 era, vocab_size=100277."""
    return _get("cl100k_base").encode(text, allowed_special="all")


def encode_o200k(text: str) -> List[int]:
    """o200k_base -- GPT-4o era, vocab_size=200019. What the original
    notebook this project started from was already using."""
    return _get("o200k_base").encode(text, allowed_special="all")
