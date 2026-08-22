#!/usr/bin/env python3
"""Compatibility entry point for the canonical v11 generator.

Use the root ``generate.py`` implementation so fixed-loop generation and
Stage-II adaptive early-exit generation share exactly one code path.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import main


if __name__ == "__main__":
    main()
