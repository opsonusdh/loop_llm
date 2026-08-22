#!/usr/bin/env python3
"""Quarantine resolver: interactive by default, or batch with --yes/--restore-all.

Interactive mode (no flags) prompts once per file -- fine for reviewing a
handful of borderline cases, but doesn't scale past a few dozen files, and
nothing is actually removed from the active corpus until every file has
been answered.

--yes deletes every quarantined file's matching source in one pass. The
quarantine folder should already only contain files the shared classifier
rejected (via audit_corpus.py + quarantine_web.py), so this is meant to be
the normal "commit the cleanup" step once you've spot-checked a sample.

--restore-all is the reverse, for undoing a quarantine batch that turns
out to have been too aggressive.

Use --dry-run with either batch flag to preview counts first without
changing anything.

Examples:
    python fix_quarantines.py --quarantine data/quarantine/web --source data/raw --yes
    python fix_quarantines.py --quarantine data/quarantine/web --source data/raw --yes --dry-run
    python fix_quarantines.py --quarantine data/quarantine/web --source data/raw --restore-all
    python fix_quarantines.py --quarantine data/quarantine/web --source data/raw   # interactive, per file
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quarantine", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--yes", "-y", action="store_true",
                       help="Batch mode: delete every quarantined file's matching source, no per-file prompts.")
    mode.add_argument("--restore-all", action="store_true",
                       help="Batch mode: restore every quarantined file back to its source path, no per-file prompts.")
    ap.add_argument("--dry-run", action="store_true",
                     help="With --yes/--restore-all: show what would happen without changing anything. "
                          "No effect in interactive mode (each prompt already asks before acting).")
    args = ap.parse_args()

    q = args.quarantine.resolve()
    s = args.source.resolve()
    if not q.is_dir():
        ap.error(f"Quarantine directory does not exist: {q}")
    if not s.is_dir():
        ap.error(f"Source directory does not exist: {s}")

    files = sorted(p for p in q.rglob("*") if p.is_file())
    print(f"Found {len(files):,} quarantined files.")
    if not files:
        return 0

    deleted = restored = skipped = 0

    if args.yes or args.restore_all:
        verb = "Deleting" if args.yes else "Restoring"
        suffix = " (dry run -- nothing will change)" if args.dry_run else ""
        print(f"Batch mode: {verb.lower()} all {len(files):,} files{suffix}.")
        for qfile in files:
            rel = qfile.relative_to(q)
            src = s / rel
            if args.dry_run:
                print(f"  [DRY] {verb}: {rel}")
                continue
            if args.yes:
                if src.exists():
                    src.unlink()
                qfile.unlink()
                deleted += 1
            else:
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(qfile, src)
                qfile.unlink()
                restored += 1
        if not args.dry_run:
            print(f"\nDone. Deleted: {deleted:,}  Restored: {restored:,}")
        return 0

    # Interactive: same per-file prompt as before, plus a final summary
    # (the original never printed one, so a run that got interrupted partway
    # left no record of what had actually been resolved).
    for qfile in files:
        rel = qfile.relative_to(q)
        src = s / rel
        print(f"\nFILE: {rel}")
        print(f"QUARANTINE: {qfile}")
        print(f"SOURCE:     {src}")

        while True:
            ans = input("Delete [d], Restore [r], Skip [s]? ").strip().lower()
            if ans in {"d", "r", "s"}:
                break

        if ans == "s":
            skipped += 1
            continue

        if ans == "d":
            qfile.unlink()
            if src.exists():
                src.unlink()
            print("Deleted quarantine copy and matching source file.")
            deleted += 1
            continue

        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(qfile, src)
        qfile.unlink()
        print("Restored to the matching relative source path.")
        restored += 1

    print(f"\nDone. Deleted: {deleted:,}  Restored: {restored:,}  Skipped: {skipped:,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
