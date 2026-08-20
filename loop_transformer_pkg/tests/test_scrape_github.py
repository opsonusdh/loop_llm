"""Tests for scrape_github.py's pure logic -- filtering and state
persistence. Deliberately excludes anything requiring live network
access (that's covered by manual/integration testing against the real
GitHub API instead, same as prepare_data.py and train.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from scrape_github import ScrapeState, is_candidate_path, looks_generated_or_minified  # noqa: E402


class TestIsCandidatePath:
    def test_accepts_normal_source_file(self):
        assert is_candidate_path("src/main.py", 1000, 200, 200_000)

    def test_rejects_binary_extension(self):
        assert not is_candidate_path("assets/logo.png", 1000, 200, 200_000)

    def test_rejects_vendored_path(self):
        assert not is_candidate_path("node_modules/lodash/index.js", 1000, 200, 200_000)
        assert not is_candidate_path("vendor/lib/thing.go", 1000, 200, 200_000)
        assert not is_candidate_path("project/dist/bundle.js", 1000, 200, 200_000)

    def test_rejects_lockfiles(self):
        assert not is_candidate_path("package-lock.json", 1000, 200, 200_000)
        assert not is_candidate_path("subdir/yarn.lock", 1000, 200, 200_000)

    def test_rejects_too_small(self):
        assert not is_candidate_path("src/main.py", 50, 200, 200_000)

    def test_rejects_too_large(self):
        assert not is_candidate_path("src/main.py", 500_000, 200, 200_000)

    def test_accepts_at_size_boundaries(self):
        assert is_candidate_path("src/main.py", 200, 200, 200_000)
        assert is_candidate_path("src/main.py", 200_000, 200, 200_000)


class TestLooksGeneratedOrMinified:
    def test_normal_source_not_flagged(self):
        text = "\n".join(f"line {i} of normal code" for i in range(20))
        assert not looks_generated_or_minified(text)

    def test_single_long_line_flagged(self):
        text = "normal line\n" + ("x" * 5000) + "\nmore normal code"
        assert looks_generated_or_minified(text)

    def test_only_checks_first_50_lines(self):
        # A long line beyond the first 50 shouldn't trigger the heuristic --
        # it's meant to catch minified headers/bundles, not scan the whole file.
        text = "\n".join("short line" for _ in range(60)) + "\n" + ("x" * 5000)
        assert not looks_generated_or_minified(text)


class TestScrapeState:
    def test_fresh_state_is_empty(self, tmp_path):
        state = ScrapeState(tmp_path)
        assert state.visited_repos == set()
        assert state.seen_hashes == set()
        assert state.total_saved == 0

    def test_record_file_updates_hashes_and_count(self, tmp_path):
        state = ScrapeState(tmp_path)
        state.record_file("abc123", {"hash": "abc123", "repo": "x/y", "path": "f.py"})
        assert "abc123" in state.seen_hashes
        assert state.total_saved == 1
        assert (tmp_path / "manifest.jsonl").exists()

    def test_save_and_reload_roundtrip(self, tmp_path):
        state = ScrapeState(tmp_path)
        state.visited_repos.add("owner/repo1")
        state.record_file("hash1", {"hash": "hash1", "repo": "owner/repo1", "path": "a.py"})
        state.save()

        reloaded = ScrapeState(tmp_path)
        assert reloaded.visited_repos == {"owner/repo1"}
        assert reloaded.seen_hashes == {"hash1"}
        assert reloaded.total_saved == 1

    def test_manifest_accumulates_across_saves(self, tmp_path):
        state = ScrapeState(tmp_path)
        state.record_file("h1", {"hash": "h1"})
        state.record_file("h2", {"hash": "h2"})
        lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
        assert len(lines) == 2
