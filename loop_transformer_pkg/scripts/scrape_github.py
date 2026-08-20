#!/usr/bin/env python3
"""Scrape source files from public GitHub repositories into a directory
of text files, ready for prepare_data.py.

Builds on the same approach as a random-blob-per-step notebook prototype
(search repos by keyword -> pick one -> list its files -> download one),
hardened for real corpus collection:

  - Multiple files per repo (not one), capped per-repo for diversity
  - Filters vendored/generated paths (node_modules, dist, build, lockfiles,
    minified bundles, ...) in addition to binary extensions -- otherwise
    these dominate a scraped corpus with low-value repeated content
  - Content-hash deduplication (GitHub is full of forks, vendored copies,
    and boilerplate -- without this, a corpus ends up full of exact repeats)
  - Resumable: a manifest + state file mean re-running with the same
    --output-dir continues instead of re-downloading everything
  - GitHub API rate-limit aware (primary limit -> sleep until reset;
    secondary/abuse-detection limit -> exponential backoff)
  - Reads the token from the GITHUB_TOKEN environment variable, never
    a hardcoded string or (by default) a CLI argument -- CLI args are
    visible in shell history and process listings on shared machines.

Usage
-----
    export GITHUB_TOKEN=ghp_...          # optional but strongly recommended:
                                          # unauthenticated requests are capped
                                          # at 60/hour (10/min for search)
                                          # vs 5000/hour (30/min) authenticated
    python scripts/scrape_github.py --output-dir data/github_raw --num-files 5000

Then feed the result straight into the existing pipeline:
    python scripts/prepare_data.py --input data/github_raw --output data/train.bin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests
from filters import DocumentClassifier, Decision, Deduplicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("scrape_github")

API_ROOT = "https://api.github.com"

# Binary/media extensions -- never useful as text training data.
BAD_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".mp3", ".wav", ".ogg", ".flac", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".zip", ".7z", ".rar", ".gz", ".xz", ".bz2", ".tar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".obj",
    ".pdf", ".ttf", ".woff", ".woff2", ".eot", ".class", ".jar", ".pyc",
    ".whl", ".db", ".sqlite", ".sqlite3",
}

# Path substrings that mark vendored, generated, or build-output content --
# technically text, but low-value and heavily duplicated across repos.
BAD_PATH_PARTS = {
    "node_modules/", "vendor/", "vendored/", "dist/", "build/", "target/",
    ".git/", "__pycache__/", ".next/", ".nuxt/", "venv/", ".venv/",
    "site-packages/", "bower_components/", "coverage/", ".pytest_cache/",
    ".egg-info/",
}

# Lockfiles and similar: valid text, but machine-generated and extremely
# repetitive across the entire corpus -- not useful signal for an LM.
BAD_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "Gemfile.lock", "composer.lock", "go.sum",
}

DEFAULT_KEYWORDS = [
    "python", "javascript", "typescript", "java", "go", "rust", "c", "cpp",
    "api", "tool", "cli", "web", "data", "machine", "model", "server", "app",
    "ai", "framework", "library", "compiler", "parser", "game", "hacking",
]


# ======================================================================
# GitHub API client: auth, rate-limit handling, retries
# ======================================================================

class GitHubClient:
    def __init__(self, token: Optional[str], request_delay: float = 0.5):
        self.session = requests.Session()
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "loop-transformer-scraper/0.1",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        else:
            log.warning(
                "No GitHub token found (GITHUB_TOKEN env var unset). Proceeding "
                "unauthenticated: 60 requests/hour, 10/minute for search. Set "
                "GITHUB_TOKEN for 5000/hour, 30/minute instead."
            )
        self.request_delay = request_delay

    def get(self, url: str, params: Optional[dict] = None, max_retries: int = 5) -> requests.Response:
        """GET with rate-limit and transient-error handling. Sleeps and
        retries on primary rate-limit exhaustion, secondary (abuse-
        detection) limits, and 5xx server errors; returns directly on
        success or on errors the caller should decide about (404, etc.)."""
        for attempt in range(max_retries):
            time.sleep(self.request_delay)
            resp = self.session.get(url, headers=self.headers, params=params, timeout=30)

            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_for = max(reset_at - time.time(), 1) + 2
                log.warning(f"Primary rate limit exhausted; sleeping {sleep_for:.0f}s until reset")
                time.sleep(sleep_for)
                continue

            if resp.status_code == 403 and "secondary rate limit" in resp.text.lower():
                sleep_for = min(2 ** attempt * 10, 300)
                log.warning(f"Secondary rate limit hit; backing off {sleep_for}s (attempt {attempt+1})")
                time.sleep(sleep_for)
                continue

            if resp.status_code >= 500:
                sleep_for = 2 ** attempt
                log.warning(f"Server error {resp.status_code}; retrying in {sleep_for}s")
                time.sleep(sleep_for)
                continue

            return resp

        raise RuntimeError(f"Exceeded {max_retries} retries for {url}")

    def search_repos(self, keyword: str, language: Optional[str], min_stars: int,
                      max_page: int) -> List[dict]:
        query = f"{keyword} in:name"
        if language:
            query += f" language:{language}"
        if min_stars > 0:
            query += f" stars:>={min_stars}"
        resp = self.get(
            f"{API_ROOT}/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc",
                    "per_page": 10, "page": random.randint(1, max_page)},
        )
        if resp.status_code != 200:
            log.warning(f"Repo search failed ({resp.status_code}) for query {query!r}: {resp.text[:200]}")
            return []
        return resp.json().get("items", [])

    def get_tree(self, owner: str, repo: str, branch: str) -> List[dict]:
        branch_resp = self.get(f"{API_ROOT}/repos/{owner}/{repo}/branches/{branch}")
        if branch_resp.status_code != 200:
            return []
        commit_sha = branch_resp.json()["commit"]["sha"]
        tree_resp = self.get(
            f"{API_ROOT}/repos/{owner}/{repo}/git/trees/{commit_sha}",
            params={"recursive": 1},
        )
        if tree_resp.status_code != 200:
            return []
        tree = tree_resp.json()
        if tree.get("truncated"):
            log.info(f"{owner}/{repo}: tree truncated by GitHub (very large repo); "
                     f"only the returned subset is considered")
        return [x for x in tree.get("tree", []) if x.get("type") == "blob"]

    def download_raw(self, owner: str, repo: str, branch: str, path: str) -> Optional[str]:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        # raw.githubusercontent.com doesn't count against the REST API's
        # rate limit, so this bypasses self.get()'s throttling deliberately --
        # a small fixed delay here is still polite.
        time.sleep(0.2)
        resp = self.session.get(raw_url, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.text


# ======================================================================
# Filtering
# ======================================================================

def is_candidate_path(path: str, size: int, min_size: int, max_size: int) -> bool:
    lower = path.lower()
    if any(lower.endswith(ext) for ext in BAD_EXTENSIONS):
        return False
    if any(part in path for part in BAD_PATH_PARTS):
        return False
    if Path(path).name in BAD_FILENAMES:
        return False
    if not (min_size <= size <= max_size):
        return False
    return True


def looks_generated_or_minified(text: str, max_line_length: int = 2000) -> bool:
    """Cheap heuristic: a single absurdly long line strongly suggests a
    minified bundle or a generated data blob rather than hand-written
    source -- filter these out even if the extension looked innocent."""
    return any(len(line) > max_line_length for line in text.splitlines()[:50])


# ======================================================================
# State (resumability + dedup)
# ======================================================================

class ScrapeState:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_path = output_dir / "_state.json"
        self.manifest_path = output_dir / "manifest.jsonl"
        self.visited_repos: Set[str] = set()
        self.deduplicator = Deduplicator()
        self.total_saved = 0
        self.stats = {
            "downloaded": 0,
            "accepted": 0,
            "rejected": 0,
            "rejected_by_reason": {},
            "accepted_by_category": {},
            "duplicates": 0,
            "bytes_kept": 0,
            "bytes_rejected": 0
        }
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.visited_repos = set(data.get("visited_repos", []))
            self.deduplicator.exact_hashes = set(data.get("seen_hashes", []))
            self.deduplicator.norm_hashes = set(data.get("norm_hashes", []))
            self.total_saved = data.get("total_saved", 0)
            self.stats = data.get("stats", self.stats)
            log.info(f"Resumed state: {len(self.visited_repos)} repos visited, "
                     f"{self.total_saved} files already saved")

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "visited_repos": sorted(self.visited_repos),
            "seen_hashes": sorted(self.deduplicator.exact_hashes),
            "norm_hashes": sorted(self.deduplicator.norm_hashes),
            "total_saved": self.total_saved,
            "stats": self.stats
        }))

    def record_file(self, content_hash: str, meta: Dict[str, Any]) -> None:
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        self.total_saved += 1

    def record_rejection(self, meta: Dict[str, Any]) -> None:
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(meta) + "\n")


# ======================================================================
# Main scrape loop
# ======================================================================

def scrape(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = args.token or os.environ.get("GITHUB_TOKEN")
    client = GitHubClient(token, request_delay=args.request_delay)
    state = ScrapeState(args.output_dir)
    random.seed(args.seed)

    interrupted = {"flag": False}

    def _handle_interrupt(signum, frame):
        log.warning("Interrupt received -- will save state and exit after this repo")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    keywords = args.keywords.split(",") if args.keywords else DEFAULT_KEYWORDS
    languages = args.languages.split(",") if args.languages else [None]

    classifier = DocumentClassifier()

    empty_search_streak = 0
    while state.total_saved < args.num_files and not interrupted["flag"]:
        if empty_search_streak > 20:
            log.warning("20 consecutive empty searches -- widen --keywords/--languages "
                        "or lower --min-stars. Stopping.")
            break

        keyword = random.choice(keywords)
        language = random.choice(languages)
        repos = client.search_repos(keyword, language, args.min_stars, args.max_search_pages)
        if not repos:
            empty_search_streak += 1
            continue
        empty_search_streak = 0

        repo = random.choice(repos)
        full_name = repo["full_name"]
        if full_name in state.visited_repos:
            continue
        owner, name = repo["owner"]["login"], repo["name"]
        branch = repo.get("default_branch", "main")

        files = client.get_tree(owner, name, branch)
        candidates = [
            f for f in files
            if is_candidate_path(f["path"], f.get("size", 0), args.min_file_size, args.max_file_size)
        ]
        state.visited_repos.add(full_name)

        if not candidates:
            log.info(f"{full_name}: no usable files after filtering ({len(files)} total)")
            continue

        random.shuffle(candidates)
        saved_this_repo = 0
        for blob in candidates:
            if saved_this_repo >= args.max_files_per_repo:
                break
            path = blob["path"]
            content = client.download_raw(owner, name, branch, path)
            if content is None:
                continue
                
            state.stats["downloaded"] += 1

            content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
            if state.deduplicator.is_duplicate(content, content_hash):
                state.stats["duplicates"] += 1
                state.stats["rejected"] += 1
                state.stats["bytes_rejected"] += len(content)
                state.record_rejection({
                    "hash": content_hash, "repo": full_name, "path": path,
                    "branch": branch, "size": len(content), "classification": Decision.REJECT_DUPLICATE.value,
                    "reason": "Duplicate content"
                })
                continue

            decision, reason, stats = classifier.classify(content, path=path, source="github")
            
            if decision.name.startswith("REJECT"):
                state.stats["rejected"] += 1
                state.stats["bytes_rejected"] += len(content)
                state.stats["rejected_by_reason"][decision.value] = state.stats["rejected_by_reason"].get(decision.value, 0) + 1
                state.record_rejection({
                    "hash": content_hash, "repo": full_name, "path": path,
                    "branch": branch, "size": len(content), "classification": decision.value,
                    "reason": reason, "stats": stats
                })
                continue

            state.stats["accepted"] += 1
            state.stats["bytes_kept"] += len(content)
            state.stats["accepted_by_category"][decision.value] = state.stats["accepted_by_category"].get(decision.value, 0) + 1
            state.deduplicator.add(content, content_hash)

            out_path = args.output_dir / f"{content_hash[:16]}.txt"
            out_path.write_text(content, encoding="utf-8", errors="ignore")
            state.record_file(content_hash, {
                "hash": content_hash, "repo": full_name, "path": path,
                "branch": branch, "size": len(content), "filename": out_path.name,
                "classification": decision.value, "stats": stats
            })
            saved_this_repo += 1

            if state.total_saved % 20 == 0:
                state.save()
                log.info(f"Progress: {state.total_saved}/{args.num_files} files saved "
                         f"({len(state.visited_repos)} repos visited)")

            if state.total_saved >= args.num_files:
                break

        log.info(f"{full_name}: saved {saved_this_repo}/{len(candidates)} candidate files")

    state.save()
    log.info(f"Done. {state.total_saved} files saved to {args.output_dir} "
             f"({len(state.visited_repos)} repos visited). ")
             
    print("\n--- Scraping Statistics ---")
    print(f"Downloaded: {state.stats['downloaded']}")
    print(f"Accepted: {state.stats['accepted']}")
    for cat, count in state.stats['accepted_by_category'].items():
        print(f"  {cat}: {count}")
    print(f"Rejected: {state.stats['rejected']}")
    for reason, count in state.stats['rejected_by_reason'].items():
        print(f"  {reason}: {count}")
    print(f"  Duplicates: {state.stats['duplicates']}")
    print(f"Bytes kept: {state.stats['bytes_kept']}")
    print(f"Bytes rejected: {state.stats['bytes_rejected']}")
    print("---------------------------\n")
    
    log.info(f"Next: python scripts/prepare_data.py --input {args.output_dir} --output data/train.bin")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--num-files", type=int, default=2000, help="target total file count")
    p.add_argument("--keywords", default=None,
                    help="comma-separated repo-name search terms; default: a broad built-in list")
    p.add_argument("--languages", default=None,
                    help="comma-separated GitHub language filters, e.g. 'python,go'; default: unfiltered")
    p.add_argument("--min-stars", type=int, default=0)
    p.add_argument("--max-search-pages", type=int, default=10,
                    help="random page in [1, this] is sampled per search, matching GitHub search's practical result cap")
    p.add_argument("--max-files-per-repo", type=int, default=10,
                    help="cap per repo so no single repo dominates the corpus")
    p.add_argument("--min-file-size", type=int, default=200, help="bytes; skip trivially small files")
    p.add_argument("--max-file-size", type=int, default=200_000, help="bytes; skip huge outlier files")
    p.add_argument("--request-delay", type=float, default=0.5,
                    help="seconds between REST API calls, on top of rate-limit backoff")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--token", default=None,
                    help="GitHub token override. Prefer the GITHUB_TOKEN env var instead -- "
                         "CLI args are visible in shell history and `ps` output on shared machines.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scrape(args)


if __name__ == "__main__":
    main()
