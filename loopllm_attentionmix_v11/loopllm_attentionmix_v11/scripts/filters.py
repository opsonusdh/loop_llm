import hashlib
import math
import re
from collections import Counter
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional


class Decision(Enum):
    KEEP_CODE = "KEEP_CODE"
    KEEP_PROSE = "KEEP_PROSE"
    KEEP_TUTORIAL = "KEEP_TUTORIAL"
    REJECT_TABULAR = "REJECT_TABULAR"
    REJECT_PROMPT = "REJECT_PROMPT"
    REJECT_AGENT_TRACE = "REJECT_AGENT_TRACE"
    REJECT_GENERATED = "REJECT_GENERATED"
    REJECT_BENCHMARK = "REJECT_BENCHMARK"
    REJECT_LOW_QUALITY = "REJECT_LOW_QUALITY"
    REJECT_DUPLICATE = "REJECT_DUPLICATE"
    REJECT_LOG = "REJECT_LOG"
    REJECT_BINARY = "REJECT_BINARY"
    REJECT_LICENSE = "REJECT_LICENSE"
    REJECT_WEB_ARTIFACT = "REJECT_WEB_ARTIFACT"
    REJECT_OTHER = "REJECT_OTHER"


_GUTENBERG_START_PATTERNS = [
    re.compile(r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}", re.I),
    re.compile(r"\*{3}\s*START OF THE PROJECT GUTENBERG[^\n]*\*{3}", re.I),
    re.compile(r"\*END\*THE SMALL PRINT! FOR PUBLIC DOMAIN EBOOKS[^\n]*\*", re.I),
    re.compile(r"<<\s*THIS ETEXT IS PREPARED BY[^\n]*>>", re.I),
]
_GUTENBERG_END_PATTERNS = [
    re.compile(r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}", re.I),
    re.compile(r"\*{3}\s*END OF THE PROJECT GUTENBERG[^\n]*\*{3}", re.I),
    re.compile(r"\*{3}END OF THE PROJECT GUTENBERG", re.I),
    re.compile(r"End of (the )?Project Gutenberg['’s]* EBook", re.I),
    re.compile(r"End of Project Gutenberg['’s]*", re.I),
    re.compile(r"\*\*\* END: FULL LICENSE \*\*\*", re.I),
]


def clean_gutenberg_text(raw_text: str) -> str:
    text = raw_text
    start_pos = -1
    for pattern in _GUTENBERG_START_PATTERNS:
        match = pattern.search(text)
        if match:
            start_pos = match.end()
            break
    if start_pos != -1:
        text = text[start_pos:]
    end_pos = -1
    for pattern in _GUTENBERG_END_PATTERNS:
        match = pattern.search(text)
        if match:
            end_pos = match.start()
            break
    if end_pos != -1:
        text = text[:end_pos]
    return text.strip()


def classify_book_type(text: str, title: str = "", subjects: Optional[List[str]] = None) -> str:
    if subjects:
        subj = " ".join(subjects).lower()
        if any(k in subj for k in ["drama", "plays", "tragedies", "comedies", "theatre"]):
            return "drama"
        if any(k in subj for k in ["poetry", "poems", "verse", "sonnets"]):
            return "poetry"
        if any(k in subj for k in ["grammar", "english language -- grammar", "linguistics"]):
            return "grammar"
        if any(k in subj for k in ["rhetoric", "composition", "oratory", "public speaking"]):
            return "rhetoric"
        if any(k in subj for k in ["philosophy", "ethics", "logic", "metaphysics"]):
            return "philosophy"
        if any(k in subj for k in ["history", "biography", "historical", "chronicle"]):
            return "history"
        if any(k in subj for k in ["science", "mathematics", "physics", "astronomy", "biology", "chemistry"]):
            return "science"
        if any(k in subj for k in ["reference", "dictionary", "encyclopedia", "handbook", "glossary"]):
            return "reference"
        if any(k in subj for k in ["education", "textbook", "reader", "school"]):
            return "educational"
        if any(k in subj for k in ["fiction", "literature", "novel", "stories", "tales"]):
            return "literature"
    t = title.lower()
    if any(k in t for k in ["grammar", "grammatical"]): return "grammar"
    if any(k in t for k in ["rhetoric", "composition"]): return "rhetoric"
    if any(k in t for k in ["poem", "poetry", "verse", "sonnet"]): return "poetry"
    if any(k in t for k in ["play", "tragedy", "comedy"]): return "drama"
    if any(k in t for k in ["dictionary", "encyclopedia", "handbook"]): return "reference"
    sample = text[:15000]
    lines = [l.strip() for l in sample.splitlines() if l.strip()]
    act_scene = sum(bool(re.match(r'^(ACT\s+[IVXLCDM\d]+|SCENE\s+[IVXLCDM\d]+)', l, re.I)) for l in lines)
    speaker_lines = sum(bool(re.match(r'^[A-Z]{2,15}[\.:]\s+', l)) for l in lines)
    if act_scene >= 2 or (speaker_lines >= 8 and len(lines) > 20 and speaker_lines / len(lines) > .15): return "drama"
    if len(lines) > 25 and sum(10 <= len(l) <= 60 for l in lines) / len(lines) > .75: return "poetry"
    grammar_terms = sum(term in sample.lower() for term in ["noun","verb","adjective","pronoun","adverb","preposition","conjunction","syntax","conjugation","inflection"])
    if grammar_terms >= 5: return "grammar"
    return "literature"


# Shared by every script that walks the corpus (scan/audit/quarantine/
# prepare_data) so "what source is this file from" is answered exactly the
# same way everywhere, rather than N slightly-different local copies
# drifting apart over time.
_BOOK_PATH_MARKERS = {"books", "book", "gutenberg", "standard_ebooks", "wikisource"}
_WEB_PATH_MARKERS = ("github", "wikipedia", "w3schools", "custom")


def infer_source(path: str) -> str:
    parts = {p.lower() for p in re.split(r"[\\/]+", str(path)) if p}
    if "github" in parts:
        return "github"
    if parts & _BOOK_PATH_MARKERS:
        return "books"
    for source in _WEB_PATH_MARKERS:
        if source in parts:
            return source
    return ""


class DocumentClassifier:
    """Conservative corpus gate. Rejects known junk while preserving real code/tutorials/books."""
    BAD_FILENAMES = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock",
        "gemfile.lock", "composer.lock", "go.sum", "package.resolved", "npm-shrinkwrap.json",
        "pipfile.lock", "uv.lock", "podfile.lock", "packages.lock.json", "mix.lock",
        "flake.lock", "shard.lock", "conan.lock", "bun.lockb", "deno.lock",
    }
    BAD_PATH_PARTS = {
        "/node_modules/", "/vendor/", "/vendored/", "/dist/", "/build/", "/target/", "/.git/",
        "/__pycache__/", "/site-packages/", "/bower_components/", "/coverage/", "/.pytest_cache/",
        "/.next/", "/.nuxt/", "/.gradle/", "/deriveddata/", "/generated/", "/snapshots/",
    }
    BINARY_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".mp3", ".wav", ".ogg", ".flac",
        ".mp4", ".mkv", ".avi", ".mov", ".webm", ".zip", ".7z", ".rar", ".gz", ".xz", ".bz2", ".tar",
        ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".obj", ".pdf", ".ttf", ".woff", ".woff2",
        ".eot", ".class", ".jar", ".pyc", ".whl", ".db", ".sqlite", ".sqlite3",
    }
    PROMPT_PATTERNS = [
        r"\bsystem prompt\b", r"\byou are (?:an? )?(?:ai|language model|assistant)\b", r"\btool call\b",
        r"\bverification (?:subagent|agent)\b", r"\bthe only job\b", r"\bcaller may\b",
        r"\b(?:human|assistant|system|user)\s*:\s*", r"\binstruction\s*:\s*",
    ]
    GENERATED_PATTERNS = [
        r"\bauto[- ]generated\b", r"\bautomatically generated\b", r"\bmachine[- ]generated\b",
        r"\bthis file (?:was|is) generated\b", r"\bdo not edit\b.*\bgenerated\b", r"^\s*DO NOT EDIT\s*$",
    ]
    LOG_PATTERNS = [
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', r'^\[\d{4}-\d{2}-\d{2}',
        r'\b(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b',
    ]
    COMPILER_IR_PATTERNS = [r"^FILE fqName:", r"^\s*FUN name:", r"^\s*BLOCK_BODY$", r"^\s*CALL '.* declared in ", r"^\s*CONST (?:Int|String|Long|Boolean)", r"^\s*VAR name:", r"^\s*\$this:"]
    LICENSE_PATTERNS = [r"permission is hereby granted", r"apache license, version 2\.0", r"mit license", r"redistribution and use in source and binary forms", r"software is provided ['\"]as is['\"]", r"without warranty of any kind"]
    BOOK_CATALOG_PATTERNS = ["download this book", "plain text utf-8", "bibliographic record", "marc record", "rdf description", "table of contents navigation", "terms of use at www.gutenberg.org"]
    MOJIBAKE_MARKERS = ["Ã", "Â", "â€", "â€™", "â€œ", "â€�", "â€“", "â€”", "ðŸ", "ï¼", "à°", "è¯", "æ–", "å¤"]

    # Machine-readable web/package metadata that is harmful as LM training text.
    # These are deliberately structural, not keyword-only, so normal source code
    # mentioning "description", "type", or GitHub URLs is preserved.
    WEB_METADATA_KEYS = {
        "name", "version", "description", "repository", "homepage", "license",
        "author", "contributors", "maintainers", "keywords", "scripts",
        "dependencies", "devdependencies", "peerdependencies", "engines",
        "directories", "dist-tags", "versions", "_id", "_rev", "_npmversion",
        "_nodeversion", "_npmuser", "_npmoperationalinternal", "dist",
        "resolved", "integrity", "shasum", "tarball",
    }
    # Fields that are, on their own, essentially unambiguous evidence of a
    # raw package-registry API response rather than authored code -- no
    # legitimate source file has a reason to contain these. A single hit
    # is enough to reject.
    STRONG_REGISTRY_MARKERS = (
        '"dist-tags"', '"_npmuser"', '"_npmoperationalinternal"',
        '"_npmversion"', '"_nodeversion"', '"_hasshrinkwrap"', '"_resolved"',
    )
    # Real signals, but more ambiguous alone (e.g. "versions" can appear in
    # prose) -- these only count combined with each other or with the
    # structural checks below.
    WEAK_REGISTRY_MARKERS = (
        '"versions"', '"package-lock"', '"readmefilename"', '"maintainers"',
        '"_id"', '"_rev"',
    )
    # Matches a JSON "version": "1.2.3"-shaped field. A handful of these can
    # be legitimate (a package.json plus a changelog snippet); many in one
    # file is what a version-history dump from ANY package registry (npm,
    # PyPI, crates.io, RubyGems, Packagist, ...) looks like, regardless of
    # which registry-specific field names happen to be present.
    VERSION_FIELD_RE = re.compile(r'"version"\s*:\s*"\d+\.\d+(?:\.\d+)?[\w.+\-]*"')
    # Markdown link syntax, including a badge-image wrapped in a link
    # ([![alt](img)](url)). Near-exclusive to markdown/web content -- real
    # source code essentially never contains this shape.
    MD_LINK_RE = re.compile(r'\[!?\[[^\]]*\]\([^)]*\)\]\([^)]*\)|\[[^\]]*\]\([^)]*\)')

    def _stats(self, text: str) -> Dict[str, Any]:
        lines = text.splitlines()
        nonempty = [x for x in lines if x.strip()]
        tokens = re.findall(r"\S+", text)
        numeric = sum(bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", t.strip(",;:[](){}\"'"))) for t in tokens)
        replacement = text.count("\ufffd")
        controls = sum(ord(c) < 32 and c not in "\n\r\t" for c in text)
        line_lengths = [len(x) for x in nonempty]
        allowed = set(".,;:!?_`~@#$%^&*()[]{}<>/\\|+-=—–…") | {chr(34), chr(39), "“", "”", "‘", "’"}
        weird = sum(not (c.isalnum() or c.isspace() or c in allowed) for c in text)
        return {
            "length": len(text), "lines": len(lines), "nonempty_lines": len(nonempty), "words": len(tokens),
            "numeric_ratio": numeric / len(tokens) if tokens else 0.0,
            "avg_line_length": sum(line_lengths)/len(line_lengths) if line_lengths else 0.0,
            "max_line_length": max(line_lengths, default=0), "replacement_chars": replacement,
            "replacement_ratio": replacement / max(len(text),1), "control_chars": controls,
            "control_ratio": controls / max(len(text),1), "weird_char_ratio": weird / max(len(text),1),
            "unique_line_ratio": len(set(nonempty)) / max(len(nonempty),1),
        }

    def _path_reject(self, path: str) -> Optional[Tuple[Decision, str]]:
        p = "/" + path.replace("\\", "/").lower().lstrip("/")
        name = p.rsplit("/", 1)[-1]
        if name in self.BAD_FILENAMES: return Decision.REJECT_GENERATED, "Generated dependency lockfile"
        if any(part in p for part in self.BAD_PATH_PARTS): return Decision.REJECT_GENERATED, "Vendored/generated/build artifact path"
        ext = "." + name.rsplit(".",1)[-1] if "." in name else ""
        if ext in self.BINARY_EXTENSIONS: return Decision.REJECT_BINARY, f"Binary/media extension {ext}"
        return None

    @staticmethod
    def _delimiter_consistency(text: str) -> float:
        rows = [x for x in text.splitlines() if x.strip()][:100]
        if len(rows) < 5: return 0.0
        best = 0.0
        for delim in [",", "\t", "|"]:
            counts = [r.count(delim) for r in rows]
            pos = [c for c in counts if c > 0]
            if len(pos) < max(5, len(rows)//2): continue
            freq = Counter(pos)
            consistency = freq.most_common(1)[0][1] / len(rows)
            if sum(pos)/len(pos) >= 1.5: best = max(best, consistency)
        return best

    def _repetition(self, text: str) -> float:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if len(lines) < 12: return 0.0
        c = Counter(lines)
        return sum(v for v in c.values() if v >= 3) / len(lines)

    def _blob_artifact(self, text: str, s: Dict[str, Any]) -> Optional[str]:
        if len(text) < 12000 or s["words"] < 10:
            return None
        # Long single-line base64/hex payloads are usually embedded assets,
        # serialized data, or build artifacts rather than useful source.
        sample = text[:20000].replace("\n", "").replace("\r", "")
        if len(sample) >= 5000:
            base64_like = sum(c.isalnum() or c in "+/=_-" for c in sample) / len(sample)
            hex_like = sum(c.lower() in "0123456789abcdef" for c in sample) / len(sample)
            if base64_like > 0.995 and s["nonempty_lines"] <= 12 and s["avg_line_length"] > 1000:
                return "Large embedded base64/data blob"
            if hex_like > 0.985 and s["nonempty_lines"] <= 20 and s["avg_line_length"] > 700:
                return "Large hexadecimal/data blob"
        return None

    def _binary_or_corrupt(self, text: str, s: Dict[str, Any]) -> Optional[str]:
        if s["replacement_ratio"] >= 0.002 and s["replacement_chars"] >= 4:
            return f"Replacement-character/binary corruption ratio={s['replacement_ratio']:.5f}"
        if s["control_chars"] >= 4 and s["control_ratio"] >= 0.0005:
            return "Embedded binary/control characters"
        # Large binary-ish payloads often survive extension filters as mislabeled files.
        if len(text) >= 10000 and (s["weird_char_ratio"] > 0.03 or s["control_ratio"] > 0.0001):
            return "High non-text/binary character density"
        return None

    def _license_only(self, text: str, s: Dict[str, Any], source: str) -> bool:
        if source != "github" or s["words"] < 30 or s["length"] > 20000:
            return False
        low = text.lower()
        hits = sum(bool(re.search(p, low)) for p in self.LICENSE_PATTERNS)
        code_markers = len(re.findall(r"\b(class|def|function|fn|struct|#include|import|package|namespace|return|if|for|while)\b", low))
        return hits >= 3 and code_markers <= 4 and len(re.findall(r"https?://", low)) <= 3

    def _prompt_or_agent(self, text: str) -> bool:
        low = text.lower()
        hits = sum(bool(re.search(p, low, re.I|re.M)) for p in self.PROMPT_PATTERNS)
        return hits >= 2 or ("tool call" in low and ("assistant" in low or "user" in low))

    def _generated(self, text: str, s: Dict[str, Any]) -> bool:
        header = "\n".join(text.splitlines()[:20])
        if any(re.search(p, header, re.I|re.M) for p in self.GENERATED_PATTERNS):
            return True
        if s["max_line_length"] > 5000 and s["avg_line_length"] > 300:
            return True
        # Minified/bundled payloads: one/two huge lines and almost no whitespace.
        if len(text) > 10000 and s["avg_line_length"] > 1200 and s["nonempty_lines"] <= 8:
            return True
        return False

    def _benchmark(self, text: str) -> bool:
        low = text.lower()
        return (("{\"input\":" in low and "\"target\":" in low) or
                ("{\"prompt\":" in low and "\"completion\":" in low))

    def _web_metadata_artifact(self, text: str, s: Dict[str, Any], source: str) -> Optional[str]:
        """Reject machine-readable registry/package/web metadata, not ordinary code."""
        if source != "github":
            return None

        stripped = text.lstrip()
        low = text.lower()

        code_markers = len(re.findall(
            r"(?m)^\s*(?:def\s+\w+|class\s+\w+|function\s+\w+|"
            r"async\s+function\s+\w+|#include\s*[<\"]|package\s+\w+|"
            r"fn\s+\w+|func\s+\w+|public\s+(?:class|static)|"
            r"import\s+\w+|from\s+\S+\s+import\s+)",
            text
        ))

        # A single unambiguous registry-API field is enough on its own --
        # nothing legitimate (code, docs, a normal package.json a human
        # wrote by hand) has a reason to contain "_npmUser" or "dist-tags".
        strong_hits = sum(m in low for m in self.STRONG_REGISTRY_MARKERS)
        if strong_hits >= 1 and code_markers <= 2:
            return "NPM/package-registry metadata dump"

        weak_hits = sum(m in low for m in self.WEAK_REGISTRY_MARKERS)
        if strong_hits + weak_hits >= 2 and code_markers <= 2:
            return "NPM/package-registry metadata dump"

        # Repeated "version": "x.y.z" fields -- the shape of a version-history
        # dump from any package registry, not just npm-branded ones. A real
        # package.json has exactly one; a registry dump has one per release
        # (often dozens to hundreds).
        version_hits = len(self.VERSION_FIELD_RE.findall(text))
        if version_hits >= 6 and code_markers <= 2:
            return f"Version-history registry dump ({version_hits} version fields)"

        # JSON/YAML-like package manifests. Require multiple independent
        # metadata keys and low evidence of real executable source code.
        key_hits = 0
        for key in self.WEB_METADATA_KEYS:
            if re.search(r'(?m)^\s*["\']?' + re.escape(key) + r'["\']?\s*:', low):
                key_hits += 1

        objectish = (
            stripped.startswith("{")
            or stripped.startswith("[")
            or (":" in stripped[:1200] and stripped.lstrip().split("\n", 1)[0].count(":") >= 1)
        )

        if objectish and key_hits >= 6 and code_markers <= 2:
            return f"Machine-readable package/web metadata ({key_hits} metadata keys)"

        # High-density JSON records: repeated quoted key/value lines with many
        # metadata keys. This catches API/registry payloads even when minified,
        # OR buried mid-file (e.g. appended after unrelated content) rather
        # than only right at the top of the file.
        if s["length"] >= 800 and s["words"] >= 25:
            quoted_pairs = len(re.findall(r'["\'][A-Za-z_$][A-Za-z0-9_$.-]*["\']\s*:', text))
            if quoted_pairs >= 12 and key_hits >= 5 and code_markers <= 3:
                return f"Machine-readable metadata record ({key_hits} metadata keys)"
            nonempty = s["nonempty_lines"]
            if nonempty >= 15:
                kv_lines = len(re.findall(r'(?m)^\s*["\']?[A-Za-z_$][\w$.\-]*["\']?\s*:\s*\S', text))
                if kv_lines / nonempty >= 0.55 and key_hits >= 3 and code_markers <= 2:
                    return f"Dense key/value JSON block ({key_hits} metadata keys, {kv_lines} kv lines)"

        return None

    def _link_list_artifact(self, text: str, s: Dict[str, Any], source: str) -> Optional[str]:
        """Reject 'awesome-list'/directory pages that are almost entirely
        badges and bullet links with little original prose -- legitimate
        markdown, but low training value and easy to mistake for
        substantial documentation given how long they can run."""
        if source not in ("github", "wikipedia", "w3schools", "custom"):
            return None
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 15:
            return None
        link_heavy = 0
        for l in lines:
            matched_len = sum(len(m.group(0)) for m in self.MD_LINK_RE.finditer(l))
            if matched_len and matched_len / len(l) >= 0.4:
                link_heavy += 1
        if link_heavy / len(lines) < 0.55:
            return None
        code_markers = len(re.findall(
            r"(?m)^\s*(?:def\s+\w+|class\s+\w+|function\s+\w+|import\s+\w+|from\s+\S+\s+import\s+)",
            text
        ))
        if code_markers <= 2:
            return f"Link/badge directory, not prose ({link_heavy}/{len(lines)} lines are mostly markdown links)"
        return None

    def _tabular(self, text: str, s: Dict[str, Any]) -> bool:
        if s["nonempty_lines"] < 5: return False
        best = self._delimiter_consistency(text)
        if best >= 0.80 and s["numeric_ratio"] >= 0.20: return True
        if best >= 0.95 and s["nonempty_lines"] >= 20: return True
        return False

    def _log(self, text: str, s: Dict[str, Any]) -> bool:
        lines = [x for x in text.splitlines() if x.strip()][:100]
        if len(lines) < 8: return False
        hits = sum(any(re.search(p, l, re.I) for p in self.LOG_PATTERNS) for l in lines)
        return hits >= 5 and hits / len(lines) >= 0.40

    def _compiler_ir(self, text: str) -> bool:
        lines = text.splitlines()
        hits = sum(any(re.search(p, l, re.I) for p in self.COMPILER_IR_PATTERNS) for l in lines[:100])
        return hits >= 3

    def _book_artifact(self, text: str, s: Dict[str, Any]) -> Optional[str]:
        low = text.lower()
        if "<rdf:rdf" in low or "xmlns:rdf=" in low or "<opf:metadata" in low:
            return "RDF/XML book metadata dump"
        catalog_hits = sum(p in low for p in self.BOOK_CATALOG_PATTERNS)
        if catalog_hits >= 3 and s["words"] < 400:
            return "Book catalog/download page"
        return None

    def classify(self, text: str, path: str = "", source: str = "") -> Tuple[Decision, str, Dict[str, Any]]:
        s = self._stats(text)
        source = source.lower()
        path_decision = self._path_reject(path)
        if path_decision:
            return path_decision[0], path_decision[1], s
        min_length = 200 if source == "books" else 50
        if s["length"] < min_length:
            return Decision.REJECT_LOW_QUALITY, "Too short", s
        corrupt = self._binary_or_corrupt(text, s)
        if corrupt:
            return Decision.REJECT_BINARY, corrupt, s
        blob = self._blob_artifact(text, s)
        if blob:
            return Decision.REJECT_GENERATED, blob, s
        if self._license_only(text, s, source):
            return Decision.REJECT_LICENSE, "License-only/boilerplate GitHub document", s
        if self._generated(text, s):
            return Decision.REJECT_GENERATED, "Generated/minified artifact", s
        if self._compiler_ir(text):
            return Decision.REJECT_GENERATED, "Compiler/intermediate representation", s
        if self._prompt_or_agent(text):
            return Decision.REJECT_PROMPT, "Prompt/agent trace", s
        if self._log(text, s):
            return Decision.REJECT_LOG, "Log file", s
        if self._benchmark(text):
            return Decision.REJECT_BENCHMARK, "Benchmark/dataset fixture", s
        metadata = self._web_metadata_artifact(text, s, source)
        if metadata:
            return Decision.REJECT_WEB_ARTIFACT, metadata, s
        link_list = self._link_list_artifact(text, s, source)
        if link_list:
            return Decision.REJECT_WEB_ARTIFACT, link_list, s
        if self._tabular(text, s):
            return Decision.REJECT_TABULAR, "Raw tabular dataset", s
        if source == "books":
            artifact = self._book_artifact(text, s)
            if artifact:
                return Decision.REJECT_WEB_ARTIFACT, artifact, s
            return Decision.KEEP_PROSE, "Book text", s
        if source in {"wikipedia", "w3schools", "custom"}:
            return (Decision.KEEP_TUTORIAL if source == "w3schools" or "```" in text else Decision.KEEP_PROSE), "Web tutorial/prose", s
        if source == "github":
            return Decision.KEEP_CODE, "Source code/documentation", s
        if path.lower().endswith((".md", ".rst")):
            return Decision.KEEP_PROSE, "Documentation", s
        return Decision.KEEP_CODE, "Source text", s


class Deduplicator:
    def __init__(self):
        self.exact_hashes = set()
        self.norm_hashes = set()
    def _norm_hash(self, text: str) -> str:
        norm_text = re.sub(r"\s+", "", text).lower()
        return hashlib.sha256(norm_text.encode("utf-8", errors="replace")).hexdigest()
    def is_duplicate(self, text: str, exact_hash: str) -> bool:
        return exact_hash in self.exact_hashes or self._norm_hash(text) in self.norm_hashes
    def add(self, text: str, exact_hash: str):
        self.exact_hashes.add(exact_hash)
        self.norm_hashes.add(self._norm_hash(text))
