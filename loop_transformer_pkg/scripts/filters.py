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


class DocumentClassifier:
    """Conservative corpus gate. Rejects known junk while preserving real code/tutorials/books."""
    BAD_FILENAMES = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "cargo.lock",
        "gemfile.lock", "composer.lock", "go.sum", "package.resolved", "npm-shrinkwrap.json",
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
    LOG_PATTERNS = [
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', r'^\[\d{4}-\d{2}-\d{2}',
        r'\b(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b',
    ]
    COMPILER_IR_PATTERNS = [r"^FILE fqName:", r"^\s*FUN name:", r"^\s*BLOCK_BODY$", r"^\s*CALL '.* declared in ", r"^\s*CONST (?:Int|String|Long|Boolean)", r"^\s*VAR name:", r"^\s*\$this:"]
    LICENSE_PATTERNS = [r"permission is hereby granted", r"apache license, version 2\.0", r"mit license", r"redistribution and use in source and binary forms", r"software is provided ['\"]as is['\"]", r"without warranty of any kind"]
    BOOK_CATALOG_PATTERNS = ["download this book", "plain text utf-8", "bibliographic record", "marc record", "rdf description", "table of contents navigation", "terms of use at www.gutenberg.org"]
    MOJIBAKE_MARKERS = ["Ã", "Â", "â€", "â€™", "â€œ", "â€�", "â€“", "â€”", "ðŸ", "ï¼", "à°", "è¯", "æ–", "å¤"]

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

    def _benchmark(self, text: str) -> bool:
        low = text.lower()
        return (("{\"input\":" in low and "\"target\":" in low) or
                ("{\"prompt\":" in low and "\"completion\":" in low))

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
        if self._compiler_ir(text):
            return Decision.REJECT_GENERATED, "Compiler/intermediate representation", s
        if self._log(text, s):
            return Decision.REJECT_LOG, "Log file", s
        if self._benchmark(text):
            return Decision.REJECT_BENCHMARK, "Benchmark/dataset fixture", s
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
