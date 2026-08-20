#!/usr/bin/env python3
"""Scrape Wikipedia, W3Schools, Books (Project Gutenberg, Standard Ebooks,
Wikisource), and custom tutorial sites into clean text files ready for
prepare_data.py.

Sources
-------
  books      -- High-quality public-domain books from Project Gutenberg,
                Standard Ebooks, and Wikisource. Extracts clean literary,
                dramatic, poetic, grammatical, and educational texts while
                aggressively stripping HTML chrome, catalog pages, and
                source boilerplate (e.g. Gutenberg headers/footers).

  wikipedia  -- MediaWiki API. Recent popular pages via pageviews API and/or
                curated Vital Articles lists, fetched as clean plaintext via
                prop=extracts.

  w3schools  -- HTML->markdown extraction discovering lesson pages via in-page
                navigation starting from DEFAULT_W3SCHOOLS_HUBS.

  custom     -- Generic tutorial crawler pointed at user-supplied --hub-urls.

  all        -- Runs books + wikipedia + w3schools, each into its own subdirectory.

Usage
-----
    python scripts/scrape_web.py --source books --output-dir data/raw/books --num-pages 20
    python scripts/scrape_web.py --source wikipedia --output-dir data/raw/wikipedia --num-pages 3000
    python scripts/scrape_web.py --source w3schools --output-dir data/raw/w3schools --num-pages 1500
    python scripts/scrape_web.py --source all --output-dir data/raw --num-pages 2000
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import logging
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from filters import (
    DocumentClassifier,
    Decision,
    Deduplicator,
    clean_gutenberg_text,
    classify_book_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("scrape_web")

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

NON_ARTICLE_NAMESPACES = {
    "Special", "Wikipedia", "Category", "Portal", "File", "Talk", "User",
    "Template", "Help", "MediaWiki", "Draft", "Module", "TimedText",
    "User_talk", "Template_talk", "Category_talk", "Portal_talk", "Book",
}

DEFAULT_W3SCHOOLS_HUBS: Dict[str, str] = {
    "html": "https://www.w3schools.com/html/default.asp",
    "css": "https://www.w3schools.com/css/default.asp",
    "javascript": "https://www.w3schools.com/js/default.asp",
    "sql": "https://www.w3schools.com/sql/default.asp",
    "python": "https://www.w3schools.com/python/default.asp",
    "java": "https://www.w3schools.com/java/default.asp",
    "php": "https://www.w3schools.com/php/default.asp",
    "c": "https://www.w3schools.com/c/index.php",
    "cpp": "https://www.w3schools.com/cpp/default.asp",
    "csharp": "https://www.w3schools.com/cs/index.php",
    "bootstrap": "https://www.w3schools.com/bootstrap/bootstrap_ver.asp",
    "react": "https://www.w3schools.com/react/default.asp",
    "mysql": "https://www.w3schools.com/mysql/default.asp",
    "jquery": "https://www.w3schools.com/jquery/default.asp",
    "xml": "https://www.w3schools.com/xml/default.asp",
    "django": "https://www.w3schools.com/django/index.php",
    "nodejs": "https://www.w3schools.com/nodejs/default.asp",
    "dsa": "https://www.w3schools.com/dsa/index.php",
    "typescript": "https://www.w3schools.com/typescript/index.php",
    "angular": "https://www.w3schools.com/angular/default.asp",
    "git": "https://www.w3schools.com/git/default.asp",
    "postgresql": "https://www.w3schools.com/postgresql/index.php",
    "mongodb": "https://www.w3schools.com/mongodb/index.php",
    "r": "https://www.w3schools.com/r/default.asp",
    "go": "https://www.w3schools.com/go/index.php",
    "kotlin": "https://www.w3schools.com/kotlin/index.php",
    "swift": "https://www.w3schools.com/swift/default.asp",
    "rust": "https://www.w3schools.com/rust/index.php",
    "bash": "https://www.w3schools.com/bash/index.php",
    "ai": "https://www.w3schools.com/ai/default.asp",
    "datascience": "https://www.w3schools.com/datascience/default.asp",
    "cybersecurity": "https://www.w3schools.com/cybersecurity/index.php",
    "numpy": "https://www.w3schools.com/python/numpy/default.asp",
    "pandas": "https://www.w3schools.com/python/pandas/default.asp",
}

# Curated high-quality seeds for robust offline/fallback operation
CURATED_GUTENBERG_SEEDS = [
    {"id": 1513, "title": "Romeo and Juliet", "author": "William Shakespeare", "type": "drama"},
    {"id": 1524, "title": "Hamlet, Prince of Denmark", "author": "William Shakespeare", "type": "drama"},
    {"id": 1533, "title": "Macbeth", "author": "William Shakespeare", "type": "drama"},
    {"id": 1532, "title": "King Lear", "author": "William Shakespeare", "type": "drama"},
    {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen", "type": "literature"},
    {"id": 1260, "title": "Jane Eyre: An Autobiography", "author": "Charlotte Brontë", "type": "literature"},
    {"id": 768, "title": "Wuthering Heights", "author": "Emily Brontë", "type": "literature"},
    {"id": 1400, "title": "Great Expectations", "author": "Charles Dickens", "type": "literature"},
    {"id": 1661, "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "type": "literature"},
    {"id": 35, "title": "The Time Machine", "author": "H. G. Wells", "type": "literature"},
    {"id": 36, "title": "The War of the Worlds", "author": "H. G. Wells", "type": "literature"},
    {"id": 11, "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "type": "literature"},
    {"id": 345, "title": "Dracula", "author": "Bram Stoker", "type": "literature"},
    {"id": 84, "title": "Frankenstein; Or, The Modern Prometheus", "author": "Mary Wollstonecraft Shelley", "type": "literature"},
    {"id": 164, "title": "Twenty Thousand Leagues under the Sea", "author": "Jules Verne", "type": "literature"},
    {"id": 2701, "title": "Moby Dick; Or, The Whale", "author": "Herman Melville", "type": "literature"},
    {"id": 2600, "title": "War and Peace", "author": "Leo Tolstoy", "type": "literature"},
    {"id": 28054, "title": "The Brothers Karamazov", "author": "Fyodor Dostoyevsky", "type": "literature"},
    {"id": 4300, "title": "Ulysses", "author": "James Joyce", "type": "literature"},
    {"id": 1952, "title": "The Yellow Wallpaper", "author": "Charlotte Perkins Gilman", "type": "literature"},
    {"id": 160, "title": "The Awakening, and Selected Short Stories", "author": "Kate Chopin", "type": "literature"},
    {"id": 158, "title": "Emma", "author": "Jane Austen", "type": "literature"},
    {"id": 2852, "title": "The Hound of the Baskervilles", "author": "Arthur Conan Doyle", "type": "literature"},
    {"id": 2097, "title": "The Sign of the Four", "author": "Arthur Conan Doyle", "type": "literature"},
    {"id": 120, "title": "Treasure Island", "author": "Robert Louis Stevenson", "type": "literature"},
    {"id": 43, "title": "The Strange Case of Dr. Jekyll and Mr. Hyde", "author": "Robert Louis Stevenson", "type": "literature"},
    {"id": 37106, "title": "A Compendious English Grammar", "author": "Alexander Bain", "type": "grammar"},
    {"id": 15474, "title": "English Grammar in Familiar Lectures", "author": "Samuel Kirkham", "type": "grammar"},
    {"id": 24205, "title": "The Elements of Style", "author": "William Strunk Jr.", "type": "rhetoric"},
    {"id": 1497, "title": "The Republic", "author": "Plato", "type": "philosophy"},
]

CURATED_STANDARD_EBOOKS_SEEDS = [
    {"slug": "david-hume/a-treatise-of-human-nature", "title": "A Treatise of Human Nature", "author": "David Hume", "type": "philosophy"},
    {"slug": "john-donne/poetry", "title": "Poetry of John Donne", "author": "John Donne", "type": "poetry"},
    {"slug": "mor-jokai/the-slaves-of-the-padishah/robert-nisbet-bain", "title": "The Slaves of the Padishah", "author": "Mór Jókai", "type": "literature"},
    {"slug": "charlotte-m-yonge/the-clever-woman-of-the-family", "title": "The Clever Woman of the Family", "author": "Charlotte M. Yonge", "type": "literature"},
    {"slug": "karel-capek/krakatit/lawrence-hyde", "title": "Krakatit", "author": "Karel Čapek", "type": "literature"},
    {"slug": "aristophanes/the-birds/the-athenian-society", "title": "The Birds", "author": "Aristophanes", "type": "drama"},
    {"slug": "henry-van-dyke-jr/the-house-of-rimmon", "title": "The House of Rimmon", "author": "Henry van Dyke", "type": "drama"},
    {"slug": "edgar-saltus/short-fiction", "title": "Short Fiction", "author": "Edgar Saltus", "type": "literature"},
    {"slug": "dorothy-m-richardson/revolving-lights", "title": "Revolving Lights", "author": "Dorothy M. Richardson", "type": "literature"},
    {"slug": "h-c-mcneile/the-third-round", "title": "The Third Round", "author": "H. C. McNeile", "type": "literature"},
    {"slug": "lewis-carroll/alices-adventures-in-wonderland/john-tenniel", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "type": "literature"},
    {"slug": "mary-shelley/frankenstein", "title": "Frankenstein", "author": "Mary Shelley", "type": "literature"},
    {"slug": "bram-stoker/dracula", "title": "Dracula", "author": "Bram Stoker", "type": "literature"},
    {"slug": "jane-austen/pride-and-prejudice", "title": "Pride and Prejudice", "author": "Jane Austen", "type": "literature"},
    {"slug": "charlotte-bronte/jane-eyre", "title": "Jane Eyre", "author": "Charlotte Brontë", "type": "literature"},
]

CURATED_WIKISOURCE_SEEDS = [
    {"title": "The Tragedy of Hamlet, Prince of Denmark", "author": "William Shakespeare", "type": "drama"},
    {"title": "The Tragedy of Macbeth", "author": "William Shakespeare", "type": "drama"},
    {"title": "The Tragedy of Julius Caesar", "author": "William Shakespeare", "type": "drama"},
    {"title": "The Comedy of Errors", "author": "William Shakespeare", "type": "drama"},
    {"title": "Alice's Adventures in Wonderland (1866)", "author": "Lewis Carroll", "type": "literature"},
    {"title": "The Adventures Of A Revolutionary Soldier", "author": "Joseph Plumb Martin", "type": "history"},
    {"title": "Aesthetic Papers/Resistance to Civil Government", "author": "Henry David Thoreau", "type": "philosophy"},
    {"title": "An English Grammar for the Higher Grades in Grammar Schools", "author": "William Dwight Whitney", "type": "grammar"},
    {"title": "A Handbook of English Grammar and Composition", "author": "Thomas J. McAvoy", "type": "grammar"},
    {"title": "The Philosophy of Composition", "author": "Edgar Allan Poe", "type": "rhetoric"},
    {"title": "Sonnet 18", "author": "William Shakespeare", "type": "poetry"},
    {"title": "The Raven", "author": "Edgar Allan Poe", "type": "poetry"},
    {"title": "Ozymandias", "author": "Percy Bysshe Shelley", "type": "poetry"},
    {"title": "Kubla Khan", "author": "Samuel Taylor Coleridge", "type": "poetry"},
    {"title": "Self-Reliance", "author": "Ralph Waldo Emerson", "type": "philosophy"},
]


# ======================================================================
# HTTP session
# ======================================================================

class WebSession:
    """Requests session with retries for transient failures."""

    def __init__(self, user_agent: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        })

    def get(self, url: str, params: Optional[dict] = None,
            max_retries: int = 4, timeout: int = 20) -> Optional[requests.Response]:
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                sleep_for = 2 ** attempt
                log.warning(f"Request error for {url}: {e}; retrying in {sleep_for}s")
                time.sleep(sleep_for)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_for = min(int(retry_after), 60)
                else:
                    sleep_for = min(2 ** attempt * 5, 60)
                log.warning(f"HTTP {resp.status_code} from {url}; retrying in {sleep_for}s")
                time.sleep(sleep_for)
                continue
            return resp

        log.warning(f"Giving up on {url} after {max_retries} attempts")
        return None


# ======================================================================
# HTML -> Markdown & Text Conversion
# ======================================================================

def parse_html(html_text: str) -> BeautifulSoup:
    return BeautifulSoup(html_text, "html.parser")


def strip_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    """Removes comments/script/style/nav/ads/etc. IN PLACE."""
    for comment in soup.find_all(string=lambda n: isinstance(n, Comment)):
        comment.extract()
    for tag in soup.select(
        "script, style, noscript, template, svg, canvas, nav, footer, aside, "
        "[hidden], [aria-hidden='true'], [style*='display:none'], [style*='display: none'], "
        ".sidebar, .menu, .ads, .ad, .advert, .advertisement, .popup, .modal, "
        ".cookie, .banner, .newsletter, .subscribe, .social-share"
    ):
        tag.decompose()
    return soup


def w3schools_preprocess_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """W3Schools-specific cleanup applied to the #main subtree."""
    for code_div in soup.select("div.w3-code, div.notranslate"):
        lang = ""
        for cls in (code_div.get("class") or []):
            cls = str(cls)
            if cls.endswith("High") and cls != "notranslate":
                lang = cls.replace("High", "").lower()
                break
            if cls in ("w3-black",):
                lang = "sh"
        parts: List[str] = []
        for node in code_div.children:
            if isinstance(node, NavigableString):
                parts.append(str(node))
            elif isinstance(node, Tag):
                if node.name == "br":
                    parts.append("\n")
                elif node.name in ("em", "i", "strong", "b", "span"):
                    parts.append(node.get_text(""))
                else:
                    parts.append(node.get_text(""))
        raw_code = "".join(parts)
        code_lines = [ln.rstrip() for ln in raw_code.split("\n")]
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        if not code_lines:
            code_div.decompose()
            continue
        code_text = "\n".join(code_lines)
        new_pre = soup.new_tag("pre")
        new_code = soup.new_tag("code", attrs={"class": f"language-{lang}"} if lang else {})
        new_code.string = code_text
        new_pre.append(new_code)
        code_div.replace_with(new_pre)

    for nav_div in soup.select("div.nextprev, div.w3-clear.nextprev, .w3-prevnext"):
        nav_div.decompose()

    for try_link in soup.select(
        "a[href*='tryit'], a[href*='trypython'], a[href*='tryphp'], "
        "a[href*='tryjava'], a[href*='trycss'], a[href*='tryhtml'], "
        "a[href*='trysql'], a[href*='tryr.asp'], a[href*='trysql'],"
        "a[href*='trycsharp'], a[href*='trykotlin'], a[href*='trygo'],"
        "a[href*='tryjquery'], a[href*='tryxquery'],"
        "a.w3-btn[href*='try']"
    ):
        try_link.decompose()

    for el in soup.select(
        "#user-profile-bottom-wrapper, .user-profile-bottom-wrapper, "
        ".user-profile-btn, .ga-bottom, "
        "a[href*='my-learning.w3schools.com'], a[href*='pathfinder.w3schools.com'], "
        "a[href*='profile.w3schools.com'], a[href*='log-in']"
    ):
        el.decompose()

    for el in soup.select(
        "a.ga-youtube, a.ga-featured, #yt_container, #yt_div, "
        "div[id^='yt_'], picture"
    ):
        el.decompose()

    for el in soup.select(
        "div[id='exercisecontainer'], div[id^='exercise'], "
        "#midcontentadcontainer, div[id*='adcontainer'], "
        "div[id*='leaderboard'], div[id*='MainLeaderboard']"
    ):
        el.decompose()

    for ex_h3 in soup.select("div.w3-example > h3"):
        if ex_h3.get_text(strip=True).lower() in ("example", "examples"):
            ex_h3.decompose()

    main_el = soup.select_one("#main")
    if main_el:
        first_h1 = main_el.find("h1")
        if first_h1:
            first_h1.decompose()

    return soup


def clean_standard_ebooks_html(html_text: str, base_url: str = "") -> Tuple[str, str, str]:
    """Clean Standard Ebooks HTML representation.

    Strips website navigation, header chrome, colophon, uncopyright,
    table-of-contents navigation, and returns (title, author, clean_text).
    """
    soup = parse_html(html_text)

    # Extract title and author from <title> or metadata
    page_title = soup.title.get_text(strip=True) if soup.title else ""
    title = page_title.split(" - ")[0].split(" | ")[0].strip()
    author = ""
    if " by " in page_title:
        parts = page_title.split(" by ", 1)
        title = parts[0].strip()
        author = parts[1].split(" - ")[0].split(" | ")[0].strip()

    # Strip Standard Ebooks website chrome and non-book sections
    for tag in soup.select(
        "header.site-header, footer.site-footer, nav, "
        "section#titlepage, section#colophon, section#uncopyright, section#imprint, "
        "section[epub\\:type='colophon'], section[epub\\:type='imprint'], "
        "section#toc, nav#toc, .ebook-hero, .downloads, .download-links, "
        ".donate, .edit-links, .page-header, .page-footer"
    ):
        tag.decompose()

    strip_boilerplate(soup)

    # Standard Ebooks puts content inside <article> or <main> or <section class="body">
    root = soup.select_one("article") or soup.select_one("main") or soup.body or soup
    rendered = render_markdown(root, base_url)
    return title, author, rendered


def clean_wikisource_html(html_text: str, base_url: str = "") -> str:
    """Clean Wikisource HTML representation.

    Strips header/footer templates, edit links, navigation bars, TOC UI,
    category bars, and interface chrome.
    """
    soup = parse_html(html_text)

    for tag in soup.select(
        ".ws-noexport, .headertemplate, .footertemplate, .wst-nav, "
        ".mw-editsection, .noprint, .toc, #toc, .portal, .category-bar, "
        ".catlinks, #catlinks, .navigation, .mw-jump-link, .printfooter, "
        ".ws-pagenum, .pagenum, .wst-badge, table.navigation"
    ):
        tag.decompose()

    strip_boilerplate(soup)

    root = soup.select_one("#mw-content-text") or soup.select_one(".mw-parser-output") or soup.body or soup
    return render_markdown(root, base_url)


def extract_page_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    links = []
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"].strip())
        if urlparse(full).scheme in ("http", "https"):
            links.append(full.split("#")[0])
    return links


def render_markdown(root: Tag, base_url: str) -> str:
    """Convert one HTML subtree into readable markdown."""
    lines: List[str] = []
    seen_urls: Set[str] = set()

    def normalize_text(text: str) -> str:
        text = html.unescape(text or "")
        return re.sub(r"\s+", " ", text).strip()

    def escape_table_cell(text: str) -> str:
        return normalize_text(text).replace("|", r"\|")

    def resolve(raw: str) -> str:
        raw = (raw or "").strip()
        if not raw:
            return ""
        full = urljoin(base_url, raw)
        scheme = urlparse(full).scheme.lower()
        return full if scheme in {"http", "https", "mailto", "tel"} else ""

    def add_line(text: str = "") -> None:
        if text is None:
            return
        for part in str(text).splitlines() or [""]:
            part = part.rstrip()
            if not part:
                add_blank()
                continue
            if lines and lines[-1] == part:
                continue
            lines.append(part)

    def add_blank() -> None:
        if lines and lines[-1] != "":
            lines.append("")

    def label_for(tag: Tag) -> str:
        for attr in ("alt", "title", "aria-label", "data-label", "data-title"):
            value = tag.get(attr)
            if value:
                return normalize_text(str(value))
        return ""

    def first_srcset_url(value: str) -> str:
        for candidate in (value or "").split(","):
            url_part = candidate.strip().split(" ")[0]
            if url_part:
                return url_part
        return ""

    def link_markdown(tag: Tag) -> str:
        text = normalize_text(tag.get_text(" ", strip=True))
        href = tag.get("href")
        if not href:
            return text
        full = resolve(href)
        if not full:
            return text
        if full in seen_urls:
            return text or f"<{full}>"
        seen_urls.add(full)
        return f"[{text}]({full})" if text else f"<{full}>"

    def image_markdown(tag: Tag) -> str:
        alt = label_for(tag)
        src = (
            tag.get("src") or tag.get("data-src") or tag.get("data-original")
            or first_srcset_url(tag.get("srcset", ""))
        )
        if not src:
            return alt
        full = resolve(src)
        return f"![{alt or 'image'}]({full})" if full else alt

    def media_markdown(tag: Tag) -> str:
        label = label_for(tag)
        src = tag.get("src") or tag.get("poster") or tag.get("data-src")
        if not src:
            source = tag.find("source", src=True)
            src = source.get("src") if source else None
        if not src:
            return label
        full = resolve(src)
        return (f"[{label}]({full})" if label else f"<{full}>") if full else label

    def render_inline(node) -> str:
        parts: List[str] = []
        for child in node.children:
            if isinstance(child, NavigableString):
                parts.append(html.unescape(str(child)))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name == "a":
                parts.append(link_markdown(child))
            elif name == "img":
                parts.append(image_markdown(child))
            elif name in {"video", "audio", "source", "iframe", "embed"}:
                parts.append(media_markdown(child))
            elif name in {"strong", "b"}:
                inner = render_inline(child)
                parts.append(f"**{inner}**" if inner else "")
            elif name in {"em", "i"}:
                inner = render_inline(child)
                parts.append(f"*{inner}*" if inner else "")
            elif name == "code":
                inner = normalize_text(child.get_text(" ", strip=True))
                parts.append(f"`{inner}`" if inner else "")
            elif name == "br":
                parts.append("\n")
            else:
                parts.append(render_inline(child))
        return normalize_text("".join(parts))

    def table_markdown(table: Tag) -> List[str]:
        rows: List[List[str]] = []
        header_seen = False
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            row = [escape_table_cell(render_inline(cell)) for cell in cells]
            if any(cell.name.lower() == "th" for cell in cells):
                header_seen = True
            rows.append(row)
        if not rows:
            return []
        col_count = max(len(row) for row in rows)
        rows = [row + [""] * (col_count - len(row)) for row in rows]
        header, body = rows[0], rows[1:]
        out = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]
        out.extend("| " + " | ".join(row) + " |" for row in body)
        if not header_seen and len(rows) == 1:
            out.append("")
        return out

    def code_lang(pre: Tag) -> str:
        code = pre.find("code")
        classes: List[str] = []
        if code:
            raw = code.get("class") or []
            classes = raw if isinstance(raw, list) else [raw]
        for cls in classes:
            cls = str(cls)
            if cls.startswith("language-"):
                return cls.split("language-", 1)[1]
        return ""

    def render_list(list_tag: Tag, ordered: bool, depth: int = 0) -> None:
        try:
            index = int(list_tag.get("start", 1))
        except (TypeError, ValueError):
            index = 1
        for li in list_tag.find_all("li", recursive=False):
            marker = f"{li.get('value') or index}." if ordered else "-"
            parts: List[str] = []
            nested_lists: List[Tag] = []
            for part in li.contents:
                if isinstance(part, NavigableString):
                    parts.append(str(part))
                elif isinstance(part, Tag):
                    name = part.name.lower()
                    if name in {"ul", "ol"}:
                        nested_lists.append(part)
                    elif name == "br":
                        parts.append(" ")
                    else:
                        parts.append(render_inline(part))
            item = normalize_text(" ".join(parts))
            if item:
                add_line(f"{'  ' * depth}{marker} {item}")
            for nested in nested_lists:
                render_list(nested, nested.name.lower() == "ol", depth + 1)
            if ordered:
                index += 1

    block_tags = {
        "article", "section", "main", "div", "header", "figure", "figcaption",
        "h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "pre",
        "ul", "ol", "li", "table", "dl", "dt", "dd",
    }

    def walk(node) -> None:
        for child in node.children:
            if isinstance(child, NavigableString) or not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                heading = render_inline(child)
                if heading and not heading.lower().startswith("video:"):
                    add_line("#" * int(name[1]) + " " + heading)
                    add_blank()
            elif name in {"ul", "ol"}:
                render_list(child, ordered=name == "ol")
                add_blank()
            elif name in {"p", "blockquote"}:
                if child.find(list(block_tags)):
                    walk(child)
                else:
                    inner = render_inline(child)
                    if inner:
                        if name == "blockquote":
                            for quote_line in inner.splitlines() or [inner]:
                                add_line(f"> {quote_line}")
                        else:
                            add_line(inner)
                        add_blank()
            elif name == "pre":
                code_text = child.get_text("\n", strip=False).strip("\n")
                if code_text:
                    add_line(f"```{code_lang(child)}")
                    for code_line in code_text.splitlines():
                        lines.append(code_line.rstrip())
                    add_line("```")
                    add_blank()
            elif name == "table":
                table_lines = table_markdown(child)
                if table_lines:
                    for table_line in table_lines:
                        add_line(table_line)
                    add_blank()
            elif name == "dl":
                for item in child.find_all(["dt", "dd"], recursive=False):
                    text = render_inline(item)
                    if text:
                        prefix = "**" if item.name.lower() == "dt" else "- "
                        suffix = "**" if item.name.lower() == "dt" else ""
                        add_line(f"{prefix}{text}{suffix}")
                add_blank()
            elif name in {"html", "body", "article", "section", "main", "div", "header", "figure", "figcaption"}:
                walk(child)
            elif name == "img":
                img = image_markdown(child)
                if img:
                    add_line(img)
            elif name in {"video", "audio", "iframe", "embed"}:
                media = media_markdown(child)
                if media:
                    add_line(media)
            else:
                if child.find(list(block_tags)):
                    walk(child)
                else:
                    text = render_inline(child)
                    if text:
                        add_line(text)
                        add_blank()

    walk(root)

    cleaned: List[str] = []
    prev_blank = False
    for line in lines:
        line = line.rstrip()
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    return "\n".join(cleaned).strip()


_LINK_ONLY_LINE = re.compile(r"^\[[^\]]*\]\([^)]+\)$")


def boilerplate_ratio(markdown_text: str) -> float:
    lines = [ln.strip() for ln in markdown_text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    link_only = sum(1 for ln in lines if _LINK_ONLY_LINE.match(ln))
    return link_only / len(lines)


def is_low_quality(text: str, min_chars: int = 400, max_boilerplate: float = 0.4) -> bool:
    return len(text) < min_chars or boilerplate_ratio(text) > max_boilerplate


# ======================================================================
# State & Manifest Management
# ======================================================================

class ScrapeState:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_path = output_dir / "_state.json"
        self.manifest_path = output_dir / "manifest.jsonl"
        self.deduplicator = Deduplicator()
        self.total_saved = 0
        self.stats = {
            "downloaded": 0,
            "accepted": 0,
            "rejected": 0,
            "rejected_by_reason": {},
            "accepted_by_category": {},
            "by_book_source": {},
            "by_book_type": {},
            "duplicates": 0,
            "bytes_kept": 0,
            "bytes_rejected": 0,
            "approx_tokens": 0,
        }
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.deduplicator.exact_hashes = set(data.get("seen_hashes", []))
            self.deduplicator.norm_hashes = set(data.get("norm_hashes", []))
            self.total_saved = data.get("total_saved", 0)
            self.stats = data.get("stats", self.stats)
            log.info(f"Resumed state: {self.total_saved} pages/books already saved")

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "seen_hashes": sorted(self.deduplicator.exact_hashes),
            "norm_hashes": sorted(self.deduplicator.norm_hashes),
            "total_saved": self.total_saved,
            "stats": self.stats
        }, indent=2), encoding="utf-8")

    def record_file(self, content_hash: str, meta: Dict[str, Any]) -> None:
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        self.total_saved += 1

    def record_rejection(self, meta: Dict[str, Any]) -> None:
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")


# ======================================================================
# Book Collectors: Project Gutenberg, Standard Ebooks, Wikisource
# ======================================================================

def scrape_gutenberg(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                     out_dir: Path, interrupted: Dict[str, bool], max_books: int) -> int:
    """Download clean plain-text books from Project Gutenberg."""
    log.info(f"[gutenberg] Discovering books (target: {max_books})...")
    classifier = DocumentClassifier()
    saved = 0

    # 1. Discover candidates via Gutendex or fallback seeds
    candidates: List[Dict[str, Any]] = []
    lang = getattr(args, "book_language", "en") or "en"
    gutendex_url = f"https://gutendex.com/books/?languages={lang}"

    resp = session.get(gutendex_url, timeout=15)
    if resp is not None and resp.status_code == 200:
        try:
            results = resp.json().get("results", [])
            for item in results:
                book_id = item.get("id")
                title = item.get("title", "")
                authors = [a.get("name", "") for a in item.get("authors", [])]
                author = ", ".join(authors) if authors else "Unknown"
                subjects = item.get("subjects", [])
                formats = item.get("formats", {})
                # Prefer text/plain; charset=utf-8 or direct txt
                text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain") or formats.get("text/plain; charset=us-ascii")
                if not text_url and book_id:
                    text_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                if text_url:
                    candidates.append({
                        "id": book_id,
                        "title": title,
                        "author": author,
                        "subjects": subjects,
                        "url": text_url,
                    })
        except Exception as e:
            log.warning(f"Error parsing Gutendex response: {e}")

    # Complement / fallback with curated seeds
    seed_ids = {c.get("id") for c in candidates}
    for seed in CURATED_GUTENBERG_SEEDS:
        if seed["id"] not in seed_ids:
            candidates.append({
                "id": seed["id"],
                "title": seed["title"],
                "author": seed["author"],
                "subjects": [seed["type"]],
                "url": f"https://www.gutenberg.org/cache/epub/{seed['id']}/pg{seed['id']}.txt",
                "preset_type": seed["type"],
            })

    log.info(f"[gutenberg] {len(candidates)} candidate books available")

    for cand in candidates:
        if interrupted["flag"] or saved >= max_books or state.total_saved >= args.num_pages:
            break

        book_id = cand["id"]
        title = cand["title"]
        author = cand["author"]
        url = cand["url"]
        preset_type = cand.get("preset_type")

        # Download plain text
        resp = session.get(url, timeout=25)
        if resp is None or resp.status_code != 200:
            alt_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
            resp = session.get(alt_url, timeout=25)
            if resp is None or resp.status_code != 200:
                continue
            url = alt_url

        raw_text = resp.text
        state.stats["downloaded"] += 1

        # Clean Gutenberg boilerplate
        cleaned_text = clean_gutenberg_text(raw_text)

        # Quality filtering
        if is_low_quality(cleaned_text, min_chars=500, max_boilerplate=0.4):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(cleaned_text)
            state.stats["rejected_by_reason"][Decision.REJECT_LOW_QUALITY.value] = (
                state.stats["rejected_by_reason"].get(Decision.REJECT_LOW_QUALITY.value, 0) + 1
            )
            state.record_rejection({
                "source": "books", "book_source": "gutenberg", "title": title,
                "author": author, "size": len(cleaned_text), "classification": Decision.REJECT_LOW_QUALITY.value,
                "reason": "Low quality or short boilerplate", "url": url
            })
            continue

        content_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        if state.deduplicator.is_duplicate(cleaned_text, content_hash):
            state.stats["duplicates"] += 1
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(cleaned_text)
            state.record_rejection({
                "hash": content_hash, "source": "books", "book_source": "gutenberg",
                "title": title, "author": author, "size": len(cleaned_text),
                "classification": Decision.REJECT_DUPLICATE.value, "reason": "Duplicate content", "url": url
            })
            continue

        decision, reason, stats = classifier.classify(cleaned_text, source="books")
        if decision.name.startswith("REJECT"):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(cleaned_text)
            state.stats["rejected_by_reason"][decision.value] = (
                state.stats["rejected_by_reason"].get(decision.value, 0) + 1
            )
            state.record_rejection({
                "hash": content_hash, "source": "books", "book_source": "gutenberg",
                "title": title, "author": author, "size": len(cleaned_text),
                "classification": decision.value, "reason": reason, "stats": stats, "url": url
            })
            continue

        # Inferred book subcategory
        book_type = preset_type or classify_book_type(cleaned_text, title=title, subjects=cand.get("subjects"))
        if getattr(args, "book_types", None):
            allowed_types = [t.strip().lower() for t in args.book_types.split(",") if t.strip()]
            if book_type not in allowed_types:
                continue

        # Accept and save clean text
        state.stats["accepted"] += 1
        state.stats["bytes_kept"] += len(cleaned_text)
        state.stats["accepted_by_category"][decision.value] = (
            state.stats["accepted_by_category"].get(decision.value, 0) + 1
        )
        state.stats["by_book_source"]["gutenberg"] = state.stats["by_book_source"].get("gutenberg", 0) + 1
        state.stats["by_book_type"][book_type] = state.stats["by_book_type"].get(book_type, 0) + 1
        approx_tokens = stats["words"]
        state.stats["approx_tokens"] += approx_tokens
        state.deduplicator.add(cleaned_text, content_hash)

        out_path = out_dir / f"{content_hash[:16]}.txt"
        out_path.write_text(cleaned_text, encoding="utf-8")
        state.record_file(content_hash, {
            "hash": content_hash,
            "source": "books",
            "book_source": "gutenberg",
            "lang": lang,
            "title": title,
            "author": author,
            "edition": f"pg_{book_id}",
            "url": url,
            "size": len(cleaned_text),
            "estimated_tokens": approx_tokens,
            "classification": decision.value,
            "book_type": book_type,
            "filename": out_path.name,
            "stats": stats,
        })
        saved += 1
        log.info(f"[gutenberg] Saved: {title} by {author} ({len(cleaned_text):,} chars, type: {book_type})")

        if state.total_saved % 5 == 0:
            state.save()

    log.info(f"[gutenberg] Finished: saved {saved} books")
    return saved


def scrape_standard_ebooks(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                          out_dir: Path, interrupted: Dict[str, bool], max_books: int) -> int:
    """Download clean literary works from Standard Ebooks."""
    log.info(f"[standard_ebooks] Discovering books (target: {max_books})...")
    classifier = DocumentClassifier()
    saved = 0

    candidates: List[Dict[str, Any]] = []
    catalog_resp = session.get("https://standardebooks.org/ebooks", timeout=15)
    if catalog_resp is not None and catalog_resp.status_code == 200:
        soup = parse_html(catalog_resp.text)
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("/ebooks/") and "?" not in href and "/text" not in href and "/downloads" not in href:
                parts = [p for p in href.strip("/").split("/") if p != "ebooks"]
                if len(parts) >= 2:
                    candidates.append({
                        "slug": "/".join(parts),
                        "url": f"https://standardebooks.org/ebooks/{'/'.join(parts)}/text/single-page",
                        "title": a.get_text(strip=True),
                    })

    # Add curated seeds
    existing_slugs = {c.get("slug") for c in candidates}
    for seed in CURATED_STANDARD_EBOOKS_SEEDS:
        if seed["slug"] not in existing_slugs:
            candidates.append({
                "slug": seed["slug"],
                "url": f"https://standardebooks.org/ebooks/{seed['slug']}/text/single-page",
                "title": seed["title"],
                "author": seed["author"],
                "preset_type": seed["type"],
            })

    log.info(f"[standard_ebooks] {len(candidates)} candidate books available")

    for cand in candidates:
        if interrupted["flag"] or saved >= max_books or state.total_saved >= args.num_pages:
            break

        url = cand["url"]
        slug = cand.get("slug", "")
        preset_type = cand.get("preset_type")

        resp = session.get(url, timeout=25)
        if resp is None or resp.status_code != 200:
            alt_url = f"https://standardebooks.org/ebooks/{slug}"
            resp = session.get(alt_url, timeout=25)
            if resp is None or resp.status_code != 200:
                continue
            url = alt_url

        state.stats["downloaded"] += 1
        extracted_title, extracted_author, clean_text = clean_standard_ebooks_html(resp.text, base_url=url)
        title = cand.get("title") or extracted_title or slug.replace("-", " ").title()
        author = cand.get("author") or extracted_author or "Unknown"

        if is_low_quality(clean_text, min_chars=500, max_boilerplate=0.4):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(clean_text)
            state.stats["rejected_by_reason"][Decision.REJECT_LOW_QUALITY.value] = (
                state.stats["rejected_by_reason"].get(Decision.REJECT_LOW_QUALITY.value, 0) + 1
            )
            state.record_rejection({
                "source": "books", "book_source": "standard_ebooks", "title": title,
                "author": author, "size": len(clean_text), "classification": Decision.REJECT_LOW_QUALITY.value,
                "reason": "Low quality or boilerplate", "url": url
            })
            continue

        content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        if state.deduplicator.is_duplicate(clean_text, content_hash):
            state.stats["duplicates"] += 1
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(clean_text)
            state.record_rejection({
                "hash": content_hash, "source": "books", "book_source": "standard_ebooks",
                "title": title, "author": author, "size": len(clean_text),
                "classification": Decision.REJECT_DUPLICATE.value, "reason": "Duplicate content", "url": url
            })
            continue

        decision, reason, stats = classifier.classify(clean_text, source="books")
        if decision.name.startswith("REJECT"):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(clean_text)
            state.stats["rejected_by_reason"][decision.value] = (
                state.stats["rejected_by_reason"].get(decision.value, 0) + 1
            )
            state.record_rejection({
                "hash": content_hash, "source": "books", "book_source": "standard_ebooks",
                "title": title, "author": author, "size": len(clean_text),
                "classification": decision.value, "reason": reason, "stats": stats, "url": url
            })
            continue

        book_type = preset_type or classify_book_type(clean_text, title=title)
        if getattr(args, "book_types", None):
            allowed_types = [t.strip().lower() for t in args.book_types.split(",") if t.strip()]
            if book_type not in allowed_types:
                continue

        state.stats["accepted"] += 1
        state.stats["bytes_kept"] += len(clean_text)
        state.stats["accepted_by_category"][decision.value] = (
            state.stats["accepted_by_category"].get(decision.value, 0) + 1
        )
        state.stats["by_book_source"]["standard_ebooks"] = state.stats["by_book_source"].get("standard_ebooks", 0) + 1
        state.stats["by_book_type"][book_type] = state.stats["by_book_type"].get(book_type, 0) + 1
        approx_tokens = stats["words"]
        state.stats["approx_tokens"] += approx_tokens
        state.deduplicator.add(clean_text, content_hash)

        out_path = out_dir / f"{content_hash[:16]}.txt"
        out_path.write_text(clean_text, encoding="utf-8")
        state.record_file(content_hash, {
            "hash": content_hash,
            "source": "books",
            "book_source": "standard_ebooks",
            "lang": getattr(args, "book_language", "en") or "en",
            "title": title,
            "author": author,
            "edition": slug,
            "url": url,
            "size": len(clean_text),
            "estimated_tokens": approx_tokens,
            "classification": decision.value,
            "book_type": book_type,
            "filename": out_path.name,
            "stats": stats,
        })
        saved += 1
        log.info(f"[standard_ebooks] Saved: {title} by {author} ({len(clean_text):,} chars, type: {book_type})")

        if state.total_saved % 5 == 0:
            state.save()

    log.info(f"[standard_ebooks] Finished: saved {saved} books")
    return saved


def scrape_wikisource(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                     out_dir: Path, interrupted: Dict[str, bool], max_books: int) -> int:
    """Download clean literature, grammar, and reference works from Wikisource."""
    log.info(f"[wikisource] Discovering works (target: {max_books})...")
    classifier = DocumentClassifier()
    lang = getattr(args, "book_language", "en") or "en"
    api_url = f"https://{lang}.wikisource.org/w/api.php"
    saved = 0

    candidates: List[Dict[str, Any]] = []

    # Query featured / literature categories from Wikisource API
    categories = [
        "Category:Featured_texts",
        "Category:Plays_by_William_Shakespeare",
        "Category:Grammar_books",
        "Category:Rhetoric",
    ]
    for cat in categories:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmlimit": 20,
            "format": "json"
        }
        resp = session.get(api_url, params=params, timeout=15)
        if resp is not None and resp.status_code == 200:
            try:
                members = resp.json().get("query", {}).get("categorymembers", [])
                for m in members:
                    if m.get("ns") == 0:  # main namespace work
                        candidates.append({
                            "title": m.get("title", ""),
                            "url": f"https://{lang}.wikisource.org/wiki/{m.get('title', '').replace(' ', '_')}",
                        })
            except Exception as e:
                log.warning(f"Error querying Wikisource category {cat}: {e}")

    # Add curated seeds
    existing_titles = {c.get("title") for c in candidates}
    for seed in CURATED_WIKISOURCE_SEEDS:
        if seed["title"] not in existing_titles:
            candidates.append({
                "title": seed["title"],
                "author": seed.get("author", "Unknown"),
                "url": f"https://{lang}.wikisource.org/wiki/{seed['title'].replace(' ', '_')}",
                "preset_type": seed.get("type"),
            })

    log.info(f"[wikisource] {len(candidates)} candidate works available")

    for cand in candidates:
        if interrupted["flag"] or saved >= max_books or state.total_saved >= args.num_pages:
            break

        work_title = cand["title"]
        url = cand["url"]
        preset_type = cand.get("preset_type")
        author = cand.get("author", "Unknown")

        # Fetch parsed HTML from Wikisource API
        params = {
            "action": "parse",
            "page": work_title,
            "prop": "text",
            "format": "json",
            "redirects": "1"
        }
        resp = session.get(api_url, params=params, timeout=25)
        if resp is None or resp.status_code != 200:
            continue

        try:
            parse_data = resp.json().get("parse", {})
            html_raw = parse_data.get("text", {}).get("*", "")
            if not html_raw:
                continue
        except Exception:
            continue

        state.stats["downloaded"] += 1
        clean_text = clean_wikisource_html(html_raw, base_url=url)

        if is_low_quality(clean_text, min_chars=200, max_boilerplate=0.5):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(clean_text)
            state.stats["rejected_by_reason"][Decision.REJECT_LOW_QUALITY.value] = (
                state.stats["rejected_by_reason"].get(Decision.REJECT_LOW_QUALITY.value, 0) + 1
            )
            state.record_rejection({
                "source": "books", "book_source": "wikisource", "title": work_title,
                "author": author, "size": len(clean_text), "classification": Decision.REJECT_LOW_QUALITY.value,
                "reason": "Low quality or boilerplate", "url": url
            })
            continue

        content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        if state.deduplicator.is_duplicate(clean_text, content_hash):
            state.stats["duplicates"] += 1
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(clean_text)
            state.record_rejection({
                "hash": content_hash, "source": "books", "book_source": "wikisource",
                "title": work_title, "author": author, "size": len(clean_text),
                "classification": Decision.REJECT_DUPLICATE.value, "reason": "Duplicate content", "url": url
            })
            continue

        decision, reason, stats = classifier.classify(clean_text, source="books")
        if decision.name.startswith("REJECT"):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(clean_text)
            state.stats["rejected_by_reason"][decision.value] = (
                state.stats["rejected_by_reason"].get(decision.value, 0) + 1
            )
            state.record_rejection({
                "hash": content_hash, "source": "books", "book_source": "wikisource",
                "title": work_title, "author": author, "size": len(clean_text),
                "classification": decision.value, "reason": reason, "stats": stats, "url": url
            })
            continue

        book_type = preset_type or classify_book_type(clean_text, title=work_title)
        if getattr(args, "book_types", None):
            allowed_types = [t.strip().lower() for t in args.book_types.split(",") if t.strip()]
            if book_type not in allowed_types:
                continue

        state.stats["accepted"] += 1
        state.stats["bytes_kept"] += len(clean_text)
        state.stats["accepted_by_category"][decision.value] = (
            state.stats["accepted_by_category"].get(decision.value, 0) + 1
        )
        state.stats["by_book_source"]["wikisource"] = state.stats["by_book_source"].get("wikisource", 0) + 1
        state.stats["by_book_type"][book_type] = state.stats["by_book_type"].get(book_type, 0) + 1
        approx_tokens = stats["words"]
        state.stats["approx_tokens"] += approx_tokens
        state.deduplicator.add(clean_text, content_hash)

        out_path = out_dir / f"{content_hash[:16]}.txt"
        out_path.write_text(clean_text, encoding="utf-8")
        state.record_file(content_hash, {
            "hash": content_hash,
            "source": "books",
            "book_source": "wikisource",
            "lang": lang,
            "title": work_title,
            "author": author,
            "edition": work_title,
            "url": url,
            "size": len(clean_text),
            "estimated_tokens": approx_tokens,
            "classification": decision.value,
            "book_type": book_type,
            "filename": out_path.name,
            "stats": stats,
        })
        saved += 1
        log.info(f"[wikisource] Saved: {work_title} ({len(clean_text):,} chars, type: {book_type})")

        if state.total_saved % 5 == 0:
            state.save()

    log.info(f"[wikisource] Finished: saved {saved} works")
    return saved


def scrape_books(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                 out_dir: Path, interrupted: Dict[str, bool]) -> None:
    """Master orchestrator for scraping high quality public domain books."""
    raw_sources = getattr(args, "book_sources", "gutenberg,standard_ebooks,wikisource")
    book_sources = [s.strip().lower() for s in raw_sources.split(",") if s.strip()]

    target_total = args.num_pages
    per_source_target = getattr(args, "max_books_per_source", None)
    if per_source_target is None:
        per_source_target = max(1, (target_total + len(book_sources) - 1) // len(book_sources))

    log.info(f"Scraping books from sources: {book_sources} (target total: {target_total}, per source: {per_source_target})")

    for src in book_sources:
        if interrupted["flag"] or state.total_saved >= target_total:
            break
        remaining = target_total - state.total_saved
        limit = min(per_source_target, remaining)
        if src in ("gutenberg", "project_gutenberg"):
            scrape_gutenberg(session, state, args, out_dir, interrupted, max_books=limit)
        elif src in ("standard_ebooks", "standardebooks"):
            scrape_standard_ebooks(session, state, args, out_dir, interrupted, max_books=limit)
        elif src in ("wikisource", "wiki_source"):
            scrape_wikisource(session, state, args, out_dir, interrupted, max_books=limit)
        else:
            log.warning(f"Unknown book source: {src}; supported: gutenberg, standard_ebooks, wikisource")

    state.save()


# ======================================================================
# Wikipedia (MediaWiki API)
# ======================================================================

def wikipedia_top_titles(session: WebSession, lang: str, date: dt.date, limit: int) -> List[str]:
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        f"{lang}.wikipedia.org/all-access/{date.year:04d}/{date.month:02d}/{date.day:02d}"
    )
    resp = session.get(url)
    if resp is None or resp.status_code != 200:
        log.debug(f"Pageviews top request failed for {date}: {getattr(resp, 'status_code', 'no response')}")
        return []
    items = resp.json().get("items", [])
    if not items:
        return []
    titles = []
    for a in items[0].get("articles", []):
        title = a.get("article", "")
        if not title or title in {"Main_Page", "-"}:
            continue
        if ":" in title and title.split(":", 1)[0] in NON_ARTICLE_NAMESPACES:
            continue
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def collect_wikipedia_top_titles(session: WebSession, lang: str, num_days: int, per_day_limit: int) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    d = dt.date.today() - dt.timedelta(days=2)
    for _ in range(num_days):
        for t in wikipedia_top_titles(session, lang, d, per_day_limit):
            if t not in seen_set:
                seen_set.add(t)
                seen.append(t)
        d -= dt.timedelta(days=1)
    return seen


def wikipedia_page_ns0_links(session: WebSession, lang: str, page_title: str) -> Tuple[List[str], List[str]]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    articles, subpages = [], []
    plcontinue = None
    while True:
        params = {"action": "parse", "page": page_title, "format": "json",
                   "prop": "links", "pllimit": "max"}
        if plcontinue:
            params["plcontinue"] = plcontinue
        resp = session.get(api, params=params)
        if resp is None or resp.status_code != 200:
            break
        data = resp.json()
        if "error" in data:
            log.warning(f"parse API error for {page_title!r}: {data['error'].get('info')}")
            break
        prefix = page_title.rsplit("/", 1)[0]
        for link in data.get("parse", {}).get("links", []):
            title = link.get("*", "")
            ns = link.get("ns")
            if ns == 0 and title:
                articles.append(title)
            elif ns == 4 and title.startswith(prefix) and title != page_title:
                subpages.append(title)
        plcontinue = data.get("continue", {}).get("plcontinue")
        if not plcontinue:
            break
    return articles, subpages


def wikipedia_vital_titles(session: WebSession, lang: str, level: int, max_subpages: int = 40) -> List[str]:
    seed = f"Wikipedia:Vital articles/Level/{level}"
    seen: List[str] = []
    seen_set: Set[str] = set()
    articles, subpages = wikipedia_page_ns0_links(session, lang, seed)
    for a in articles:
        if a not in seen_set:
            seen_set.add(a)
            seen.append(a)
    for sub in subpages[:max_subpages]:
        sub_articles, _ = wikipedia_page_ns0_links(session, lang, sub)
        for a in sub_articles:
            if a not in seen_set:
                seen_set.add(a)
                seen.append(a)
    return seen


def wikipedia_article_text(session: WebSession, lang: str, title: str) -> Optional[Tuple[str, str]]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "prop": "extracts", "explaintext": "1",
              "format": "json", "titles": title, "redirects": "1"}
    resp = session.get(api, params=params)
    if resp is None or resp.status_code != 200:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        extract = (page.get("extract") or "").strip()
        if extract:
            return page.get("title", title), extract
    return None


def scrape_wikipedia(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                      out_dir: Path, interrupted: Dict[str, bool]) -> None:
    titles: List[str] = []
    if args.wikipedia_mode in ("top", "both"):
        top = collect_wikipedia_top_titles(session, args.lang, args.top_days, per_day_limit=1000)
        titles.extend(top)
        log.info(f"Pageviews API: {len(top)} candidate titles from the last {args.top_days} days")
    if args.wikipedia_mode in ("vital", "both"):
        vital = wikipedia_vital_titles(session, args.lang, args.vital_level)
        added = [t for t in vital if t not in titles]
        titles.extend(added)
        log.info(f"Vital articles (level {args.vital_level}): {len(vital)} titles, {len(added)} new")

    log.info(f"{len(titles)} candidate Wikipedia titles queued")
    classifier = DocumentClassifier()

    for title in titles:
        if interrupted["flag"] or state.total_saved >= args.num_pages:
            break
        result = wikipedia_article_text(session, args.lang, title)
        if result is None:
            continue
        real_title, text = result
        content = f"# {real_title}\n\n{text}"
        
        state.stats["downloaded"] += 1
        
        if is_low_quality(content, min_chars=300, max_boilerplate=1.0):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(content)
            state.stats["rejected_by_reason"][Decision.REJECT_LOW_QUALITY.value] = state.stats["rejected_by_reason"].get(Decision.REJECT_LOW_QUALITY.value, 0) + 1
            state.record_rejection({
                "source": "wikipedia", "lang": args.lang, "title": real_title,
                "size": len(content), "classification": Decision.REJECT_LOW_QUALITY.value,
                "reason": "Low quality or boilerplate"
            })
            continue
            
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if state.deduplicator.is_duplicate(content, content_hash):
            state.stats["duplicates"] += 1
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(content)
            state.record_rejection({
                "hash": content_hash, "source": "wikipedia", "lang": args.lang,
                "title": real_title, "size": len(content), "classification": Decision.REJECT_DUPLICATE.value,
                "reason": "Duplicate content"
            })
            continue
            
        decision, reason, stats = classifier.classify(content, source="wikipedia")
        if decision.name.startswith("REJECT"):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(content)
            state.stats["rejected_by_reason"][decision.value] = state.stats["rejected_by_reason"].get(decision.value, 0) + 1
            state.record_rejection({
                "hash": content_hash, "source": "wikipedia", "lang": args.lang,
                "title": real_title, "size": len(content), "classification": decision.value,
                "reason": reason, "stats": stats
            })
            continue

        state.stats["accepted"] += 1
        state.stats["bytes_kept"] += len(content)
        state.stats["accepted_by_category"][decision.value] = state.stats["accepted_by_category"].get(decision.value, 0) + 1
        approx_tokens = stats["words"]
        state.stats["approx_tokens"] += approx_tokens
        state.deduplicator.add(content, content_hash)

        out_path = out_dir / f"{content_hash[:16]}.txt"
        out_path.write_text(content, encoding="utf-8")
        state.record_file(content_hash, {
            "hash": content_hash, "source": "wikipedia", "lang": args.lang,
            "title": real_title, "size": len(content), "filename": out_path.name,
            "classification": decision.value, "stats": stats
        })
        if state.total_saved % 20 == 0:
            state.save()
            log.info(f"Progress: {state.total_saved}/{args.num_pages} pages saved")

    state.save()
    log.info(f"wikipedia: {state.total_saved} pages saved to {out_dir}")


# ======================================================================
# Generic Tutorial Crawler: W3Schools & Custom
# ======================================================================

def crawl_hub(session: WebSession, state: ScrapeState, out_dir: Path, hub_url: str,
              label: str, selector: Optional[str], max_pages: int, num_pages_cap: int,
              interrupted: Dict[str, bool], source_name: str) -> int:
    topic_prefix = "/" + urlparse(hub_url).path.strip("/").split("/")[0] + "/"
    hub_host = urlparse(hub_url).netloc
    frontier: List[str] = [hub_url]
    visited: Set[str] = set()
    saved = 0
    classifier = DocumentClassifier()

    while frontier and saved < max_pages:
        if interrupted["flag"] or state.total_saved >= num_pages_cap:
            break
        url = frontier.pop(0)
        if url in visited:
            continue
        visited.add(url)

        resp = session.get(url)
        if resp is None or resp.status_code != 200:
            continue
        if "html" not in resp.headers.get("Content-Type", "").lower():
            continue

        final_url = resp.url or url
        soup = parse_html(resp.text)
        is_seed = (url == hub_url)

        raw_links = extract_page_links(soup, final_url)
        links_considered = 0
        newly_accepted = []
        rejected_breakdown = {
            "different_host": 0,
            "wrong_topic_prefix": 0,
            "unsupported_extension": 0,
            "already_seen": 0,
        }

        for link in raw_links:
            links_considered += 1
            parsed = urlparse(link)
            path = parsed.path
            if parsed.netloc != hub_host:
                rejected_breakdown["different_host"] += 1
                continue
            if not path.startswith(topic_prefix):
                rejected_breakdown["wrong_topic_prefix"] += 1
                continue
            if not (path.endswith(".asp") or path.endswith(".php") or path.endswith(".html")):
                rejected_breakdown["unsupported_extension"] += 1
                continue
            if link in visited or link in frontier:
                rejected_breakdown["already_seen"] += 1
                continue
            newly_accepted.append(link)
            frontier.append(link)

        if is_seed:
            log.info(f"[{label}] Seed URL: {hub_url}")
            log.info(f"[{label}] Links extracted: {len(raw_links)}")
            log.info(f"[{label}] Links considered: {links_considered}")
            log.info(f"[{label}] Links rejected: {sum(rejected_breakdown.values())} (by reason: {rejected_breakdown})")
            log.info(f"[{label}] Links accepted as candidate lesson pages: {len(newly_accepted)}")

        if source_name == "w3schools":
            w3schools_preprocess_soup(soup)

        strip_boilerplate(soup)
        root = soup.select_one(selector) if selector else None
        if selector and root is None:
            log.debug(f"Selector {selector!r} not found on {url}; using generic extraction")
        if root is None:
            root = soup.select_one("article") or soup.select_one("main") or soup.body or soup

        title = soup.title.get_text(strip=True) if soup.title else ""
        body_md = render_markdown(root, final_url)
        header = f"# {title}\n\n" if title else ""
        content = header + body_md
        
        state.stats["downloaded"] += 1

        if is_low_quality(content):
            log.debug(f"Low-content/nav-heavy, skipping: {url}")
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(content)
            state.stats["rejected_by_reason"][Decision.REJECT_LOW_QUALITY.value] = state.stats["rejected_by_reason"].get(Decision.REJECT_LOW_QUALITY.value, 0) + 1
            state.record_rejection({
                "source": source_name, "topic": label, "url": url, "title": title,
                "size": len(content), "classification": Decision.REJECT_LOW_QUALITY.value,
                "reason": "Low quality or boilerplate"
            })
            continue

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if state.deduplicator.is_duplicate(content, content_hash):
            state.stats["duplicates"] += 1
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(content)
            state.record_rejection({
                "hash": content_hash, "source": source_name, "topic": label,
                "url": url, "title": title, "size": len(content), "classification": Decision.REJECT_DUPLICATE.value,
                "reason": "Duplicate content"
            })
            continue
            
        decision, reason, stats = classifier.classify(content, source=source_name)
        if decision.name.startswith("REJECT"):
            state.stats["rejected"] += 1
            state.stats["bytes_rejected"] += len(content)
            state.stats["rejected_by_reason"][decision.value] = state.stats["rejected_by_reason"].get(decision.value, 0) + 1
            state.record_rejection({
                "hash": content_hash, "source": source_name, "topic": label,
                "url": url, "title": title, "size": len(content), "classification": decision.value,
                "reason": reason, "stats": stats
            })
            continue

        state.stats["accepted"] += 1
        state.stats["bytes_kept"] += len(content)
        state.stats["accepted_by_category"][decision.value] = state.stats["accepted_by_category"].get(decision.value, 0) + 1
        approx_tokens = stats["words"]
        state.stats["approx_tokens"] += approx_tokens
        state.deduplicator.add(content, content_hash)

        out_path = out_dir / f"{content_hash[:16]}.txt"
        out_path.write_text(content, encoding="utf-8")
        state.record_file(content_hash, {
            "hash": content_hash, "source": source_name, "topic": label,
            "url": url, "title": title, "size": len(content), "filename": out_path.name,
            "classification": decision.value, "stats": stats
        })
        saved += 1

        if state.total_saved % 20 == 0:
            state.save()
            log.info(f"Progress: {state.total_saved}/{num_pages_cap} pages saved")

    log.info(f"{label}: saved {saved} pages ({len(visited)} visited)")
    return saved


def scrape_w3schools(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                      out_dir: Path, interrupted: Dict[str, bool]) -> None:
    requested = [t.strip() for t in args.topics.split(",") if t.strip()]
    unknown = [t for t in requested if t not in DEFAULT_W3SCHOOLS_HUBS]
    if unknown:
        log.warning(f"Unknown --topics (skipped): {unknown}; known: {sorted(DEFAULT_W3SCHOOLS_HUBS)}")

    for topic in requested:
        if topic not in DEFAULT_W3SCHOOLS_HUBS:
            continue
        if interrupted["flag"] or state.total_saved >= args.num_pages:
            break
        crawl_hub(session, state, out_dir, DEFAULT_W3SCHOOLS_HUBS[topic], topic,
                   args.selector, args.max_pages_per_topic, args.num_pages, interrupted, "w3schools")

    state.save()
    log.info(f"w3schools: {state.total_saved} pages saved to {out_dir}")


def scrape_custom(session: WebSession, state: ScrapeState, args: argparse.Namespace,
                   out_dir: Path, interrupted: Dict[str, bool]) -> None:
    hub_urls = [u.strip() for u in (args.hub_urls or "").split(",") if u.strip()]
    if not hub_urls:
        log.error("--source custom requires --hub-urls (comma-separated seed URLs)")
        return

    for hub_url in hub_urls:
        if interrupted["flag"] or state.total_saved >= args.num_pages:
            break
        parsed = urlparse(hub_url)
        label = parsed.netloc + "/" + parsed.path.strip("/").split("/")[0]
        crawl_hub(session, state, out_dir, hub_url, label,
                   args.selector, args.max_pages_per_topic, args.num_pages, interrupted, "custom")

    state.save()
    log.info(f"custom: {state.total_saved} pages saved to {out_dir}")


# ======================================================================
# CLI & Main Entry Point
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True, choices=["books", "wikipedia", "w3schools", "custom", "all"],
                    help="'all' runs books + wikipedia + w3schools, each into its own subdirectory")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="with --source all, each source gets its own subdirectory under this path")
    p.add_argument("--num-pages", type=int, default=2000, help="target page/book count, per source")
    p.add_argument("--user-agent", default=None, help="HTTP User-Agent to send with requests")

    bk = p.add_argument_group("books")
    bk.add_argument("--book-sources", default="gutenberg,standard_ebooks,wikisource",
                    help="comma-separated book sources: gutenberg, standard_ebooks, wikisource")
    bk.add_argument("--book-language", default="en", help="Language for book collection (default: en)")
    bk.add_argument("--book-types", default=None,
                    help="optional comma-separated book subcategories (e.g. literature,drama,poetry,grammar,philosophy)")
    bk.add_argument("--max-books-per-source", type=int, default=None,
                    help="optional per-source book limit")

    wiki = p.add_argument_group("wikipedia")
    wiki.add_argument("--lang", default="en", help="Wikipedia language subdomain, e.g. 'en', 'es', 'bn'")
    wiki.add_argument("--wikipedia-mode", choices=["top", "vital", "both"], default="both",
                        help="'top' = recent popular-pages API, 'vital' = curated Vital Articles list, 'both' = union")
    wiki.add_argument("--top-days", type=int, default=14,
                        help="sample the top-pageviews list from this many distinct recent days")
    wiki.add_argument("--vital-level", type=int, default=4, choices=[1, 2, 3, 4, 5],
                        help="Wikipedia:Vital_articles/Level/N")

    tut = p.add_argument_group("w3schools / custom (shared crawler)")
    tut.add_argument("--topics", default=",".join(DEFAULT_W3SCHOOLS_HUBS),
                        help="comma-separated topic keys for --source w3schools")
    tut.add_argument("--hub-urls", default=None,
                        help="comma-separated seed URLs for --source custom")
    tut.add_argument("--selector", default="#main",
                        help="CSS selector for lesson content")
    tut.add_argument("--max-pages-per-topic", type=int, default=80)

    return p.parse_args()


def print_statistics(source: str, state: ScrapeState) -> None:
    stats = state.stats
    print(f"\n--- Scraping Statistics for {source} ---")
    print(f"Downloaded: {stats.get('downloaded', 0)}")
    print(f"Accepted: {stats.get('accepted', 0)}")
    for cat, count in stats.get("accepted_by_category", {}).items():
        print(f"  {cat}: {count}")
    print(f"Rejected: {stats.get('rejected', 0)}")
    for reason, count in stats.get("rejected_by_reason", {}).items():
        print(f"  {reason}: {count}")
    print(f"Duplicates: {stats.get('duplicates', 0)}")
    print(f"Bytes kept: {stats.get('bytes_kept', 0):,}")
    print(f"Bytes rejected: {stats.get('bytes_rejected', 0):,}")

    if source == "books" or stats.get("by_book_source"):
        print("\nBy source:")
        for b_src, count in stats.get("by_book_source", {}).items():
            print(f"  {b_src}: {count}")
        print("\nBy type:")
        for b_type, count in stats.get("by_book_type", {}).items():
            print(f"  {b_type}: {count}")
        if stats.get("approx_tokens"):
            print(f"\nApproximate tokens: {stats.get('approx_tokens', 0):,}")
    print("---------------------------\n")


def main() -> None:
    args = parse_args()
    ua = args.user_agent or DEFAULT_USER_AGENT
    session = WebSession(ua)

    interrupted = {"flag": False}

    def _handle_interrupt(signum, frame):
        log.warning("Interrupt received -- will save state and exit after this item")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    sources = ["books", "wikipedia", "w3schools"] if args.source == "all" else [args.source]

    for source in sources:
        out_dir = (args.output_dir / source) if args.source == "all" else args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        state = ScrapeState(out_dir)

        if source == "books":
            scrape_books(session, state, args, out_dir, interrupted)
        elif source == "wikipedia":
            scrape_wikipedia(session, state, args, out_dir, interrupted)
        elif source == "w3schools":
            scrape_w3schools(session, state, args, out_dir, interrupted)
        elif source == "custom":
            scrape_custom(session, state, args, out_dir, interrupted)

        print_statistics(source, state)

        if interrupted["flag"]:
            break


    log.info(f"Next: python scripts/prepare_data.py --input {args.output_dir} --output data/train.bin")


if __name__ == "__main__":
    main()
