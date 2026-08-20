"""Regression and integration tests for scrape_web.py -- W3Schools crawler,
Wikipedia, and Books (Project Gutenberg, Standard Ebooks, Wikisource).
"""

from __future__ import annotations

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from scrape_web import (  # noqa: E402
    parse_html,
    strip_boilerplate,
    w3schools_preprocess_soup,
    clean_standard_ebooks_html,
    clean_wikisource_html,
    extract_page_links,
    render_markdown,
    is_low_quality,
    crawl_hub,
    scrape_gutenberg,
    scrape_standard_ebooks,
    scrape_wikisource,
    scrape_books,
    ScrapeState,
)
from filters import DocumentClassifier, Decision  # noqa: E402


REPRESENTATIVE_W3SCHOOLS_HUB_HTML = """
<!DOCTYPE html>
<html>
<head><title>Python Tutorial</title></head>
<body>
<div id="main">
  <h1>Python <span class="color_h1">Tutorial</span></h1>
  <div class="w3-clear nextprev">
    <a class="w3-left w3-btn" href="/default.asp">❮ Home</a>
    <a class="w3-right w3-btn" href="python_intro.asp">Next ❯</a>
  </div>
  <h2>Learn Python</h2>
  <p>Python is a popular general-purpose programming language.</p>
  <div class="w3-example">
    <h3>Example</h3>
    <div class="w3-code notranslate pythonHigh">
      print("Hello, World!")<br/>
      x = 10<br/>
      print(x)
    </div>
    <a class="w3-btn" href="trypython.asp?filename=demo_default">Try it Yourself »</a>
  </div>
  <a href="python_intro.asp">Python Intro</a>
  <a href="python_syntax.asp">Python Syntax</a>
  <a href="python_variables.asp">Python Variables</a>
  <a href="/css/default.asp">CSS Tutorial</a>
  <a href="https://external.com/page.html">External</a>
  <div class="user-profile-bottom-wrapper" id="user-profile-bottom-wrapper">
    <a href="https://profile.w3schools.com/log-in">Sign in to track progress</a>
  </div>
</div>
</body>
</html>
"""

REPRESENTATIVE_W3SCHOOLS_LESSON_HTML = """
<!DOCTYPE html>
<html>
<head><title>Python Syntax</title></head>
<body>
<div id="main">
  <h1>Python <span class="color_h1">Syntax</span></h1>
  <div class="w3-clear nextprev">
    <a class="w3-left w3-btn" href="python_intro.asp">❮ Previous</a>
    <a class="w3-right w3-btn" href="python_variables.asp">Next ❯</a>
  </div>
  <h2>Execute Python Syntax</h2>
  <p>As we learned in the previous page, Python syntax can be executed directly.</p>
  <div class="w3-example">
    <h3>Example</h3>
    <div class="w3-code notranslate pythonHigh">
      if 5 > 2:<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;print("Five is greater than two!")
    </div>
    <a class="w3-btn" href="trypython.asp?filename=demo_syntax">Try it Yourself »</a>
  </div>
  <h2>Python Indentation</h2>
  <p>Indentation refers to the spaces at the beginning of a code line.</p>
  <p>Where in other programming languages indentation in code is for readability only, the indentation in Python is very important. Python uses indentation to indicate a block of code.</p>
  <div class="w3-example">
    <h3>Example</h3>
    <div class="w3-code notranslate pythonHigh">
      if 5 > 2:<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;print("Five is greater than two!")<br/>
      if 5 > 2:<br/>
      &nbsp;&nbsp;&nbsp;&nbsp;print("Five is greater than two!")
    </div>
    <a class="w3-btn" href="trypython.asp?filename=demo_indentation">Try it Yourself »</a>
  </div>
  <div class="user-profile-bottom-wrapper" id="user-profile-bottom-wrapper">
    <a href="https://profile.w3schools.com/log-in">Sign in to track progress</a>
  </div>
</div>
</body>
</html>
"""

REPRESENTATIVE_STANDARD_EBOOKS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head><title>A Treatise of Human Nature by David Hume - Standard Ebooks</title></head>
<body>
<header class="site-header">
  <nav><a href="/">Home</a> <a href="/ebooks">Ebooks</a></nav>
</header>
<section id="titlepage">
  <h1>A Treatise of Human Nature</h1>
  <p class="author">David Hume</p>
</section>
<nav id="toc">
  <ol>
    <li><a href="#book-1">Book 1</a></li>
    <li><a href="#book-2">Book 2</a></li>
  </ol>
</nav>
<main>
  <article>
    <section id="book-1" class="body">
      <h2>BOOK I: OF THE UNDERSTANDING</h2>
      <h3>PART I: OF IDEAS, THEIR ORIGIN, COMPOSITION, CONNEXION, ABSTRACTION, ETC.</h3>
      <h4>SECTION I: OF THE ORIGIN OF OUR IDEAS</h4>
      <p>All the perceptions of the human mind resolve themselves into two distinct kinds, which I shall call IMPRESSIONS and IDEAS. The difference betwixt these consists in the degrees of force and liveliness, with which they strike upon the mind, and make their way into our thought or consciousness.</p>
      <p>Those perceptions, which enter with most force and violence, we may name impressions; and under this name I comprehend all our sensations, passions and emotions, as they make their first appearance in the soul.</p>
    </section>
  </article>
</main>
<section id="colophon">
  <p>This ebook was produced for the Standard Ebooks project.</p>
</section>
<section id="uncopyright">
  <p>The public domain status of this work is uncopyrighted.</p>
</section>
<footer class="site-footer">
  <p>Standard Ebooks is a volunteer-driven project.</p>
</footer>
</body>
</html>
"""

REPRESENTATIVE_WIKISOURCE_HTML = """
<!DOCTYPE html>
<html>
<head><title>The Tragedy of Hamlet, Prince of Denmark - Wikisource</title></head>
<body>
<div id="mw-content-text" class="mw-body-content">
  <table class="headertemplate">
    <tr><td>Header Navigation Bar</td></tr>
  </table>
  <div class="ws-noexport wst-nav">Navigation links</div>
  <div class="mw-parser-output">
    <div id="toc" class="toc"><h2>Contents</h2><ul><li><a href="#act-1">Act I</a></li></ul></div>
    <h2><span class="mw-headline" id="ACT_I">ACT I</span><span class="mw-editsection"><a href="#">edit</a></span></h2>
    <h3><span class="mw-headline" id="SCENE_I">SCENE I</span></h3>
    <p><b>Elsinore. A platform before the castle.</b></p>
    <p><i>FRANCISCO at his post. Enter to him BERNARDO.</i></p>
    <p><b>BERNARDO</b><br/>Who’s there?</p>
    <p><b>FRANCISCO</b><br/>Nay, answer me: stand, and unfold yourself.</p>
    <p><b>BERNARDO</b><br/>Long live the king!</p>
  </div>
  <table class="footertemplate">
    <tr><td>Footer Navigation Bar</td></tr>
  </table>
  <div class="catlinks" id="catlinks">Category:Tragedies</div>
</div>
</body>
</html>
"""


def test_extracts_same_topic_lesson_links():
    hub_url = "https://www.w3schools.com/python/default.asp"
    soup = parse_html(REPRESENTATIVE_W3SCHOOLS_HUB_HTML)
    raw_links = extract_page_links(soup, hub_url)
    
    hub_host = urlparse(hub_url).netloc
    topic_prefix = "/python/"
    
    accepted = []
    for link in raw_links:
        parsed = urlparse(link)
        if (parsed.netloc == hub_host
                and parsed.path.startswith(topic_prefix)
                and (parsed.path.endswith(".asp") or parsed.path.endswith(".php") or parsed.path.endswith(".html"))):
            accepted.append(link)
            
    assert "https://www.w3schools.com/python/python_intro.asp" in accepted
    assert "https://www.w3schools.com/python/python_syntax.asp" in accepted
    assert "https://www.w3schools.com/python/python_variables.asp" in accepted
    assert "https://www.w3schools.com/css/default.asp" not in accepted
    assert "https://external.com/page.html" not in accepted


def test_converts_w3_code_to_fenced_code_blocks():
    soup = parse_html(REPRESENTATIVE_W3SCHOOLS_LESSON_HTML)
    w3schools_preprocess_soup(soup)
    strip_boilerplate(soup)
    
    main = soup.select_one("#main")
    md = render_markdown(main, "https://www.w3schools.com/python/python_syntax.asp")
    
    assert "```python" in md
    assert 'print("Five is greater than two!")' in md
    assert "```" in md


def test_strips_try_it_yourself_and_nav_noise():
    soup = parse_html(REPRESENTATIVE_W3SCHOOLS_LESSON_HTML)
    w3schools_preprocess_soup(soup)
    strip_boilerplate(soup)
    
    main = soup.select_one("#main")
    md = render_markdown(main, "https://www.w3schools.com/python/python_syntax.asp")
    
    assert "Try it Yourself" not in md
    assert "❮ Previous" not in md
    assert "Next ❯" not in md
    assert "Sign in to track progress" not in md


def test_lesson_content_passes_quality_and_classifier():
    soup = parse_html(REPRESENTATIVE_W3SCHOOLS_LESSON_HTML)
    w3schools_preprocess_soup(soup)
    strip_boilerplate(soup)
    
    main = soup.select_one("#main")
    body_md = render_markdown(main, "https://www.w3schools.com/python/python_syntax.asp")
    title = soup.title.get_text(strip=True) if soup.title else ""
    header = f"# {title}\n\n" if title else ""
    content = header + body_md
    
    assert not is_low_quality(content)
    
    classifier = DocumentClassifier()
    decision, reason, stats = classifier.classify(content, source="w3schools")
    assert decision == Decision.KEEP_TUTORIAL


def test_crawl_hub_discovers_and_accepts_pages():
    hub_url = "https://www.w3schools.com/python/default.asp"
    lesson_url = "https://www.w3schools.com/python/python_syntax.asp"
    
    mock_session = MagicMock()
    
    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        if url == hub_url:
            mock_resp.url = hub_url
            mock_resp.text = REPRESENTATIVE_W3SCHOOLS_HUB_HTML
        elif url == lesson_url:
            mock_resp.url = lesson_url
            mock_resp.text = REPRESENTATIVE_W3SCHOOLS_LESSON_HTML
        else:
            mock_resp.url = url
            mock_resp.text = REPRESENTATIVE_W3SCHOOLS_LESSON_HTML
        return mock_resp
        
    mock_session.get.side_effect = mock_get
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        state = ScrapeState(tmp_path)
        interrupted = {"flag": False}
        
        saved = crawl_hub(
            session=mock_session,
            state=state,
            out_dir=tmp_path,
            hub_url=hub_url,
            label="python",
            selector="#main",
            max_pages=5,
            num_pages_cap=5,
            interrupted=interrupted,
            source_name="w3schools",
        )
        
        assert saved > 0
        assert state.stats["downloaded"] > 0
        assert state.stats["accepted"] > 0
        assert state.total_saved > 0
        assert (tmp_path / "manifest.jsonl").exists()


# ======================================================================
# Book Extraction & Pipeline Tests
# ======================================================================

def test_standard_ebooks_extraction_and_cleaning():
    title, author, text = clean_standard_ebooks_html(
        REPRESENTATIVE_STANDARD_EBOOKS_HTML,
        base_url="https://standardebooks.org/ebooks/david-hume/a-treatise-of-human-nature/text/single-page"
    )
    assert "David Hume" in author or "Hume" in title
    assert "BOOK I: OF THE UNDERSTANDING" in text
    assert "All the perceptions of the human mind resolve themselves" in text
    # Verify chrome and colophon are stripped
    assert "This ebook was produced for the Standard Ebooks project" not in text
    assert "Standard Ebooks is a volunteer-driven project" not in text
    assert "Table of Contents" not in text


def test_wikisource_extraction_and_cleaning():
    text = clean_wikisource_html(
        REPRESENTATIVE_WIKISOURCE_HTML,
        base_url="https://en.wikisource.org/wiki/The_Tragedy_of_Hamlet,_Prince_of_Denmark"
    )
    assert "ACT I" in text
    assert "SCENE I" in text
    assert "FRANCISCO at his post" in text
    assert "Who’s there?" in text
    # Verify templates, TOC and category bar are stripped
    assert "Header Navigation Bar" not in text
    assert "Footer Navigation Bar" not in text
    assert "Category:Tragedies" not in text
    assert "edit" not in text


def test_scrape_gutenberg_mock():
    raw_gutenberg = """The Project Gutenberg eBook of Romeo and Juliet, by William Shakespeare

*** START OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET ***

PROLOGUE

Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventur'd piteous overthrows
Doth with their death bury their parents' strife.

ACT I
SCENE I. Verona. A public place.

SAMPSON
Gregory, o' my word, we'll not carry coals.

GREGORY
No, for then we should be colliers.

SAMPSON
I mean, an we be in choler, we'll draw.

GREGORY
Ay, while you live, draw your neck out o' the collar.

SAMPSON
I strike quickly, being moved.

*** END OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET ***

Project Gutenberg Legal Notices and Licensing Terms
"""
    mock_session = MagicMock()
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "gutendex.com" in url:
            resp.json.return_value = {"results": []}
            resp.text = '{"results": []}'
        else:
            resp.text = raw_gutenberg
            resp.json.return_value = {}
        return resp
    mock_session.get.side_effect = mock_get

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        state = ScrapeState(tmp_path)
        args = Namespace(num_pages=1, book_language="en", book_types=None)
        interrupted = {"flag": False}

        saved = scrape_gutenberg(mock_session, state, args, tmp_path, interrupted, max_books=1)
        assert saved == 1
        assert state.total_saved == 1
        assert state.stats["accepted"] == 1
        assert state.stats["by_book_source"]["gutenberg"] == 1

        manifest_lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(manifest_lines) == 1
        meta = json.loads(manifest_lines[0])

        assert meta["source"] == "books"
        assert meta["book_source"] == "gutenberg"
        assert "Romeo and Juliet" in meta["title"]
        assert meta["book_type"] == "drama"
        assert meta["classification"] == Decision.KEEP_PROSE.value

        # Ensure file content does not inject unnecessary manifest metadata
        saved_file = tmp_path / meta["filename"]
        content = saved_file.read_text(encoding="utf-8")
        assert "Two households, both alike in dignity" in content
        assert "source = books" not in content
        assert "*** START OF THE PROJECT GUTENBERG" not in content
        assert "*** END OF THE PROJECT GUTENBERG" not in content


def test_scrape_standard_ebooks_mock():
    mock_session = MagicMock()
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = REPRESENTATIVE_STANDARD_EBOOKS_HTML
        return resp
    mock_session.get.side_effect = mock_get

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        state = ScrapeState(tmp_path)
        args = Namespace(num_pages=1, book_language="en", book_types=None)
        interrupted = {"flag": False}

        saved = scrape_standard_ebooks(mock_session, state, args, tmp_path, interrupted, max_books=1)
        assert saved == 1
        assert state.stats["by_book_source"]["standard_ebooks"] == 1

        manifest_lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
        meta = json.loads(manifest_lines[0])
        assert meta["source"] == "books"
        assert meta["book_source"] == "standard_ebooks"
        assert "Treatise of Human Nature" in meta["title"]


def test_scrape_wikisource_mock():
    mock_session = MagicMock()
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "query": {"categorymembers": []},
            "parse": {
                "text": {
                    "*": REPRESENTATIVE_WIKISOURCE_HTML
                }
            }
        }
        return resp
    mock_session.get.side_effect = mock_get

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        state = ScrapeState(tmp_path)
        args = Namespace(num_pages=1, book_language="en", book_types=None)
        interrupted = {"flag": False}

        saved = scrape_wikisource(mock_session, state, args, tmp_path, interrupted, max_books=1)
        assert saved == 1
        assert state.stats["by_book_source"]["wikisource"] == 1

        manifest_lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
        meta = json.loads(manifest_lines[0])
        assert meta["source"] == "books"
        assert meta["book_source"] == "wikisource"


def test_scrape_books_orchestrator():
    raw_gutenberg = """The Project Gutenberg eBook of Romeo and Juliet

*** START OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET ***

Two households, both alike in dignity,
In fair Verona, where we lay our scene,
From ancient grudge break to new mutiny,
Where civil blood makes civil hands unclean.
From forth the fatal loins of these two foes
A pair of star-cross'd lovers take their life;
Whose misadventur'd piteous overthrows
Doth with their death bury their parents' strife.

ACT I
SCENE I. Verona. A public place.

SAMPSON
Gregory, o' my word, we'll not carry coals.

GREGORY
No, for then we should be colliers.

SAMPSON
I mean, an we be in choler, we'll draw.

GREGORY
Ay, while you live, draw your neck out o' the collar.

SAMPSON
I strike quickly, being moved.

*** END OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET ***
"""
    mock_session = MagicMock()
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "standardebooks" in url:
            resp.text = REPRESENTATIVE_STANDARD_EBOOKS_HTML
        elif "wikisource" in url:
            resp.json.return_value = {
                "query": {"categorymembers": []},
                "parse": {
                    "text": {
                        "*": REPRESENTATIVE_WIKISOURCE_HTML
                    }
                }
            }
        else:
            resp.json.return_value = {"results": []}
            resp.text = raw_gutenberg
        return resp
    mock_session.get.side_effect = mock_get

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        state = ScrapeState(tmp_path)
        args = Namespace(
            num_pages=3,
            book_sources="gutenberg,standard_ebooks,wikisource",
            book_language="en",
            book_types=None,
            max_books_per_source=1
        )
        interrupted = {"flag": False}

        scrape_books(mock_session, state, args, tmp_path, interrupted)
        assert state.total_saved == 3
        assert state.stats["by_book_source"]["gutenberg"] == 1
        assert state.stats["by_book_source"]["standard_ebooks"] == 1
        assert state.stats["by_book_source"]["wikisource"] == 1


if __name__ == "__main__":
    print("Running scrape_web regression tests...")
    test_extracts_same_topic_lesson_links()
    print("  [PASS] test_extracts_same_topic_lesson_links")
    test_converts_w3_code_to_fenced_code_blocks()
    print("  [PASS] test_converts_w3_code_to_fenced_code_blocks")
    test_strips_try_it_yourself_and_nav_noise()
    print("  [PASS] test_strips_try_it_yourself_and_nav_noise")
    test_lesson_content_passes_quality_and_classifier()
    print("  [PASS] test_lesson_content_passes_quality_and_classifier")
    test_crawl_hub_discovers_and_accepts_pages()
    print("  [PASS] test_crawl_hub_discovers_and_accepts_pages")
    test_standard_ebooks_extraction_and_cleaning()
    print("  [PASS] test_standard_ebooks_extraction_and_cleaning")
    test_wikisource_extraction_and_cleaning()
    print("  [PASS] test_wikisource_extraction_and_cleaning")
    test_scrape_gutenberg_mock()
    print("  [PASS] test_scrape_gutenberg_mock")
    test_scrape_standard_ebooks_mock()
    print("  [PASS] test_scrape_standard_ebooks_mock")
    test_scrape_wikisource_mock()
    print("  [PASS] test_scrape_wikisource_mock")
    test_scrape_books_orchestrator()
    print("  [PASS] test_scrape_books_orchestrator")
    print("\nAll scrape_web tests passed successfully!")
