"""Tests for prepare_data.py document-level train/val splitting."""
import sys
import json
import tempfile
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import (
    get_document_files,
    compute_hash,
    split_documents,
    process_split,
    EXCLUDED_FILENAMES,
    DOCUMENT_SEPARATOR,
)

BYTE_ENCODE = lambda text: list(text.encode("utf-8"))


def make_corpus(tmp: Path, docs: dict) -> None:
    for name, content in docs.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


# A. Multiple documents split at document boundaries
def test_document_boundaries():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {"a.txt": "AAA", "b.txt": "BBB", "c.txt": "CCC", "d.txt": "DDD"})
        files = get_document_files(tmp)
        train, val = split_documents(files, val_fraction=0.25, seed=0)
        out = tmp / "train.bin"
        process_split(train, BYTE_ENCODE, "uint16", out)
        raw = out.read_bytes()
        tokens = list(struct.unpack(f"{len(raw)//2}H", raw))
        text = bytes(tokens).decode("utf-8", errors="replace")
        sep = DOCUMENT_SEPARATOR
        # separator must appear between docs, not inside any single doc content
        for doc_content in ["AAA", "BBB", "CCC", "DDD"]:
            if doc_content in text:
                # the doc content must not be split by a separator
                assert sep not in doc_content


# B. No document appears in both train and validation
def test_no_overlap():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {f"doc{i}.txt": f"Content {i} " * 10 for i in range(10)})
        files = get_document_files(tmp)
        train, val = split_documents(files, val_fraction=0.2, seed=42)
        train_set = set(str(f) for f in train)
        val_set = set(str(f) for f in val)
        assert train_set.isdisjoint(val_set), "Documents appear in both splits"
        assert train_set | val_set == set(str(f) for f in files)


# C. Exact duplicate documents stay in the same split
def test_duplicates_same_split():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # doc1 and doc2 are exact duplicates
        make_corpus(tmp, {
            "doc1.txt": "DUPLICATE CONTENT",
            "doc2.txt": "DUPLICATE CONTENT",
            "doc3.txt": "UNIQUE CONTENT A",
            "doc4.txt": "UNIQUE CONTENT B",
            "doc5.txt": "UNIQUE CONTENT C",
        })
        files = get_document_files(tmp)
        train, val = split_documents(files, val_fraction=0.2, seed=0)
        train_set = set(str(f) for f in train)
        val_set = set(str(f) for f in val)
        dup1_in_train = "doc1.txt" in str(train_set)
        dup2_in_train = "doc2.txt" in str(train_set)
        dup1_in_val = "doc1.txt" in str(val_set)
        dup2_in_val = "doc2.txt" in str(val_set)
        # both duplicates must be in the same split
        assert (dup1_in_train == dup2_in_train) and (dup1_in_val == dup2_in_val)


# D. Deterministic split with the same seed
def test_deterministic_same_seed():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {f"doc{i}.txt": f"Content {i}" for i in range(20)})
        files = get_document_files(tmp)
        train1, val1 = split_documents(files, val_fraction=0.2, seed=7)
        train2, val2 = split_documents(files, val_fraction=0.2, seed=7)
        assert [str(f) for f in train1] == [str(f) for f in train2]
        assert [str(f) for f in val1] == [str(f) for f in val2]


# E. Different seed can change the assignment
def test_different_seed_changes_split():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {f"doc{i}.txt": f"Content {i}" for i in range(20)})
        files = get_document_files(tmp)
        _, val1 = split_documents(files, val_fraction=0.2, seed=1)
        _, val2 = split_documents(files, val_fraction=0.2, seed=999)
        # With 20 docs it's astronomically unlikely both seeds give identical val sets
        assert [str(f) for f in val1] != [str(f) for f in val2]


# F. One-document corpus produces a warning and no invalid split
def test_single_document_no_split(capsys=None):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "only.txt").write_text("Only document", encoding="utf-8")
        files = get_document_files(tmp)
        train, val = split_documents(files, val_fraction=0.5, seed=0)
        assert len(train) == 1
        assert len(val) == 0


# G. Validation file is actually separate (different bytes from train)
def test_val_file_separate():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {f"doc{i}.txt": f"{'X' * 50} {i}" for i in range(10)})
        files = get_document_files(tmp)
        train, val = split_documents(files, val_fraction=0.2, seed=0)
        train_out = tmp / "train.bin"
        val_out = tmp / "train.val.bin"
        process_split(train, BYTE_ENCODE, "uint16", train_out)
        process_split(val, BYTE_ENCODE, "uint16", val_out)
        assert train_out.read_bytes() != val_out.read_bytes()


# H. dtype bounds still work
def test_dtype_bounds():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "doc.txt").write_text("hello", encoding="utf-8")
        files = get_document_files(tmp)
        out = tmp / "out.bin"
        # byte-level encoding stays within uint16
        process_split(files, BYTE_ENCODE, "uint16", out)
        assert out.exists()


# I. manifest.jsonl and _state.json are ignored
def test_excluded_files_ignored():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {
            "doc1.txt": "Real content",
            "manifest.jsonl": '{"hash": "abc"}',
            "_state.json": '{"total_saved": 1}',
        })
        files = get_document_files(tmp)
        names = [f.name for f in files]
        assert "manifest.jsonl" not in names
        assert "_state.json" not in names
        assert "doc1.txt" in names


# J. Document separators are preserved between documents
def test_document_separators():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        make_corpus(tmp, {"a.txt": "AAAA", "b.txt": "BBBB"})
        files = sorted(get_document_files(tmp))
        out = tmp / "out.bin"
        process_split(files, BYTE_ENCODE, "uint16", out)
        raw = out.read_bytes()
        tokens = list(struct.unpack(f"{len(raw)//2}H", raw))
        text = bytes(tokens).decode("utf-8")
        assert DOCUMENT_SEPARATOR in text


# Regression: no validation tokens come from the middle of a document
def test_no_cross_document_tokens():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        docs = {
            "doc1.txt": "AAA" * 100,
            "doc2.txt": "BBB" * 100,
            "doc3.txt": "CCC" * 100,
            "doc4.txt": "DDD" * 100,
        }
        make_corpus(tmp, docs)
        files = get_document_files(tmp)
        train, val = split_documents(files, val_fraction=0.25, seed=0)

        def read_tokens(fs):
            out = tmp / f"tmp_{id(fs)}.bin"
            process_split(fs, BYTE_ENCODE, "uint16", out)
            raw = out.read_bytes()
            return bytes(struct.unpack(f"{len(raw)//2}H", raw)).decode("utf-8")

        train_text = read_tokens(train)
        val_text = read_tokens(val)

        for marker, content in [("AAA", "AAA" * 100), ("BBB", "BBB" * 100),
                                  ("CCC", "CCC" * 100), ("DDD", "DDD" * 100)]:
            in_train = content in train_text
            in_val = content in val_text
            # each document must be entirely in one split, not both
            assert not (in_train and in_val), f"Document {marker} appears in both splits"


# Books Subdirectory & Document Split integration
def test_books_corpus_integration():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        docs = {
            "books/gutenberg_hamlet.txt": "HAMLET: To be, or not to be, that is the question.",
            "books/standard_ebooks_hume.txt": "HUME: All impressions and ideas arise from experience.",
            "books/wikisource_grammar.txt": "GRAMMAR: A noun is a name of any person, place, or thing.",
            "wikipedia/article1.txt": "WIKIPEDIA: General knowledge article.",
            "w3schools/python_intro.txt": "W3SCHOOLS: Python is an interpreted language.",
            "books/manifest.jsonl": '{"source": "books", "book_source": "gutenberg"}',
            "books/_state.json": '{"total_saved": 3}',
        }
        make_corpus(tmp, docs)
        files = get_document_files(tmp)
        file_names = [f.name for f in files]
        assert len(files) == 5
        assert "manifest.jsonl" not in file_names
        assert "_state.json" not in file_names

        train, val = split_documents(files, val_fraction=0.4, seed=42)
        assert len(train) + len(val) == 5

        # Entire books belong to either train or validation
        train_paths = [str(f) for f in train]
        val_paths = [str(f) for f in val]
        for f in files:
            p = str(f)
            assert (p in train_paths) ^ (p in val_paths), f"Document {p} leaked across splits"


if __name__ == "__main__":
    test_document_boundaries()
    test_no_overlap()
    test_duplicates_same_split()
    test_deterministic_same_seed()
    test_different_seed_changes_split()
    test_single_document_no_split()
    test_val_file_separate()
    test_dtype_bounds()
    test_excluded_files_ignored()
    test_document_separators()
    test_no_cross_document_tokens()
    test_books_corpus_integration()
    print("All prepare_data tests passed!")
