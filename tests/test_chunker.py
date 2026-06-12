import pytest
from app.chunker import Chunk, split_into_chunks


def test_basic_chunking():
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = split_into_chunks(text, source="test.txt", chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.source == "test.txt"
        assert len(chunk.text.split()) <= 100


def test_short_text_single_chunk():
    text = "Hello world this is a short document."
    chunks = split_into_chunks(text, source="short.txt", chunk_size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_overlap_shares_words():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = split_into_chunks(text, source="overlap.txt", chunk_size=20, overlap=5)
    if len(chunks) > 1:
        last_words_of_first = set(chunks[0].text.split()[-5:])
        first_words_of_second = set(chunks[1].text.split()[:5])
        assert last_words_of_first & first_words_of_second, "Overlap should share words"


def test_empty_text_returns_no_chunks():
    chunks = split_into_chunks("", source="empty.txt")
    assert chunks == []
