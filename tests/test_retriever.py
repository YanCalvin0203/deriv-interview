import pytest
from app.chunker import Chunk
from app.retriever import RetrievalIndex


def _make_chunks(texts: list[str], source: str = "test.txt") -> list[Chunk]:
    return [Chunk(text=t, source=source, chunk_index=i) for i, t in enumerate(texts)]


def test_query_returns_relevant_chunk():
    chunks = _make_chunks([
        "NovaPay supports Visa and Mastercard payments.",
        "Refunds take 5 to 10 business days to process.",
        "The platform is PCI DSS Level 1 compliant.",
    ])
    index = RetrievalIndex()
    index.build(chunks)

    results = index.query("How long do refunds take?", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert "refund" in top_chunk.text.lower()
    assert score > 0


def test_query_returns_top_k():
    chunks = _make_chunks([f"Document about topic {i}" for i in range(10)])
    index = RetrievalIndex()
    index.build(chunks)
    results = index.query("topic document", top_k=3)
    assert len(results) <= 3


def test_empty_index_returns_empty():
    index = RetrievalIndex()
    results = index.query("anything")
    assert results == []


def test_unrelated_query_returns_zero_score():
    chunks = _make_chunks(["NovaPay is a payment platform."])
    index = RetrievalIndex()
    index.build(chunks)
    results = index.query("zzz xyzzy gobbledygook")
    for _, score in results:
        assert score == 0.0
