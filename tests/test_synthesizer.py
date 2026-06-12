import pytest
from app.chunker import Chunk
from app.synthesizer import synthesize_answer


def _result(text: str, score: float) -> tuple:
    return (Chunk(text=text, source="doc.txt", chunk_index=0), score)


def test_high_score_returns_answered_from_docs():
    results = [_result("Refunds take 5 to 10 business days.", 0.8)]
    answer, confidence = synthesize_answer("How long do refunds take?", results)
    assert confidence == "answered_from_docs"
    assert "doc.txt" in answer


def test_low_score_returns_partial_context():
    results = [_result("Some loosely related text.", 0.06)]
    answer, confidence = synthesize_answer("specific question", results)
    assert confidence == "partial_context"


def test_empty_results_returns_insufficient():
    answer, confidence = synthesize_answer("What is NovaPay?", [])
    assert confidence == "insufficient_context"
    assert "could not find" in answer.lower()


def test_below_threshold_returns_insufficient():
    results = [_result("Barely related text.", 0.01)]
    answer, confidence = synthesize_answer("question", results)
    assert confidence == "insufficient_context"
