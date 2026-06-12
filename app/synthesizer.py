from __future__ import annotations
from app.chunker import Chunk


MIN_SCORE_THRESHOLD = 0.05
CONFIDENT_SCORE_THRESHOLD = 0.15


def synthesize_answer(
    question: str,
    results: list[tuple[Chunk, float]],
) -> tuple[str, str]:
    """
    Produce a grounded answer from retrieved chunks without an external LLM.

    Returns (answer_text, confidence_label).
    confidence_label is one of:
        "answered_from_docs"  — at least one chunk has a strong relevance score
        "partial_context"     — weak matches found, answer may be incomplete
        "insufficient_context" — no useful matches
    """
    if not results or results[0][1] < MIN_SCORE_THRESHOLD:
        return (
            "I could not find relevant information in the documents to answer your question.",
            "insufficient_context",
        )

    best_score = results[0][1]
    confidence = "answered_from_docs" if best_score >= CONFIDENT_SCORE_THRESHOLD else "partial_context"

    # Build answer by presenting extracted passages from the top chunks
    parts: list[str] = [f"Based on the NovaPay documentation:\n"]
    for chunk, score in results:
        parts.append(f"[{chunk.source}] {chunk.text.strip()}")

    answer = "\n\n".join(parts)
    return answer, confidence
