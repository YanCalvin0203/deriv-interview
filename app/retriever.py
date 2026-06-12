from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunker import Chunk


@dataclass
class RetrievalIndex:
    chunks: list[Chunk] = field(default_factory=list)
    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english")
    )
    matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def query(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
