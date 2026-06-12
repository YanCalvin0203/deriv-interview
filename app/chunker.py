from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def split_into_chunks(text: str, source: str, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split text into overlapping word-window chunks."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(text=" ".join(window), source=source, chunk_index=i))
    return chunks
