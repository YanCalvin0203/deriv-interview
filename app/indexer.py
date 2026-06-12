from __future__ import annotations
import glob
import os

from app.chunker import Chunk, split_into_chunks
from app.retriever import RetrievalIndex


def load_and_index(docs_dir: str, chunk_size: int = 200, overlap: int = 40) -> tuple[RetrievalIndex, dict]:
    """Read all .txt files from docs_dir, chunk them, and build the index."""
    pattern = os.path.join(docs_dir, "*.txt")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No .txt files found in {docs_dir!r}")

    all_chunks: list[Chunk] = []
    doc_count = len(paths)

    for path in paths:
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        all_chunks.extend(split_into_chunks(text, source=source, chunk_size=chunk_size, overlap=overlap))

    index = RetrievalIndex()
    index.build(all_chunks)

    stats = {
        "documents_indexed": doc_count,
        "chunks_indexed": len(all_chunks),
        "sources": [os.path.basename(p) for p in paths],
    }
    return index, stats
