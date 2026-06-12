# NovaPay RAG API

A lightweight Retrieval-Augmented Generation (RAG) service that answers questions from a local document set — no external paid APIs required.

## Architecture

```
docs/                  ← plain-text knowledge base (5 .txt files)
  privacy_policy.txt
  product_overview.txt
  refund_policy.txt
  security_policy.txt
  support_faq.txt
app/
  chunker.py           ← splits docs into overlapping word windows
  retriever.py         ← TF-IDF index (scikit-learn) + cosine similarity
  synthesizer.py       ← grounded answer builder + confidence classifier
  indexer.py           ← loads & indexes all .txt files
  main.py              ← FastAPI app with /index and /ask endpoints
tests/                 ← pytest suite (17 tests, 100 % passing)
pytest.ini             ← pytest config (testpaths = tests)
requirements.txt       ← pinned dependencies
run.sh                 ← venv bootstrap + uvicorn launcher
```

### RAG pipeline

1. **POST /index** reads every `docs/*.txt` file, splits each into overlapping 200-word chunks (40-word overlap), and fits a TF-IDF vectorizer over all chunks.
2. **POST /ask** transforms the question with the same vectorizer, scores every chunk via cosine similarity, and returns up to the top-K chunks with a non-zero score plus a synthesized answer.
3. The **confidence field** is determined by the top retrieval score:
   - `answered_from_docs` — score ≥ 0.15 (strong lexical match)
   - `partial_context` — score in [0.05, 0.15) (weak match, answer may be incomplete)
   - `insufficient_context` — score < 0.05 (no useful match found)

## Quick start

```bash
# 1. Start the server (creates .venv and installs deps automatically on first run)
./run.sh

# 2. Build the index
curl -X POST http://localhost:8000/index

# 3. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do credit card refunds take?"}'
```

## API reference

### `POST /index`

Reads `docs/*.txt`, chunks, and builds the in-memory TF-IDF index.

**Response**
```json
{
  "message": "Index built successfully.",
  "documents_indexed": 5,
  "chunks_indexed": 9,
  "sources": ["privacy_policy.txt", "product_overview.txt", "refund_policy.txt", "security_policy.txt", "support_faq.txt"]
}
```

### `POST /ask`

**Request body**
```json
{ "question": "What are the transaction fees?" }
```

**Response**
```json
{
  "answer": "Based on the NovaPay documentation:\n\n[product_overview.txt] ...",
  "confidence": "answered_from_docs",
  "retrieved_chunks": [
    {
      "source": "product_overview.txt",
      "chunk_index": 0,
      "text": "...",
      "score": 0.2341
    }
  ]
}
```

**Confidence values**

| Value | Meaning |
|---|---|
| `answered_from_docs` | Strong match found (score ≥ 0.15) |
| `partial_context` | Weak match; answer may be incomplete |
| `insufficient_context` | No relevant content found |

## Configuration

| Env var | Default | Description |
|---|---|---|
| `DOCS_DIR` | `docs` | Directory containing `.txt` source files |
| `TOP_K` | `3` | Number of chunks to retrieve per query |
| `PORT` | `8000` | Server port |

## Running tests

```bash
.venv/bin/pytest -v
```

Or with coverage:

```bash
.venv/bin/pytest --cov=app --cov-report=term-missing
```

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `uvicorn[standard]` | ASGI server |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `numpy` | Array operations |
| `pydantic` | Request/response validation |
| `httpx` | HTTP client used by FastAPI test client |
| `pytest` | Test framework |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reporting |

## Tradeoffs and what I'd improve next

### Current approach: TF-IDF + extractive synthesis

**Why TF-IDF?**
Zero external dependencies, instant startup, deterministic output, and easy to reason about. For a small document set it works well. The downside is purely lexical matching — a query like "settlement duration" won't hit a chunk that says "funds arrive within 24 hours" unless those exact tokens overlap.

**Why extractive synthesis?**
Combining retrieved passages verbatim guarantees the answer is fully grounded in the documents. There is no risk of hallucination. The downside is that the output can be verbose and may include irrelevant sentences from the same chunk window.

### What I'd improve with more time

| Area | Current | Improvement |
|---|---|---|
| Retrieval quality | TF-IDF cosine similarity | Swap to a small bi-encoder (e.g. `all-MiniLM-L6-v2`) for semantic search — queries match by meaning, not just tokens |
| Answer quality | Raw chunk concatenation | Sentence-level extraction: score individual sentences within top chunks and return only the most relevant 2-3 sentences |
| Index persistence | Rebuilt on every `/index` call | Serialize the vectorizer and matrix to disk so the server can warm-start without re-reading all files |
| Concurrency | Module-level `_state` dict | Replace with a proper dependency-injected store; the current approach is unsafe under multiple workers |
| Confidence thresholds | Hard-coded `0.05` / `0.15` | Calibrate thresholds empirically against a labelled eval set; expose as env vars |
| Chunk strategy | Fixed word-count windows | Sentence-boundary chunking to avoid mid-sentence splits that hurt both retrieval and readability |
| Observability | None | Add request-level logging (query, top scores, latency) for debugging retrieval quality in production |
