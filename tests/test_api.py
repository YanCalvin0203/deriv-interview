import os
import tempfile
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_with_docs(tmp_path, monkeypatch):
    """TestClient wired to a temp docs directory."""
    # Create sample docs
    (tmp_path / "product.txt").write_text(
        "NovaPay is a digital payment platform. "
        "It supports Visa, Mastercard, and Apple Pay. "
        "Instant settlements are processed within 24 hours."
    )
    (tmp_path / "refunds.txt").write_text(
        "Refunds can be issued within 30 days. "
        "Credit card refunds take 5 to 10 business days. "
        "Transaction fees are non-refundable."
    )
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))

    # Re-import app after env var is set so DOCS_DIR is picked up
    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    main_module._state["index"] = None

    from app.main import app
    return TestClient(app)


def test_index_returns_stats(client_with_docs):
    resp = client_with_docs.post("/index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents_indexed"] == 2
    assert data["chunks_indexed"] > 0
    assert len(data["sources"]) == 2


def test_ask_before_index_returns_400(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCS_DIR", str(tmp_path))
    import importlib
    import app.main as main_module
    importlib.reload(main_module)
    main_module._state["index"] = None
    from app.main import app
    client = TestClient(app)
    resp = client.post("/ask", json={"question": "What is NovaPay?"})
    assert resp.status_code == 400


def test_ask_returns_answer(client_with_docs):
    client_with_docs.post("/index")
    resp = client_with_docs.post("/ask", json={"question": "How long do refunds take?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["confidence"] in ("answered_from_docs", "partial_context", "insufficient_context")
    assert isinstance(data["retrieved_chunks"], list)


def test_ask_empty_question_returns_422(client_with_docs):
    client_with_docs.post("/index")
    resp = client_with_docs.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


def test_ask_irrelevant_question_returns_insufficient(client_with_docs):
    client_with_docs.post("/index")
    resp = client_with_docs.post("/ask", json={"question": "zzz xyzzy gobbledygook nonsense"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence"] == "insufficient_context"
