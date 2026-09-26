from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_status_lists_documents():
    data = client.get("/api/status").json()
    assert {d["source"] for d in data["documents"]} >= {"01_work_rules.md", "03_expenses.md"}


def test_retrieve_returns_results_and_prompt():
    data = client.post("/api/retrieve", json={"question": "経費の締め日は？", "top_k": 2}).json()
    assert data["results"][0]["source"] == "03_expenses.md"
    assert len(data["results"]) <= 2
    assert "<question>" in data["prompt"]


def test_retrieve_rejects_overlap_not_smaller_than_chunk_size():
    res = client.post("/api/retrieve", json={"question": "x", "chunk_size": 100, "overlap": 100})
    assert res.status_code == 422
