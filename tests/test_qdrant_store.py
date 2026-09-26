import pytest
from qdrant_client import QdrantClient

from evaluate import EVAL_SET
from rag import RAGConfig, RAGPipeline
from rag.chunker import chunk_documents
from rag.loader import load_documents
from rag.qdrant_store import QdrantVectorStore


def make_pipeline(client: QdrantClient, **kwargs) -> RAGPipeline:
    config = RAGConfig(store="qdrant", **kwargs)
    return RAGPipeline(config, store=QdrantVectorStore(config.collection, client))


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def test_qdrant_returns_same_results_as_in_memory(client):
    # ベクトル DB に置き換えても、検索結果 (順位とスコア) は numpy の全件探索と一致する
    memory = RAGPipeline(RAGConfig(store="memory"))
    memory.build_index()
    qdrant = make_pipeline(client)
    qdrant.build_index()
    for question, _, _ in EVAL_SET:
        expected = memory.retrieve(question)
        actual = qdrant.retrieve(question)
        assert [r.chunk.id for r in actual] == [r.chunk.id for r in expected], question
        assert [r.score for r in actual] == pytest.approx([r.score for r in expected], abs=1e-6)


def test_index_is_loaded_instead_of_rebuilt(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    first = make_pipeline(client)
    first.embedder_path = tmp_path / "tfidf.json"
    assert first.build_index() == "built"

    second = make_pipeline(client)  # 新しいプロセスで起動し直した想定: 埋め込みは未学習の状態
    second.embedder_path = tmp_path / "tfidf.json"
    assert second.build_index() == "loaded"
    question = "経費の締め日は？"
    assert [r.chunk.id for r in second.retrieve(question)] == [r.chunk.id for r in first.retrieve(question)]
    client.close()


def test_adding_same_chunks_again_does_not_duplicate_points(client):
    rag = make_pipeline(client)
    rag.build_index()
    count = rag.store.count()
    # 同じチャンクをもう一度入れても、ID が決定的なので upsert で上書きされるだけ
    chunks = chunk_documents(load_documents(rag.config.data_dir), rag.config.chunk_size, rag.config.overlap)
    rag.store.add(chunks, rag.embedder.embed_documents([c.text_for_embedding() for c in chunks]))
    assert rag.store.count() == count


def test_filter_by_source(client):
    rag = make_pipeline(client)
    rag.build_index()
    results = rag.retrieve("申請の期限は？", top_k=5, sources=["03_expenses.md"])
    assert results
    assert {r.chunk.source for r in results} == {"03_expenses.md"}


def test_query_with_no_known_tokens_returns_nothing(client):
    rag = make_pipeline(client)
    rag.build_index()
    assert rag.retrieve("xyzzy") == []
