"""全ステップをつなぐ RAG パイプライン

  [インデックス作成 (事前に 1 回)]
    ドキュメント読み込み → チャンク分割 → 埋め込み → ベクトルストアに保存

  [質問応答 (質問のたび)]
    質問を埋め込み → 類似チャンクを検索 → プロンプトに詰める → LLM が回答
"""

from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_documents
from .embedder import Embedder, create_embedder
from .generator import DEFAULT_MODEL, generate_answer
from .loader import load_documents
from .vector_store import InMemoryVectorStore, SearchResult


@dataclass
class RAGConfig:
    data_dir: str = str(Path(__file__).resolve().parent.parent / "data")
    chunk_size: int = 300
    overlap: int = 50
    top_k: int = 3
    min_score: float = 0.05
    embedder: str = "tfidf"  # "tfidf" or "neural"
    model: str = DEFAULT_MODEL


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None, embedder: Embedder | None = None) -> None:
        self.config = config or RAGConfig()
        self.embedder = embedder or create_embedder(self.config.embedder)
        self.store = InMemoryVectorStore()

    def build_index(self) -> None:
        c = self.config
        docs = load_documents(c.data_dir)
        chunks = chunk_documents(docs, c.chunk_size, c.overlap)
        texts = [ch.text_for_embedding() for ch in chunks]
        self.embedder.fit(texts)
        self.store.add(chunks, self.embedder.embed_documents(texts))

    def retrieve(self, question: str) -> list[SearchResult]:
        query_vec = self.embedder.embed_query(question)
        return self.store.search(query_vec, self.config.top_k, self.config.min_score)

    def ask(self, question: str) -> tuple[str, list[SearchResult]]:
        results = self.retrieve(question)
        return generate_answer(question, results, self.config.model), results
