"""全ステップをつなぐ RAG パイプライン

  [インデックス作成 (事前に 1 回 / 資料が変わったとき)]  python ingest.py
    ドキュメント読み込み → チャンク分割 → 埋め込み → ベクトルストアに保存

  [質問応答 (質問のたび)]
    質問を埋め込み → 類似チャンクを検索 → プロンプトに詰める → LLM が回答

ベクトル DB (Qdrant) を使う場合、インデックスはディスクに残るので、
2 回目以降の起動では作り直さずに読み込むだけになります。
"""

from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_documents
from .embedder import Embedder, create_embedder
from .generator import DEFAULT_MODEL, generate_answer
from .loader import load_documents
from .vector_store import INDEX_DIR, InMemoryVectorStore, SearchResult, VectorStore


@dataclass
class RAGConfig:
    data_dir: str = str(Path(__file__).resolve().parent.parent / "data")
    chunk_size: int = 300
    overlap: int = 50
    top_k: int = 3
    min_score: float = 0.05
    embedder: str = "tfidf"  # "tfidf" or "neural"
    store: str = "qdrant"    # "qdrant" (ベクトル DB) or "memory" (numpy の全件探索)
    model: str = DEFAULT_MODEL

    @property
    def collection(self) -> str:
        # チャンク分割や埋め込みの設定が違えば別のベクトル空間になるので、設定ごとに別コレクションにする
        return f"rag_{self.embedder}_c{self.chunk_size}_o{self.overlap}"


def create_store(config: RAGConfig) -> VectorStore:
    if config.store == "memory":
        return InMemoryVectorStore()
    if config.store == "qdrant":
        from .qdrant_store import QdrantVectorStore  # qdrant-client は使うときだけ import

        return QdrantVectorStore(config.collection)
    raise ValueError(f"unknown store: {config.store}")


class RAGPipeline:
    def __init__(
        self, config: RAGConfig | None = None, embedder: Embedder | None = None, store: VectorStore | None = None
    ) -> None:
        self.config = config or RAGConfig()
        self.embedder = embedder or create_embedder(self.config.embedder)
        self.store = store or create_store(self.config)
        self.embedder_path = INDEX_DIR / "embedders" / f"{self.config.collection}.json"

    def build_index(self, rebuild: bool = False) -> str:
        """インデックスを用意する。既に DB にあれば読み込むだけ。戻り値は "loaded" か "built"。"""
        if not rebuild and self.store.exists() and self.embedder.load(self.embedder_path):
            return "loaded"

        c = self.config
        docs = load_documents(c.data_dir)
        chunks = chunk_documents(docs, c.chunk_size, c.overlap)
        texts = [ch.text_for_embedding() for ch in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed_documents(texts)
        self.store.reset(dim=vectors.shape[1])
        self.store.add(chunks, vectors)
        if not isinstance(self.store, InMemoryVectorStore):
            self.embedder.save(self.embedder_path)  # 永続化するストアのときだけ、検索に必要な状態も保存する
        return "built"

    def retrieve(self, question: str, top_k: int | None = None, sources: list[str] | None = None) -> list[SearchResult]:
        query_vec = self.embedder.embed_query(question)
        return self.store.search(query_vec, top_k or self.config.top_k, self.config.min_score, sources)

    def ask(self, question: str) -> tuple[str, list[SearchResult]]:
        results = self.retrieve(question)
        return generate_answer(question, results, self.config.model), results
