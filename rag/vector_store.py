"""STEP 4: ベクトルストア & 検索 (Retrieval)

チャンクのベクトルを保存し、質問ベクトルに「近い」ものを取り出します。

このリポジトリには 2 つの実装があり、同じインターフェース (VectorStore) で差し替えられます。

  InMemoryVectorStore (このファイル)
    numpy 配列で全件の内積 (= コサイン類似度) を計算し、上位 k 件を返すだけ (全件探索)。
    仕組みを理解するための実装。プロセスが終わるとインデックスは消える。

  QdrantVectorStore (rag/qdrant_store.py)
    本物のベクトル DB。永続化・メタデータでの絞り込み・大規模データ向けの近似最近傍探索 (HNSW) を
    DB 側が面倒を見てくれる。

数千〜数万チャンク程度なら全件探索でも十分速い。「最初からベクトル DB が必要」とは限らない、
というのも大事な設計判断です (docs/02_design_decisions.md 参照)。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunker import Chunk

# 永続化するインデックス (ベクトル DB のデータ、TF-IDF の語彙など) の置き場所
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR") or Path(__file__).resolve().parent.parent / "index")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # コサイン類似度 (-1〜1、TF-IDF なら 0〜1)。大きいほど似ている


class VectorStore(Protocol):
    def exists(self) -> bool: ...
    def reset(self, dim: int) -> None: ...
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def count(self) -> int: ...
    def search(
        self, query_vector: np.ndarray, top_k: int = 3, min_score: float = 0.0, sources: list[str] | None = None
    ) -> list[SearchResult]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape: (チャンク数, 次元数)

    def exists(self) -> bool:
        return self.vectors is not None

    def reset(self, dim: int) -> None:
        self.chunks = []
        self.vectors = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def count(self) -> int:
        return len(self.chunks)

    def search(
        self, query_vector: np.ndarray, top_k: int = 3, min_score: float = 0.0, sources: list[str] | None = None
    ) -> list[SearchResult]:
        if self.vectors is None:
            return []
        scores = self.vectors @ query_vector       # 全チャンクとの類似度を一括計算
        if sources is not None:                    # メタデータでの絞り込み: 対象外の文書は候補から外す
            allowed = np.array([c.source in sources for c in self.chunks])
            scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores)[:top_k]         # 類似度の高い順に top_k 件
        # min_score: どれも似ていないときに無関係なチャンクを LLM に渡さないための足切り
        return [SearchResult(self.chunks[i], float(scores[i])) for i in order if scores[i] > min_score]
