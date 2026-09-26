"""STEP 4: ベクトルストア & 検索 (Retrieval)

チャンクのベクトルを保存し、質問ベクトルに「近い」ものを取り出します。

仕組みは驚くほど単純で、正規化済みベクトル同士の内積 (= コサイン類似度) を
全チャンクについて計算し、上位 k 件を返すだけです (全件探索 / brute force)。

数千〜数万チャンク程度ならこれで十分速い。それ以上になると
  - 近似最近傍探索 (ANN): FAISS, HNSW など。少し精度を犠牲に桁違いに速くする
  - 専用 DB: Chroma, Qdrant, pgvector など。永続化・メタデータ絞り込み・更新も面倒を見てくれる
を検討します。「最初からベクトル DB が必要」とは限らない、というのが大事な設計判断です。
"""

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # コサイン類似度 (-1〜1、TF-IDF なら 0〜1)。大きいほど似ている


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None  # shape: (チャンク数, 次元数)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 3, min_score: float = 0.0) -> list[SearchResult]:
        if self.vectors is None:
            return []
        scores = self.vectors @ query_vector       # 全チャンクとの類似度を一括計算
        order = np.argsort(-scores)[:top_k]         # 類似度の高い順に top_k 件
        # min_score: どれも似ていないときに無関係なチャンクを LLM に渡さないための足切り
        return [SearchResult(self.chunks[i], float(scores[i])) for i in order if scores[i] > min_score]
