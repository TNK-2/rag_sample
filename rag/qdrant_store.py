"""ベクトル DB (Qdrant) を使ったベクトルストア

Qdrant を選んだ理由:
  - 同じコードで「組み込みモード (サーバー不要・ローカルファイルに保存)」と
    「サーバーモード (Docker やクラウド)」を切り替えられる → 学習から本番まで同じ API
  - メタデータ (payload) での絞り込み、HNSW による高速な近似最近傍探索に対応

  組み込みモード (デフォルト): index/qdrant/ に保存。サーバー不要
  サーバーモード:             QDRANT_URL=http://localhost:6333 を設定
                              (docker run -p 6333:6333 qdrant/qdrant で起動できる)

ベクトル DB の基本用語:
  - コレクション: テーブルのようなもの。ベクトルの次元数と距離関数を決めて作る
  - ポイント:     1 件のデータ。id + ベクトル + payload (任意の JSON = メタデータ)
  - upsert:       同じ id があれば上書き、なければ追加。再実行しても重複しない (冪等)
"""

import atexit
import os
import uuid
import warnings
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from .chunker import Chunk
from .vector_store import INDEX_DIR, SearchResult


@lru_cache(maxsize=1)
def default_client() -> QdrantClient:
    """プロセス内で 1 つのクライアントを共有する。

    組み込みモードはファイルをロックするので、同じフォルダを 2 つのクライアント (や 2 つのプロセス) で
    同時に開けない。複数プロセスから使いたくなったら、それがサーバーモードに移るタイミング。
    """
    url = os.environ.get("QDRANT_URL")
    if url:
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    else:
        client = QdrantClient(path=str(INDEX_DIR / "qdrant"))
    atexit.register(client.close)  # 終了時に閉じてファイルロックを確実に解放する
    return client


class QdrantVectorStore:
    def __init__(self, collection: str, client: QdrantClient | None = None) -> None:
        self.collection = collection
        self.client = client or default_client()

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def reset(self, dim: int) -> None:
        # 次元数や距離関数はコレクション作成時に固定される。
        # 埋め込みモデルや語彙が変わったら (= ベクトル空間が変わったら) 作り直すしかない
        if self.exists():
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        # 絞り込みに使うフィールドにはインデックスを張っておくと、データが増えても速い
        # (組み込みモードでは効果がなく警告が出るだけなので、警告は抑えてサーバーモードと同じコードにしておく)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index(self.collection, "source", models.PayloadSchemaType.KEYWORD)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [
            models.PointStruct(
                # チャンク ID から決定的に UUID を作る → 同じチャンクを入れ直しても重複せず上書きされる
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                vector=vec.tolist(),
                payload={"text": chunk.text, "source": chunk.source, "title": chunk.title, "index": chunk.index},
            )
            for chunk, vec in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points)

    def count(self) -> int:
        return self.client.count(self.collection).count

    def search(
        self, query_vector: np.ndarray, top_k: int = 3, min_score: float = 0.0, sources: list[str] | None = None
    ) -> list[SearchResult]:
        if sources is not None and not sources:
            return []  # 検索対象の文書が 1 つも選ばれていない
        query_filter = None
        if sources is not None:
            # SQL の WHERE source IN (...) に相当。ベクトル検索と同時に DB 側で絞り込む
            query_filter = models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchAny(any=sources))]
            )
        if np.linalg.norm(query_vector) == 0:
            return []  # 質問の語がどれも語彙に無いと零ベクトルになり、コサイン類似度が定義できない
        response = self.client.query_points(
            self.collection,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=min_score,
            with_payload=True,
        )
        return [
            SearchResult(
                Chunk(text=p.payload["text"], source=p.payload["source"], title=p.payload["title"],
                      index=p.payload["index"]),
                p.score,
            )
            for p in response.points
        ]
