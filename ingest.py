"""インデックスを作成して、ベクトル DB に保存する (API キー不要)

  python ingest.py                          # デフォルト設定 (chunk_size=300, overlap=50) で作り直す
  python ingest.py --chunk-size 200 --overlap 40

RAG では「インデックス作成 (重い・たまに)」と「検索 (軽い・毎回)」を分けるのが基本です。
資料 (data/) を更新したら、このスクリプトを実行し直してください。
"""

import argparse
import time

from rag import RAGConfig, RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="インデックスを作成してベクトル DB に保存する")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--embedder", choices=["tfidf", "neural"], default="tfidf")
    args = parser.parse_args()

    config = RAGConfig(chunk_size=args.chunk_size, overlap=args.overlap, embedder=args.embedder, store="qdrant")
    rag = RAGPipeline(config)
    start = time.perf_counter()
    rag.build_index(rebuild=True)
    elapsed = time.perf_counter() - start
    print(f"コレクション '{config.collection}' に {rag.store.count()} チャンクを保存しました ({elapsed:.2f} 秒)")


if __name__ == "__main__":
    main()
