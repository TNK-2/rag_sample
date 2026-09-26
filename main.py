"""RAG に質問する CLI

  python main.py "リモートワークは週何日まで？"
  python main.py "..." --retrieve-only   # LLM を呼ばず、検索結果とプロンプトだけ表示 (API キー不要)
  python main.py                          # 対話モード
"""

import argparse

from rag import RAGConfig, RAGPipeline
from rag.generator import build_prompt, generate_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="シンプルな RAG のデモ")
    parser.add_argument("question", nargs="?", help="質問 (省略すると対話モード)")
    parser.add_argument("--retrieve-only", action="store_true", help="LLM を呼ばずに検索結果とプロンプトを表示")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--embedder", choices=["tfidf", "neural"], default="tfidf")
    parser.add_argument("--store", choices=["qdrant", "memory"], default="qdrant",
                        help="qdrant: ベクトル DB に保存 / memory: numpy で毎回作り直す")
    args = parser.parse_args()

    config = RAGConfig(top_k=args.top_k, chunk_size=args.chunk_size, overlap=args.overlap,
                       embedder=args.embedder, store=args.store)
    rag = RAGPipeline(config)
    how = rag.build_index()
    label = "既存のインデックスを読み込みました" if how == "loaded" else "インデックスを作成しました"
    print(f"{label} ({args.store}): {rag.store.count()} チャンク\n")

    if args.question:
        answer_one(rag, args.question, args.retrieve_only)
        return
    while True:
        try:
            question = input("質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question:
            answer_one(rag, question, args.retrieve_only)


def answer_one(rag: RAGPipeline, question: str, retrieve_only: bool) -> None:
    results = rag.retrieve(question)
    print("── 検索結果 ──")
    if not results:
        print("(関連するチャンクが見つかりませんでした)")
    for i, r in enumerate(results, start=1):
        preview = r.chunk.text.replace("\n", " ")[:60]
        print(f"[{i}] score={r.score:.3f}  {r.chunk.id}  {preview}…")
    print()

    if retrieve_only:
        print("── LLM に送るプロンプト ──")
        print(build_prompt(question, results))
        return
    print("── 回答 ──")
    print(generate_answer(question, results, rag.config.model))
    print()


if __name__ == "__main__":
    main()
