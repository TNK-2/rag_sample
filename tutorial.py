"""RAG の各ステップを 1 つずつ実行して中身を覗くチュートリアル (API キー不要)

  python tutorial.py

Enter キーを押すごとに次のステップへ進みます。
"""

import numpy as np

from rag.chunker import chunk_documents
from rag.embedder import TfidfEmbedder, tokenize
from rag.generator import build_prompt
from rag.loader import load_documents
from rag.pipeline import RAGConfig
from rag.vector_store import InMemoryVectorStore

QUESTION = "リモートワークは週に何日までできますか？"


def step(title: str) -> None:
    input(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n(Enter で実行)")


def main() -> None:
    config = RAGConfig()

    step("STEP 1: ドキュメントを読み込む")
    docs = load_documents(config.data_dir)
    for d in docs:
        print(f"- {d.source}: 「{d.title}」 {len(d.text)} 文字")

    step("STEP 2: チャンクに分割する")
    chunks = chunk_documents(docs, config.chunk_size, config.overlap)
    print(f"{len(docs)} 文書 → {len(chunks)} チャンク (chunk_size={config.chunk_size}, overlap={config.overlap})\n")
    for c in chunks[:3]:
        print(f"--- {c.id} ({len(c.text)} 文字) ---\n{c.text}\n")
    print("… 以下略。overlap により、前のチャンクの末尾が次のチャンクの先頭に重なっている点に注目。")

    step("STEP 3: テキストをベクトルに変換する (TF-IDF)")
    print(f"トークン化の例: 「有給休暇の繰り越し」 → {tokenize('有給休暇の繰り越し')}\n")
    embedder = TfidfEmbedder()
    texts = [c.text_for_embedding() for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed_documents(texts)
    print(f"ベクトル行列の形: {vectors.shape}  (チャンク数 × 語彙数)")
    nonzero = np.count_nonzero(vectors[0])
    print(f"チャンク 0 の非ゼロ要素: {nonzero} / {vectors.shape[1]}  → ほとんどが 0 の「疎ベクトル」\n")
    inv_vocab = {i: t for t, i in embedder.vocab.items()}
    top = np.argsort(-vectors[0])[:8]
    print("チャンク 0 で重みの大きいトークン:", [(inv_vocab[i], round(float(vectors[0][i]), 3)) for i in top])

    step("STEP 4: 質問をベクトル化し、似たチャンクを検索する")
    store = InMemoryVectorStore()
    store.add(chunks, vectors)
    print(f"質問: {QUESTION}\n")
    all_scores = vectors @ embedder.embed_query(QUESTION)
    for i in np.argsort(-all_scores)[:6]:
        print(f"score={all_scores[i]:.3f}  {chunks[i].id}  {chunks[i].text[:40].replace(chr(10), ' ')}…")
    print(f"\n→ 上位 {config.top_k} 件を LLM に渡す資料として採用します。")
    results = store.search(embedder.embed_query(QUESTION), config.top_k, config.min_score)

    step("STEP 5: 検索結果をプロンプトに詰める")
    print(build_prompt(QUESTION, results))
    print("\nこのプロンプト + システムプロンプトを LLM に送ると、資料を根拠にした回答が返ってきます。")
    print('実際に回答させるには: python main.py "リモートワークは週に何日までできますか？"')


if __name__ == "__main__":
    main()
