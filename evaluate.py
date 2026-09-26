"""検索 (Retrieval) の精度を測る簡易評価スクリプト (API キー不要)

RAG の回答品質は「正しいチャンクを取ってこられるか」にほぼ依存します。
LLM がどれだけ賢くても、正解が書かれたチャンクを渡さなければ正しく答えられません。
そこで、質問と「正解が書かれている文書」のペアを用意し、上位 k 件に正解が含まれる割合
(Hit Rate@k) と、正解が何位に来たか (MRR: Mean Reciprocal Rank) を計測します。

  python evaluate.py                         # デフォルト設定で評価
  python evaluate.py --sweep                 # chunk_size / top_k を変えて比較

パラメータを変えるときは「感覚」ではなく、こうした数字で比較するのが設計判断の基本です。
"""

import argparse

from rag import RAGConfig, RAGPipeline

# (質問, 正解が書かれている文書, 正解チャンクに含まれるべきキーワード)
# わざと資料と違う言い回しの質問 (言い換え) も混ぜている
EVAL_SET = [
    ("リモートワークは週に何日まで？", "01_work_rules.md", "週 3 日"),
    ("在宅勤務の申請期限はいつ？", "01_work_rules.md", "前日の 18:00"),
    ("出社しないといけない曜日は？", "01_work_rules.md", "水曜日"),
    ("コアタイムは何時から何時？", "01_work_rules.md", "11:00〜15:00"),
    ("有給休暇は入社後いつもらえる？", "02_leave.md", "3 か月"),
    ("ソラマメ休暇の旅行補助はいくら？", "02_leave.md", "5 万円"),
    ("年休は何年で失効する？", "02_leave.md", "2 年で失効"),
    ("経費の締め日はいつ？", "03_expenses.md", "25 日"),
    ("取引先との会食の上限金額は？", "03_expenses.md", "8,000 円"),
    ("タクシーを使っていいのはどんなとき？", "03_expenses.md", "22:00 以降"),
    ("本を買うのに会社はいくらまで出してくれる？", "03_expenses.md", "3 万円"),
    ("パスワードは何文字以上必要？", "04_it_security.md", "12 文字"),
    ("PC をなくしたらどうすればいい？", "04_it_security.md", "1 時間以内"),
    ("ChatGPT に顧客情報を入れてもいい？", "04_it_security.md", "個人情報"),
]


def evaluate(config: RAGConfig, verbose: bool = False) -> tuple[float, float]:
    rag = RAGPipeline(config)
    rag.build_index()
    hits, rr_sum = 0, 0.0
    for question, source, keyword in EVAL_SET:
        results = rag.retrieve(question)
        rank = next(
            (i for i, r in enumerate(results, start=1) if r.chunk.source == source and keyword in r.chunk.text),
            None,
        )
        if rank:
            hits += 1
            rr_sum += 1 / rank
        if verbose:
            mark = f"○ {rank}位" if rank else "× 圏外"
            print(f"{mark:6}  {question}")
    n = len(EVAL_SET)
    return hits / n, rr_sum / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="パラメータを変えて比較する")
    parser.add_argument("--embedder", choices=["tfidf", "neural"], default="tfidf")
    args = parser.parse_args()

    if not args.sweep:
        config = RAGConfig(embedder=args.embedder)
        hit, mrr = evaluate(config, verbose=True)
        print(f"\nHit Rate@{config.top_k} = {hit:.2f}   MRR = {mrr:.2f}")
        return

    print(f"{'chunk_size':>10} {'overlap':>8} {'top_k':>6} {'Hit@k':>7} {'MRR':>6}")
    for chunk_size, overlap in [(100, 20), (200, 40), (300, 50), (600, 100)]:
        for top_k in [1, 3, 5]:
            config = RAGConfig(chunk_size=chunk_size, overlap=overlap, top_k=top_k, embedder=args.embedder)
            hit, mrr = evaluate(config)
            print(f"{chunk_size:>10} {overlap:>8} {top_k:>6} {hit:>7.2f} {mrr:>6.2f}")


if __name__ == "__main__":
    main()
