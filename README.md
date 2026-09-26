# rag_sample — 作りながら学ぶシンプルな RAG

架空の会社「ソラマメ・テクノロジーズ」の社内規程 (`data/`) に答える、社内ヘルプデスク RAG です。
LLM が**知らないはずの**情報を使うので、「検索して渡したから答えられた」ことが実感できます。

- 検索部分 (チャンク分割・埋め込み・ベクトル検索) は **numpy だけで一から実装**。中身が全部読めます
- 回答生成は Claude API を使います
- 各ファイルの冒頭に、そのステップの**概念と設計判断のポイント**を書いています

## クイックスタート

```bash
pip install -r requirements.txt

# 1. 各ステップの中身を順に覗く (API キー不要)
python tutorial.py

# 2. 検索結果と LLM に送るプロンプトを確認する (API キー不要)
python main.py "リモートワークは週何日まで？" --retrieve-only

# 3. 検索精度を測る / パラメータを変えて比較する (API キー不要)
python evaluate.py
python evaluate.py --sweep

# 4. Claude に回答させる (API キーが必要)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py "リモートワークは週何日まで？"
python main.py            # 対話モード

# テスト
python -m pytest
```

モデルは `RAG_MODEL` 環境変数で変えられます (デフォルト `claude-opus-5`)。

## 全体像

```
[インデックス作成: 事前に 1 回]
 data/*.md ─▶ loader ─▶ chunker ─▶ embedder ─▶ vector_store
             (読込)     (分割)      (ベクトル化)   (保存)

[質問応答: 質問のたび]
 質問 ─▶ embedder ─▶ vector_store.search ─▶ generator ─▶ 回答
        (ベクトル化)  (似たチャンク top-k)    (プロンプト + Claude)
```

| ファイル | ステップ | 学べること |
|---|---|---|
| `rag/loader.py` | 1. 読み込み | データ取り込みが品質を左右する理由 |
| `rag/chunker.py` | 2. チャンク分割 | chunk_size / overlap / 自然な区切り |
| `rag/embedder.py` | 3. 埋め込み | TF-IDF (疎ベクトル) とニューラル (密ベクトル) の違い、日本語のトークン化 |
| `rag/vector_store.py` | 4. 検索 | コサイン類似度、top-k、足切り、ベクトル DB がいつ必要か |
| `rag/generator.py` | 5. 生成 | 根拠付き回答のプロンプト設計 |
| `rag/pipeline.py` | 全体 | 各ステップのつなぎ方 |
| `evaluate.py` | 評価 | Hit Rate / MRR で検索精度を測る |

## ドキュメント

- [docs/01_concepts.md](docs/01_concepts.md) — RAG の概念 (なぜ必要か、各ステップの意味)
- [docs/02_design_decisions.md](docs/02_design_decisions.md) — 技術選定と設計判断のポイント
- [docs/03_exercises.md](docs/03_exercises.md) — 手を動かして理解を深める演習
