# rag_sample — 作りながら学ぶシンプルな RAG

架空の会社「ソラマメ・テクノロジーズ」の社内規程 (`data/`) に答える、社内ヘルプデスク RAG です。
LLM が**知らないはずの**情報を使うので、「検索して渡したから答えられた」ことが実感できます。

- 検索部分 (チャンク分割・埋め込み・ベクトル検索) は **numpy だけで一から実装**。中身が全部読めます
- インデックスは**ベクトル DB (Qdrant)** に保存。サーバー不要の組み込みモードで動きます
- 回答生成は Claude API を使います
- 各ファイルの冒頭に、そのステップの**概念と設計判断のポイント**を書いています

## クイックスタート

```bash
pip install -r requirements.txt

# 1. 各ステップの中身を順に覗く (API キー不要)
python tutorial.py

# 2. インデックスを作ってベクトル DB (index/qdrant/) に保存する (API キー不要)
python ingest.py

# 3. 検索結果と LLM に送るプロンプトを確認する (API キー不要)
python main.py "リモートワークは週何日まで？" --retrieve-only

# 4. 検索精度を測る / パラメータを変えて比較する (API キー不要)
python evaluate.py
python evaluate.py --sweep

# 5. Claude に回答させる (API キーが必要)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py "リモートワークは週何日まで？"
python main.py            # 対話モード

# テスト
python -m pytest
```

## ベクトル DB (Qdrant)

| モード | 設定 | 用途 |
|---|---|---|
| 組み込み (デフォルト) | なし。`index/qdrant/` にファイル保存 | 学習・ローカル開発。1 プロセスからしか開けない |
| サーバー | `QDRANT_URL=http://localhost:6333` (`docker run -p 6333:6333 qdrant/qdrant`) | 複数プロセスから使う・本番 |

- 資料 (`data/`) を更新したら `python ingest.py` で作り直してください
- `main.py` と `api.py` を組み込みモードで**同時に**起動すると、ファイルロックでエラーになります
  (「already accessed by another instance」)。同時に使うならサーバーモードにしてください
- `--store memory` を付けると、ベクトル DB を使わずに numpy の全件探索で動きます (比較用)

## Web UI (React)

検索結果・スコア・ハイライト・プロンプト・回答を 1 画面で見られる UI です。
チャンクサイズや top_k をスライダーで変えながら、検索結果がどう変わるかを観察できます。

```bash
# ターミナル 1: API サーバー (FastAPI)
export ANTHROPIC_API_KEY=sk-ant-...   # 検索だけ試すなら不要 (UI で「回答を生成」をオフに)
uvicorn api:app --reload --port 8000

# ターミナル 2: フロントエンド開発サーバー (Vite)
cd web
npm install
npm run dev     # → http://localhost:5173
```

`cd web && npm run build` でビルドしておけば、`uvicorn api:app --port 8000` だけで
http://localhost:8000 から UI も配信されます。

| 画面の要素 | 対応するステップ |
|---|---|
| ① 検索: スコアバーとハイライト (質問と共通する文字 2-gram) | `store.search` (Qdrant) / TF-IDF |
| ② プロンプト: Claude に送られる内容そのもの | `generator.build_prompt` |
| ③ 回答: 出典番号 [1] にマウスを乗せると根拠のチャンクが強調される | `generator.generate_answer` |

モデルは `RAG_MODEL` 環境変数で変えられます (デフォルト `claude-opus-5`)。

## 全体像

```
[インデックス作成: 事前に 1 回 / 資料更新時]  python ingest.py
 data/*.md ─▶ loader ─▶ chunker ─▶ embedder ─▶ Qdrant (index/qdrant/)
             (読込)     (分割)      (ベクトル化)   (保存)

[質問応答: 質問のたび]
 質問 ─▶ embedder ─▶ Qdrant で検索 (+ フィルタ) ─▶ generator ─▶ 回答
        (ベクトル化)  (似たチャンク top-k)    (プロンプト + Claude)
```

| ファイル | ステップ | 学べること |
|---|---|---|
| `rag/loader.py` | 1. 読み込み | データ取り込みが品質を左右する理由 |
| `rag/chunker.py` | 2. チャンク分割 | chunk_size / overlap / 自然な区切り |
| `rag/embedder.py` | 3. 埋め込み | TF-IDF (疎ベクトル) とニューラル (密ベクトル) の違い、日本語のトークン化 |
| `rag/vector_store.py` | 4. 検索 | コサイン類似度、top-k、足切り (numpy の全件探索) |
| `rag/qdrant_store.py` | 4. 検索 | ベクトル DB: コレクション、upsert、メタデータフィルタ、永続化 |
| `ingest.py` | インデックス作成 | 「作成 (重い・たまに)」と「検索 (軽い・毎回)」を分ける |
| `rag/generator.py` | 5. 生成 | 根拠付き回答のプロンプト設計 |
| `rag/pipeline.py` | 全体 | 各ステップのつなぎ方 |
| `evaluate.py` | 評価 | Hit Rate / MRR で検索精度を測る |
| `api.py` / `web/` | UI | FastAPI で RAG を API 化し、React から呼ぶ |

## ドキュメント

- [docs/01_concepts.md](docs/01_concepts.md) — RAG の概念 (なぜ必要か、各ステップの意味)
- [docs/02_design_decisions.md](docs/02_design_decisions.md) — 技術選定と設計判断のポイント
- [docs/03_exercises.md](docs/03_exercises.md) — 手を動かして理解を深める演習
