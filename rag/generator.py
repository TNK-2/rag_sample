"""STEP 5: 生成 (Generation) — 検索結果をプロンプトに詰めて LLM に回答させる

RAG の "Augmented Generation" の部分です。LLM 自体は社内規程を知りませんが、
プロンプトに関連チャンクを「資料」として渡すことで、それを根拠に答えられるようになります。

プロンプト設計のポイント:
  - 資料と質問を明確に区別する (XML タグで囲むと Claude は構造を理解しやすい)
  - 「資料に無いことは答えない」と指示し、知ったかぶり (ハルシネーション) を抑える
  - 出典番号を付けさせ、ユーザーが根拠を確認できるようにする
"""

import os

from .vector_store import SearchResult

DEFAULT_MODEL = os.environ.get("RAG_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """\
あなたは株式会社ソラマメ・テクノロジーズの社内ヘルプデスクです。
<documents> 内の社内資料だけを根拠に、社員からの質問に日本語で簡潔に答えてください。

- 回答の根拠にした資料の番号を、文末に [1] のように付けてください。
- 資料から答えが分からない場合は、推測せずに「資料からは分かりませんでした」と答え、
  問い合わせ先の候補があれば添えてください。
"""


def build_prompt(question: str, results: list[SearchResult]) -> str:
    docs = "\n".join(
        f'<document index="{i}" source="{r.chunk.source}" title="{r.chunk.title}">\n'
        f"{r.chunk.text}\n</document>"
        for i, r in enumerate(results, start=1)
    )
    return f"<documents>\n{docs}\n</documents>\n\n<question>\n{question}\n</question>"


def generate_answer(question: str, results: list[SearchResult], model: str = DEFAULT_MODEL) -> str:
    import anthropic  # LLM を呼ぶときだけ必要

    client = anthropic.Anthropic()  # 認証情報は環境変数 ANTHROPIC_API_KEY などから自動で読まれる
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, results)}],
        # 安全性分類器がリクエストを断った場合に、サーバー側で別モデルに自動で切り替える設定
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "(モデルがこのリクエストへの回答を控えました)"
    return "".join(block.text for block in response.content if block.type == "text")
