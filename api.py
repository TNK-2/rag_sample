"""Web UI 用の API サーバー (FastAPI)

  uvicorn api:app --reload --port 8000

  GET  /api/status    ... 文書一覧・使用モデル・ベクトルストアの情報
  POST /api/retrieve  ... 検索だけ行い、検索結果と LLM に送るプロンプトを返す (API キー不要)
  POST /api/answer    ... 検索 + Claude による回答生成

web/dist (npm run build の出力) があれば、それも同じサーバーから配信する。
"""

import os
from functools import lru_cache
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import RAGConfig, RAGPipeline
from rag.generator import DEFAULT_MODEL, build_prompt, generate_answer
from rag.loader import load_documents
from rag.vector_store import INDEX_DIR

app = FastAPI(title="rag_sample")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    chunk_size: int = Field(default=300, ge=50, le=2000)
    overlap: int = Field(default=50, ge=0, le=500)
    # 検索対象の文書を絞り込む (ベクトル DB のメタデータフィルタ)。None なら全文書
    sources: list[str] | None = None


class ChunkResult(BaseModel):
    id: str
    source: str
    title: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    results: list[ChunkResult]
    prompt: str
    num_chunks: int
    collection: str


class AnswerResponse(BaseModel):
    answer: str
    model: str


@lru_cache(maxsize=16)
def get_pipeline(chunk_size: int, overlap: int) -> RAGPipeline:
    # チャンク分割の設定ごとにコレクションが分かれる。初めての設定なら作成し、既にあれば読み込むだけ
    rag = RAGPipeline(RAGConfig(chunk_size=chunk_size, overlap=overlap))
    rag.build_index()
    return rag


def _retrieve(req: QueryRequest):
    if req.overlap >= req.chunk_size:
        raise HTTPException(422, "overlap は chunk_size より小さくしてください")
    rag = get_pipeline(req.chunk_size, req.overlap)
    return rag, rag.retrieve(req.question, req.top_k, req.sources)


@app.get("/api/status")
def status():
    config = RAGConfig()
    docs = load_documents(config.data_dir)
    qdrant_url = os.environ.get("QDRANT_URL")
    return {
        "model": DEFAULT_MODEL,
        "documents": [{"source": d.source, "title": d.title} for d in docs],
        "store": {
            "kind": config.store,
            "location": qdrant_url or f"{INDEX_DIR.name}/qdrant (組み込みモード)",
        },
    }


@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve(req: QueryRequest):
    rag, results = _retrieve(req)
    return RetrieveResponse(
        results=[
            ChunkResult(id=r.chunk.id, source=r.chunk.source, title=r.chunk.title, text=r.chunk.text, score=r.score)
            for r in results
        ],
        prompt=build_prompt(req.question, results),
        num_chunks=rag.store.count(),
        collection=rag.config.collection,
    )


@app.post("/api/answer", response_model=AnswerResponse)
def answer(req: QueryRequest):
    rag, results = _retrieve(req)
    try:
        text = generate_answer(req.question, results, rag.config.model)
    except anthropic.AuthenticationError:
        raise HTTPException(401, "Claude API の認証に失敗しました。ANTHROPIC_API_KEY を確認してください")
    except anthropic.RateLimitError:
        raise HTTPException(429, "レート制限に達しました。少し待ってから再度お試しください")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API エラー ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Claude API に接続できませんでした")
    except anthropic.AnthropicError as e:
        raise HTTPException(500, f"Claude API を呼び出せませんでした: {e}")
    except TypeError as e:
        # 認証情報がどこにも設定されていないと、SDK はリクエスト組み立て時に TypeError を投げる
        if "authentication" not in str(e):
            raise
        raise HTTPException(401, "Claude API の認証情報がありません。ANTHROPIC_API_KEY を設定してサーバーを再起動してください")
    return AnswerResponse(answer=text, model=rag.config.model)


_dist = Path(__file__).parent / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
