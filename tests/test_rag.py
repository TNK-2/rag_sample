import numpy as np
import pytest

from rag import RAGConfig, RAGPipeline
from rag.chunker import split_text
from rag.embedder import TfidfEmbedder, tokenize
from rag.generator import build_prompt


def test_tokenize_uses_bigrams_for_japanese_and_words_for_ascii():
    assert tokenize("有給休暇") == ["有給", "給休", "休暇"]
    assert tokenize("ＭＦＡ設定") == ["mfa", "設定"]  # NFKC で全角英字が半角・小文字に


def test_split_text_respects_chunk_size_and_overlap():
    text = "\n\n".join(f"段落{i}。" + "あ" * 40 for i in range(10))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # overlap のぶん chunk_size を超えることはあるが、それ以上には膨らまない
    assert all(len(c) <= 100 + 20 + 1 for c in chunks)
    assert chunks[1].startswith(chunks[0][-20:])


def test_split_text_keeps_heading_with_body():
    chunks = split_text("## 見出し\n\n本文です。", chunk_size=100, overlap=0)
    assert chunks == ["## 見出し\n本文です。"]


def test_split_text_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_tfidf_vectors_are_normalized_and_similar_texts_score_higher():
    emb = TfidfEmbedder()
    docs = ["有給休暇は10日付与", "経費精算の締め日は25日", "パスワードは12文字以上"]
    emb.fit(docs)
    vecs = emb.embed_documents(docs)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    scores = vecs @ emb.embed_query("有給休暇は何日？")
    assert int(np.argmax(scores)) == 0


def test_pipeline_retrieves_relevant_chunk():
    rag = RAGPipeline(RAGConfig())
    rag.build_index()
    results = rag.retrieve("リモートワークは週に何日まで？")
    assert results[0].chunk.source == "01_work_rules.md"
    assert "週 3 日" in results[0].chunk.text


def test_build_prompt_numbers_documents():
    rag = RAGPipeline(RAGConfig())
    rag.build_index()
    prompt = build_prompt("経費の締め日は？", rag.retrieve("経費の締め日は？"))
    assert '<document index="1"' in prompt
    assert "<question>\n経費の締め日は？\n</question>" in prompt
