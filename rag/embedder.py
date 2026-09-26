"""STEP 3: 埋め込み (Embedding) — テキストをベクトル(数値の配列)に変換する

「意味が近いテキストほど、ベクトル空間上で近い位置に来る」ように変換できれば、
検索は「質問ベクトルに近いチャンクベクトルを探す」という幾何の問題になります。

このファイルには 2 種類の Embedder があり、同じインターフェースで差し替えられます。

  TfidfEmbedder (デフォルト)
    numpy だけで一から実装した「疎な」ベクトル。単語(ここでは文字 2-gram)の出現頻度ベース。
    - 利点: 依存ゼロ・高速・中身が全部見える → 学習用に最適。固有名詞や型番の完全一致に強い
    - 欠点: 言い換え・同義語に弱い (「有給」と「年休」は別物として扱われる)

  SentenceTransformerEmbedder (オプション)
    ニューラルネットの「密な」ベクトル。意味の近さを捉えられる。
    - 利点: 言い換え・同義語に強い
    - 欠点: モデルのダウンロードが必要、計算が重い、なぜヒットしたか説明しにくい

実務では両方を組み合わせる「ハイブリッド検索」もよく使われます (docs/02_design_decisions.md 参照)。
"""

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> bool: ...


# ---------------------------------------------------------------------------
# TF-IDF (from scratch)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """日本語向けの超シンプルなトークナイザ。

    日本語は単語の間にスペースがないため、形態素解析 (MeCab, Sudachi など) を使うのが本格的。
    ここでは依存を増やさないために「文字 2-gram」を使う:
        "有給休暇" -> ["有給", "給休", "休暇"]
    意外と検索ではよく効く手法です。英数字は単語単位で扱います。
    """
    text = unicodedata.normalize("NFKC", text).lower()  # 全角英数→半角、大文字→小文字 など表記ゆれを吸収
    tokens: list[str] = []
    # 「英数字の連続」と「それ以外の文字 (かな・漢字など) の連続」に分け、記号や空白は捨てる
    for span in re.findall(r"[a-z0-9]+|[^\W\da-z_]+", text):
        if span.isascii() or len(span) == 1:
            tokens.append(span)
        else:
            tokens.extend(span[i:i + 2] for i in range(len(span) - 1))
    return tokens


class TfidfEmbedder:
    """TF-IDF: 「その文書によく出て (TF)、他の文書にはあまり出ない (IDF)」語を重視する。

    例: 「です」「ます」はどの文書にも出る → IDF が低い → 重要でない
        「ソラマメ休暇」は特定の文書にしか出ない → IDF が高い → 手がかりとして重要
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def fit(self, texts: list[str]) -> None:
        # 検索対象コーパス全体から語彙と IDF を学習する (ニューラルモデルの「学習」に相当)
        doc_freq: Counter[str] = Counter()
        for t in texts:
            doc_freq.update(set(tokenize(t)))
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        n = len(texts)
        # smooth idf (scikit-learn と同じ式): log((1+N)/(1+df)) + 1
        self.idf = np.array([math.log((1 + n) / (1 + doc_freq[tok])) + 1 for tok in self.vocab])

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for tok, count in Counter(tokenize(text)).items():
            if tok in self.vocab:  # 未知語 (コーパスに無い語) は無視される → TF-IDF の限界の 1 つ
                vec[self.vocab[tok]] = count
        vec *= self.idf
        norm = np.linalg.norm(vec)
        # L2 正規化しておくと、内積 = コサイン類似度 になり検索が内積だけで済む
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vectorize(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)

    # TF-IDF は「どの語が何番目の次元か (vocab)」と「IDF」をコーパスから学習している。
    # インデックスを DB に永続化しても、この 2 つが無いと質問を同じベクトル空間に変換できないので一緒に保存する。
    # (事前学習済みのニューラル埋め込みモデルなら、モデル名さえ同じならこの心配はない)
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = sorted(self.vocab, key=self.vocab.__getitem__)
        path.write_text(json.dumps({"tokens": tokens, "idf": self.idf.tolist()}, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self.vocab = {tok: i for i, tok in enumerate(data["tokens"])}
        self.idf = np.array(data["idf"])
        return True


# ---------------------------------------------------------------------------
# Neural embedding (optional)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """多言語対応の埋め込みモデルを使う版。`pip install sentence-transformers` が必要。

    intfloat/multilingual-e5-* 系モデルは、学習時の形式に合わせて
    質問に "query: "、検索対象に "passage: " を付ける必要がある点に注意。
    (モデルごとの「お作法」を守らないと精度が落ちる、という良い例)
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer  # 使うときだけ import (重いので)

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        pass  # 事前学習済みなので、コーパスに合わせた学習は不要

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(f"query: {text}", normalize_embeddings=True)

    def save(self, path: Path) -> None:
        pass  # モデルの重みは Hugging Face のキャッシュにあるので保存不要

    def load(self, path: Path) -> bool:
        return True


def create_embedder(name: str) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "neural":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder: {name}")
