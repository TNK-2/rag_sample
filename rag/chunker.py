"""STEP 2: チャンク分割 (Chunking)

ドキュメントを丸ごと検索・プロンプト投入するのではなく、小さな「チャンク」に分けます。

なぜ分けるのか:
  - 検索の精度: 長い文書を 1 本のベクトルにすると、色々な話題が混ざって意味がぼやける
  - コスト/ノイズ: 質問に関係する部分だけを LLM に渡せば、トークン代も誤答も減る

設計判断のポイント:
  - chunk_size: 小さすぎると文脈が切れ、大きすぎると検索がぼやける (日本語なら 200〜800 文字が出発点)
  - overlap: 境界で情報が分断されるのを防ぐため、前のチャンクの末尾を少し重ねる
  - 区切り位置: 固定長で機械的に切るより、段落 → 文 → 文字 の順に「自然な境界」を優先する
"""

import re
from dataclasses import dataclass

from .loader import Document

# 日本語と英語の文末。区切り文字自体は直前の文に残す
_SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    index: int  # ドキュメント内での通し番号

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"

    def text_for_embedding(self) -> str:
        # チャンク単体だと「何の話か」が失われがちなので、文書タイトルを前置してから埋め込む。
        # (例: 「週 3 日まで利用できます」だけでは、何が週 3 日なのか分からない)
        return f"{self.title}\n{self.text}"


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """段落を単位に chunk_size 文字を目安として詰め込み、隣接チャンクを overlap 文字重ねる。"""
    if overlap >= chunk_size:
        raise ValueError("overlap は chunk_size より小さくしてください")

    # 1) 段落に分割。chunk_size を超える段落だけ、さらに文単位 → 文字単位に割る
    #    Markdown の見出しだけの段落は、次の段落とくっつける (見出しと本文が別チャンクに泣き別れないように)
    pieces: list[str] = []
    heading = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if re.fullmatch(r"#+ .*", para):
            heading = f"{heading}\n{para}" if heading else para
            continue
        para = f"{heading}\n{para}" if heading else para
        heading = ""
        pieces.extend([para] if len(para) <= chunk_size else _split_long(para, chunk_size))
    if heading:
        pieces.append(heading)

    # 2) 上限に達するまで段落を詰めていく
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > chunk_size:
            chunks.append(current)
            # 3) 直前チャンクの末尾を次のチャンクの頭に重ねる (overlap)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n{piece}" if tail else piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for sentence in filter(None, _SENTENCE_END.split(paragraph)):
        if len(sentence) > chunk_size:  # 1 文が長すぎる場合は文字数で強制的に切る
            if current:
                parts.append(current)
                current = ""
            parts.extend(sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size))
        elif len(current) + len(sentence) > chunk_size:
            parts.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def chunk_documents(docs: list[Document], chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    return [
        Chunk(text=t, source=doc.source, title=doc.title, index=i)
        for doc in docs
        for i, t in enumerate(split_text(doc.text, chunk_size, overlap))
    ]
