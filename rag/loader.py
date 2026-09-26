"""STEP 1: ドキュメントの読み込み (Load)

RAG の最初の一歩は「検索対象にしたい知識」をテキストとして集めることです。
ここでは data/ 配下の Markdown ファイルを読むだけですが、実務では
PDF・HTML・Word・Notion・Slack など多様なソースからテキストを抽出する部分が
一番泥臭く、品質にも大きく効きます(表や画像の扱い、ヘッダー/フッターのノイズ除去など)。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    source: str  # ファイル名。回答の「出典」として使う
    title: str   # 先頭の見出し。チャンクに文脈を付けるために使う
    text: str


def load_documents(data_dir: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(source=path.name, title=_extract_title(text, path.stem), text=text))
    return docs


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback
