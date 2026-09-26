import os
import tempfile

# テストがリポジトリの index/ を汚さないよう、永続化先を一時ディレクトリに向ける (rag を import する前に設定)
os.environ["RAG_INDEX_DIR"] = tempfile.mkdtemp(prefix="rag_index_")
os.environ.pop("QDRANT_URL", None)
