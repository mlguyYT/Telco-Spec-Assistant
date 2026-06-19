from __future__ import annotations

import os
from pathlib import Path

from retrieval.base import Retriever
from retrieval.local import LocalRetriever


def get_retriever(chunks_path: Path | None = None, kind: str | None = None) -> Retriever:
    retriever_kind = (kind or os.environ.get("RETRIEVER", "bm25")).lower()
    if retriever_kind == "bm25":
        if chunks_path is None:
            chunks_path = Path(os.environ.get("CHUNKS_PATH", ".data/chunks/rlc_v1.jsonl"))
        return LocalRetriever.from_jsonl(chunks_path)
    if retriever_kind == "vertex":
        raise RuntimeError("RETRIEVER=vertex is not implemented yet. Use RETRIEVER=bm25.")
    raise ValueError(f"unsupported retriever: {retriever_kind}")
