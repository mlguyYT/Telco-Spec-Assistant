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
        from retrieval.vertex import VertexRetriever

        return VertexRetriever(chunks_path=chunks_path)
    if retriever_kind == "hybrid":
        from retrieval.hybrid import HybridRetriever
        from retrieval.vertex import VertexRetriever

        if chunks_path is None:
            chunks_path = Path(os.environ.get("CHUNKS_PATH", ".data/chunks/rlc_v1.jsonl"))
        return HybridRetriever(
            [
                LocalRetriever.from_jsonl(chunks_path),
                VertexRetriever(chunks_path=chunks_path),
            ],
            retriever_weights=[1.0, float(os.environ.get("HYBRID_VERTEX_WEIGHT", "1.02"))],
        )
    raise ValueError(f"unsupported retriever: {retriever_kind}")
