from __future__ import annotations

import os

from retrieval.base import RetrievedChunk, Retriever
from retrieval.local import is_out_of_scope_query

DEFAULT_RRF_C = 40
DEFAULT_SOURCE_K = 100
DEFAULT_VECTOR_WEIGHT = 2.0


class HybridRetriever:
    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        retriever_weights: list[float] | None = None,
        source_k: int | None = None,
        rrf_c: int | None = None,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever")
        if retriever_weights is not None and len(retriever_weights) != len(retrievers):
            raise ValueError("retriever_weights must match retrievers")
        self.retrievers = retrievers
        self.retriever_weights = retriever_weights or [1.0] * len(retrievers)
        self.source_k = source_k or int(os.environ.get("HYBRID_SOURCE_K", str(DEFAULT_SOURCE_K)))
        self.rrf_c = rrf_c or int(os.environ.get("HYBRID_RRF_C", str(DEFAULT_RRF_C)))

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if is_out_of_scope_query(query):
            return []

        source_k = max(self.source_k, k)
        fused: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        for retriever, weight in zip(self.retrievers, self.retriever_weights):
            for rank, chunk in enumerate(retriever.retrieve(query, k=source_k), start=1):
                chunk_id = _chunk_id(chunk)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (self.rrf_c + rank)
                if chunk_id not in fused or rank < best_rank[chunk_id]:
                    fused[chunk_id] = chunk
                    best_rank[chunk_id] = rank

        ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id))
        return [
            RetrievedChunk(
                text=fused[chunk_id].text,
                score=scores[chunk_id],
                metadata=fused[chunk_id].metadata,
            )
            for chunk_id in ordered_ids[:k]
        ]


def _chunk_id(chunk: RetrievedChunk) -> str:
    value = chunk.metadata.get("chunk_id")
    if value:
        return str(value)
    return f"{chunk.metadata.get('document_id', '<unknown>')}:{chunk.metadata.get('section', '<unknown>')}"
