from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from retrieval.base import RetrievedChunk
from retrieval.embedding import DEFAULT_EMBEDDING_MODEL, DEFAULT_REGION, GenAIEmbedder

DEFAULT_CHUNKS_PATH = ".data/chunks/rlc_v1.jsonl"


class VertexRetriever:
    """Vertex AI Vector Search retriever.

    Cloud dependencies are imported lazily so local BM25 development and CI do not
    require Google credentials or the optional cloud dependency set.
    """

    def __init__(self, chunks_path: Path | None = None, top_k: int | None = None) -> None:
        aiplatform = _load_vertex_dependencies()
        self.project_id = _required_env("GCP_PROJECT_ID")
        self.region = os.environ.get("REGION", DEFAULT_REGION)
        self.embedding_model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.endpoint_id = _required_env("VS_ENDPOINT_ID")
        self.deployed_index_id = _required_env("VS_DEPLOYED_INDEX_ID")
        self.default_top_k = top_k or int(os.environ.get("TOP_K", "5"))

        path = chunks_path or Path(os.environ.get("CHUNKS_PATH", DEFAULT_CHUNKS_PATH))
        self.by_id = _load_chunk_map(path)

        aiplatform.init(project=self.project_id, location=self.region)
        self.embedder = GenAIEmbedder(
            project_id=self.project_id,
            region=self.region,
            model_name=self.embedding_model_name,
        )
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=self.endpoint_id)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        top_k = k or self.default_top_k
        query_vector = self.embedder.embed_query(query)
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_vector],
            num_neighbors=top_k,
        )
        if not response:
            return []

        results: list[RetrievedChunk] = []
        for neighbor in response[0]:
            chunk_id = _neighbor_id(neighbor)
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                continue
            metadata = dict(chunk)
            text = str(metadata.pop("text"))
            results.append(
                RetrievedChunk(
                    text=text,
                    score=float(getattr(neighbor, "distance", 0.0)),
                    metadata=metadata,
                )
            )
        return results


def _load_vertex_dependencies() -> Any:
    try:
        from google.cloud import aiplatform
    except ImportError as exc:
        raise RuntimeError(
            "RETRIEVER=vertex requires optional cloud dependencies. "
            "Install them with: pip install -r requirements-cloud.txt"
        ) from exc
    return aiplatform


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required when RETRIEVER=vertex")
    return value


def _load_chunk_map(path: Path) -> dict[str, dict[str, Any]]:
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError(f"duplicate chunk_id values in {path}")
    if not by_id:
        raise ValueError(f"chunk file has no chunks: {path}")
    return by_id


def _neighbor_id(neighbor: Any) -> str:
    direct_id = getattr(neighbor, "id", None)
    if direct_id:
        return str(direct_id)
    datapoint = getattr(neighbor, "datapoint", None)
    datapoint_id = getattr(datapoint, "datapoint_id", None)
    if datapoint_id:
        return str(datapoint_id)
    raise ValueError("Vector Search neighbor did not include a datapoint id")
