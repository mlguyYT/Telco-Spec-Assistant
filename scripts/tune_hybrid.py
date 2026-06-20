from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run import evaluate_with_retriever
from retrieval.base import RetrievedChunk
from retrieval.embedding import DEFAULT_EMBEDDING_MODEL, DEFAULT_REGION, GenAIEmbedder
from retrieval.local import LocalRetriever, is_out_of_scope_query

DEFAULT_CHUNKS_PATH = ".data/chunks/telco_v1.jsonl"
DEFAULT_DATASET_PATH = "eval/datasets/telco_retrieval_v1.jsonl"
DEFAULT_VECTOR_DIR = ".data/vector"
DEFAULT_TOP_K = 5
DEFAULT_SOURCE_K_VALUES = "20,50,100"
DEFAULT_VECTOR_WEIGHT_VALUES = "0.8,1.0,1.02,1.1,1.25,1.5,2.0"
DEFAULT_RRF_C_VALUES = "20,40,60,80"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep hybrid RRF settings using local chunk vectors and cached query embeddings."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--vector-dir", default=os.environ.get("VECTOR_DATA_DIR", DEFAULT_VECTOR_DIR))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--source-k-values", default=DEFAULT_SOURCE_K_VALUES)
    parser.add_argument("--vector-weight-values", default=DEFAULT_VECTOR_WEIGHT_VALUES)
    parser.add_argument("--rrf-c-values", default=DEFAULT_RRF_C_VALUES)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    np = _load_numpy()
    dataset_path = Path(args.dataset)
    chunks_path = Path(args.chunks)
    vector_dir = Path(args.vector_dir)
    rows = _load_dataset(dataset_path)
    query_to_id = {str(row["question"]): str(row["id"]) for row in rows}
    chunks = _load_chunks(chunks_path)
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    chunk_vectors, vector_chunk_ids = _load_chunk_vectors(np, vector_dir)
    query_vectors = _load_or_embed_query_vectors(
        np=np,
        rows=rows,
        vector_dir=vector_dir,
        dataset_path=dataset_path,
    )

    source_k_values = _parse_ints(args.source_k_values)
    vector_weight_values = _parse_floats(args.vector_weight_values)
    rrf_c_values = _parse_ints(args.rrf_c_values)
    max_source_k = max(max(source_k_values), args.top_k)

    bm25_rankings = _bm25_rankings(chunks, rows, source_k=max_source_k)
    vector_rankings = _vector_rankings(
        np=np,
        rows=rows,
        query_vectors=query_vectors,
        chunk_vectors=chunk_vectors,
        vector_chunk_ids=vector_chunk_ids,
        chunk_by_id=chunk_by_id,
        source_k=max_source_k,
    )

    reports = []
    for source_k in source_k_values:
        for vector_weight in vector_weight_values:
            for rrf_c in rrf_c_values:
                retriever = _FusedRetriever(
                    bm25_rankings=bm25_rankings,
                    vector_rankings=vector_rankings,
                    query_to_id=query_to_id,
                    source_k=source_k,
                    vector_weight=vector_weight,
                    rrf_c=rrf_c,
                )
                report = evaluate_with_retriever(dataset_path, retriever, top_k=args.top_k)
                reports.append(_summary(source_k, vector_weight, rrf_c, report))

    reports.sort(
        key=lambda row: (
            -(row["answerable_recall_at_k"] or 0.0),
            -(row["subset_recall_at_k"].get("paraphrase") or 0.0),
            -(row["non_paraphrase_recall_at_k"] or 0.0),
            row["source_k"],
            row["rrf_c"],
            row["vector_weight"],
        )
    )
    for row in reports[: args.limit]:
        print(json.dumps(row, sort_keys=True))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"dataset has no rows: {path}")
    return rows


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError(f"chunk file has no chunks: {path}")
    return chunks


def _load_chunk_vectors(np: Any, vector_dir: Path) -> tuple[Any, list[str]]:
    rows = [
        json.loads(line)
        for line in (vector_dir / "chunk_vectors.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"vector file has no rows: {vector_dir / 'chunk_vectors.jsonl'}")
    chunk_ids = [str(row["chunk_id"]) for row in rows]
    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    return vectors, chunk_ids


def _load_or_embed_query_vectors(np: Any, rows: list[dict[str, Any]], vector_dir: Path, dataset_path: Path) -> Any:
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    cache_path = vector_dir / f"query_vectors_{dataset_path.stem}_{model_name}.jsonl"
    cached = _read_query_vector_cache(cache_path)
    missing = [row for row in rows if str(row["id"]) not in cached]
    if missing:
        project_id = _required_env("GCP_PROJECT_ID")
        embedder = GenAIEmbedder(
            project_id=project_id,
            region=os.environ.get("REGION", DEFAULT_REGION),
            model_name=model_name,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as file:
            for row in missing:
                embedding = embedder.embed_query(str(row["question"]))
                cached[str(row["id"])] = embedding
                file.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "question": row["question"],
                            "embedding_model": model_name,
                            "embedding": embedding,
                        }
                    )
                    + "\n"
                )
    return np.asarray([cached[str(row["id"])] for row in rows], dtype=np.float32)


def _read_query_vector_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[str(row["id"])] = [float(value) for value in row["embedding"]]
    return cache


def _bm25_rankings(chunks: list[dict[str, Any]], rows: list[dict[str, Any]], source_k: int) -> dict[str, list[RetrievedChunk]]:
    retriever = LocalRetriever(chunks)
    return {str(row["id"]): retriever.retrieve(str(row["question"]), k=source_k) for row in rows}


def _vector_rankings(
    *,
    np: Any,
    rows: list[dict[str, Any]],
    query_vectors: Any,
    chunk_vectors: Any,
    vector_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    source_k: int,
) -> dict[str, list[RetrievedChunk]]:
    rankings: dict[str, list[RetrievedChunk]] = {}
    scores = query_vectors @ chunk_vectors.T
    for row_index, row in enumerate(rows):
        question = str(row["question"])
        if is_out_of_scope_query(question):
            rankings[str(row["id"])] = []
            continue
        row_scores = scores[row_index]
        limit = min(source_k, len(row_scores))
        candidate_indexes = np.argpartition(-row_scores, limit - 1)[:limit]
        ordered_indexes = candidate_indexes[np.argsort(-row_scores[candidate_indexes])]
        results = []
        for chunk_index in ordered_indexes:
            chunk_id = vector_chunk_ids[int(chunk_index)]
            chunk = chunk_by_id[chunk_id]
            metadata = dict(chunk)
            text = str(metadata.pop("text"))
            results.append(RetrievedChunk(text=text, score=float(row_scores[int(chunk_index)]), metadata=metadata))
        rankings[str(row["id"])] = results
    return rankings


class _FusedRetriever:
    def __init__(
        self,
        *,
        bm25_rankings: dict[str, list[RetrievedChunk]],
        vector_rankings: dict[str, list[RetrievedChunk]],
        query_to_id: dict[str, str],
        source_k: int,
        vector_weight: float,
        rrf_c: int,
    ) -> None:
        self.bm25_rankings = bm25_rankings
        self.vector_rankings = vector_rankings
        self.query_to_id = query_to_id
        self.source_k = source_k
        self.vector_weight = vector_weight
        self.rrf_c = rrf_c

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_id = self.query_to_id[query]
        return _fuse(
            [self.bm25_rankings.get(query_id, [])[: self.source_k], self.vector_rankings.get(query_id, [])[: self.source_k]],
            weights=[1.0, self.vector_weight],
            rrf_c=self.rrf_c,
            k=k,
        )


def _fuse(
    rankings: list[list[RetrievedChunk]],
    *,
    weights: list[float],
    rrf_c: int,
    k: int,
) -> list[RetrievedChunk]:
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = str(chunk.metadata["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_c + rank)
            if chunk_id not in fused or rank < best_rank[chunk_id]:
                fused[chunk_id] = chunk
                best_rank[chunk_id] = rank
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id))
    return [
        RetrievedChunk(text=fused[chunk_id].text, score=scores[chunk_id], metadata=fused[chunk_id].metadata)
        for chunk_id in ordered_ids[:k]
    ]


def _summary(source_k: int, vector_weight: float, rrf_c: int, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_k": source_k,
        "vector_weight": vector_weight,
        "rrf_c": rrf_c,
        "answerable_recall_at_k": report["answerable_recall_at_k"],
        "non_paraphrase_recall_at_k": report["non_paraphrase_recall_at_k"],
        "subset_recall_at_k": report["subset_recall_at_k"],
        "per_spec_recall_at_k": report["per_spec_recall_at_k"],
        "abstention_accuracy": report["abstention_accuracy"],
    }


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid tuning requires numpy. Install optional cloud dependencies with: "
            "pip install -r requirements-cloud.txt"
        ) from exc
    return np


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


if __name__ == "__main__":
    main()
