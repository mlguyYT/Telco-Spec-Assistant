from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.embedding import DEFAULT_EMBEDDING_MODEL, DEFAULT_REGION, GenAIEmbedder

DEFAULT_BATCH_SIZE = 50
DEFAULT_CHUNKS_PATH = ".data/chunks/rlc_v1.jsonl"
DEFAULT_VECTOR_DIR = ".data/vector"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed existing clause chunks for Vertex AI Vector Search.")
    parser.add_argument("--chunks", default=os.environ.get("CHUNKS_PATH", DEFAULT_CHUNKS_PATH))
    parser.add_argument("--out-dir", default=os.environ.get("VECTOR_DATA_DIR", DEFAULT_VECTOR_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    chunks = _load_chunks(Path(args.chunks))
    vectors = embed_texts(
        texts=[str(chunk["text"]) for chunk in chunks],
        batch_size=args.batch_size,
    )
    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedded {len(vectors)} vectors for {len(chunks)} chunks")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "chunk_vectors.jsonl"
    metadata_path = out_dir / "chunk_metadata.json"
    manifest_path = out_dir / "manifest.json"

    with vectors_path.open("w", encoding="utf-8") as file:
        for chunk, vector in zip(chunks, vectors):
            file.write(json.dumps({"chunk_id": chunk["chunk_id"], "embedding": vector}) + "\n")

    metadata = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "chunk_count": len(chunks),
        "embedding_dimension": len(vectors[0]) if vectors else 0,
        "embedding_model": os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "region": os.environ.get("REGION", DEFAULT_REGION),
        "vectors_path": str(vectors_path),
        "metadata_path": str(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"embedded_chunks: {len(chunks)}")
    print(f"embedding_dimension: {manifest['embedding_dimension']}")
    print(f"wrote_vectors: {vectors_path}")
    print(f"wrote_metadata: {metadata_path}")
    print(f"wrote_manifest: {manifest_path}")


def embed_texts(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    project_id = _required_env("GCP_PROJECT_ID")
    region = os.environ.get("REGION", DEFAULT_REGION)
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedder = GenAIEmbedder(project_id=project_id, region=region, model_name=model_name)
    return embedder.embed_documents(texts, batch_size=batch_size)


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError(f"chunk file has no chunks: {path}")
    return chunks


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


if __name__ == "__main__":
    main()
