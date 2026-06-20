from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DEPLOYED_INDEX_ID = "telco_deployed"
DEFAULT_INDEX_DISPLAY_NAME = "telco-spec-index"
DEFAULT_ENDPOINT_DISPLAY_NAME = "telco-spec-endpoint"
DEFAULT_REGION = "us-central1"
DEFAULT_VECTOR_DIR = ".data/vector"
DEFAULT_UPSERT_BATCH_SIZE = 1000


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a brute-force Vertex AI Vector Search index, upsert local vectors, "
            "and deploy it to a public endpoint."
        )
    )
    parser.add_argument("--vector-dir", default=os.environ.get("VECTOR_DATA_DIR", DEFAULT_VECTOR_DIR))
    parser.add_argument("--index-display-name", default=os.environ.get("VECTOR_INDEX_DISPLAY_NAME", DEFAULT_INDEX_DISPLAY_NAME))
    parser.add_argument(
        "--endpoint-display-name",
        default=os.environ.get("VECTOR_ENDPOINT_DISPLAY_NAME", DEFAULT_ENDPOINT_DISPLAY_NAME),
    )
    parser.add_argument("--deployed-index-id", default=os.environ.get("VS_DEPLOYED_INDEX_ID", DEFAULT_DEPLOYED_INDEX_ID))
    parser.add_argument("--upsert-batch-size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE)
    parser.add_argument("--min-replica-count", type=int, default=int(os.environ.get("VS_MIN_REPLICA_COUNT", "1")))
    parser.add_argument("--max-replica-count", type=int, default=int(os.environ.get("VS_MAX_REPLICA_COUNT", "1")))
    args = parser.parse_args()

    project_id = _required_env("GCP_PROJECT_ID")
    region = os.environ.get("REGION", DEFAULT_REGION)
    vectors_path = Path(args.vector_dir) / "chunk_vectors.jsonl"
    manifest_path = Path(args.vector_dir) / "manifest.json"
    vectors = _load_vectors(vectors_path)
    manifest = _load_manifest(manifest_path)
    dimension = int(manifest["embedding_dimension"])

    aiplatform, aiplatform_v1, json_format, struct_pb2 = _load_cloud_dependencies()
    aiplatform.init(project=project_id, location=region)

    index_name = _create_streaming_brute_force_index(
        aiplatform_v1=aiplatform_v1,
        json_format=json_format,
        struct_pb2=struct_pb2,
        project_id=project_id,
        region=region,
        display_name=args.index_display_name,
        dimension=dimension,
    )
    print(f"created_index: {index_name}")

    index = aiplatform.MatchingEngineIndex(index_name=index_name)
    datapoints = [
        aiplatform_v1.types.index.IndexDatapoint(
            datapoint_id=str(row["chunk_id"]),
            feature_vector=[float(value) for value in row["embedding"]],
        )
        for row in vectors
    ]
    total_upserted = 0
    for batch in _batched(datapoints, args.upsert_batch_size):
        index.upsert_datapoints(datapoints=batch)
        total_upserted += len(batch)
        print(f"upserted_datapoints: {total_upserted}/{len(datapoints)}")

    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=args.endpoint_display_name,
        public_endpoint_enabled=True,
    )
    print(f"created_endpoint: {endpoint.resource_name}")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=args.deployed_index_id,
        min_replica_count=args.min_replica_count,
        max_replica_count=args.max_replica_count,
    )
    print(f"deployed_index_id: {args.deployed_index_id}")
    print()
    print("Add these values to your local .env:")
    print(f"VS_INDEX_ID={index_name}")
    print(f"VS_ENDPOINT_ID={endpoint.resource_name}")
    print(f"VS_DEPLOYED_INDEX_ID={args.deployed_index_id}")
    print("RETRIEVER=vertex")


def _create_streaming_brute_force_index(
    *,
    aiplatform_v1: Any,
    json_format: Any,
    struct_pb2: Any,
    project_id: str,
    region: str,
    display_name: str,
    dimension: int,
) -> str:
    client_options = {"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    client = aiplatform_v1.IndexServiceClient(client_options=client_options)
    parent = f"projects/{project_id}/locations/{region}"
    metadata = struct_pb2.Value()
    json_format.ParseDict(
        {
            "config": {
                "dimensions": dimension,
                "approximateNeighborsCount": 5,
                "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
                "algorithmConfig": {"bruteForceConfig": {}},
            }
        },
        metadata,
    )
    index = aiplatform_v1.Index(
        display_name=display_name,
        description="Telecom specification clause chunk brute-force vector index",
        metadata=metadata,
        index_update_method=aiplatform_v1.Index.IndexUpdateMethod.STREAM_UPDATE,
    )
    operation = client.create_index(parent=parent, index=index)
    created = operation.result()
    return str(created.name)


def _load_vectors(path: Path) -> list[dict[str, Any]]:
    vectors = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not vectors:
        raise ValueError(f"vector file has no datapoints: {path}")
    return vectors


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("embedding_dimension"):
        raise ValueError(f"manifest is missing embedding_dimension: {path}")
    return manifest


def _batched(values: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _load_cloud_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        from google.cloud import aiplatform
        from google.cloud import aiplatform_v1
        from google.protobuf import json_format
        from google.protobuf import struct_pb2
    except ImportError as exc:
        raise RuntimeError(
            "Vector index creation requires optional cloud dependencies. Install them with: "
            "pip install -r requirements-cloud.txt"
        ) from exc
    return aiplatform, aiplatform_v1, json_format, struct_pb2


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


if __name__ == "__main__":
    main()
