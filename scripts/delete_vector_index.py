from __future__ import annotations

import argparse
import os
from typing import Any

DEFAULT_REGION = "us-central1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Undeploy and delete the Vertex AI Vector Search test resources.")
    parser.add_argument("--keep-endpoint", action="store_true")
    parser.add_argument("--keep-index", action="store_true")
    args = parser.parse_args()

    project_id = _required_env("GCP_PROJECT_ID")
    region = os.environ.get("REGION", DEFAULT_REGION)
    index_id = _required_env("VS_INDEX_ID")
    endpoint_id = _required_env("VS_ENDPOINT_ID")
    deployed_index_id = os.environ.get("VS_DEPLOYED_INDEX_ID")

    aiplatform = _load_cloud_dependencies()
    aiplatform.init(project=project_id, location=region)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
    index = aiplatform.MatchingEngineIndex(index_name=index_id)

    _undeploy(endpoint, deployed_index_id)
    if not args.keep_endpoint:
        endpoint.delete()
        print(f"deleted_endpoint: {endpoint_id}")
    if not args.keep_index:
        index.delete()
        print(f"deleted_index: {index_id}")


def _undeploy(endpoint: Any, deployed_index_id: str | None) -> None:
    if hasattr(endpoint, "undeploy_all"):
        endpoint.undeploy_all()
        print("undeployed_all_indexes")
        return
    if not deployed_index_id:
        raise RuntimeError("VS_DEPLOYED_INDEX_ID is required because this SDK does not expose undeploy_all()")
    endpoint.undeploy_index(deployed_index_id=deployed_index_id)
    print(f"undeployed_index: {deployed_index_id}")


def _load_cloud_dependencies() -> Any:
    try:
        from google.cloud import aiplatform
    except ImportError as exc:
        raise RuntimeError(
            "Vector index deletion requires optional cloud dependencies. Install them with: "
            "pip install -r requirements-cloud.txt"
        ) from exc
    return aiplatform


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


if __name__ == "__main__":
    main()
