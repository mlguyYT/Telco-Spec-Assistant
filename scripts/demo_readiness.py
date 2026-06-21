from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_CHUNKS_PATH = ".data/chunks/telco_v1.jsonl"
RECOMMENDED_ENV = {
    "RETRIEVER": "hybrid",
    "GENERATOR": "gemini",
    "TOP_K": "5",
    "MIN_SCORE": "auto",
    "HYBRID_SOURCE_K": "100",
    "HYBRID_RRF_C": "40",
    "HYBRID_VERTEX_WEIGHT": "2.0",
}
REQUIRED_ENV = [
    "GCP_PROJECT_ID",
    "REGION",
    "GEMINI_MODEL",
    "EMBEDDING_MODEL",
    "VS_ENDPOINT_ID",
    "VS_DEPLOYED_INDEX_ID",
]
RECOMMENDED_ENV_PRESENT = ["VS_INDEX_ID"]


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local readiness for an expert-review serving session.")
    parser.add_argument("--chunks", default=os.environ.get("CHUNKS_PATH", DEFAULT_CHUNKS_PATH))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings as well as errors.")
    args = parser.parse_args()

    checks = readiness_checks(env=os.environ, chunks_path=Path(args.chunks))
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")

    has_error = any(check.status == "FAIL" for check in checks)
    has_warning = any(check.status == "WARN" for check in checks)
    if has_error or (args.strict and has_warning):
        raise SystemExit(1)


def readiness_checks(env: Mapping[str, str], chunks_path: Path) -> list[Check]:
    checks: list[Check] = []
    checks.extend(_chunk_checks(chunks_path))
    checks.extend(_recommended_env_checks(env))
    checks.extend(_required_env_checks(env))
    checks.extend(_dependency_checks())
    checks.append(
        Check(
            "PASS",
            "access model",
            "serve the browser UI behind an identity-aware access layer; do not expose /ask publicly",
        )
    )
    return checks


def _chunk_checks(path: Path) -> list[Check]:
    if not path.exists():
        return [Check("FAIL", "chunks", f"missing chunk file: {path}")]
    try:
        count = _jsonl_count(path)
    except OSError as exc:
        return [Check("FAIL", "chunks", f"could not read {path}: {exc}")]
    if count == 0:
        return [Check("FAIL", "chunks", f"chunk file is empty: {path}")]
    return [Check("PASS", "chunks", f"{count} chunks at {path}")]


def _recommended_env_checks(env: Mapping[str, str]) -> list[Check]:
    checks: list[Check] = []
    for name, expected in RECOMMENDED_ENV.items():
        actual = env.get(name)
        if actual == expected:
            checks.append(Check("PASS", name, actual))
        elif actual:
            checks.append(Check("WARN", name, f"{actual}; recommended {expected}"))
        else:
            checks.append(Check("WARN", name, f"unset; recommended {expected}"))
    return checks


def _required_env_checks(env: Mapping[str, str]) -> list[Check]:
    checks: list[Check] = []
    for name in REQUIRED_ENV:
        value = env.get(name)
        if value:
            checks.append(Check("PASS", name, "set"))
        else:
            checks.append(Check("FAIL", name, "required for Hybrid + Gemini runtime"))
    for name in RECOMMENDED_ENV_PRESENT:
        value = env.get(name)
        if value:
            checks.append(Check("PASS", name, "set"))
        else:
            checks.append(Check("WARN", name, "recommended for cleanup and audit scripts"))
    return checks


def _dependency_checks() -> list[Check]:
    checks = []
    packages = {
        "google.cloud.aiplatform": "Vertex AI Vector Search",
        "google.genai": "Gemini generation",
    }
    for package, purpose in packages.items():
        if importlib.util.find_spec(package):
            checks.append(Check("PASS", package, f"available for {purpose}"))
        else:
            checks.append(Check("FAIL", package, f"missing; install requirements-cloud.txt for {purpose}"))
    return checks


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()
