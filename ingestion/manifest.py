from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SpecDocument:
    id: str
    spec_id: str
    title: str
    release: str
    version: str
    local_seed_filename: str
    source_url: str
    source_archive_hint: str | None = None


def load_manifest(path: Path) -> list[SpecDocument]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a mapping")
    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("manifest must contain a non-empty documents list")

    parsed: list[SpecDocument] = []
    for index, item in enumerate(documents):
        if not isinstance(item, dict):
            raise ValueError(f"document entry {index} must be a mapping")
        parsed.append(_parse_document(item, index))
    return parsed


def _parse_document(item: dict[str, Any], index: int) -> SpecDocument:
    required = [
        "id",
        "spec_id",
        "title",
        "release",
        "version",
        "local_seed_filename",
        "source_url",
    ]
    missing = [field for field in required if not item.get(field)]
    if missing:
        raise ValueError(f"document entry {index} missing required field(s): {', '.join(missing)}")

    return SpecDocument(
        id=str(item["id"]),
        spec_id=str(item["spec_id"]),
        title=str(item["title"]),
        release=str(item["release"]),
        version=str(item["version"]),
        local_seed_filename=str(item["local_seed_filename"]),
        source_url=str(item["source_url"]),
        source_archive_hint=str(item["source_archive_hint"]) if item.get("source_archive_hint") else None,
    )
