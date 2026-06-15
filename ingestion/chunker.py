from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ingestion.docx_parser import ParsedDocument, Section
from ingestion.manifest import SpecDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    spec_id: str
    release: str
    version: str
    section: str
    section_title: str
    page: int | None
    source_url: str
    chunk_hash: str
    doc_title: str
    document_id: str
    chunk_index: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)


def chunks_from_document(
    parsed: ParsedDocument,
    spec: SpecDocument,
    max_chars: int = 2400,
    overlap_paragraphs: int = 1,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in parsed.sections:
        for text in _split_section(section, max_chars=max_chars, overlap_paragraphs=overlap_paragraphs):
            chunk_hash = _hash_text(text)
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{spec.id}:{section.section_id}:{chunk_hash[:12]}",
                    text=text,
                    spec_id=spec.spec_id,
                    release=spec.release,
                    version=spec.version,
                    section=section.section_id,
                    section_title=section.title,
                    page=None,
                    source_url=spec.source_url,
                    chunk_hash=f"sha256:{chunk_hash}",
                    doc_title=spec.title,
                    document_id=spec.id,
                    chunk_index=chunk_index,
                )
            )
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(chunk.to_json() + "\n")


def _split_section(section: Section, max_chars: int, overlap_paragraphs: int) -> list[str]:
    header = f"{section.section_id} {section.title}".strip()
    parts: list[str] = []
    current: list[str] = []

    for paragraph in section.paragraphs:
        candidate = "\n".join([header, *current, paragraph]).strip()
        if current and len(candidate) > max_chars:
            parts.append("\n".join([header, *current]).strip())
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
        current.append(paragraph)

    if current:
        parts.append("\n".join([header, *current]).strip())
    return parts


def _hash_text(text: str) -> str:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
