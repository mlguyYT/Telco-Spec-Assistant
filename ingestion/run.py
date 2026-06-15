from __future__ import annotations

import argparse
from pathlib import Path

from ingestion.chunker import chunks_from_document, write_chunks_jsonl
from ingestion.docx_parser import parse_docx
from ingestion.manifest import load_manifest
from ingestion.staging import stage_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse staged specifications into citation chunks.")
    parser.add_argument("--manifest", default="specs/manifest.example.yaml", help="Path to the spec manifest.")
    parser.add_argument("--data-dir", default=".data", help="Ignored local data directory.")
    parser.add_argument("--seed-dir", default=None, help="Optional local directory containing seed documents.")
    parser.add_argument("--output", default=".data/chunks/rlc_v1.jsonl", help="Output JSONL chunk path.")
    parser.add_argument("--no-download", action="store_true", help="Do not download missing documents.")
    args = parser.parse_args()

    documents = load_manifest(Path(args.manifest))
    staged = stage_documents(
        documents,
        data_dir=Path(args.data_dir),
        seed_dir=Path(args.seed_dir) if args.seed_dir else None,
        allow_download=not args.no_download,
    )

    all_chunks = []
    for staged_doc in staged:
        parsed = parse_docx(staged_doc.path)
        chunks = chunks_from_document(parsed, staged_doc.spec)
        all_chunks.extend(chunks)
        print(
            f"parsed {staged_doc.path}: {len(parsed.sections)} sections, "
            f"{len(chunks)} chunks ({staged_doc.source})"
        )

    output = Path(args.output)
    write_chunks_jsonl(all_chunks, output)
    print(f"wrote {len(all_chunks)} chunks to {output}")


if __name__ == "__main__":
    main()
