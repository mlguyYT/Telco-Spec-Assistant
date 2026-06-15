#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.manifest import load_manifest  # noqa: E402
from ingestion.staging import stage_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or stage public specification documents.")
    parser.add_argument("--manifest", default="specs/manifest.example.yaml", help="Path to the spec manifest.")
    parser.add_argument("--data-dir", default=".data", help="Ignored local data directory.")
    parser.add_argument("--seed-dir", default=None, help="Optional local directory containing seed documents.")
    parser.add_argument("--no-download", action="store_true", help="Do not download missing documents.")
    args = parser.parse_args()

    documents = load_manifest(Path(args.manifest))
    staged = stage_documents(
        documents,
        data_dir=Path(args.data_dir),
        seed_dir=Path(args.seed_dir) if args.seed_dir else None,
        allow_download=not args.no_download,
    )

    for item in staged:
        print(f"{item.spec.id}: {item.path} ({item.source})")


if __name__ == "__main__":
    main()
