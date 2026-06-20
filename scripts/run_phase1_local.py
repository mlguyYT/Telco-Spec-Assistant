#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run import evaluate  # noqa: E402
from ingestion.chunker import chunks_from_document, write_chunks_jsonl  # noqa: E402
from ingestion.docx_parser import parse_docx  # noqa: E402
from ingestion.manifest import load_manifest  # noqa: E402
from ingestion.staging import stage_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Phase 1 telecom-spec pipeline.")
    parser.add_argument("--manifest", default="specs/manifest.example.yaml")
    parser.add_argument("--dataset", default="eval/datasets/telco_retrieval_v1.jsonl")
    parser.add_argument("--data-dir", default=".data")
    parser.add_argument("--seed-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    chunks_path = data_dir / "chunks" / "telco_v1.jsonl"
    report_path = data_dir / "eval" / "telco_retrieval_report.json"

    documents = load_manifest(Path(args.manifest))
    staged = stage_documents(
        documents,
        data_dir=data_dir,
        seed_dir=Path(args.seed_dir) if args.seed_dir else None,
        allow_download=not args.no_download,
    )

    section_count = 0
    all_chunks = []
    for staged_doc in staged:
        parsed = parse_docx(staged_doc.path)
        chunks = chunks_from_document(parsed, staged_doc.spec)
        section_count += len(parsed.sections)
        all_chunks.extend(chunks)
        print(
            f"staged {staged_doc.path.name} from {staged_doc.source}; "
            f"{len(parsed.sections)} sections, {len(chunks)} chunks"
        )

    write_chunks_jsonl(all_chunks, chunks_path)
    report = evaluate(Path(args.dataset), chunks_path, top_k=args.top_k)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\nPhase 1 local summary")
    print(f"sections: {section_count}")
    print(f"chunks: {len(all_chunks)}")
    print(f"questions: {report['question_count']}")
    print(f"answerable questions: {report['answerable_question_count']}")
    print(f"out-of-scope questions: {report['unanswerable_question_count']}")
    print(f"answerable recall@{args.top_k}: {report['answerable_recall_at_k']:.3f}")
    if report["non_paraphrase_recall_at_k"] is not None:
        print(f"non-paraphrase recall@{args.top_k}: {report['non_paraphrase_recall_at_k']:.3f}")
    for subset, recall in report["subset_recall_at_k"].items():
        print(f"{subset} recall@{args.top_k}: {recall:.3f}")
    print(f"abstention accuracy: {report['abstention_accuracy']:.3f}")
    if report["answer_quality_accuracy"] is not None:
        print(f"answer quality accuracy: {report['answer_quality_accuracy']:.3f}")
    if report["answer_assertion_group_accuracy"] is not None:
        print(f"answer assertion group accuracy: {report['answer_assertion_group_accuracy']:.3f}")
    if report["answer_citation_accuracy"] is not None:
        print(f"answer citation accuracy: {report['answer_citation_accuracy']:.3f}")
    if report["answer_refusal_accuracy"] is not None:
        print(f"answer refusal accuracy: {report['answer_refusal_accuracy']:.3f}")
    print(f"latency p50 / p95 ms: {report['latency_ms_p50']:.2f} / {report['latency_ms_p95']:.2f}")
    print(f"chunks: {chunks_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
