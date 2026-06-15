from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from retrieval.local import LocalRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval over chunk JSONL.")
    parser.add_argument("--dataset", default="eval/datasets/rlc_retrieval_v1.jsonl")
    parser.add_argument("--chunks", default=".data/chunks/rlc_v1.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", default=".data/eval/rlc_retrieval_report.json")
    args = parser.parse_args()

    report = evaluate(
        dataset_path=Path(args.dataset),
        chunks_path=Path(args.chunks),
        top_k=args.top_k,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"questions: {report['question_count']}")
    print(f"recall@{args.top_k}: {report['recall_at_k']:.3f}")
    print(f"latency_ms_p50: {report['latency_ms_p50']:.2f}")
    print(f"latency_ms_p95: {report['latency_ms_p95']:.2f}")
    print(f"wrote {report_path}")


def evaluate(dataset_path: Path, chunks_path: Path, top_k: int = 5) -> dict[str, Any]:
    retriever = LocalRetriever.from_jsonl(chunks_path)
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"dataset has no questions: {dataset_path}")

    results = []
    latencies: list[float] = []
    hits = 0
    for row in rows:
        expected_sections = _expected_sections(row)
        start = time.perf_counter()
        retrieved = retriever.search(row["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        retrieved_sections = [item.chunk["section"] for item in retrieved]
        hit = bool(set(expected_sections).intersection(retrieved_sections))
        hits += int(hit)
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected_sections": expected_sections,
                "retrieved_sections": retrieved_sections,
                "hit": hit,
                "top_score": retrieved[0].score if retrieved else 0.0,
            }
        )

    return {
        "question_count": len(rows),
        "top_k": top_k,
        "recall_at_k": hits / len(rows),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": _percentile(latencies, 95),
        "citation_support": "approximated_by_expected_section_hit",
        "cost_per_request": "local baseline only; no model or cloud cost",
        "results": results,
    }


def _expected_sections(row: dict[str, Any]) -> list[str]:
    if "expected_sections" in row:
        return [str(section) for section in row["expected_sections"]]
    if "expected_section" in row:
        return [str(row["expected_section"])]
    raise ValueError(f"question {row.get('id', '<unknown>')} is missing expected sections")


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


if __name__ == "__main__":
    main()
