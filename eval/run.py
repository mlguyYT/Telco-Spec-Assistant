from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from retrieval.base import Retriever
from retrieval.factory import get_retriever
from serving.app import build_evidence_answer


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
    if report["non_paraphrase_recall_at_k"] is not None:
        print(f"non_paraphrase_recall@{args.top_k}: {report['non_paraphrase_recall_at_k']:.3f}")
    for subset, recall in report["subset_recall_at_k"].items():
        print(f"{subset}_recall@{args.top_k}: {recall:.3f}")
    if report["answer_quality_accuracy"] is not None:
        print(f"answer_quality_accuracy: {report['answer_quality_accuracy']:.3f}")
    if report["answer_assertion_group_accuracy"] is not None:
        print(f"answer_assertion_group_accuracy: {report['answer_assertion_group_accuracy']:.3f}")
    print(f"latency_ms_p50: {report['latency_ms_p50']:.2f}")
    print(f"latency_ms_p95: {report['latency_ms_p95']:.2f}")
    print(f"wrote {report_path}")


def evaluate(dataset_path: Path, chunks_path: Path, top_k: int = 5) -> dict[str, Any]:
    retriever = get_retriever(chunks_path=chunks_path)
    return evaluate_with_retriever(dataset_path=dataset_path, retriever=retriever, top_k=top_k)


def evaluate_with_retriever(dataset_path: Path, retriever: Retriever, top_k: int = 5) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"dataset has no questions: {dataset_path}")

    results = []
    latencies: list[float] = []
    hits = 0
    answerable_count = 0
    answerable_hits = 0
    unanswerable_count = 0
    abstention_hits = 0
    answer_quality_count = 0
    answer_quality_hits = 0
    assertion_group_count = 0
    assertion_group_hits = 0
    non_paraphrase_count = 0
    non_paraphrase_hits = 0
    subset_counts: dict[str, int] = {}
    subset_hits: dict[str, int] = {}
    for row in rows:
        expected_sections = _expected_sections(row)
        expected_answerable = _expected_answerable(row)
        start = time.perf_counter()
        retrieved = retriever.retrieve(row["question"], k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        retrieved_sections = [item.chunk["section"] for item in retrieved]
        if expected_answerable:
            hit = bool(set(expected_sections).intersection(retrieved_sections))
            answerable_count += 1
            answerable_hits += int(hit)
        else:
            hit = not retrieved_sections
            unanswerable_count += 1
            abstention_hits += int(hit)
        hits += int(hit)
        phrasing = row.get("phrasing")
        if phrasing:
            key = str(phrasing)
            subset_counts[key] = subset_counts.get(key, 0) + 1
            subset_hits[key] = subset_hits.get(key, 0) + int(hit)
        else:
            non_paraphrase_count += 1
            non_paraphrase_hits += int(hit)
        answer = build_evidence_answer(retrieved, question=row["question"])
        assertion_result = _check_required_terms(answer, row.get("required_answer_terms", []))
        if assertion_result["required_group_count"]:
            answer_quality_count += 1
            answer_quality_hits += int(assertion_result["all_required_groups_hit"])
            assertion_group_count += assertion_result["required_group_count"]
            assertion_group_hits += assertion_result["required_group_hits"]
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected_answerable": expected_answerable,
                "expected_sections": expected_sections,
                "retrieved_sections": retrieved_sections,
                "hit": hit,
                "phrasing": row.get("phrasing"),
                "answer": answer,
                "answer_assertions": assertion_result,
                "top_score": retrieved[0].score if retrieved else 0.0,
            }
        )

    return {
        "question_count": len(rows),
        "answerable_question_count": answerable_count,
        "unanswerable_question_count": unanswerable_count,
        "top_k": top_k,
        "recall_at_k": hits / len(rows),
        "answerable_recall_at_k": answerable_hits / answerable_count if answerable_count else None,
        "abstention_accuracy": abstention_hits / unanswerable_count if unanswerable_count else None,
        "answer_quality_question_count": answer_quality_count,
        "answer_quality_accuracy": answer_quality_hits / answer_quality_count if answer_quality_count else None,
        "answer_assertion_group_accuracy": assertion_group_hits / assertion_group_count if assertion_group_count else None,
        "non_paraphrase_recall_at_k": (
            non_paraphrase_hits / non_paraphrase_count if non_paraphrase_count else None
        ),
        "subset_recall_at_k": {
            key: subset_hits[key] / count for key, count in sorted(subset_counts.items())
        },
        "subset_question_counts": dict(sorted(subset_counts.items())),
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


def _expected_answerable(row: dict[str, Any]) -> bool:
    if "expected_answerable" in row:
        return bool(row["expected_answerable"])
    return bool(_expected_sections(row))


def _check_required_terms(answer: str | None, required_groups: Any) -> dict[str, Any]:
    if not required_groups:
        return {
            "required_group_count": 0,
            "required_group_hits": 0,
            "all_required_groups_hit": False,
            "groups": [],
        }

    normalized_answer = _normalize_answer(answer or "")
    groups = []
    hits = 0
    for group in required_groups:
        terms = [str(term).lower() for term in group]
        missing = [term for term in terms if term not in normalized_answer]
        hit = not missing
        hits += int(hit)
        groups.append({"terms": terms, "hit": hit, "missing": missing})

    return {
        "required_group_count": len(groups),
        "required_group_hits": hits,
        "all_required_groups_hit": hits == len(groups),
        "groups": groups,
    }


def _normalize_answer(answer: str) -> str:
    return " ".join(answer.lower().replace("-", " ").split())


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


if __name__ == "__main__":
    main()
