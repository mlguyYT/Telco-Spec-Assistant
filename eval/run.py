from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generation.base import AnswerGenerator
from generation.factory import get_generator
from retrieval.base import Retriever
from retrieval.factory import get_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval over chunk JSONL.")
    parser.add_argument("--dataset", default="eval/datasets/telco_retrieval_v1.jsonl")
    parser.add_argument("--chunks", default=".data/chunks/telco_v1.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--report", default=".data/eval/telco_retrieval_report.json")
    parser.add_argument("--generator", default=None)
    args = parser.parse_args()

    report = evaluate(
        dataset_path=Path(args.dataset),
        chunks_path=Path(args.chunks),
        top_k=args.top_k,
        generator=get_generator(args.generator),
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
    for spec_id, recall in report["per_spec_recall_at_k"].items():
        print(f"{spec_id}_recall@{args.top_k}: {recall:.3f}")
    if report["answer_quality_accuracy"] is not None:
        print(f"answer_quality_accuracy: {report['answer_quality_accuracy']:.3f}")
    if report["answer_assertion_group_accuracy"] is not None:
        print(f"answer_assertion_group_accuracy: {report['answer_assertion_group_accuracy']:.3f}")
    if report["answer_citation_accuracy"] is not None:
        print(f"answer_citation_accuracy: {report['answer_citation_accuracy']:.3f}")
    if report["answer_refusal_accuracy"] is not None:
        print(f"answer_refusal_accuracy: {report['answer_refusal_accuracy']:.3f}")
    print(f"latency_ms_p50: {report['latency_ms_p50']:.2f}")
    print(f"latency_ms_p95: {report['latency_ms_p95']:.2f}")
    print(f"wrote {report_path}")


def evaluate(
    dataset_path: Path,
    chunks_path: Path,
    top_k: int = 5,
    generator: AnswerGenerator | None = None,
) -> dict[str, Any]:
    retriever = get_retriever(chunks_path=chunks_path)
    return evaluate_with_retriever(dataset_path=dataset_path, retriever=retriever, top_k=top_k, generator=generator)


def evaluate_with_retriever(
    dataset_path: Path,
    retriever: Retriever,
    top_k: int = 5,
    generator: AnswerGenerator | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"dataset has no questions: {dataset_path}")
    if generator is None:
        generator = get_generator()

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
    answer_citation_count = 0
    answer_citation_hits = 0
    answer_refusal_count = 0
    answer_refusal_hits = 0
    non_paraphrase_count = 0
    non_paraphrase_hits = 0
    subset_counts: dict[str, int] = {}
    subset_hits: dict[str, int] = {}
    per_spec_counts: dict[str, int] = {}
    per_spec_hits: dict[str, int] = {}
    for row in rows:
        expected_sections = _expected_sections(row)
        expected_answerable = _expected_answerable(row)
        start = time.perf_counter()
        retrieved = retriever.retrieve(row["question"], k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        retrieved_sections = [item.chunk["section"] for item in retrieved]
        retrieved_refs = [_retrieved_ref(item.chunk) for item in retrieved]
        if expected_answerable:
            if row.get("expected_spec_id"):
                hit = bool(set(_expected_refs(row)).intersection(retrieved_refs))
            else:
                hit = bool(set(expected_sections).intersection(retrieved_sections))
            answerable_count += 1
            answerable_hits += int(hit)
            expected_spec_id = row.get("expected_spec_id")
            if expected_spec_id:
                spec_key = str(expected_spec_id)
                per_spec_counts[spec_key] = per_spec_counts.get(spec_key, 0) + 1
                per_spec_hits[spec_key] = per_spec_hits.get(spec_key, 0) + int(hit)
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
        generated = generator.generate(row["question"], retrieved, min_score=0.0)
        generated_refs = [f"{citation.get('spec_id')}#{citation.get('section')}" for citation in generated.citations]
        generated_sections = [str(citation.get("section")) for citation in generated.citations]
        if expected_answerable and generated.supported and generated_refs:
            answer_citation_count += 1
            if row.get("expected_spec_id"):
                citation_hit = bool(set(_expected_refs(row)).intersection(generated_refs))
            else:
                citation_hit = bool(set(expected_sections).intersection(generated_sections))
            answer_citation_hits += int(citation_hit)
        if not expected_answerable:
            answer_refusal_count += 1
            answer_refusal_hits += int(not generated.supported)

        assertion_result = _check_required_terms(generated.answer, row.get("required_answer_terms", []))
        if assertion_result["required_group_count"]:
            answer_quality_count += 1
            answer_quality_hits += int(assertion_result["all_required_groups_hit"])
            assertion_group_count += assertion_result["required_group_count"]
            assertion_group_hits += assertion_result["required_group_hits"]
        results.append(
            {
                "id": row["id"],
                "question": row["question"],
                "expected_spec_id": row.get("expected_spec_id"),
                "expected_answerable": expected_answerable,
                "expected_sections": expected_sections,
                "retrieved_sections": retrieved_sections,
                "retrieved_refs": retrieved_refs,
                "hit": hit,
                "phrasing": row.get("phrasing"),
                "answer": generated.answer,
                "answer_supported": generated.supported,
                "generator": generated.generator,
                "generated_citations": generated_refs,
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
        "answer_citation_question_count": answer_citation_count,
        "answer_citation_accuracy": answer_citation_hits / answer_citation_count if answer_citation_count else None,
        "answer_refusal_question_count": answer_refusal_count,
        "answer_refusal_accuracy": answer_refusal_hits / answer_refusal_count if answer_refusal_count else None,
        "non_paraphrase_recall_at_k": (
            non_paraphrase_hits / non_paraphrase_count if non_paraphrase_count else None
        ),
        "subset_recall_at_k": {
            key: subset_hits[key] / count for key, count in sorted(subset_counts.items())
        },
        "subset_question_counts": dict(sorted(subset_counts.items())),
        "per_spec_recall_at_k": {
            key: per_spec_hits[key] / count for key, count in sorted(per_spec_counts.items())
        },
        "per_spec_question_counts": dict(sorted(per_spec_counts.items())),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": _percentile(latencies, 95),
        "citation_support": "generated_citations_checked_against_expected_sections",
        "generator": generator.name,
        "cost_per_request": "not estimated by local eval",
        "results": results,
    }


def _expected_sections(row: dict[str, Any]) -> list[str]:
    if "expected_sections" in row:
        return [str(section) for section in row["expected_sections"]]
    if "expected_section" in row:
        return [str(row["expected_section"])]
    raise ValueError(f"question {row.get('id', '<unknown>')} is missing expected sections")


def _expected_refs(row: dict[str, Any]) -> list[str]:
    sections = _expected_sections(row)
    expected_spec_id = row.get("expected_spec_id")
    if not expected_spec_id:
        return sections
    return [f"{expected_spec_id}#{section}" for section in sections]


def _retrieved_ref(chunk: dict[str, Any]) -> str:
    spec_id = str(chunk.get("spec_id", ""))
    section = str(chunk.get("section", ""))
    return f"{spec_id}#{section}"


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
