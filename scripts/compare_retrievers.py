from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run import evaluate_with_retriever
from retrieval.factory import get_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval recall across configured retrievers.")
    parser.add_argument("--dataset", default="eval/datasets/rlc_retrieval_v1.jsonl")
    parser.add_argument("--chunks", default=".data/chunks/rlc_v1.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrievers", default="bm25,vertex,hybrid")
    args = parser.parse_args()

    for kind in [value.strip() for value in args.retrievers.split(",") if value.strip()]:
        os.environ["RETRIEVER"] = kind
        try:
            retriever = get_retriever(chunks_path=Path(args.chunks), kind=kind)
            report = evaluate_with_retriever(Path(args.dataset), retriever, top_k=args.top_k)
        except RuntimeError as exc:
            print(f"{kind}: skipped ({exc})")
            continue
        print(json.dumps(_summary(kind, report), sort_keys=True))


def _summary(kind: str, report: dict[str, object]) -> dict[str, object]:
    return {
        "retriever": kind,
        "questions": report["question_count"],
        "answerable_recall_at_k": report["answerable_recall_at_k"],
        "non_paraphrase_recall_at_k": report["non_paraphrase_recall_at_k"],
        "subset_recall_at_k": report["subset_recall_at_k"],
        "abstention_accuracy": report["abstention_accuracy"],
        "latency_ms_p50": report["latency_ms_p50"],
        "latency_ms_p95": report["latency_ms_p95"],
    }


if __name__ == "__main__":
    main()
