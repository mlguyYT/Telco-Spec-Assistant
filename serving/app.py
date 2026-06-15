from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from retrieval.local import LocalRetriever, tokenize


class AskHandler(BaseHTTPRequestHandler):
    retriever: LocalRetriever
    top_k: int
    min_score: float

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response({"status": "ok"})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        question = str(payload.get("q") or payload.get("question") or "").strip()
        if not question:
            self._json_response({"error": "missing question"}, status=400)
            return

        results = self.retriever.search(question, top_k=self.top_k)
        answer = build_evidence_answer(results, question=question, min_score=self.min_score)
        if answer is None:
            self._json_response(
                {
                    "answer": "Insufficient evidence in the indexed corpus.",
                    "citations": [],
                }
            )
            return

        citations = [_citation(result, question=question) for result in _order_results_for_answer(results, question)]
        self._json_response(
            {
                "answer": answer,
                "citations": citations,
            }
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json_response(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local evidence-serving API.")
    parser.add_argument("--chunks", default=".data/chunks/rlc_v1.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=1.0)
    args = parser.parse_args()

    handler = AskHandler
    handler.retriever = LocalRetriever.from_jsonl(Path(args.chunks))
    handler.top_k = args.top_k
    handler.min_score = args.min_score

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving http://{args.host}:{args.port}")
    server.serve_forever()


def _citation(result: Any, question: str = "") -> dict[str, Any]:
    chunk = result.chunk
    snippet = _citation_snippet(result, question)
    return {
        "score": result.score,
        "spec_id": chunk["spec_id"],
        "release": chunk["release"],
        "version": chunk["version"],
        "section": chunk["section"],
        "section_title": chunk["section_title"],
        "source_url": chunk["source_url"],
        "chunk_id": chunk["chunk_id"],
        "snippet": snippet,
    }


def _citation_snippet(result: Any, question: str) -> str:
    evidence = _select_evidence_items([result], question=question, limit=1)
    if evidence:
        return evidence[0][0]
    return result.chunk["text"][:500].strip()


def _order_results_for_answer(results: list[Any], question: str) -> list[Any]:
    evidence_items = _select_evidence_items(results, question=question, limit=1)
    if not evidence_items:
        return results

    evidence_chunk_id = evidence_items[0][1].get("chunk_id")
    matching = [result for result in results if result.chunk.get("chunk_id") == evidence_chunk_id]
    rest = [result for result in results if result.chunk.get("chunk_id") != evidence_chunk_id]
    return matching + rest


def build_evidence_answer(results: list[Any], question: str = "", min_score: float = 1.0) -> str | None:
    if not results or results[0].score < min_score:
        return None

    top = results[0].chunk
    evidence_items = _select_evidence_items(results, question=question)
    if evidence_items:
        top = evidence_items[0][1]
        evidence = [item[0] for item in evidence_items]
    else:
        evidence = [_first_body_line(top["text"])]

    evidence_text = " ".join(evidence)
    return (
        f"According to {top['spec_id']} {top['version']}, clause {top['section']} "
        f"({top['section_title']}): {evidence_text}"
    )


def _select_evidence_items(results: list[Any], question: str, limit: int = 2) -> list[tuple[str, dict[str, Any]]]:
    query_terms = set(tokenize(question))
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for rank, result in enumerate(results[:3]):
        for line in _evidence_candidates(result.chunk["text"]):
            line_terms = set(tokenize(line))
            overlap = len(query_terms.intersection(line_terms))
            if overlap == 0 and rank > 0:
                continue
            score = overlap + (result.score / 100) - (rank * 0.25)
            candidates.append((score, line, result.chunk))

    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for _, line, chunk in sorted(candidates, key=lambda item: item[0], reverse=True):
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append((line, chunk))
        if len(selected) >= limit:
            break
    return selected


def _evidence_candidates(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Figure ") or line.startswith("Table "):
            lines.append(line)
            continue
        line = line.lstrip("-").strip()
        if len(line) < 24:
            continue
        if len(line) <= 260:
            lines.append(line)
            continue
        lines.extend(_split_sentences(line))
    return lines


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.;])\s+", text) if 24 <= len(part.strip()) <= 260]


def _first_body_line(text: str) -> str:
    for line in _evidence_candidates(text):
        return line
    return text[:260].strip()


if __name__ == "__main__":
    main()
