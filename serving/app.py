from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from retrieval.base import Retriever
from retrieval.factory import get_retriever
from retrieval.local import tokenize


class AskHandler(BaseHTTPRequestHandler):
    retriever: Retriever
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

        payload = self._read_json_payload()
        if payload is None:
            self._json_response({"error": "invalid json"}, status=400)
            return

        question = str(payload.get("q") or payload.get("question") or "").strip()
        if not question:
            self._json_response({"error": "missing question"}, status=400)
            return

        results = self.retriever.retrieve(question, k=self.top_k)
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

    def _read_json_payload(self) -> dict[str, Any] | None:
        length = int(self.headers.get("content-length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _json_response(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local evidence-serving API.")
    parser.add_argument("--chunks", default=os.environ.get("CHUNKS_PATH", ".data/chunks/rlc_v1.jsonl"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "5")))
    parser.add_argument("--min-score", type=float, default=float(os.environ.get("MIN_SCORE", "1.0")))
    args = parser.parse_args()

    server = create_server(
        chunks_path=Path(args.chunks),
        host=args.host,
        port=args.port,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print(f"serving http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def create_server(
    chunks_path: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    top_k: int = 5,
    min_score: float = 1.0,
) -> ThreadingHTTPServer:
    return create_server_from_retriever(
        retriever=get_retriever(chunks_path=chunks_path),
        host=host,
        port=port,
        top_k=top_k,
        min_score=min_score,
    )


def create_server_from_retriever(
    retriever: Retriever,
    host: str = "127.0.0.1",
    port: int = 8080,
    top_k: int = 5,
    min_score: float = 1.0,
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredAskHandler",
        (AskHandler,),
        {
            "retriever": retriever,
            "top_k": top_k,
            "min_score": min_score,
        },
    )
    return ThreadingHTTPServer((host, port), handler)


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


def _select_evidence_items(results: list[Any], question: str, limit: int = 3) -> list[tuple[str, dict[str, Any]]]:
    query_terms = set(tokenize(question))
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for rank, result in enumerate(results[:3]):
        for line in _evidence_candidates(result.chunk["text"]):
            line_terms = set(tokenize(line))
            overlap = len(query_terms.intersection(line_terms))
            if overlap == 0 and rank > 0:
                continue
            score = _score_evidence_line(
                query_terms=query_terms,
                line=line,
                line_terms=line_terms,
                retrieval_score=result.score,
                rank=rank,
            )
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


def _score_evidence_line(
    query_terms: set[str],
    line: str,
    line_terms: set[str],
    retrieval_score: float,
    rank: int,
) -> float:
    overlap = len(query_terms.intersection(line_terms))
    score = overlap + (retrieval_score / 100) - (rank * 0.25)
    normalized = line.lower()

    asks_field_meaning = bool({"field", "encode", "offset"}.intersection(query_terms))
    if asks_field_meaning and normalized.startswith("length:"):
        score += 3.0
    if asks_field_meaning and "numbering starts at zero" in normalized:
        score += 3.0

    asks_error_recovery = bool({"error", "recovery", "failure", "retransmission"}.intersection(query_terms))
    if asks_error_recovery and "arq" in normalized:
        score += 10.0
    if asks_error_recovery and "status" in line_terms and "pdu" in line_terms:
        score += 3.0
    if asks_error_recovery and "status pdu" in normalized:
        score += 6.0
    if asks_error_recovery and "negative" in line_terms and "acknowledgment" in line_terms:
        score += 2.0
    if asks_error_recovery and "consider" in normalized and "retransmission" in line_terms:
        score += 3.0
    if asks_error_recovery and "negative acknowledgement" in normalized and "retransmission" in line_terms:
        score += 4.0
    if asks_error_recovery and normalized.startswith("detection of "):
        score -= 3.0

    if " | " in line and overlap <= 1:
        score -= 2.0

    return score


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
        if line.lower().startswith("length:"):
            lines.append(line)
            continue
        if line.endswith("PDU from its peer AM RLC entity."):
            lines.append(line)
            continue
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
