from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from generation.base import AnswerGenerator
from generation.extractive import (
    build_evidence_answer,
    citation as _citation,
    order_results_for_answer as _order_results_for_answer,
)
from generation.factory import get_generator
from retrieval.base import Retriever
from retrieval.factory import get_retriever


class AskHandler(BaseHTTPRequestHandler):
    retriever: Retriever
    generator: AnswerGenerator
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
        generated = self.generator.generate(question, results, min_score=self.min_score)
        self._json_response(
            {
                "answer": generated.answer,
                "citations": generated.citations,
                "supported": generated.supported,
                "generator": generated.generator,
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
    parser.add_argument("--chunks", default=os.environ.get("CHUNKS_PATH", ".data/chunks/telco_v1.jsonl"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "5")))
    parser.add_argument("--min-score", default=os.environ.get("MIN_SCORE", "auto"))
    args = parser.parse_args()
    retriever_kind = os.environ.get("RETRIEVER", "bm25")

    server = create_server(
        chunks_path=Path(args.chunks),
        host=args.host,
        port=args.port,
        top_k=args.top_k,
        min_score=_resolve_min_score(args.min_score, retriever_kind),
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
    min_score: float | None = None,
) -> ThreadingHTTPServer:
    if min_score is None:
        min_score = _default_min_score(os.environ.get("RETRIEVER", "bm25"))
    return create_server_from_retriever(
        retriever=get_retriever(chunks_path=chunks_path),
        generator=get_generator(),
        host=host,
        port=port,
        top_k=top_k,
        min_score=min_score,
    )


def create_server_from_retriever(
    retriever: Retriever,
    generator: AnswerGenerator | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    top_k: int = 5,
    min_score: float = 1.0,
) -> ThreadingHTTPServer:
    if generator is None:
        generator = get_generator()
    handler = type(
        "ConfiguredAskHandler",
        (AskHandler,),
        {
            "retriever": retriever,
            "generator": generator,
            "top_k": top_k,
            "min_score": min_score,
        },
    )
    return ThreadingHTTPServer((host, port), handler)


def _resolve_min_score(value: str | float | None, retriever_kind: str) -> float:
    if value is None:
        return _default_min_score(retriever_kind)
    if isinstance(value, (float, int)):
        return float(value)
    normalized = value.strip().lower()
    if normalized == "auto":
        return _default_min_score(retriever_kind)
    return float(value)


def _default_min_score(retriever_kind: str) -> float:
    if retriever_kind.lower() == "bm25":
        return 1.0
    return 0.0


if __name__ == "__main__":
    main()
