from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from retrieval.local import LocalRetriever


class AskHandler(BaseHTTPRequestHandler):
    retriever: LocalRetriever
    top_k: int

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
        if not results:
            self._json_response(
                {
                    "answer": "Insufficient evidence in the indexed corpus.",
                    "citations": [],
                }
            )
            return

        citations = [_citation(result) for result in results]
        self._json_response(
            {
                "answer": "Retrieved evidence from the indexed corpus. Generation is not enabled in the local baseline.",
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
    args = parser.parse_args()

    handler = AskHandler
    handler.retriever = LocalRetriever.from_jsonl(Path(args.chunks))
    handler.top_k = args.top_k

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving http://{args.host}:{args.port}")
    server.serve_forever()


def _citation(result: Any) -> dict[str, Any]:
    chunk = result.chunk
    snippet = chunk["text"][:500].strip()
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


if __name__ == "__main__":
    main()
