from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.run import evaluate
from retrieval.local import LocalRetriever, tokenize
from serving.app import build_evidence_answer


class RetrievalEvalTests(unittest.TestCase):
    def test_tokenize_expands_common_rlc_terms(self) -> None:
        tokens = tokenize("polling in acknowledged mode with missing PDUs")

        self.assertIn("poll", tokens)
        self.assertIn("am", tokens)
        self.assertIn("pdu", tokens)
        self.assertIn("nack", tokens)

    def test_local_retriever_ranks_expected_section(self) -> None:
        retriever = LocalRetriever(
            [
                _chunk("4.2.1.3.1", "AM RLC entity general logical channels."),
                _chunk(
                    "5.3.3.1",
                    "An AM RLC entity can poll its peer AM RLC entity in order to trigger STATUS reporting.",
                ),
            ]
        )

        results = retriever.search("What is the purpose of the polling mechanism in acknowledged mode RLC?")

        self.assertEqual(results[0].chunk["section"], "5.3.3.1")

    def test_evaluate_reports_recall_at_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            dataset_path = root / "dataset.jsonl"
            chunks_path.write_text(
                "\n".join(
                    [
                        json.dumps(_chunk("4.4", "segmentation and reassembly of RLC SDUs")),
                        json.dumps(_chunk("7.1", "state variables and sequence number handling")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Which function supports reassembly?",
                        "expected_sections": ["4.4"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate(dataset_path, chunks_path, top_k=1)

            self.assertEqual(report["question_count"], 1)
            self.assertEqual(report["recall_at_k"], 1.0)

    def test_evidence_answer_refuses_low_score(self) -> None:
        result = type("Result", (), {"score": 0.1, "chunk": _chunk("4.4", "text")})()

        self.assertIsNone(build_evidence_answer([result], min_score=1.0))

    def test_evidence_answer_cites_top_clause(self) -> None:
        result = type("Result", (), {"score": 2.0, "chunk": _chunk("4.4", "text")})()

        answer = build_evidence_answer([result], min_score=1.0)

        self.assertIsNotNone(answer)
        self.assertIn("clause 4.4", answer or "")


def _chunk(section: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": f"test:{section}",
        "text": text,
        "spec_id": "3GPP TS 38.322",
        "release": "Rel-19",
        "version": "v19.2.0",
        "section": section,
        "section_title": "Test",
        "page": None,
        "source_url": "https://example.invalid/spec.zip",
        "chunk_hash": "sha256:test",
        "doc_title": "Test",
        "document_id": "test",
        "chunk_index": 0,
    }


if __name__ == "__main__":
    unittest.main()
