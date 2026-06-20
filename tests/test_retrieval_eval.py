from __future__ import annotations

import json
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request
from pathlib import Path

from eval.run import evaluate
from retrieval.embedding import _effective_batch_size
from retrieval.factory import get_retriever
from retrieval.hybrid import HybridRetriever
from retrieval.local import LocalRetriever, is_out_of_scope_query, tokenize
from retrieval.vertex import VertexRetriever, _load_chunk_map, _neighbor_id, _required_env
from serving.app import _citation, _order_results_for_answer, build_evidence_answer, create_server_from_retriever


class RetrievalEvalTests(unittest.TestCase):
    def test_tokenize_expands_common_rlc_terms(self) -> None:
        tokens = tokenize("polling in acknowledged mode with missing PDUs")

        self.assertIn("poll", tokens)
        self.assertIn("am", tokens)
        self.assertIn("pdu", tokens)
        self.assertIn("nack", tokens)

    def test_scope_guard_rejects_external_layer_question(self) -> None:
        self.assertTrue(is_out_of_scope_query("Which PDCP entity performs ciphering?"))
        self.assertFalse(is_out_of_scope_query("Which RLC service is provided to upper layers?"))

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

    def test_factory_returns_default_bm25_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(json.dumps(_chunk("4.4", "4.4 Functions\nsegmentation")) + "\n", encoding="utf-8")

            retriever = get_retriever(chunks_path=chunks_path)

            self.assertIsInstance(retriever, LocalRetriever)

    def test_hybrid_retriever_uses_rank_fusion_not_raw_scores(self) -> None:
        first = _StaticRetriever(
            [
                _retrieved("a", "4.1", "first raw score is tiny", 0.01),
                _retrieved("b", "4.2", "second raw score is huge", 1000.0),
            ]
        )
        second = _StaticRetriever(
            [
                _retrieved("b", "4.2", "second raw score is huge", 0.02),
                _retrieved("c", "4.3", "third raw score is larger", 500.0),
            ]
        )
        retriever = HybridRetriever([first, second], source_k=2, rrf_c=60)

        results = retriever.retrieve("rank fusion query", k=3)

        self.assertEqual([item.metadata["chunk_id"] for item in results], ["b", "a", "c"])
        self.assertGreater(results[0].score, results[1].score)
        self.assertLess(results[0].score, 1.0)

    def test_hybrid_retriever_applies_out_of_scope_guard_before_children(self) -> None:
        child = _FailingRetriever()
        retriever = HybridRetriever([child])

        results = retriever.retrieve("Which PDCP entity performs ciphering?", k=5)

        self.assertEqual(results, [])

    def test_factory_rejects_unknown_retriever(self) -> None:
        with self.assertRaises(ValueError):
            get_retriever(kind="unknown")

    def test_vertex_helpers_load_chunks_without_cloud_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunks_path.write_text(json.dumps(_chunk("4.4", "4.4 Functions\nsegmentation")) + "\n", encoding="utf-8")

            chunks = _load_chunk_map(chunks_path)

            self.assertEqual(chunks["test:4.4"]["section"], "4.4")

    def test_vertex_helper_detects_duplicate_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            chunk = _chunk("4.4", "4.4 Functions\nsegmentation")
            chunks_path.write_text(json.dumps(chunk) + "\n" + json.dumps(chunk) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                _load_chunk_map(chunks_path)

    def test_vertex_helper_reads_neighbor_id_shapes(self) -> None:
        direct = type("Neighbor", (), {"id": "chunk-a"})()
        nested = type(
            "Neighbor",
            (),
            {"id": "", "datapoint": type("Datapoint", (), {"datapoint_id": "chunk-b"})()},
        )()

        self.assertEqual(_neighbor_id(direct), "chunk-a")
        self.assertEqual(_neighbor_id(nested), "chunk-b")

    def test_vertex_required_env_reports_missing_value(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                _required_env("GCP_PROJECT_ID")

    def test_vertex_retriever_applies_out_of_scope_guard_before_cloud_calls(self) -> None:
        retriever = object.__new__(VertexRetriever)

        results = retriever.retrieve("Which PDCP entity performs ciphering?", k=5)

        self.assertEqual(results, [])

    def test_embedding_batch_size_respects_model_limits(self) -> None:
        self.assertEqual(_effective_batch_size("text-embedding-005", 50), 5)
        self.assertEqual(_effective_batch_size("gemini-embedding-001", 50), 1)
        with self.assertRaises(ValueError):
            _effective_batch_size("text-embedding-005", 0)

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
            self.assertEqual(report["answerable_recall_at_k"], 1.0)

    def test_evaluate_reports_phrasing_subset_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            dataset_path = root / "dataset.jsonl"
            chunks_path.write_text(
                json.dumps(_chunk("5.3.2", "5.3.2 Retransmission\nRLC retransmission procedure.")) + "\n",
                encoding="utf-8",
            )
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "q1",
                                "question": "Which procedure handles retransmission?",
                                "expected_sections": ["5.3.2"],
                                "phrasing": "paraphrase",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "q2",
                                "question": "Which clause describes ciphering?",
                                "expected_sections": ["5.3.2"],
                                "phrasing": "paraphrase",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate(dataset_path, chunks_path, top_k=1)

            self.assertEqual(report["subset_question_counts"]["paraphrase"], 2)
            self.assertEqual(report["subset_recall_at_k"]["paraphrase"], 0.5)
            self.assertIsNone(report["non_paraphrase_recall_at_k"])

    def test_evaluate_counts_out_of_scope_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            dataset_path = root / "dataset.jsonl"
            chunks_path.write_text(json.dumps(_chunk("4.4", "RLC segmentation and reassembly")) + "\n", encoding="utf-8")
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Which PDCP entity performs ciphering?",
                        "expected_sections": [],
                        "expected_answerable": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate(dataset_path, chunks_path, top_k=1)

            self.assertEqual(report["unanswerable_question_count"], 1)
            self.assertEqual(report["abstention_accuracy"], 1.0)

    def test_evaluate_reports_answer_assertion_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            dataset_path = root / "dataset.jsonl"
            chunks_path.write_text(
                json.dumps(
                    _chunk(
                        "6.2.3.5",
                        "6.2.3.5 Segment Offset (SO) field\n"
                        "Length: 16 bits\n"
                        "The SO field indicates the position of the RLC SDU segment in bytes within the original RLC SDU. "
                        "The first byte of the original RLC SDU is referred by the SO field value "
                        '"0000000000000000", i.e., numbering starts at zero.',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "What does the Segment Offset field encode?",
                        "expected_sections": ["6.2.3.5"],
                        "required_answer_terms": [["length", "16", "bit"], ["position"], ["zero"]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate(dataset_path, chunks_path, top_k=1)

            self.assertEqual(report["answer_quality_question_count"], 1)
            self.assertEqual(report["answer_quality_accuracy"], 1.0)
            self.assertEqual(report["answer_assertion_group_accuracy"], 1.0)

    def test_evidence_answer_refuses_low_score(self) -> None:
        result = type("Result", (), {"score": 0.1, "chunk": _chunk("4.4", "text")})()

        self.assertIsNone(build_evidence_answer([result], min_score=1.0))

    def test_evidence_answer_cites_top_clause(self) -> None:
        result = type(
            "Result",
            (),
            {
                "score": 2.0,
                "chunk": _chunk("4.4", "4.4 Functions\nsegmentation and reassembly of RLC SDUs"),
            },
        )()

        answer = build_evidence_answer([result], question="Which function supports reassembly?", min_score=1.0)

        self.assertIsNotNone(answer)
        self.assertIn("clause 4.4", answer or "")
        self.assertIn("segmentation and reassembly", answer or "")

    def test_evidence_answer_attributes_selected_evidence_clause(self) -> None:
        weak = type("Result", (), {"score": 3.0, "chunk": _chunk("4.1", "4.1 Intro\nRLC architecture overview")})()
        strong_evidence = type(
            "Result",
            (),
            {
                "score": 2.0,
                "chunk": _chunk(
                    "4.2.1",
                    "4.2.1 RLC entities\nAn RLC entity can be configured to perform data transfer in one of the following three modes: Transparent Mode (TM), Unacknowledged Mode (UM) or Acknowledged Mode (AM).",
                ),
            },
        )()

        answer = build_evidence_answer([weak, strong_evidence], question="What are the three RLC modes?")

        self.assertIsNotNone(answer)
        self.assertIn("clause 4.2.1", answer or "")
        self.assertIn("Transparent Mode", answer or "")

    def test_evidence_citation_order_puts_selected_chunk_first(self) -> None:
        weak = type("Result", (), {"score": 3.0, "chunk": _chunk("4.1", "4.1 Intro\nRLC architecture overview")})()
        strong_evidence = type(
            "Result",
            (),
            {
                "score": 2.0,
                "chunk": _chunk(
                    "4.2.1",
                    "4.2.1 RLC entities\nAn RLC entity can be configured to perform data transfer in one of the following three modes: Transparent Mode (TM), Unacknowledged Mode (UM) or Acknowledged Mode (AM).",
                ),
            },
        )()

        ordered = _order_results_for_answer([weak, strong_evidence], "What are the three RLC modes?")

        self.assertEqual(ordered[0].chunk["section"], "4.2.1")

    def test_evidence_answer_selects_arq_status_and_retransmission_evidence(self) -> None:
        retransmission = type(
            "Result",
            (),
            {
                "score": 20.0,
                "chunk": _chunk(
                    "5.3.2",
                    "5.3.2 Retransmission\n"
                    "The transmitting side can receive a negative acknowledgement for an RLC SDU.\n"
                    "When receiving a negative acknowledgement for an RLC SDU by a STATUS PDU from its peer AM RLC entity, the transmitting side shall:\n"
                    "-consider the RLC SDU for which a negative acknowledgement was received for retransmission.",
                ),
            },
        )()
        functions = type(
            "Result",
            (),
            {
                "score": 16.0,
                "chunk": _chunk(
                    "4.4",
                    "4.4 Functions\n"
                    "-error correction through ARQ (only for AM data transfer);\n"
                    "-Protocol error detection (only for AM data transfer).",
                ),
            },
        )()

        answer = build_evidence_answer(
            [retransmission, functions],
            question="How does AM RLC do error recovery by retransmission after reception failure?",
        )

        self.assertIsNotNone(answer)
        self.assertIn("ARQ", answer or "")
        self.assertIn("STATUS PDU", answer or "")
        self.assertIn("retransmission", answer or "")
        self.assertNotIn("Protocol error detection", answer or "")

    def test_citation_snippet_uses_selected_evidence_line(self) -> None:
        result = type(
            "Result",
            (),
            {
                "score": 2.0,
                "chunk": _chunk(
                    "4.2.1",
                    "4.2.1 RLC entities\nRRC is generally in control of RLC configuration.\nAn RLC entity can be configured to perform data transfer in one of the following three modes: Transparent Mode (TM), Unacknowledged Mode (UM) or Acknowledged Mode (AM).",
                ),
            },
        )()

        citation = _citation(result, question="What are the three RLC modes?")

        self.assertIn("three modes", citation["snippet"])

    def test_serving_http_health_and_ask(self) -> None:
        retriever = LocalRetriever(
            [
                _chunk(
                    "4.2.1",
                    "4.2.1 RLC entities\n"
                    "An RLC entity can be configured to perform data transfer in one of the following three modes: "
                    "Transparent Mode (TM), Unacknowledged Mode (UM) or Acknowledged Mode (AM).",
                )
            ]
        )
        server = create_server_from_retriever(retriever, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            health = _request_json(f"{base_url}/health")
            answer = _request_json(
                f"{base_url}/ask",
                payload={"question": "What are the three RLC modes?"},
            )

            self.assertEqual(health["status"], "ok")
            self.assertIn("Transparent Mode", answer["answer"])
            self.assertEqual(answer["citations"][0]["section"], "4.2.1")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_serving_http_rejects_invalid_json(self) -> None:
        retriever = LocalRetriever([_chunk("4.4", "4.4 Functions\nsegmentation and reassembly")])
        server = create_server_from_retriever(retriever, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/ask",
                data=b"{not-json",
                headers={"content-type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=5)

            self.assertEqual(raised.exception.code, 400)
            payload = json.loads(raised.exception.read().decode("utf-8"))
            self.assertEqual(payload["error"], "invalid json")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


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


def _retrieved(chunk_id: str, section: str, text: str, score: float) -> object:
    from retrieval.base import RetrievedChunk

    return RetrievedChunk(
        text=text,
        score=score,
        metadata={
            "chunk_id": chunk_id,
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
        },
    )


class _StaticRetriever:
    def __init__(self, results: list[object]) -> None:
        self.results = results

    def retrieve(self, query: str, k: int = 5) -> list[object]:
        return self.results[:k]


class _FailingRetriever:
    def retrieve(self, query: str, k: int = 5) -> list[object]:
        raise AssertionError("child retriever should not be called")


def _request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    if payload is None:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
