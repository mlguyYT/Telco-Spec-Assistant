from __future__ import annotations

import re
from typing import Any

from generation.base import GeneratedAnswer
from retrieval.base import RetrievedChunk
from retrieval.local import tokenize

INSUFFICIENT_EVIDENCE_ANSWER = "Insufficient evidence in the indexed corpus."


class ExtractiveGenerator:
    name = "extractive"

    def generate(
        self,
        question: str,
        results: list[RetrievedChunk],
        *,
        min_score: float = 1.0,
    ) -> GeneratedAnswer:
        answer = build_evidence_answer(results, question=question, min_score=min_score)
        if answer is None:
            return GeneratedAnswer(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=[],
                supported=False,
                generator=self.name,
            )
        return GeneratedAnswer(
            answer=answer,
            citations=[citation(result, question=question) for result in order_results_for_answer(results, question)],
            supported=True,
            generator=self.name,
        )


def citation(result: Any, question: str = "") -> dict[str, Any]:
    chunk = result.chunk
    snippet = citation_snippet(result, question)
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


def citation_snippet(result: Any, question: str) -> str:
    evidence = select_evidence_items([result], question=question, limit=1)
    if evidence:
        return evidence[0][0]
    return result.chunk["text"][:500].strip()


def order_results_for_answer(results: list[Any], question: str) -> list[Any]:
    evidence_items = select_evidence_items(results, question=question, limit=1)
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
    evidence_items = select_evidence_items(results, question=question)
    if evidence_items:
        top = evidence_items[0][1]
        evidence = [item[0] for item in evidence_items]
    else:
        evidence = [first_body_line(top["text"])]

    evidence_text = " ".join(evidence)
    return (
        f"According to {top['spec_id']} {top['version']}, clause {top['section']} "
        f"({top['section_title']}): {evidence_text}"
    )


def select_evidence_items(results: list[Any], question: str, limit: int = 3) -> list[tuple[str, dict[str, Any]]]:
    query_terms = set(tokenize(question))
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for rank, result in enumerate(results[:3]):
        for line in evidence_candidates(result.chunk["text"]):
            line_terms = set(tokenize(line))
            overlap = len(query_terms.intersection(line_terms))
            if overlap == 0 and rank > 0:
                continue
            score = score_evidence_line(
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


def score_evidence_line(
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


def evidence_candidates(text: str) -> list[str]:
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
        lines.extend(split_sentences(line))
    return lines


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.;])\s+", text) if 24 <= len(part.strip()) <= 260]


def first_body_line(text: str) -> str:
    for line in evidence_candidates(text):
        return line
    return text[:260].strip()

