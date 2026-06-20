from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from retrieval.base import RetrievedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
    "with",
}
OUT_OF_SCOPE_TERMS = {
    "ciphering",
    "compression",
    "integrity",
    "nas",
    "pdcp",
}
HIGH_CONFIDENCE_OUT_OF_SCOPE_TERMS = {
    "5gmm",
    "amf",
    "ngap",
    "nas",
    "smf",
}
EXPLICIT_CORPUS_TERMS = {
    "mac",
    "rlc",
    "rrc",
}
IN_SCOPE_TERMS = {
    "access",
    "am",
    "amd",
    "harq",
    "logical",
    "mac",
    "measurement",
    "mobility",
    "random",
    "rlc",
    "rrc",
    "scheduling",
    "sdu",
    "security",
    "service",
    "pdu",
    "sn",
    "status",
    "tm",
    "tmd",
    "um",
    "umd",
}


RetrievalResult = RetrievedChunk


class LocalRetriever:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.doc_lengths = [sum(counts.values()) for counts in self.term_counts]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.document_frequency = self._document_frequency()

    @classmethod
    def from_jsonl(cls, path: Path) -> "LocalRetriever":
        chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not chunks:
            raise ValueError(f"chunk file has no chunks: {path}")
        return cls(chunks)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if is_out_of_scope_query(query):
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[RetrievedChunk] = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)
            score = _adjust_score_for_query_context(query_terms, chunk, score)
            if score > 0:
                scored.append(_to_retrieved_chunk(chunk, score))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:k]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.retrieve(query, k=top_k)

    def _document_frequency(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for term_counts in self.term_counts:
            counts.update(term_counts.keys())
        return counts

    def _score(self, query_terms: list[str], chunk_index: int) -> float:
        counts = self.term_counts[chunk_index]
        doc_length = self.doc_lengths[chunk_index]
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = self.document_frequency.get(term, 0)
            idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * (tf * (self.k1 + 1)) / denominator
        return score


def tokenize(text: str) -> list[str]:
    raw_tokens = [_normalize_token(match.group(0).lower()) for match in TOKEN_RE.finditer(text)]
    tokens = [token for token in raw_tokens if token and token not in STOPWORDS]
    return _expand_domain_terms(tokens)


def is_out_of_scope_query(text: str) -> bool:
    tokens = set(tokenize(text))
    if tokens.intersection(HIGH_CONFIDENCE_OUT_OF_SCOPE_TERMS) and not tokens.intersection(EXPLICIT_CORPUS_TERMS):
        return True
    return bool(tokens.intersection(OUT_OF_SCOPE_TERMS)) and not bool(tokens.intersection(IN_SCOPE_TERMS))


def _normalize_token(token: str) -> str:
    canonical = {
        "acknowledgement": "acknowledgment",
        "acknowledgements": "acknowledgment",
        "acknowledges": "acknowledge",
        "acknowledged": "acknowledged",
        "delivered": "deliver",
        "delivers": "deliver",
        "delivery": "deliver",
        "failures": "failure",
        "pdus": "pdu",
        "polling": "poll",
        "reassembled": "reassembly",
        "reassemble": "reassembly",
        "reassembles": "reassembly",
        "receives": "receive",
        "received": "receive",
        "receiving": "receive",
        "reports": "report",
        "sdus": "sdu",
        "segments": "segment",
        "segmentation": "segment",
        "segmented": "segment",
        "retransmissions": "retransmission",
    }
    if token in canonical:
        return canonical[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _expand_domain_terms(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    token_set = set(tokens)
    if {"acknowledged", "mode"}.issubset(token_set):
        expanded.extend(["am", "am"])
    if {"unacknowledged", "mode"}.issubset(token_set):
        expanded.extend(["um", "um"])
    if {"transparent", "mode"}.issubset(token_set):
        expanded.extend(["tm", "tm"])
    if "missing" in token_set:
        expanded.extend(["lost", "loss", "nack"])
    if "poll" in token_set:
        expanded.extend(["polling", "poll"])
    if "sequence" in token_set:
        expanded.extend(["sn", "sequence"])
    if "error" in token_set or "recovery" in token_set:
        expanded.extend(["arq", "correction"])
    if "failure" in token_set:
        expanded.extend(["negative", "acknowledgment", "status", "nack"])
    if "retransmission" in token_set:
        expanded.extend(["retransmit", "negative", "acknowledgment"])
    if {"random", "access"}.issubset(token_set):
        expanded.extend(["ra", "preamble", "response"])
    if {"logical", "channel"}.issubset(token_set):
        expanded.extend(["lch", "prioritization"])
    if {"transport", "channel"}.issubset(token_set):
        expanded.extend(["ul", "sch", "dl", "sch"])
    return expanded


def _adjust_score_for_query_context(query_terms: list[str], chunk: dict[str, Any], score: float) -> float:
    section = str(chunk.get("section", ""))
    spec_id = str(chunk.get("spec_id", ""))
    text = str(chunk.get("text", "")).lower()
    query_set = set(query_terms)
    asks_for_format = bool({"field", "header", "format", "bit"}.intersection(query_set))
    asks_rlc = "rlc" in query_set
    asks_error_recovery = bool({"error", "recovery", "failure", "retransmission"}.intersection(query_set))

    if section.startswith("6.2.3") and not asks_for_format:
        score *= 0.65
    if asks_rlc and spec_id == "3GPP TS 38.322":
        score *= 1.35
    if asks_rlc and spec_id in {"3GPP TS 38.321", "3GPP TS 38.331"}:
        score *= 0.75
    if {"purpose", "segment"}.issubset(query_set) and str(chunk.get("section")) == "4.4":
        score *= 2.0
    if "function" in query_set and str(chunk.get("section")) == "4.4":
        score *= 2.0
    if asks_rlc and asks_error_recovery and section == "4.4" and "arq" in text:
        score *= 2.5
    if {"missing", "receive"}.issubset(query_set) and section.startswith(("5.2", "5.3")):
        score *= 1.2
    return score


def _to_retrieved_chunk(chunk: dict[str, Any], score: float) -> RetrievedChunk:
    metadata = dict(chunk)
    text = str(metadata.pop("text"))
    return RetrievedChunk(text=text, score=score, metadata=metadata)
