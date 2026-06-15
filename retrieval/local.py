from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
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


@dataclass(frozen=True)
class RetrievalResult:
    chunk: dict[str, Any]
    score: float


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

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[RetrievalResult] = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)
            if score > 0:
                scored.append(RetrievalResult(chunk=chunk, score=score))
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

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


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
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
    return expanded
