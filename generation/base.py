from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from retrieval.base import RetrievedChunk


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    supported: bool
    generator: str


class AnswerGenerator(Protocol):
    name: str

    def generate(
        self,
        question: str,
        results: list[RetrievedChunk],
        *,
        min_score: float = 0.0,
    ) -> GeneratedAnswer:
        ...

