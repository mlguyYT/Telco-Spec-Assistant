from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def chunk(self) -> dict[str, Any]:
        return {"text": self.text, **self.metadata}


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        ...
