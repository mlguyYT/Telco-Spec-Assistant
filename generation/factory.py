from __future__ import annotations

import os

from generation.base import AnswerGenerator
from generation.extractive import ExtractiveGenerator


def get_generator(kind: str | None = None) -> AnswerGenerator:
    generator_kind = (kind or os.environ.get("GENERATOR", "extractive")).lower()
    if generator_kind == "extractive":
        return ExtractiveGenerator()
    if generator_kind == "gemini":
        from generation.gemini import GeminiGenerator

        return GeminiGenerator()
    raise ValueError(f"unsupported generator: {generator_kind}")

