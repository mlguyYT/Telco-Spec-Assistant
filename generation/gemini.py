from __future__ import annotations

import json
import os
from typing import Any

from generation.base import GeneratedAnswer
from generation.extractive import INSUFFICIENT_EVIDENCE_ANSWER, citation
from retrieval.base import RetrievedChunk

DEFAULT_REGION = "us-central1"


class GeminiGenerator:
    name = "gemini"

    def __init__(
        self,
        *,
        project_id: str | None = None,
        region: str | None = None,
        model_name: str | None = None,
    ) -> None:
        genai, generate_content_config = _load_genai_dependencies()
        self.project_id = project_id or _required_env("GCP_PROJECT_ID")
        self.region = region or os.environ.get("REGION", DEFAULT_REGION)
        self.model_name = model_name or _required_env("GEMINI_MODEL")
        self.client = genai.Client(vertexai=True, project=self.project_id, location=self.region)
        self.generate_content_config = generate_content_config

    def generate(
        self,
        question: str,
        results: list[RetrievedChunk],
        *,
        min_score: float = 0.0,
    ) -> GeneratedAnswer:
        if not results or results[0].score < min_score:
            return _unsupported(self.name)

        evidence = _build_evidence(results)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=_prompt(question=question, evidence=evidence),
            config=self.generate_content_config(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        payload = _parse_json_response(getattr(response, "text", "") or "")
        if not payload.get("supported"):
            return _unsupported(self.name)

        citation_ids = [str(value) for value in payload.get("citation_ids", [])]
        valid_ids = {item["citation_id"] for item in evidence}
        selected_ids = [value for value in citation_ids if value in valid_ids]
        if not selected_ids:
            return _unsupported(self.name)

        by_id = {item["citation_id"]: item["result"] for item in evidence}
        return GeneratedAnswer(
            answer=str(payload.get("answer", "")).strip() or INSUFFICIENT_EVIDENCE_ANSWER,
            citations=[citation(by_id[citation_id], question=question) for citation_id in selected_ids],
            supported=True,
            generator=self.name,
        )


def _build_evidence(results: list[RetrievedChunk]) -> list[dict[str, Any]]:
    evidence = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        evidence.append(
            {
                "citation_id": f"C{index}",
                "result": result,
                "spec_id": chunk.get("spec_id"),
                "release": chunk.get("release"),
                "version": chunk.get("version"),
                "section": chunk.get("section"),
                "section_title": chunk.get("section_title"),
                "text": chunk.get("text", ""),
            }
        )
    return evidence


def _prompt(question: str, evidence: list[dict[str, Any]]) -> str:
    excerpts = []
    for item in evidence:
        excerpts.append(
            "\n".join(
                [
                    f"[{item['citation_id']}] {item['spec_id']} {item['version']} clause {item['section']} ({item['section_title']})",
                    str(item["text"])[:3500],
                ]
            )
        )
    return (
        "You are a telecom standards assistant. Answer only from the provided excerpts. "
        "Do not use outside knowledge. If the excerpts do not support the answer, return supported=false. "
        "Preserve important technical terms, acronyms, field names, procedure names, and quoted values from the excerpts "
        "when they are relevant to the question. If an excerpt names a mechanism or acronym that directly answers the "
        "question, include that term in the answer. Do not replace specification terms only with broad paraphrases. "
        "Use only citation_ids from the provided excerpts. Return strict JSON with keys: "
        "supported, answer, citation_ids.\n\n"
        f"Question: {question}\n\n"
        "Excerpts:\n"
        + "\n\n".join(excerpts)
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini response JSON must be an object")
    return payload


def _unsupported(generator_name: str) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        citations=[],
        supported=False,
        generator=generator_name,
    )


def _load_genai_dependencies() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
    except ImportError as exc:
        raise RuntimeError(
            "Gemini generation requires optional cloud dependencies. Install them with: "
            "pip install -r requirements-cloud.txt"
        ) from exc
    return genai, GenerateContentConfig


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value
