from __future__ import annotations

from typing import Any

DEFAULT_EMBEDDING_MODEL = "text-embedding-005"
DEFAULT_REGION = "us-central1"


class GenAIEmbedder:
    def __init__(
        self,
        *,
        project_id: str,
        region: str = DEFAULT_REGION,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        genai, embed_content_config = _load_genai_dependencies()
        self.model_name = model_name
        self.client = genai.Client(vertexai=True, project=project_id, location=region)
        self.embed_content_config = embed_content_config

    def embed_documents(self, texts: list[str], batch_size: int = 5) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT", batch_size=batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY", batch_size=1)[0]

    def _embed(self, texts: list[str], *, task_type: str, batch_size: int) -> list[list[float]]:
        vectors: list[list[float]] = []
        effective_batch_size = _effective_batch_size(self.model_name, batch_size)
        for index in range(0, len(texts), effective_batch_size):
            batch = texts[index : index + effective_batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=self.embed_content_config(task_type=task_type),
            )
            vectors.extend([list(embedding.values or []) for embedding in response.embeddings or []])
        return vectors


def _effective_batch_size(model_name: str, requested_batch_size: int) -> int:
    if requested_batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if model_name == "gemini-embedding-001":
        return 1
    return min(requested_batch_size, 5)


def _load_genai_dependencies() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai.types import EmbedContentConfig
    except ImportError as exc:
        raise RuntimeError(
            "Embedding requires optional cloud dependencies. Install them with: "
            "pip install -r requirements-cloud.txt"
        ) from exc
    return genai, EmbedContentConfig
