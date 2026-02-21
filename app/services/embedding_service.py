import json
import math
import urllib.error
import urllib.request
from typing import Protocol

from app.core.config import settings
from app.core.errors import OllamaBadResponseError, OllamaUnavailableError


class EmbeddingClient(Protocol):
    def get_embedding(self, text: str) -> list[float]: ...


class OllamaEmbeddingClient:
    """Embedding client that calls a local Ollama server.

    Requires Ollama running locally (default: http://localhost:11434).
    """
    provider_name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_embedding_model
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds

    def get_embedding(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        try:
            data = self._post_json("/api/embeddings", payload)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    data = self._post_json("/api/embed", {"model": self.model, "input": text})
                except urllib.error.HTTPError as e2:
                    raise OllamaUnavailableError(
                        f"Ollama endpoint not found: tried /api/embeddings and /api/embed on {self.host} (last HTTP {e2.code})"
                    ) from e2
            else:
                raise OllamaUnavailableError(f"Ollama HTTP {e.code} at {self.host}/api/embeddings") from e

        embedding = data.get("embedding")
        if embedding is None and "embeddings" in data:
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                embedding = embeddings[0]

        if not isinstance(embedding, list) or not embedding:
            raise OllamaBadResponseError(
                f"Ollama returned unexpected embedding payload keys={list(data.keys())}"
            )
        return [float(x) for x in embedding]

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.host}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError:
            # Let caller handle HTTP status codes (e.g. 404 fallback from /api/embeddings -> /api/embed)
            raise
        except urllib.error.URLError as e:
            raise OllamaUnavailableError(f"Ollama unreachable at {self.host}: {e}") from e


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)
