import math

from sqlalchemy.orm import Session

from app.models.label import Label
from app.repositories.text_entry_repository import TextEntryRepository
from app.services.embedding_service import EmbeddingClient


class LabelEmbeddingService:
    def __init__(self, db: Session, embedding_client: EmbeddingClient):
        self.db = db
        self.embedding_client = embedding_client
        self.entries = TextEntryRepository(db)

    def recompute_for_label(self, label: Label) -> None:
        definition_embedding = self._normalize(self.embedding_client.get_embedding(label.definition))

        label_entries = self.entries.list_by_label(label.id)
        label.usage_count = len(label_entries)

        if not label_entries:
            label.centroid = definition_embedding
            return

        entry_vectors: list[list[float]] = []
        for entry in label_entries:
            cached = entry.embedding
            if isinstance(cached, list) and len(cached) == len(definition_embedding):
                entry_vectors.append(cached)
                continue

            vector = self.embedding_client.get_embedding(entry.text)
            if len(vector) == len(definition_embedding):
                entry.embedding = vector
                entry_vectors.append(vector)

        if not entry_vectors:
            label.centroid = definition_embedding
            return

        entries_centroid = [
            sum(vector[i] for vector in entry_vectors) / len(entry_vectors)
            for i in range(len(definition_embedding))
        ]
        entries_centroid = self._normalize(entries_centroid)

        blended = [
            0.5 * definition_embedding[i] + 0.5 * entries_centroid[i]
            for i in range(len(definition_embedding))
        ]
        label.centroid = self._normalize(blended)

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return [0.0 for _ in vector]
        return [v / norm for v in vector]
