from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.label import Label
from app.repositories.label_repository import LabelRepository
from app.repositories.text_entry_repository import TextEntryRepository
from app.services.embedding_service import EmbeddingClient, cosine_similarity
from app.core.label_utils import normalize_label_name, best_label_match
from app.services.label_embedding_service import LabelEmbeddingService


@dataclass
class ClassificationResult:
    assigned_label: str | None
    similarity_score: float | None
    created_new_label: bool
    reason: str
    best_match_label: str | None = None
    best_match_score: float | None = None


class ClassificationService:
    def __init__(self, db: Session, embedding_client: EmbeddingClient):
        self.db = db
        self.labels = LabelRepository(db)
        self.entries = TextEntryRepository(db)
        self.embedding_client = embedding_client
        self.label_embeddings = LabelEmbeddingService(db, embedding_client)

    def classify(self, text: str, label: str | None = None, label_id: int | None = None) -> ClassificationResult:
        vector = self.embedding_client.get_embedding(text)

        forced_label = None
        if label_id is not None:
            forced_label = self.labels.get_by_id(label_id)
        elif label is not None and label.strip() != "":
            normalized = normalize_label_name(label)
            forced_label = self.labels.get_by_name(normalized)

        if label_id is not None or (label is not None and label.strip() != ""):
            existing = forced_label
            if not existing:
                raise ValueError("label_not_found")

            score = cosine_similarity(vector, existing.centroid)
            self.entries.create(
                text=text,
                label_id=existing.id,
                similarity_score=score,
                confidence="forced",
                embedding=vector,
            )
            self.label_embeddings.recompute_for_label(existing)
            self.db.commit()
            return ClassificationResult(
                assigned_label=existing.name,
                similarity_score=round(score, 4),
                created_new_label=False,
                reason="forced_label_assigned",
            )

        known_labels = self.labels.list_labels()
        best_label, best_score = best_label_match(vector, known_labels)
        best_match_label = best_label.name if best_label else None
        best_match_score = round(best_score, 4) if best_label else None

        if best_label and best_score >= settings.similarity_threshold:
            self.entries.create(
                text=text,
                label_id=best_label.id,
                similarity_score=best_score,
                confidence="high",
                embedding=vector,
            )
            self.label_embeddings.recompute_for_label(best_label)
            self.db.commit()
            return ClassificationResult(
                assigned_label=best_label.name,
                similarity_score=round(best_score, 4),
                created_new_label=False,
                reason="matched_existing_label",
                best_match_label=best_match_label,
                best_match_score=best_match_score,
            )

        # Safety rule: never store an unlabelled entry.
        raise ValueError(
            f"no_label_fit: best_match_label={best_match_label!r} best_match_score={best_match_score!r}"
        )





