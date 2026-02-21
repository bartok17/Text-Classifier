from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.text_entry import TextEntry


class TextEntryRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        text: str,
        label_id: int | None,
        similarity_score: float | None,
        confidence: str | None = None,
        embedding: list[float] | None = None,
    ) -> TextEntry:
        entry = TextEntry(text=text, label_id=label_id, similarity_score=similarity_score, confidence=confidence)
        if embedding is not None:
            entry.embedding = embedding
        self.db.add(entry)
        self.db.flush()
        return entry

    def count_classified(self) -> int:
        return self.db.execute(select(func.count(TextEntry.id)).where(TextEntry.label_id.is_not(None))).scalar_one()

    def count_unclassified(self) -> int:
        return self.db.execute(select(func.count(TextEntry.id)).where(TextEntry.label_id.is_(None))).scalar_one()

    def examples_for_label(self, label_id: int, limit: int = 5) -> list[str]:
        rows = self.db.execute(
            select(TextEntry.text).where(TextEntry.label_id == label_id).order_by(TextEntry.created_at.desc()).limit(limit)
        ).all()
        return [r[0] for r in rows]

    def get_by_id(self, entry_id: int) -> TextEntry | None:
        return self.db.execute(select(TextEntry).where(TextEntry.id == entry_id)).scalar_one_or_none()

    def list_by_label(self, label_id: int) -> list[TextEntry]:
        return self.db.execute(select(TextEntry).where(TextEntry.label_id == label_id)).scalars().all()

    def delete(self, entry: TextEntry) -> None:
        self.db.delete(entry)

    def detach_label(self, label_id: int) -> int:
        entries = self.db.execute(select(TextEntry).where(TextEntry.label_id == label_id)).scalars().all()
        for entry in entries:
            entry.label_id = None
        return len(entries)
