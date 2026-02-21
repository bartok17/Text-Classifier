import json
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TextEntry(Base):
    __tablename__ = "text_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(Text)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    label_id: Mapped[int | None] = mapped_column(ForeignKey("labels.id"), nullable=True, index=True)
    label = relationship("Label", back_populates="entries")

    @property
    def embedding(self) -> list[float] | None:
        if self.embedding_json is None:
            return None
        return json.loads(self.embedding_json)

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        if value is None:
            self.embedding_json = None
            return
        self.embedding_json = json.dumps(value)
