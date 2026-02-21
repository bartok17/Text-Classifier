import json
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    definition: Mapped[str] = mapped_column(Text)
    centroid_json: Mapped[str] = mapped_column(Text)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    entries = relationship("TextEntry", back_populates="label")

    @property
    def centroid(self) -> list[float]:
        return json.loads(self.centroid_json)

    @centroid.setter
    def centroid(self, value: list[float]) -> None:
        self.centroid_json = json.dumps(value)
