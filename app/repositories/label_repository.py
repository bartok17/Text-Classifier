from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.label import Label


class LabelRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_labels(self) -> list[Label]:
        return self.db.execute(select(Label).order_by(Label.usage_count.desc(), Label.name.asc())).scalars().all()

    def get_by_name(self, name: str) -> Label | None:
        return self.db.execute(select(Label).where(Label.name == name)).scalar_one_or_none()

    def get_by_id(self, label_id: int) -> Label | None:
        return self.db.execute(select(Label).where(Label.id == label_id)).scalar_one_or_none()

    def create(self, name: str, definition: str, centroid: list[float]) -> Label:
        label = Label(name=name, definition=definition)
        label.centroid = centroid
        label.usage_count = 0
        self.db.add(label)
        self.db.flush()
        return label

    def delete(self, label: Label) -> None:
        self.db.delete(label)

    def count(self) -> int:
        return self.db.execute(select(func.count(Label.id))).scalar_one()
