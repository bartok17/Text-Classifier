import hashlib
import math

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes.entries import delete_entry
from app.api.routes.labels import create_label, delete_label
from app.db.base import Base
from app.repositories.label_repository import LabelRepository
from app.repositories.text_entry_repository import TextEntryRepository
from app.schemas.classification import CreateLabelRequest


class FakeEmbeddingClient:
    provider_name = "fake"

    def __init__(self, dim: int = 8):
        self.dim = dim

    def get_embedding(self, text: str) -> list[float]:
        vals = [0.0] * self.dim
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return vals

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self.dim):
                vals[i] += (digest[i % len(digest)] / 255.0) - 0.5

        norm = sum(v * v for v in vals) ** 0.5
        return vals if norm == 0 else [v / norm for v in vals]


def _new_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def test_can_create_label_directly() -> None:
    db = _new_db()
    resp = create_label(
        CreateLabelRequest(name="My New Label", definition="All grocery shopping and food items"),
        db=db,
        embedding_client=FakeEmbeddingClient(),
    )

    assert resp.created is True
    assert resp.name == "my_new_label"

    label = LabelRepository(db).get_by_name("my_new_label")
    assert label is not None
    assert label.usage_count == 0
    assert len(label.centroid) > 0
    assert any(abs(v) > 0 for v in label.centroid)
    norm = math.sqrt(sum(v * v for v in label.centroid))
    assert abs(norm - 1.0) < 1e-6
    db.close()


def test_create_label_conflict() -> None:
    db = _new_db()
    create_label(
        CreateLabelRequest(name="dup", definition="Duplicate definition"),
        db=db,
        embedding_client=FakeEmbeddingClient(),
    )

    with pytest.raises(HTTPException) as exc:
        create_label(
            CreateLabelRequest(name="dup", definition="Duplicate definition"),
            db=db,
            embedding_client=FakeEmbeddingClient(),
        )

    assert exc.value.status_code == 409
    db.close()


def test_delete_entry_hard_deletes_entry() -> None:
    db = _new_db()

    label = LabelRepository(db).create(name="keep", definition="Keep label", centroid=[0.0] * 8)
    fake_embedding = FakeEmbeddingClient()
    entry = TextEntryRepository(db).create(text="hello", label_id=label.id, similarity_score=0.1)
    db.commit()

    resp = delete_entry(entry.id, db=db, embedding_client=fake_embedding)
    assert resp.deleted is True
    assert resp.entry_id == entry.id

    still_there = TextEntryRepository(db).get_by_id(entry.id)
    assert still_there is None

    refreshed = LabelRepository(db).get_by_id(label.id)
    assert refreshed is not None
    assert refreshed.usage_count == 0
    norm = math.sqrt(sum(v * v for v in refreshed.centroid))
    assert abs(norm - 1.0) < 1e-6
    db.close()


def test_delete_label_detaches_entries_before_delete() -> None:
    db = _new_db()

    label = LabelRepository(db).create(name="to_remove", definition="To remove", centroid=[0.0] * 8)
    label.usage_count = 1
    entry = TextEntryRepository(db).create(text="x", label_id=label.id, similarity_score=0.2)
    db.commit()

    resp = delete_label("to_remove", force=True, db=db)
    assert resp.deleted is True

    remaining = TextEntryRepository(db).get_by_id(entry.id)
    assert remaining is not None
    assert remaining.label_id is None

    gone_label = LabelRepository(db).get_by_name("to_remove")
    assert gone_label is None
    db.close()
