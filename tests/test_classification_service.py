import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.core.config import settings
from app.api.routes.labels import create_label
from app.repositories.label_repository import LabelRepository
from app.schemas.classification import ClassifyRequest, CreateLabelRequest
from app.services.classification_service import ClassificationService


class FakeEmbeddingClient:
    provider_name = "fake"

    def __init__(self, dim: int = 16):
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

        # Normalize for cosine similarity.
        norm = sum(v * v for v in vals) ** 0.5
        return vals if norm == 0 else [v / norm for v in vals]


from app.models.text_entry import TextEntry


engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base.metadata.create_all(bind=engine)


def test_errors_when_no_labels_exist() -> None:
    db: Session = TestingSessionLocal()

    service = ClassificationService(db, embedding_client=FakeEmbeddingClient(dim=16))
    with pytest.raises(ValueError):
        service.classify("Server outage reported")
    db.close()


def test_can_classify_to_existing_label() -> None:
    db: Session = TestingSessionLocal()

    old_threshold = settings.similarity_threshold
    settings.similarity_threshold = 0.05

    embedding = FakeEmbeddingClient(dim=16)
    create_label(
        CreateLabelRequest(
            name="database_incidents",
            definition="database timeout incident outages and incidents",
        ),
        db=db,
        embedding_client=embedding,
    )

    service = ClassificationService(db, embedding_client=embedding)

    first = service.classify("Database timeout incident")
    second = service.classify("Database timeout detected")

    assert first.assigned_label is not None
    assert second.assigned_label == first.assigned_label
    db.close()

    settings.similarity_threshold = old_threshold


def test_forced_label_requires_existing_label() -> None:
    db: Session = TestingSessionLocal()

    service = ClassificationService(db, embedding_client=FakeEmbeddingClient(dim=16))

    with pytest.raises(ValueError) as exc:
        service.classify("Buy bananas and grapes", label="groceries")

    assert str(exc.value) == "label_not_found"
    db.close()


def test_returns_no_label_fit_when_threshold_not_met() -> None:
    old_threshold = settings.similarity_threshold
    settings.similarity_threshold = 0.99

    db: Session = TestingSessionLocal()
    embedding = FakeEmbeddingClient(dim=16)
    create_label(
        CreateLabelRequest(name="groceries", definition="buy groceries buy food groceries and shopping"),
        db=db,
        embedding_client=embedding,
    )
    service = ClassificationService(db, embedding_client=embedding)
    before_count = db.query(TextEntry).count()

    with pytest.raises(ValueError) as exc:
        service.classify("Buy bananas")

    assert str(exc.value).startswith("no_label_fit:")
    after_count = db.query(TextEntry).count()
    assert after_count == before_count
    db.close()

    settings.similarity_threshold = old_threshold


def test_manually_created_label_can_be_matched_in_classification() -> None:
    old_threshold = settings.similarity_threshold
    settings.similarity_threshold = 0.2

    db: Session = TestingSessionLocal()
    embedding = FakeEmbeddingClient(dim=16)
    create_label(
        CreateLabelRequest(name="groceries and shopping", definition="Groceries and shopping errands"),
        db=db,
        embedding_client=embedding,
    )

    service = ClassificationService(db, embedding_client=embedding)
    result = service.classify("shopping list for groceries")

    assert result.assigned_label == "groceries_and_shopping"
    assert result.created_new_label is False
    db.close()

    settings.similarity_threshold = old_threshold


def test_forced_label_by_id_works() -> None:
    db: Session = TestingSessionLocal()
    embedding = FakeEmbeddingClient(dim=16)
    create_label(
        CreateLabelRequest(name="forced_groceries", definition="Groceries and shopping errands"),
        db=db,
        embedding_client=embedding,
    )
    label = LabelRepository(db).get_by_name("forced_groceries")
    assert label is not None

    service = ClassificationService(db, embedding_client=embedding)
    result = service.classify("Buy bananas and grapes", label_id=label.id)

    assert result.assigned_label == "forced_groceries"
    assert result.created_new_label is False
    db.close()


def test_empty_label_is_treated_as_missing() -> None:
    payload = ClassifyRequest(text="Buy bananas", label="")
    assert payload.label is None
