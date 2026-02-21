from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.repositories.label_repository import LabelRepository
from app.repositories.text_entry_repository import TextEntryRepository
from app.schemas.classification import CreateLabelRequest, CreateLabelResponse, DeleteLabelResponse, LabelDetailOut, LabelOut
from app.core.errors import OllamaBadResponseError, OllamaUnavailableError
from app.core.label_utils import normalize_label_name
from app.services.embedding_service import EmbeddingClient
from app.services.label_embedding_service import LabelEmbeddingService
from app.services.service_factory import build_embedding_client

router = APIRouter(tags=["labels"])


@router.get("/labels", response_model=list[LabelOut])
def list_labels(db: Session = Depends(get_db)) -> list[LabelOut]:
    labels = LabelRepository(db).list_labels()
    return [LabelOut.model_validate(l) for l in labels]


@router.get("/labels/{name}", response_model=LabelDetailOut)
def get_label(name: str, db: Session = Depends(get_db)) -> LabelDetailOut:
    label_repo = LabelRepository(db)
    text_repo = TextEntryRepository(db)

    normalized = normalize_label_name(name)
    label = label_repo.get_by_name(normalized)
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")

    examples = text_repo.examples_for_label(label.id)
    return LabelDetailOut(name=label.name, definition=label.definition, usage_count=label.usage_count, examples=examples)


@router.post("/labels", response_model=CreateLabelResponse)
def create_label_endpoint(
    payload: CreateLabelRequest,
    db: Session = Depends(get_db),
    embedding_client: EmbeddingClient = Depends(build_embedding_client),
) -> CreateLabelResponse:
    return create_label(payload=payload, db=db, embedding_client=embedding_client)


def create_label(payload: CreateLabelRequest, db: Session, embedding_client: EmbeddingClient) -> CreateLabelResponse:
    repo = LabelRepository(db)
    normalized = normalize_label_name(payload.name)

    existing = repo.get_by_name(normalized)
    if existing:
        raise HTTPException(status_code=409, detail="Label already exists")

    try:
        centroid = embedding_client.get_embedding(payload.definition)
    except OllamaUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except OllamaBadResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    label = repo.create(name=normalized, definition=payload.definition, centroid=centroid)
    LabelEmbeddingService(db, embedding_client).recompute_for_label(label)
    db.commit()
    return CreateLabelResponse(created=True, name=normalized)


@router.delete("/labels/{name}", response_model=DeleteLabelResponse)
def delete_label(
    name: str,
    force: bool = Query(default=False, description="Delete even when usage_count > 0"),
    db: Session = Depends(get_db),
) -> DeleteLabelResponse:
    repo = LabelRepository(db)
    normalized = normalize_label_name(name)
    label = repo.get_by_name(normalized)
    if not label:
        raise HTTPException(status_code=404, detail="Label not found")

    if label.usage_count > 0 and not force:
        raise HTTPException(
            status_code=400,
            detail="Safety rule: label has usage and cannot be deleted without force=true",
        )

    # Avoid leaving orphaned label_id values on existing entries.
    TextEntryRepository(db).detach_label(label.id)

    repo.delete(label)
    db.commit()
    return DeleteLabelResponse(deleted=True, name=normalized, reason="deleted")
