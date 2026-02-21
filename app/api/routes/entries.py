import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.errors import OllamaBadResponseError, OllamaUnavailableError
from app.core.label_utils import parse_no_label_fit,best_label_match,normalize_label_name
from app.core.config import settings
from app.repositories.label_repository import LabelRepository
from app.repositories.text_entry_repository import TextEntryRepository
from app.schemas.classification import DeleteEntryResponse, ReclassifiedItemRequest,ReclassifiedItemResponse,ReclassifyResponse
from app.services.embedding_service import EmbeddingClient,cosine_similarity
from app.services.label_embedding_service import LabelEmbeddingService
from app.services.service_factory import build_embedding_client
from app.services.classification_service import ClassificationService



router = APIRouter(tags=["entries"])


@router.post("/entries/reclassify/{entry_id}", response_model=ReclassifiedItemResponse)
def reclasify_entry(
    entry_id: int,
    payload: ReclassifiedItemRequest,
    db: Session = Depends(get_db),
    embedding_client: EmbeddingClient = Depends(build_embedding_client),
) -> ReclassifiedItemResponse:
    entry_repo = TextEntryRepository(db)
    label_repo = LabelRepository(db)
    label_embeddings = LabelEmbeddingService(db, embedding_client)
    
    entry = entry_repo.get_by_id(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    affected_label_id = entry.label_id
    entry.label_id = None
    db.flush()

    if affected_label_id is not None:
        label = label_repo.get_by_id(affected_label_id)
        if label is not None:
            try:
                label_embeddings.recompute_for_label(label)
            except OllamaUnavailableError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except OllamaBadResponseError as e:
                raise HTTPException(status_code=502, detail=str(e)) from e
    
    reason = "no reason"
    try:
        vector = embedding_client.get_embedding(entry.text)

        forced_label = None
        if payload.label_id is not None:
            forced_label = label_repo.get_by_id(payload.label_id)
        elif payload.label is not None and payload.label.strip() != "":
            normalized = normalize_label_name(payload.label)
            forced_label = label_repo.get_by_name(normalized)

        if payload.label_id is not None or (payload.label is not None and payload.label.strip() != ""):
            existing = forced_label
            if not existing:
                raise ValueError("label_not_found")

            score = cosine_similarity(vector, existing.centroid)

            entry.label_id = existing.id
            entry.similarity_score = score
            reason = "forced_label_assigned"

        else:
            known_labels = label_repo.list_labels()
            best_label, best_score = best_label_match(vector, known_labels)
            best_match_label = best_label.name if best_label else None
            best_match_score = round(best_score, 4) if best_label else None

            if best_label and best_score < settings.similarity_threshold:
                raise ValueError(
                f"no_label_fit: best_match_label={best_match_label!r} best_match_score={best_match_score!r}"
                )
            entry.label_id = best_label.id
            reason = "matched_existing_label "
        
        new_label = label_repo.get_by_id(entry.label_id)
        if not new_label:
            raise ValueError("label_not_found")

        try:
            label_embeddings.recompute_for_label(new_label)
        except OllamaUnavailableError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except OllamaBadResponseError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        db.commit()
    except ValueError as e:
        msg = str(e)
        if msg == "label_not_found":
            raise HTTPException(status_code=404, detail="Label not found") from e
        if msg.startswith("no_label_fit:"):
            best_match_label, best_match_score = parse_no_label_fit(msg)
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No existing label fit this text",
                    "best_match_label": best_match_label,
                    "best_match_score": best_match_score,
                },
            ) from e
        raise HTTPException(status_code=400, detail=msg) from e
    except OllamaUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except OllamaBadResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ReclassifiedItemResponse(
        entry_id=entry_id,
        text=entry.text,
        assigned_label= new_label.name,
        similarity_score= entry.similarity_score,
        reason=reason,
    )




@router.delete("/entries/{entry_id}", response_model=DeleteEntryResponse)
def delete_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    embedding_client: EmbeddingClient = Depends(build_embedding_client),
) -> DeleteEntryResponse:
    entry_repo = TextEntryRepository(db)
    label_repo = LabelRepository(db)
    label_embeddings = LabelEmbeddingService(db, embedding_client)

    entry = entry_repo.get_by_id(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    affected_label_id = entry.label_id
    entry_repo.delete(entry)
    db.flush()

    if affected_label_id is not None:
        label = label_repo.get_by_id(affected_label_id)
        if label is not None:
            try:
                label_embeddings.recompute_for_label(label)
            except OllamaUnavailableError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except OllamaBadResponseError as e:
                raise HTTPException(status_code=502, detail=str(e)) from e

    db.commit()

    return DeleteEntryResponse(deleted=True, entry_id=entry_id)
