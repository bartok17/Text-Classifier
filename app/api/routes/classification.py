from fastapi import APIRouter, Depends, HTTPException
import re
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.classification import ClassificationResponse, ClassifyRequest
from app.services.classification_service import ClassificationService
from app.services.service_factory import build_embedding_client
from app.core.errors import OllamaBadResponseError, OllamaUnavailableError
from app.core.label_utils import parse_no_label_fit

router = APIRouter(tags=["classification"])





@router.post("/classify", response_model=ClassificationResponse)
def classify(payload: ClassifyRequest, db: Session = Depends(get_db)) -> ClassificationResponse:
    service = ClassificationService(
        db=db,
        embedding_client=build_embedding_client(),
    )
    try:
        result = service.classify(text=payload.text, label=payload.label, label_id=payload.label_id)
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
    return ClassificationResponse(
        text=payload.text,
        assigned_label=result.assigned_label,
        similarity_score=result.similarity_score,
        created_new_label=result.created_new_label,
        reason=result.reason,
        best_match_label=result.best_match_label,
        best_match_score=result.best_match_score,
    )
