from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.repositories.label_repository import LabelRepository
from app.repositories.text_entry_repository import TextEntryRepository
from app.schemas.classification import StatsResponse

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)) -> StatsResponse:
    label_repo = LabelRepository(db)
    text_repo = TextEntryRepository(db)
    return StatsResponse(
        labels_count=label_repo.count(),
        classified_entries_count=text_repo.count_classified(),
        unclassified_entries_count=text_repo.count_unclassified(),
    )
