from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes.classification import router as classification_router
from app.api.routes.entries import router as entries_router
from app.api.routes.labels import router as labels_router
from app.api.routes.stats import router as stats_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.models import Label, TextEntry  # noqa: F401

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(classification_router)
app.include_router(entries_router)
app.include_router(labels_router)
app.include_router(stats_router)
