from app.db.base import Base
from app.db.session import engine
from app.models import Label, TextEntry  # noqa: F401


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Database initialized")
