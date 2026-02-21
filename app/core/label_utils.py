import re
from app.services.embedding_service import cosine_similarity
from app.models.label import Label


def normalize_label_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Label name must not be empty")
    return cleaned[:120]

def parse_no_label_fit(msg: str) -> tuple[str | None, float | None]:
    match = re.search(r"best_match_label=(?P<label>.+?)\s+best_match_score=(?P<score>.+)$", msg)
    if not match:
        return None, None

    raw_label = match.group("label").strip()
    raw_score = match.group("score").strip()

    label: str | None
    if raw_label == "None":
        label = None
    else:
        label = raw_label.strip("'\"")

    score: float | None
    if raw_score == "None":
        score = None
    else:
        try:
            score = float(raw_score)
        except ValueError:
            score = None

    return label, score

def best_label_match(vector: list[float], labels: list[Label]) -> tuple[Label | None, float]:
    best_label = None
    best_score = -1.0
    for label in labels:
        score = cosine_similarity(vector, label.centroid)
        if score > best_score:
            best_label = label
            best_score = score
    if best_score < 0:
        return None, 0.0
    return best_label, best_score
