from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)
    label: str | None = Field(
        default=None,
        max_length=120,
        description="Optional existing label name to force-assign. If omitted (and label_id omitted), request uses similarity matching.",
    )
    label_id: int | None = Field(
        default=None,
        ge=1,
        description="Optional existing label id to force-assign. If omitted (and label omitted), request uses similarity matching.",
    )

    @field_validator("label", mode="before")
    @classmethod
    def normalize_empty_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class ClassificationResponse(BaseModel):
    text: str
    assigned_label: str | None
    similarity_score: float | None
    created_new_label: bool = False
    reason: str
    best_match_label: str | None = None
    best_match_score: float | None = None


class LabelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    definition: str
    usage_count: int


class LabelDetailOut(BaseModel):
    name: str
    definition: str
    usage_count: int
    examples: list[str]


class DeleteLabelResponse(BaseModel):
    deleted: bool
    name: str
    reason: str


class CreateLabelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    definition: str = Field(min_length=1, max_length=2000)


class CreateLabelResponse(BaseModel):
    created: bool
    name: str


class DeleteEntryResponse(BaseModel):
    deleted: bool
    entry_id: int

class ReclassifiedItemRequest(BaseModel):
    label: str | None = Field(default=None, max_length=120)
    label_id: int | None = Field(default=None, ge=1)

    @field_validator("label", mode="before")
    @classmethod
    def normalize_empty_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

  

class ReclassifiedItemResponse(BaseModel):
    entry_id: int
    text: str
    assigned_label: str
    similarity_score: float | None = None
    reason: str


class ReclassifyResponse(BaseModel):
    scanned_count: int
    reclassified_count: int
    failed_count: int
    reclassified: list[ReclassifiedItemResponse] = Field(default_factory=list)
    failed: list[ReclassifiedItemResponse] = Field(default_factory=list) 




class StatsResponse(BaseModel):
    labels_count: int
    classified_entries_count: int
    unclassified_entries_count: int
