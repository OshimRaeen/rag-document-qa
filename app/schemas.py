from typing import Optional

from pydantic import BaseModel, Field


class DocumentUnit(BaseModel):
    text: str
    source: str
    page: Optional[int] = None


class UploadResponse(BaseModel):
    filename: str
    chunks_stored: int
    message: str


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class SourceChunk(BaseModel):
    text: str
    source: str
    page: Optional[int] = None
    chunk_index: int
    distance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]