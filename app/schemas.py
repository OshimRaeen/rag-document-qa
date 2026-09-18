from typing import Optional

from pydantic import BaseModel


class DocumentUnit(BaseModel):
    text: str
    source: str
    page: Optional[int] = None


class UploadResponse(BaseModel):
    filename: str
    units_extracted: int
    message: str