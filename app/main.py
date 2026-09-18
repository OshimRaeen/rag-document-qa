import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.loaders import load_document
from app.schemas import UploadResponse


app = FastAPI(
    title="Document RAG API",
    description="Question answering grounded in uploaded documents",
    version="0.1.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files are supported.",
        )

    file_path = UPLOAD_DIR / filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        document_units = load_document(str(file_path))

    except ValueError as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except OSError as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Could not store the uploaded document.",
        ) from exc

    finally:
        await file.close()

    return UploadResponse(
        filename=filename,
        units_extracted=len(document_units),
        message="Document uploaded and parsed successfully.",
    )