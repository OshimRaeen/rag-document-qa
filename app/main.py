import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.chunker import chunk_document_units
from app.embeddings import embed_documents, embed_query
from app.generator import generate_answer
from app.loaders import load_document
from app.schemas import (
    AnswerResponse,
    QuestionRequest,
    SourceChunk,
    UploadResponse,
)
from app.vector_store import add_chunks, search_chunks


app = FastAPI(
    title="Document RAG API",
    description="Question answering grounded in uploaded documents",
    version="0.1.0",
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 3


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document, extract its text,
    split it into chunks, generate embeddings,
    and store the chunks in the vector database.
    """

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
        # Save uploaded file locally.
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from the PDF/TXT.
        document_units = load_document(str(file_path))

        # Split extracted text into overlapping chunks.
        chunks = chunk_document_units(
            document_units,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError(
                "No usable text chunks could be created from the document."
            )

        # Generate an embedding for every chunk.
        embeddings = embed_documents(
            [chunk["text"] for chunk in chunks]
        )

        # Store chunks, metadata, and embeddings in Chroma.
        stored_count = add_chunks(
            chunks,
            embeddings,
        )

    except ValueError as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=502,
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
        chunks_stored=stored_count,
        message=(
            "Document uploaded, chunked, embedded, "
            "and stored successfully."
        ),
    )


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Embed the user's question, retrieve the most relevant
    document chunks, and generate a grounded answer.
    """

    try:
        # Convert the question into an embedding.
        query_embedding = embed_query(
            request.question
        )

        # Retrieve the closest document chunks.
        retrieved_chunks = search_chunks(
            query_embedding,
            top_k=TOP_K,
        )

        # No documents/chunks are available.
        if not retrieved_chunks:
            return AnswerResponse(
                answer=(
                    "I couldn't find enough information in the "
                    "uploaded documents to answer that question."
                ),
                sources=[],
            )

        # Ask Gemini to answer using only retrieved context.
        answer = generate_answer(
            request.question,
            retrieved_chunks,
        )

        # Return the evidence used for generation.
        sources = [
            SourceChunk(
                text=chunk["text"],
                source=chunk["source"],
                page=chunk.get("page"),
                chunk_index=chunk["chunk_index"],
                distance=chunk["distance"],
            )
            for chunk in retrieved_chunks
        ]

        return AnswerResponse(
            answer=answer,
            sources=sources,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc