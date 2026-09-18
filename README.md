# RAG Document Question Answering API

A lightweight Retrieval-Augmented Generation (RAG) API that answers questions using only information from uploaded PDF and TXT documents.

The system extracts text from uploaded documents, splits it into overlapping chunks, generates embeddings using Gemini, stores them in a local ChromaDB vector store, retrieves the most relevant chunks for a question, and uses Gemini to generate a grounded answer.

If the retrieved context does not contain enough information to answer a question, the model is instructed to say so instead of inventing an answer.

## Features

- Upload PDF and TXT documents
- Extract text while preserving source and page metadata
- Overlapping text chunking
- Gemini embeddings for semantic retrieval
- Local persistent ChromaDB vector store
- Top-k cosine-distance retrieval
- Grounded Gemini answer generation
- Source chunks returned with every answer
- Pydantic request/response validation
- Error handling for document parsing and Gemini API failures
- Retrieval evaluation across multiple chunk configurations
- Unit tests for document loading and chunking

## Architecture

```text
                 DOCUMENT INGESTION

PDF / TXT
    |
    v
Text Extraction
    |
    v
Chunking (400 chars, 60 overlap)
    |
    v
Gemini Embeddings
    |
    v
ChromaDB


                 QUESTION ANSWERING

User Question
    |
    v
Gemini Query Embedding
    |
    v
Top-3 Semantic Retrieval
    |
    v
Retrieved Document Chunks
    |
    +------------------+
    |                  |
    v                  v
Question          Grounding Context
    |                  |
    +--------+---------+
             |
             v
          Gemini
             |
             v
     Grounded Answer
             +
        Source Chunks
```

## Tech Stack

- Python
- FastAPI
- Pydantic
- PyMuPDF
- ChromaDB
- Gemini API
- pytest

## Project Structure

```text
rag-document-qa/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── loaders.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── vector_store.py
│   └── schemas.py
│
├── evaluation/
│   ├── eval_document.txt
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   └── results.json
│
├── sample_docs/
│   ├── company.txt
│   └── test.pdf
│
├── tests/
│   ├── test_chunker.py
│   └── test_loaders.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

`uploads/` and `chroma_db/` are created locally at runtime and are intentionally excluded from Git.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/OshimRaeen/rag-document-qa.git
cd rag-document-qa
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

`.env.example` is included as a template. The real `.env` file is ignored by Git.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## API Usage

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Upload a Document

```http
POST /documents
```

Upload a `.pdf` or `.txt` file using the `file` field.

The document is:

1. stored locally,
2. parsed,
3. split into chunks,
4. embedded,
5. stored in ChromaDB.

Example response:

```json
{
  "filename": "company.txt",
  "chunks_stored": 1,
  "message": "Document uploaded, chunked, embedded, and stored successfully."
}
```

### Ask a Question

```http
POST /ask
```

Example request:

```json
{
  "question": "What year did the business start?"
}
```

Example response:

```json
{
  "answer": "2021",
  "sources": [
    {
      "text": "Acme Analytics builds software for retail businesses. The company was founded in 2021...",
      "source": "company.txt",
      "page": null,
      "chunk_index": 0,
      "distance": 0.43
    }
  ]
}
```

The returned source chunks make it possible to inspect the evidence retrieved for the generated answer.

For a question whose answer is not supported by the retrieved document context, the model is instructed to return:

```text
I couldn't find enough information in the uploaded documents to answer that question.
```

## Chunking and Retrieval Evaluation

Chunk size was treated as an experimental parameter rather than chosen only by intuition.

I evaluated three character-based chunking configurations on a small controlled document with 10 manually labeled, paraphrased questions.

The metric checks whether a retrieved chunk contains the expected evidence.

| Chunk Size | Overlap | Chunks | Hit@1 | Hit@3 |
|---:|---:|---:|---:|---:|
| 400 | 60 | 7 | 90% | 100% |
| 800 | 120 | 4 | 70% | 90% |
| 1200 | 180 | 3 | 80% | 100% |

`400 / 60` performed best on this diagnostic set and is therefore used by the API.

With `800 / 120`, the question:

> How frequently is dashboard information updated?

failed to retrieve the chunk containing the expected evidence within the top three results, even though the answer was present in the document.

This experiment is intentionally small and should be treated as a diagnostic comparison, not as evidence that 400-character chunks are universally optimal.

Detailed results are stored in `evaluation/results.json`.

Run the evaluation with:

```bash
python -m evaluation.evaluate_retrieval
```

## Why Top-3 Retrieval?

The evaluation tracks both Hit@1 and Hit@3.

Hit@1 measures whether the required evidence is the first retrieved result. Hit@3 measures whether the evidence appears anywhere in the three chunks supplied as context to the generator.

For the selected `400 / 60` configuration, Hit@1 was 90% and Hit@3 was 100% on the 10-question diagnostic set.

The application therefore currently retrieves the top three chunks for generation.

## Grounding

The generation prompt explicitly instructs Gemini to:

- use only the supplied document context,
- avoid outside knowledge,
- avoid guessing,
- return a fixed fallback response when the context is insufficient.

Retrieval and generation are kept as separate steps so the retrieved evidence can be inspected independently from the generated answer.

No global cosine-distance rejection threshold is currently used. During testing, an unanswerable question could still receive a relatively close vector match, so a threshold selected from a single similarity value would not have been sufficiently justified.

## Error Handling

The API handles several failure cases, including:

- unsupported file formats,
- empty TXT documents,
- PDFs with no extractable text,
- invalid chunking parameters,
- empty questions through Pydantic validation,
- Gemini embedding failures,
- Gemini generation failures.

Failures from the external AI service are converted into API errors instead of being silently ignored.

## Tests

Run:

```bash
python -m pytest -v
```

The current tests cover:

- short-text chunking,
- multi-chunk text,
- invalid chunk parameters,
- TXT loading,
- empty TXT rejection,
- unsupported file rejection.

The tests intentionally avoid external Gemini calls so they can run without consuming API quota.

## What Works

- Text-based PDF ingestion
- UTF-8 TXT ingestion
- Character-based overlapping chunks
- Semantic embeddings
- Persistent local vector storage
- Semantic top-k retrieval
- Grounded question answering
- Source chunk reporting
- Explicit handling of unsupported answers
- Retrieval evaluation
- Basic automated tests

## Current Limitations

- Scanned/image-only PDFs are not supported because OCR is not implemented.
- Chunking is character-based rather than structure-aware or sentence-aware.
- The retrieval evaluation currently uses one controlled document and 10 questions, so it is useful for comparison but not a large benchmark.
- The vector store is local and intended for this take-home scope rather than multi-user production deployment.
- There is no authentication or frontend.
- Re-uploading a document with the same source/chunk identifiers updates existing entries rather than maintaining document versions.

## Possible Next Steps

Given more time, I would:

- evaluate retrieval on a larger and more varied document set,
- compare structure-aware or sentence-aware chunking,
- add document-level filtering and lifecycle management,
- evaluate answer faithfulness in addition to retrieval hit rate,
- add integration tests with mocked AI responses,
- add OCR support for scanned PDFs.

## API Models

Embeddings are generated using:

```text
gemini-embedding-001
```

Answer generation uses the Gemini model configured in `app/generator.py`.

API keys are loaded through environment variables and are never committed to the repository.