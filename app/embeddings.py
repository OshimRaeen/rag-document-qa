import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to generate document embeddings."
        ) from exc

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def embed_query(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to generate query embedding."
        ) from exc

    return response.embeddings[0].values