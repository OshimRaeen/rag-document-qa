import logging
import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=API_KEY)

GENERATION_MODEL = "gemini-3.6-flash"

NOT_FOUND_MESSAGE = (
    "I couldn't find enough information in the uploaded documents "
    "to answer that question."
)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return NOT_FOUND_MESSAGE

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["source"]
        page = chunk.get("page")

        source_label = source

        if page is not None:
            source_label += f", page {page}"

        context_parts.append(
            f"[Source {index}: {source_label}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are answering questions about uploaded reference documents.

Use only the information in the context below.

Rules:
- Do not use outside knowledge.
- Do not guess or invent missing information.
- If the context does not contain enough information to answer the question, say exactly:
  "{NOT_FOUND_MESSAGE}"
- Keep the answer concise and directly answer the question.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )

    except Exception as exc:
        logger.exception("Gemini answer generation failed")

        raise RuntimeError(
            "Failed to generate an answer."
        ) from exc

    answer = response.text

    if not answer:
        raise RuntimeError(
            "The model returned an empty response."
        )

    return answer.strip()