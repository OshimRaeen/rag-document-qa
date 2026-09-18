import json
from pathlib import Path

import chromadb

from app.chunker import chunk_document_units
from app.embeddings import embed_documents, embed_query
from app.loaders import load_document


DOCUMENT_PATH = "evaluation/eval_document.txt"
QUESTIONS_PATH = "evaluation/questions.json"

TOP_K = 3

CONFIGURATIONS = [
    {
        "chunk_size": 400,
        "overlap": 60,
    },
    {
        "chunk_size": 800,
        "overlap": 120,
    },
    {
        "chunk_size": 1200,
        "overlap": 180,
    },
]


def load_questions() -> list[dict]:
    path = Path(QUESTIONS_PATH)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_configuration(
    document_units: list[dict],
    questions: list[dict],
    chunk_size: int,
    overlap: int,
) -> dict:

    chunks = chunk_document_units(
        document_units,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunk_embeddings = embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    client = chromadb.EphemeralClient()

    collection = client.create_collection(
    name=f"evaluation_{chunk_size}_{overlap}",
    metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[
            f"chunk-{index}"
            for index in range(len(chunks))
        ],
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        embeddings=chunk_embeddings,
    )

    hits_at_1 = 0
    hits_at_3 = 0

    question_results = []

    for item in questions:
        question = item["question"]
        expected_text = item["expected_text"]

        query_embedding = embed_query(question)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, len(chunks)),
            include=["documents", "distances"],
        )

        retrieved_documents = results["documents"][0]
        distances = results["distances"][0]

        expected_lower = expected_text.lower()

        hit_at_1 = (
            len(retrieved_documents) > 0
            and expected_lower in retrieved_documents[0].lower()
        )

        hit_at_3 = any(
            expected_lower in document.lower()
            for document in retrieved_documents
        )

        if hit_at_1:
            hits_at_1 += 1

        if hit_at_3:
            hits_at_3 += 1

        question_results.append(
            {
                "question": question,
                "expected_text": expected_text,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "top_distance": (
                    round(distances[0], 4)
                    if distances
                    else None
                ),
            }
        )

    total_questions = len(questions)

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks_created": len(chunks),
        "hit_at_1": hits_at_1 / total_questions,
        "hit_at_3": hits_at_3 / total_questions,
        "questions": question_results,
    }


def main():
    document_units = load_document(DOCUMENT_PATH)
    questions = load_questions()

    all_results = []

    for config in CONFIGURATIONS:
        print(
            f"\nEvaluating chunk_size={config['chunk_size']}, "
            f"overlap={config['overlap']}"
        )

        result = evaluate_configuration(
            document_units=document_units,
            questions=questions,
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
        )

        all_results.append(result)

        print(
            f"Chunks created: {result['chunks_created']}"
        )

        print(
            f"Hit@1: {result['hit_at_1']:.2%}"
        )

        print(
            f"Hit@3: {result['hit_at_3']:.2%}"
        )

        for question_result in result["questions"]:
            if question_result["hit_at_1"]:
                status = "HIT@1"
            elif question_result["hit_at_3"]:
                status = "HIT@3"
            else:
                status = "MISS"

            print(
                f"  [{status}] "
                f"{question_result['question']} "
                f"(top distance="
                f"{question_result['top_distance']})"
            )

    output_path = Path(
        "evaluation/results.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_results,
            file,
            indent=2,
        )

    print(
        f"\nDetailed results saved to {output_path}"
    )


if __name__ == "__main__":
    main()