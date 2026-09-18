import chromadb


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"


client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:

    if not chunks:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        chunk_id = (
            f"{chunk['source']}:"
            f"{chunk['page']}:"
            f"{chunk['chunk_index']}"
        )

        ids.append(chunk_id)
        documents.append(chunk["text"])

        metadata = {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }

        if chunk["page"] is not None:
            metadata["page"] = chunk["page"]

        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)
def search_chunks(
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for chunk_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "page": metadata.get("page"),
                "chunk_index": metadata["chunk_index"],
                "distance": distance,
            }
        )

    return retrieved
        