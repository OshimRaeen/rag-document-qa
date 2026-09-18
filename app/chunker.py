def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        # Avoid cutting a word when possible.
        if end < len(text):
            last_space = chunk.rfind(" ")

            if last_space > 0:
                end = start + last_space
                chunk = text[start:end]

        chunk = chunk.strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def chunk_document_units(
    document_units: list[dict],
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[dict]:

    chunks = []

    for unit in document_units:
        text_chunks = chunk_text(
            unit["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "text": text,
                    "source": unit["source"],
                    "page": unit["page"],
                    "chunk_index": index,
                }
            )

    return chunks    