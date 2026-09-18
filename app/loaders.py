import fitz
from pathlib import Path


def load_txt(file_path: str) -> list[dict]:
    path = Path(file_path)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode '{path.name}' as UTF-8 text."
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Could not read '{path.name}'."
        ) from exc

    text = text.strip()

    if not text:
        raise ValueError(f"'{path.name}' contains no readable text.")

    return [
        {
            "text": text,
            "source": path.name,
            "page": None,
        }
    ]
def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError(
            f"Could not open PDF '{path.name}'."
        ) from exc

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "text": text,
                    "source": path.name,
                    "page": page_number,
                }
            )
    finally:
        document.close()

    if not pages:
        raise ValueError(
            f"'{path.name}' contains no extractable text."
        )

    return pages

def load_document(file_path: str) -> list[dict]:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".pdf":
        return load_pdf(file_path)

    raise ValueError(
        f"Unsupported file type '{extension}'. Only PDF and TXT files are supported."
    )        