import pytest

from app.chunker import chunk_text


def test_short_text_creates_single_chunk():
    text = "NovaCart provides analytics for online retailers."

    chunks = chunk_text(
        text,
        chunk_size=400,
        overlap=60,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_creates_multiple_chunks():
    text = "word " * 300

    chunks = chunk_text(
        text,
        chunk_size=400,
        overlap=60,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.strip()


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            chunk_size=0,
            overlap=0,
        )


def test_overlap_cannot_equal_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            chunk_size=100,
            overlap=100,
        )