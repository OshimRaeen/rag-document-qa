from pathlib import Path

import pytest

from app.loaders import load_document


def test_load_txt(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "This is a test document.",
        encoding="utf-8",
    )

    units = load_document(str(file_path))

    assert len(units) == 1
    assert units[0]["text"] == "This is a test document."
    assert units[0]["source"] == "sample.txt"
    assert units[0]["page"] is None


def test_empty_txt_is_rejected(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(str(file_path))


def test_unsupported_file_type(tmp_path: Path):
    file_path = tmp_path / "document.csv"
    file_path.write_text(
        "name,value",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_document(str(file_path))