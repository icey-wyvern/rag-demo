"""Local PDF parsing for snapshot-bound passages. No fetching or persistence."""

from dataclasses import dataclass

from .chunker import make_chunks
from .pdf_loader import load_pdf_from_bytes


@dataclass(frozen=True)
class ParsedPassage:
    page: int | None
    text: str


def _word_count(text: str) -> int:
    return len(text.split())


def _word_tail(text: str, overlap: int) -> str:
    return " ".join(text.split()[-overlap:]) if overlap > 0 else ""


def parse_pdf_passages(content: bytes) -> list[ParsedPassage]:
    document = load_pdf_from_bytes(name="observed.pdf", data=content)
    pages: dict[int, list[str]] = {}
    for paragraph in document["paragraphs"]:
        page = int(paragraph["page"])
        pages.setdefault(page, []).append(str(paragraph["text"]))

    passages: list[ParsedPassage] = []
    for page, paragraphs in sorted(pages.items()):
        for _, text in make_chunks(paragraphs, token_length=_word_count, tail_text=_word_tail):
            passages.append(ParsedPassage(page=page, text=text))
    return passages
