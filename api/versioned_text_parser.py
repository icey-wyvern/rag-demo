"""Deterministic local parser for normalized UTF-8 guidance text."""

from .chunker import make_chunks
from .versioned_parser import ParsedPassage, _word_count, _word_tail


def parse_guidance_passages(content: bytes) -> list[ParsedPassage]:
    text = content.decode("utf-8")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return [
        ParsedPassage(page=None, text=chunk)
        for _, chunk in make_chunks(paragraphs, token_length=_word_count, tail_text=_word_tail)
    ]
