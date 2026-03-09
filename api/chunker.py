from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import tiktoken


@lru_cache(maxsize=1)
def _get_encoder():
	try:
		return tiktoken.get_encoding("cl100k_base")
	except Exception:
		return None


def _tok_len(text: str) -> int:
	enc = _get_encoder()
	if enc is None:
		return len((text or "").split())
	return len(enc.encode(text or ""))


def _tail_text(text: str, overlap_tokens: int) -> str:
	if overlap_tokens <= 0 or not text:
		return ""

	enc = _get_encoder()
	if enc is None:
		words = text.split()
		return " ".join(words[-overlap_tokens:])
	return enc.decode(enc.encode(text)[-overlap_tokens:])


def make_chunks(paragraphs: List[str], max_tokens: int = 350, overlap_tokens: int = 40) -> List[Tuple[int, str]]:
	chunks: List[Tuple[int, str]] = []
	if not paragraphs:
		return chunks

	buffer: List[str] = []
	buffer_tokens = 0
	index = 0

	for paragraph in paragraphs:
		if not paragraph:
			continue

		paragraph_tokens = _tok_len(paragraph)
		if paragraph_tokens > max_tokens:
			words = paragraph.split()
			current_words: List[str] = []
			current_tokens = 0
			for word in words:
				word_tokens = _tok_len((" " if current_words else "") + word)
				if current_tokens + word_tokens > max_tokens:
					if current_words:
						chunks.append((index, " ".join(current_words)))
						index += 1
					if overlap_tokens and chunks:
						tail = _tail_text(chunks[-1][1], overlap_tokens)
						current_words = [tail]
						current_tokens = _tok_len(tail)
					else:
						current_words = []
						current_tokens = 0
				if word:
					current_words.append(word)
					current_tokens += word_tokens
			if current_words:
				chunks.append((index, " ".join(current_words)))
				index += 1
			continue

		if buffer and (buffer_tokens + 1 + paragraph_tokens) > max_tokens:
			chunks.append((index, "\n\n".join(buffer)))
			index += 1
			if overlap_tokens:
				tail = _tail_text(chunks[-1][1], overlap_tokens)
				buffer = [tail] if tail else []
				buffer_tokens = _tok_len(tail) if tail else 0
			else:
				buffer = []
				buffer_tokens = 0

		if buffer:
			buffer.append(paragraph)
			buffer_tokens += 1 + paragraph_tokens
		else:
			buffer = [paragraph]
			buffer_tokens = paragraph_tokens

	if buffer:
		chunks.append((index, "\n\n".join(buffer)))

	return chunks
