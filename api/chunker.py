import tiktoken
from typing import List, Tuple

# Use OpenAI's tokenizer to size chunks for embeddings/generation
_enc = tiktoken.get_encoding("cl100k_base")

def _tok_len(text: str) -> int:
	return len(_enc.encode(text or ""))

def make_chunks(paragraphs: List[str], max_tokens: int = 350, overlap_tokens: int = 40) -> List[Tuple[int, str]]:
	"""
	Pack cleaned paragraphs into ~max_tokens chunks with small token overlap.
	Returns a list of (ord, chunk_text).
	"""
	chunks: List[Tuple[int, str]] = []
	if not paragraphs:
		return chunks

	buf: List[str] = []
	buf_tokens = 0
	ord_idx = 0

	for para in paragraphs:
		if not para:
			continue
		ptok = _tok_len(para)
		# If single paragraph is huge, hard-split it.
		if ptok > max_tokens:
			words = para.split()
			cur: List[str] = []
			cur_tok = 0
			for w in words:
				wt = _tok_len((" " if cur else "") + w)
				if cur_tok + wt > max_tokens:
					if cur:
						chunks.append((ord_idx, " ".join(cur)))
						ord_idx += 1
					# start next segment with overlap from end of previous
					if overlap_tokens and chunks:
						tail = _enc.decode(_enc.encode(chunks[-1][1])[-overlap_tokens:])
						cur = [tail]
						cur_tok = _tok_len(tail)
					else:
						cur, cur_tok = [], 0
				if w:
					cur.append(w)
					cur_tok += wt
			if cur:
				chunks.append((ord_idx, " ".join(cur)))
				ord_idx += 1
			continue

		# Normal packing: append para if it fits, else flush buffer.
		if buf and (buf_tokens + 1 + ptok) > max_tokens:
			chunks.append((ord_idx, "\n\n".join(buf)))
			ord_idx += 1
			# start new buffer with an overlap tail
			if overlap_tokens:
				tail = _enc.decode(_enc.encode(chunks[-1][1])[-overlap_tokens:])
				buf = [tail] if tail else []
				buf_tokens = _tok_len(tail) if tail else 0
			else:
				buf, buf_tokens = [], 0

		# add paragraph to buffer
		if buf:
			buf.append(para)
			buf_tokens += 1 + ptok  # +1 approx for the joiner/newlines
		else:
			buf = [para]
			buf_tokens = ptok

	# flush remainder
	if buf:
		chunks.append((ord_idx, "\n\n".join(buf)))

	return chunks
