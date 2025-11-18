from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
import re


DOCS_ROOT = Path("/app/docs")


def _simple_title(name: str) -> str:
	"""Turn 'highway-code.pdf' into 'Highway Code'."""
	base = Path(name).stem
	title = re.sub(r"[_\-]+", " ", base).strip()
	if not title:
		return "Untitled"
	return title[:1].upper() + title[1:]


def _split_paragraphs(text: str) -> List[str]:
	"""
	Convert a page's raw text into a list of paragraphs.

	Strategy:
	- split on blank lines
	- normalise whitespace
	- drop very short fragments
	"""
	if not text:
		return []

	# split on blank lines
	chunks = re.split(r"\n\s*\n+", text)

	out: List[str] = []
	for chunk in chunks:
		# collapse "wrapped" lines where pages have hard line breaks
		s = chunk.strip()
		if not s:
			continue

		# Replace multiple spaces/tabs with single space
		s = re.sub(r"[ \t]+", " ", s)
		# Normalise stray internal newlines (keep paragraph feel but avoid crazy breaks)
		s = re.sub(r"\s*\n\s*", " ", s)

		if len(s) < 20:
			# probably a header/page number etc.
			continue

		out.append(s)

	return out


def list_pdf_paths(root: Path = DOCS_ROOT) -> List[Path]:
	"""Return all PDF paths under /app/docs (non-recursive)."""
	if not root.exists():
		return []
	return sorted(
		[p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
	)


def load_pdf(path: Path) -> Dict:
	"""
	Read a single PDF from disk and return a dict:

	{
		"title": str,
		"path": str,                 # basename, e.g. 'highway-code.pdf'
		"paragraphs": List[Dict],    # [{ "page": int, "text": str }, ...]
	}
	"""
	reader = PdfReader(str(path))
	all_paras: List[Dict] = []

	for i, page in enumerate(reader.pages, start=1):
		raw = page.extract_text() or ""
		for para in _split_paragraphs(raw):
			all_paras.append({
				"page": i,
				"text": para,
			})

	return {
		"title": _simple_title(path.name),
		"path": path.name,
		"paragraphs": all_paras,
	}


def load_pdf_from_bytes(name: str, data: bytes) -> Dict:
	"""
	Same shape as load_pdf(), but reads from an in-memory bytes object
	instead of a file on disk. This is what we'll use for HTTP-fetched PDFs.

	{
		"title": str,
		"path": str,                 # logical name, e.g. 'National_Risk_Register_2025.pdf'
		"paragraphs": List[Dict],    # [{ "page": int, "text": str }, ...]
	}
	"""
	reader = PdfReader(BytesIO(data))
	all_paras: List[Dict] = []

	for i, page in enumerate(reader.pages, start=1):
		raw = page.extract_text() or ""
		for para in _split_paragraphs(raw):
			all_paras.append({
				"page": i,
				"text": para,
			})

	return {
		"title": _simple_title(name),
		"path": name,
		"paragraphs": all_paras,
	}


def load_all_pdfs(root: Path = DOCS_ROOT) -> List[Dict]:
	"""Load every PDF in /app/docs and return a list of doc dicts (see load_pdf)."""
	docs: List[Dict] = []
	for p in list_pdf_paths(root):
		docs.append(load_pdf(p))
	return docs
