from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

DOCS_ROOT = Path("/app/docs")


def _simple_title(name: str) -> str:
	return re.sub(r"[_\-]+", " ", Path(name).stem).strip().title()


def _split_paragraphs(text: str) -> List[str]:
	if not text:
		return []

	chunks = re.split(r"\n\s*\n+", text)
	paragraphs: List[str] = []
	for chunk in chunks:
		cleaned = chunk.strip()
		if not cleaned:
			continue

		cleaned = re.sub(r"[ \t]+", " ", cleaned)
		cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
		if len(cleaned) >= 20:
			paragraphs.append(cleaned)

	return paragraphs


def list_pdf_paths(root: Path = DOCS_ROOT) -> List[Path]:
	if not root.exists():
		return []
	return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def load_pdf(path: Path) -> Dict:
	reader = PdfReader(str(path))
	paragraphs: List[Dict] = []
	for page_num, page in enumerate(reader.pages, start=1):
		for paragraph in _split_paragraphs(page.extract_text() or ""):
			paragraphs.append({"page": page_num, "text": paragraph})

	return {"title": _simple_title(path.name), "path": path.name, "paragraphs": paragraphs}


def load_pdf_from_bytes(name: str, data: bytes) -> Dict:
	reader = PdfReader(BytesIO(data))
	paragraphs: List[Dict] = []
	for page_num, page in enumerate(reader.pages, start=1):
		for paragraph in _split_paragraphs(page.extract_text() or ""):
			paragraphs.append({"page": page_num, "text": paragraph})

	return {"title": _simple_title(name), "path": name, "paragraphs": paragraphs}


def load_all_pdfs(root: Path = DOCS_ROOT) -> List[Dict]:
	return [load_pdf(path) for path in list_pdf_paths(root)]
