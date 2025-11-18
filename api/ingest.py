from typing import Tuple, List, Dict
from sqlalchemy.orm import Session
import httpx

from .chunker import make_chunks
from .store import upsert_document, count_documents  # keep docs table to gate first-boot ingest
from .vectorstore import get_vectorstore
from .pdf_loader import load_pdf_from_bytes


# Seed PDFs we’ll download over HTTP and ingest
SEED_PDFS: List[Dict[str, str]] = [
	{
		"title": "National Risk Register 2025",
		"base_path": "national-risk-register-2025",
		"url": "https://assets.publishing.service.gov.uk/media/67b5f85732b2aab18314bbe4/National_Risk_Register_2025.pdf",
	},
	{
		"title": "The UK Government Resilience Framework",
		"base_path": "uk-government-resilience-framework-2022",
		"url": "https://assets.publishing.service.gov.uk/media/63cff056e90e071ba7b41d54/UKG_Resilience_Framework_FINAL_v2.pdf",
	},
	{
		"title": "British Energy Security Strategy",
		"base_path": "british-energy-security-strategy-2022",
		"url": "https://assets.publishing.service.gov.uk/media/626112c0e90e07168e3fdba3/british-energy-security-strategy-web-accessible.pdf",
	},
	{
		"title": "Government Cyber Security Strategy 2022–2030",
		"base_path": "government-cyber-security-strategy-2022-2030",
		"url": "https://assets.publishing.service.gov.uk/media/61f0169de90e070375c230a8/government-cyber-security-strategy.pdf",
	},
	{
		"title": "Net Zero Strategy: Build Back Greener",
		"base_path": "net-zero-strategy-build-back-greener",
		"url": "https://assets.publishing.service.gov.uk/media/6194dfa4d3bf7f0555071b1b/net-zero-strategy-beis.pdf",
	},
]


async def ingest_seed_docs(db: Session) -> Tuple[int, int]:
	"""
	Download the predefined PDFs over HTTP, chunk them, and ingest into
	the LangChain PGVector store. Also upserts rows in our 'documents'
	table to prevent re-ingest on restarts.

	Returns: (num_docs, num_chunks)
	"""
	total_docs = 0
	total_chunks = 0

	vstore = get_vectorstore()
	timeout = httpx.Timeout(60.0)

	async with httpx.AsyncClient(timeout=timeout) as client:
		for seed in SEED_PDFS:
			title = seed["title"]
			base_path = seed["base_path"]
			url = seed["url"]

			# Fetch the entire PDF into memory
			resp = await client.get(url)
			resp.raise_for_status()
			pdf_bytes = resp.content

			# Turn pages -> paragraphs using our PDF loader
			filename = url.rsplit("/", 1)[-1] or base_path
			doc = load_pdf_from_bytes(name=filename, data=pdf_bytes)
			paragraphs = [p["text"] for p in doc.get("paragraphs", [])]

			if not paragraphs:
				continue

			# Paragraphs -> token-aware chunks
			ordered_chunks = make_chunks(paragraphs)  # List[(ord, text)]
			if not ordered_chunks:
				continue

			texts = [t for (_ord, t) in ordered_chunks]
			metas = [
				{
					"title": title,
					"url": url,
				}
				for (_ord, _t) in ordered_chunks
			]

			# Store in PGVector (embeddings handled by LangChain/OpenAI)
			vstore.add_texts(texts=texts, metadatas=metas)

			# Mark document ingested (for needs_ingest)
			upsert_document(db, title=title, base_path=base_path, url=url)
			db.commit()

			total_docs += 1
			total_chunks += len(texts)

	return total_docs, total_chunks


def needs_ingest(db: Session) -> bool:
	"""
	Gate re-ingest across restarts: if any documents exist,
	we assume seeding has already happened.
	"""
	return count_documents(db) == 0
