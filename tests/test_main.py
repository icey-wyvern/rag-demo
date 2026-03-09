import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


class MainApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_reports_pending_corpus(self):
        with patch("api.main.SessionLocal") as session_local:
            session = session_local.return_value
            with patch("api.main._get_corpus_state", return_value={"status": "pending", "is_ready": False, "documents_total": 0, "documents_ready": 0, "last_error": None, "ingest_version": 1}):
                resp = self.client.get("/health")

        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["corpus"]["is_ready"])
        session.close.assert_called_once()

    def test_query_rejects_when_corpus_not_ready(self):
        with patch("api.main.SessionLocal") as session_local:
            session = session_local.return_value
            with patch("api.main._get_corpus_state", return_value={"status": "running", "is_ready": False, "documents_total": 5, "documents_ready": 2, "last_error": None, "ingest_version": 1}):
                resp = self.client.post("/api/query", json={"q": "test", "mode": "chunks"})

        self.assertEqual(resp.status_code, 503)
        self.assertIn("still running", resp.json()["detail"])
        session.close.assert_called_once()

    def test_chunks_mode_returns_serialized_results(self):
        docs = [
            {"title": "Doc A", "url": "https://example.com/a.pdf", "page": 4, "content": "Example text."},
        ]
        with patch("api.main.SessionLocal") as session_local:
            session = session_local.return_value
            with patch("api.main._get_corpus_state", return_value={"status": "ready", "is_ready": True, "documents_total": 5, "documents_ready": 5, "last_error": None, "ingest_version": 1}):
                with patch("api.main.similarity_search", return_value=docs):
                    resp = self.client.post("/api/query", json={"q": "test", "mode": "chunks"})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["chunks"][0]["title"], "Doc A")
        self.assertNotIn("content", body["chunks"][0])
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
