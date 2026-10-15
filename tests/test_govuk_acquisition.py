"""Offline contract tests for the GOV.UK acquisition boundary."""

import copy
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

from api.govuk_acquisition import (
    AcquiredGovUkContent,
    GovUkAcquisitionError,
    GovUkClient,
    GovUkNormalizer,
    USER_AGENT,
)
from api.govuk_sources import SERVICE_STANDARD_PATHS
from api.versioned_ingest import VersionedIngestionService
from api.versioned_text_parser import parse_guidance_passages
from tests.test_versioned_ingest import MemorySessionFactory, MemoryStore


PATH = SERVICE_STANDARD_PATHS[1]
OBSERVED_AT = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / f"govuk_service_standard_observation_{name}.json").read_text(encoding="utf-8"))


def acquired(payload):
    return AcquiredGovUkContent(PATH, f"https://www.gov.uk/api/content{PATH}", payload, OBSERVED_AT)


class GovUkNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = GovUkNormalizer()

    def test_identity_urls_and_source_timestamps(self):
        result = self.normalizer.normalize(acquired(fixture("a")))
        self.assertEqual(result.content_id, "2b578bcd-775b-45d0-8198-e78507594b9c")
        self.assertEqual(result.base_path, PATH)
        self.assertEqual(result.schema_name, "service_manual_guide")
        self.assertEqual(result.document_type, "service_manual_guide")
        self.assertEqual(result.observed.source_key, f"govuk:{result.content_id}")
        self.assertEqual(result.observed.canonical_url, f"https://www.gov.uk{PATH}")
        self.assertEqual(result.observed.retrieved_url, f"https://www.gov.uk/api/content{PATH}")
        self.assertEqual(result.observed.title, "2. Solve a whole problem for users")
        self.assertEqual(result.observed.observed_at, OBSERVED_AT)
        self.assertEqual(result.observed.source_published_at, datetime.fromisoformat("2019-05-08T09:29:12+01:00"))
        self.assertEqual(result.observed.source_updated_at, datetime.fromisoformat("2026-01-29T13:48:49+00:00"))
        self.assertIsNone(result.observed.effective_from)

    def test_content_is_deterministic_guidance_text_without_navigation_metadata(self):
        payload = fixture("a")
        payload["links"] = {"organisations": [{"title": "navigation sentinel"}]}
        first = self.normalizer.normalize(acquired(payload)).observed.content
        second = self.normalizer.normalize(acquired(payload)).observed.content
        self.assertEqual(first, second)
        text = first.decode("utf-8")
        self.assertIn("Why it’s important\n\nServices that do not work well", text)
        self.assertIn("What it means\n\nService teams should:", text)
        self.assertNotIn("navigation sentinel", text)
        self.assertEqual(len(parse_guidance_passages(first)), 1)

    def test_absent_timestamp_remains_absent_and_bad_input_fails_clearly(self):
        payload = fixture("a")
        del payload["first_published_at"]
        del payload["public_updated_at"]
        observed = self.normalizer.normalize(acquired(payload)).observed
        self.assertIsNone(observed.source_published_at)
        self.assertIsNone(observed.source_updated_at)
        for key, value, message in [
            ("content_id", "bad", "content_id"),
            ("schema_name", "unsupported", "Unsupported"),
            ("base_path", "/wrong-path", "differs"),
            ("details", {}, "body"),
            ("public_updated_at", "yesterday", "timestamp"),
        ]:
            bad = copy.deepcopy(payload)
            bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(GovUkAcquisitionError, message):
                self.normalizer.normalize(acquired(bad))

    def test_two_fixture_observations_preserve_predecessor_and_repeat_is_no_op(self):
        sessions = MemorySessionFactory()
        service = VersionedIngestionService(sessions, parser=parse_guidance_passages, store_factory=MemoryStore)
        first_observed = self.normalizer.normalize(acquired(fixture("a"))).observed
        second_observed = self.normalizer.normalize(acquired(fixture("b"))).observed
        first = service.ingest(first_observed)
        repeat = service.ingest(first_observed)
        second = service.ingest(second_observed)
        self.assertEqual((first.status, repeat.status, second.status), ("new_source", "unchanged", "new_snapshot"))
        self.assertEqual((first.source_id, repeat.source_id, second.source_id), (first.source_id,) * 3)
        self.assertEqual(repeat.snapshot_id, first.snapshot_id)
        self.assertNotEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(len(sessions.state.sources), 1)
        self.assertEqual(len(sessions.state.snapshots), 2)
        self.assertEqual(sessions.state.snapshots[1].predecessor_snapshot_id, first.snapshot_id)
        first_passages = [p for p in sessions.state.passages if p.snapshot_id == first.snapshot_id]
        second_passages = [p for p in sessions.state.passages if p.snapshot_id == second.snapshot_id]
        self.assertEqual(len(first_passages), first.passage_count)
        self.assertEqual(len(second_passages), second.passage_count)
        self.assertNotEqual(first_passages[0].text, second_passages[0].text)
        self.assertIn("legislative constraints", first_passages[0].text)
        self.assertEqual(sessions.commits, 3)


class GovUkClientTests(unittest.TestCase):
    def test_request_construction_timeout_and_bounded_retry(self):
        seen = []
        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=fixture("a"))
        client = GovUkClient(transport=httpx.MockTransport(handler), timeout_seconds=4, clock=lambda: OBSERVED_AT)
        result = client.fetch(PATH)
        self.assertEqual(len(seen), 2)
        self.assertEqual(str(seen[0].url), f"https://www.gov.uk/api/content{PATH}")
        self.assertEqual(seen[0].headers["user-agent"], USER_AGENT)
        self.assertEqual(seen[0].headers["accept"], "application/json")
        self.assertEqual(seen[0].extensions["timeout"]["read"], 4)
        self.assertEqual(result.observed_at, OBSERVED_AT)
        self.assertEqual(result.payload["content_id"], fixture("a")["content_id"])

    def test_error_responses_bad_json_and_oversize_are_rejected(self):
        for response, message in [
            (httpx.Response(404), "HTTP 404"),
            (httpx.Response(200, text="{bad"), "malformed JSON"),
            (httpx.Response(200, content=b"x" * 101), "size limit"),
        ]:
            with self.subTest(message=message):
                client = GovUkClient(transport=httpx.MockTransport(lambda _: response), max_response_bytes=100)
                with self.assertRaisesRegex(GovUkAcquisitionError, message):
                    client.fetch(PATH)

    def test_non_transient_failure_is_not_retried_and_path_is_validated(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(410)
        client = GovUkClient(transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(GovUkAcquisitionError, "HTTP 410"):
            client.fetch(PATH)
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(GovUkAcquisitionError, "base path"):
            client.fetch("https://example.com/bad")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
