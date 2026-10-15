"""Offline tests of the internal evidence-to-answer boundary."""

import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
from uuid import UUID

from api.evidence_answer import (
    AnswerProviderError, EvidenceFirstAnswerService, InvalidAnswerStructureError,
    MalformedProviderOutputError, MissingCitationError, QuoteMismatchError,
    StructuredAnswer, UnknownEvidenceIdError, assign_evidence_ids, format_evidence,
)
from api.openai_answer_provider import OpenAIAnswerProvider
from api.versioned_answer import VersionedAnswerService
from tests.test_versioned_retrieval import T1, evidence


def response(status="answered", claims=None, reason=None):
    return {
        "status": status,
        "claims": [] if claims is None else claims,
        "abstention_reason": reason,
    }


def claim(text="A fact.", citations=None):
    return {"text": text, "citations": [] if citations is None else citations}


def cite(evidence_id="E1", quote="service users"):
    return {"evidence_id": evidence_id, "supporting_quote": quote}


class FakeProvider:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate(self, question, formatted_evidence, instructions):
        self.calls.append((question, formatted_evidence, instructions))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class EvidenceAnswerTests(unittest.TestCase):
    def setUp(self):
        self.first = evidence(text="Service users must be involved. Feedback should be reviewed.")
        self.second = replace(
            evidence(text="Teams should publish accessibility findings."),
            source_id=UUID(int=9), snapshot_id=UUID(int=10), passage_id=UUID(int=11),
            source_title="Accessibility guidance", source_key="govuk:accessibility",
            canonical_url="https://www.gov.uk/accessibility", rank=2,
        )

    def answer(self, output, items=None):
        provider = FakeProvider(output)
        result = EvidenceFirstAnswerService(provider).answer("What should teams do?", [self.first, self.second] if items is None else items)
        return result, provider

    def test_multiple_claims_are_validated_and_answer_text_is_assembled(self):
        output = response(claims=[
            claim("Involve users.", [cite(quote="Service users must be involved.")]),
            claim("Publish findings.", [cite("E2", "publish accessibility findings")]),
        ])
        result, provider = self.answer(output)
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer_text, "Involve users.\n\nPublish findings.")
        self.assertEqual(result.claims[0].citations[0].evidence, self.first)
        self.assertEqual(result.claims[1].citations[0].evidence.snapshot_id, self.second.snapshot_id)
        self.assertEqual(result.claims[1].citations[0].evidence.canonical_url, self.second.canonical_url)
        self.assertIn("untrusted", provider.calls[0][2].lower())

    def test_zero_evidence_abstains_without_provider_call(self):
        provider = FakeProvider(RuntimeError("must not be called"))
        result = EvidenceFirstAnswerService(provider).answer("Question", [])
        self.assertEqual(result.status, "abstained")
        self.assertIn("Insufficient retrieved evidence", result.abstention_reason)
        self.assertEqual(result.answer_text, "")
        self.assertEqual(provider.calls, [])

    def test_blank_passage_is_not_usable_evidence(self):
        provider = FakeProvider(RuntimeError("must not be called"))
        result = EvidenceFirstAnswerService(provider).answer("Question", [replace(self.first, text="  ")])
        self.assertEqual(result.status, "abstained")
        self.assertEqual(provider.calls, [])

    def test_model_requested_abstention(self):
        result, _ = self.answer(response("abstained", reason="The passages do not address this."))
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.claims, ())
        self.assertEqual(result.abstention_reason, "The passages do not address this.")

    def test_fabricated_evidence_id_rejects_entire_response(self):
        with self.assertRaises(UnknownEvidenceIdError):
            self.answer(response(claims=[claim(citations=[cite("E1", "Service users")]), claim(citations=[cite("E99")])]))

    def test_missing_citation(self):
        with self.assertRaises(MissingCitationError):
            self.answer(response(claims=[claim()]))

    def test_fabricated_and_modified_quotes(self):
        for quote in ("This quotation was invented.", "Service user must be involved.", "  "):
            with self.subTest(quote=quote), self.assertRaises(QuoteMismatchError):
                self.answer(response(claims=[claim(citations=[cite(quote=quote)])]))

    def test_multiple_citations_resolve_to_distinct_evidence(self):
        result, _ = self.answer(response(claims=[claim(citations=[
            cite("E1", "Feedback should be reviewed."),
            cite("E2", "Teams should publish accessibility findings."),
        ])]))
        self.assertEqual([item.evidence.passage_id for item in result.claims[0].citations],
                         [self.first.passage_id, self.second.passage_id])

    def test_identical_duplicate_citations_are_deduplicated_in_first_seen_order(self):
        result, _ = self.answer(response(claims=[claim(citations=[
            cite("E1", "Feedback"), cite("E1", "Feedback"), cite("E2", "Teams"),
        ])]))
        self.assertEqual([item.evidence_id for item in result.claims[0].citations], ["E1", "E2"])

    def test_malformed_and_extra_unconstrained_answer_rejected(self):
        for output in ("plain prose", {"status": "answered"},
                       response(claims=[claim(citations=[cite()])]) | {"answer": "Unsupported prose"}):
            with self.subTest(output=output), self.assertRaises(MalformedProviderOutputError):
                self.answer(output)

    def test_contradictory_status_and_empty_claims_rejected(self):
        outputs = [
            response("abstained", [claim(citations=[cite(quote="Service users")])], "No answer"),
            response("abstained", reason=" "),
            response("answered", []),
            response("answered", [claim(citations=[cite(quote="Service users")])], "Also abstained"),
            response("answered", [claim(" ", [cite(quote="Service users")])]),
        ]
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(InvalidAnswerStructureError):
                self.answer(output)

    def test_provider_failure_is_domain_error_without_fabricated_result(self):
        with self.assertRaisesRegex(AnswerProviderError, "provider failed"):
            self.answer(RuntimeError("private provider internals"))

    def test_evidence_ids_and_format_are_stable(self):
        mapped = assign_evidence_ids([self.first, self.second])
        self.assertEqual(list(mapped), ["E1", "E2"])
        self.assertEqual(mapped["E1"], self.first)
        first_render = format_evidence(mapped)
        self.assertEqual(first_render, format_evidence(mapped))
        blocks = json.loads(first_render)
        self.assertEqual([block["evidence_id"] for block in blocks], ["E1", "E2"])
        self.assertEqual(blocks[0]["snapshot_observed_at"], T1.isoformat())
        self.assertEqual(blocks[0]["canonical_url"], self.first.canonical_url)
        self.assertEqual(blocks[0]["passage_text"], self.first.text)

    def test_internal_wrapper_forwards_retrieval_arguments(self):
        retriever = Mock()
        retriever.search.return_value = [self.first]
        service = VersionedAnswerService(retriever, EvidenceFirstAnswerService(FakeProvider(
            response(claims=[claim(citations=[cite(quote="Service users")])])
        )))
        result = service.answer("Question", mode="hybrid", as_of=T1, k=3)
        retriever.search.assert_called_once_with("Question", mode="hybrid", as_of=T1, k=3)
        self.assertEqual(result.status, "answered")

    def test_openai_adapter_uses_structured_output_without_network(self):
        parsed = StructuredAnswer.model_validate(response("abstained", reason="Insufficient."))
        fake_llm = Mock()
        fake_llm.with_structured_output.return_value.invoke.return_value = parsed
        adapter = OpenAIAnswerProvider(model="test-model", api_key="test-placeholder", timeout_seconds=12)
        with patch("langchain_openai.ChatOpenAI", return_value=fake_llm) as constructor:
            self.assertEqual(adapter.generate("Question", "[]", "Instructions"), parsed)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 12)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 2)
        fake_llm.with_structured_output.assert_called_once_with(StructuredAnswer, method="json_schema", strict=True)


if __name__ == "__main__":
    unittest.main()
