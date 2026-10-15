"""Deterministic generation/evidence metrics over real answer services."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from api.generation_eval import (
    EvidenceRequirement, evaluate_cases, load_cases,
)


def answer(text="Service users must be involved.", evidence_id="E1", quote="Service users must be involved."):
    return {"status": "answered", "claims": [{"text": text, "citations": [
        {"evidence_id": evidence_id, "supporting_quote": quote},
    ]}], "abstention_reason": None}


class GenerationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.by_id = {case.case_id: case for case in cls.cases}

    def test_fixture_report_separates_valid_citations_from_expected_evidence(self):
        report = evaluate_cases(self.cases)
        self.assertEqual((report.cases, report.passed, report.failed), (15, 13, 2))
        self.assertEqual(report.failed_case_ids, ("wrong_source", "wrong_snapshot"))
        self.assertEqual(report.answer_status_accuracy, 1.0)
        self.assertEqual(report.abstention_accuracy, 1.0)
        self.assertEqual(report.correct_abstentions, 2)
        self.assertEqual(report.citation_coverage, 1.0)
        self.assertEqual(report.citation_validity, 1.0)
        self.assertEqual(report.expected_evidence_accuracy, 0.75)
        self.assertEqual(report.temporal_evidence_accuracy, 0.8)
        self.assertAlmostEqual(report.contract_pass_rate, 13 / 15)
        observations = {item.case_id: item for item in report.observations}
        self.assertEqual(observations["wrong_source"].diagnostics, ("evidence_selection_failure",))
        self.assertEqual(observations["wrong_snapshot"].diagnostics, ("temporal_version_failure",))
        self.assertEqual(observations["wrong_source"].observed_status, "answered")
        self.assertEqual(observations["wrong_snapshot"].temporal_match, False)
        self.assertNotIn("retrieval_failure", report.diagnostic_counts)

    def test_perfect_answer_and_multi_claim_require_all_evidence(self):
        perfect = evaluate_cases((self.by_id["answerable"], self.by_id["multiple_evidence"]))
        self.assertEqual(perfect.contract_pass_rate, 1.0)
        self.assertEqual(perfect.citation_coverage, 1.0)
        self.assertEqual(perfect.citation_validity, 1.0)
        self.assertEqual(perfect.expected_evidence_accuracy, 1.0)
        missing_second = replace(self.by_id["multiple_evidence"], provider_response=answer())
        result = evaluate_cases((missing_second,))
        self.assertEqual(result.expected_evidence_accuracy, 0.0)
        self.assertEqual(result.citation_validity, 1.0)
        self.assertEqual(result.failed_case_ids, ("multiple_evidence",))
        any_one = replace(missing_second, requirement_mode="any")
        self.assertEqual(evaluate_cases((any_one,)).expected_evidence_accuracy, 1.0)

    def test_correct_and_incorrect_abstention_and_incorrect_answer(self):
        correct = evaluate_cases((self.by_id["insufficient_evidence"], self.by_id["no_evidence"]))
        self.assertEqual((correct.correct_abstentions, correct.incorrect_abstentions,
                          correct.incorrect_answers), (2, 0, 0))
        self.assertEqual(correct.observations[1].provider_calls, 0)
        wrong_abstention = replace(self.by_id["answerable"], provider_response={
            "status": "abstained", "claims": [], "abstention_reason": "Insufficient evidence."})
        report = evaluate_cases((wrong_abstention,))
        self.assertEqual((report.incorrect_abstentions, report.abstention_accuracy), (1, 0.0))
        self.assertEqual(report.observations[0].diagnostics, ("generation_status_failure",))
        wrong_answer = replace(self.by_id["insufficient_evidence"], provider_response=answer())
        report = evaluate_cases((wrong_answer,))
        self.assertEqual((report.incorrect_answers, report.abstention_accuracy), (1, 0.0))

    def test_expected_and_unexpected_rejections_are_distinct(self):
        expected = evaluate_cases((self.by_id["fabricated_citation"], self.by_id["invalid_quote"],
                                   self.by_id["missing_citation"]))
        self.assertEqual(expected.passed, 3)
        self.assertEqual(expected.rejection_categories, {
            "unknown_evidence": 1, "invalid_quote": 1, "missing_citation": 1})
        self.assertEqual(expected.diagnostic_counts["expected_rejection"], 3)
        unexpected = replace(self.by_id["answerable"], provider_response=answer(evidence_id="E99"))
        report = evaluate_cases((unexpected,))
        self.assertEqual(report.failed_case_ids, ("answerable",))
        self.assertEqual(report.observations[0].diagnostics, ("unexpected_rejection",))
        self.assertEqual(report.observations[0].rejection_category, "unknown_evidence")

    def test_temporal_and_change_side_requirements(self):
        report = evaluate_cases((self.by_id["temporal_older"], self.by_id["version_modified"],
                                 self.by_id["version_added"], self.by_id["version_removed"]))
        self.assertEqual(report.temporal_evidence_accuracy, 1.0)
        self.assertEqual(report.expected_evidence_accuracy, 1.0)
        one_sided = replace(self.by_id["version_modified"], expected_status="rejected",
                            expected_category="invalid_structure", requirements=(), expected_snapshot_ids=(),
                            provider_response=answer(quote="Service users must be involved."))
        rejected = evaluate_cases((one_sided,))
        self.assertEqual(rejected.passed, 1)
        self.assertEqual(rejected.observations[0].rejection_category, "invalid_structure")
        future = replace(self.by_id["wrong_snapshot"], requirements=(),
                         expected_snapshot_ids=(self.by_id["wrong_snapshot"].evidence[1].snapshot_id,))
        self.assertEqual(evaluate_cases((future,)).temporal_evidence_accuracy, 0.0)

    def test_provenance_requirement_fields_and_empty_denominators(self):
        item = self.by_id["answerable"].evidence[0]
        self.assertTrue(EvidenceRequirement(source_key=item.source_key).matches(item))
        self.assertTrue(EvidenceRequirement(snapshot_id=item.snapshot_id,
                                            snapshot_hash=item.snapshot_content_sha256).matches(item))
        self.assertTrue(EvidenceRequirement(passage_id=item.passage_id,
                                            passage_hash=item.text_sha256).matches(item))
        self.assertFalse(EvidenceRequirement(source_key="synthetic:other").matches(item))
        empty = evaluate_cases(())
        self.assertEqual((empty.cases, empty.passed, empty.failed), (0, 0, 0))
        for metric in (empty.answer_status_accuracy, empty.abstention_accuracy,
                       empty.citation_coverage, empty.citation_validity,
                       empty.expected_evidence_accuracy, empty.temporal_evidence_accuracy,
                       empty.contract_pass_rate):
            self.assertIsNone(metric)

    def test_malformed_dataset_fails_clearly(self):
        original = json.loads((Path(__file__).resolve().parent.parent / "eval" / "generation_cases.json")
                              .read_text(encoding="utf-8"))
        variants = []
        duplicate = json.loads(json.dumps(original))
        duplicate["cases"].append(dict(duplicate["cases"][0]))
        variants.append((duplicate, "duplicate generation case ID"))
        unknown = json.loads(json.dumps(original))
        unknown["cases"][0]["evidence"] = ["missing"]
        variants.append((unknown, "unknown passage reference"))
        invalid = json.loads(json.dumps(original))
        invalid["cases"][0]["expected_status"] = "maybe"
        variants.append((invalid, "invalid expected_status"))
        naive = json.loads(json.dumps(original))
        naive["cases"][4]["as_of"] = "2026-10-15T12:00:00"
        variants.append((naive, "timezone-aware"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            file = Path(folder) / "cases.json"
            for data, message in variants:
                file.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_cases(file)


if __name__ == "__main__":
    unittest.main()
