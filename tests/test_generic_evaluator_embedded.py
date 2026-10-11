"""Generic scoring gives partial credit for correct JSON wrapped in prose."""

from evolora.evaluation.evaluator import generic_evaluate_sample
from evolora.models.core import EvalSample

SAMPLE = EvalSample(sample_id="s1", prompt="p", expected={"name": "Alex", "total": 245.75})


def test_strict_json_still_scores_full_marks():
    result = generic_evaluate_sample(SAMPLE, '{"name": "Alex", "total": 245.75}')
    assert result.score == 1.0 and result.passed


def test_fenced_json_with_prose_gets_partial_credit():
    response = '```json\n{"name": "Alex", "total": 245.75}\n```\n\nThis summarizes the spending.'
    result = generic_evaluate_sample(SAMPLE, response)
    assert result.details["checks"] == {"name": True, "total": True, "json_only": False}
    assert round(result.score, 3) == round(2 / 3, 3)
    assert not result.passed


def test_prose_without_json_scores_zero():
    result = generic_evaluate_sample(SAMPLE, "The customer is Alex.")
    assert result.score == 0.0
    assert "invalid JSON" in result.details["error"]
