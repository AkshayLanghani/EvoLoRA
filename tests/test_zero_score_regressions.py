"""Distinguish evaluator outages from model failures and keep retry feedback useful."""

import json

import pytest

from evolora.agent.planner import MiniMaxPlanner
from evolora.evaluation.evaluator import GenericEvaluator, evaluate_sample, generic_evaluate_sample
from evolora.evaluation.llm_judge import LLMJudgeEvaluator
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import ArtifactMeta, EvalSample, RunConfig, RunStatus
from evolora.orchestration.orchestrator import Orchestrator


def make_orchestrator(judge):
    sample = EvalSample(sample_id="locked-1", prompt="Extract the number 7.", expected={"n": 7})
    orch = Orchestrator(RunConfig(), LockedEvalSet([sample]), llm_judge=judge)
    orch._evaluator = GenericEvaluator()
    return orch


class BrokenJudge(LLMJudgeEvaluator):
    def __init__(self):
        super().__init__(api_key="test")

    def _make_client(self):
        return None

    async def _score(self, client, record):
        raise RuntimeError("judge offline")


@pytest.mark.asyncio
@pytest.mark.parametrize("judge", [LLMJudgeEvaluator(api_key=""), BrokenJudge()])
async def test_judge_outage_uses_locked_answers_not_remote_expected(judge):
    orch = make_orchestrator(judge)
    score, results, responses = await orch._judge_remote_evals([
        {"input": "Extract the number 7.", "expected": '{"n":999}', "actual": '{"n":7}'},
    ])
    assert score == 1.0
    assert results[0].details["scoring_source"] == "objective fallback"
    assert responses == {"locked-1": '{"n":7}'}


@pytest.mark.asyncio
async def test_partial_judge_outage_does_not_drop_ungraded_samples():
    class PartialJudge:
        async def judge(self, records):
            return 1.0, [{**records[0], "score": 10}, {**records[1], "score": None}]

    orch = make_orchestrator(PartialJudge())
    orch._eval_set = LockedEvalSet([
        EvalSample(sample_id="a", prompt="p1", expected={"n": 7}),
        EvalSample(sample_id="b", prompt="p2", expected={"n": 8}),
    ])
    score, results, _ = await orch._judge_remote_evals([
        {"input": "p1", "actual": '{"n":7}'}, {"input": "p2", "actual": '{"n":0}'},
    ])
    assert score == 0.5 and len(results) == 2


@pytest.mark.asyncio
async def test_unmatched_remote_records_are_an_error():
    orch = make_orchestrator(LLMJudgeEvaluator(api_key=""))
    with pytest.raises(RuntimeError, match="locked prompts"):
        await orch._judge_remote_evals([{"input": "stale prompt", "actual": '{"n":7}'}])
    with pytest.raises(RuntimeError, match="no records"):
        await orch._judge_remote_evals([])


@pytest.mark.asyncio
async def test_missing_runner_outputs_fail_instead_of_scoring_zero():
    class MissingRunner:
        async def run_batch(self, prompts, adapter_path=None):
            return {}

    orch = make_orchestrator(None)
    orch._runner = MissingRunner()
    with pytest.raises(RuntimeError, match="omitted evaluation outputs"):
        await orch._eval(None)


@pytest.mark.asyncio
async def test_baseline_failures_include_output_and_feed_first_plan():
    class WrongRunner:
        async def run_batch(self, prompts, adapter_path=None):
            return {"locked-1": "invalid JSON"}

    orch = make_orchestrator(None)
    orch._runner = WrongRunner()
    _, orch._baseline_results, _ = await orch._eval(None)
    failures = orch._get_failures(orch._record)
    assert failures[0].details["actual"] == "invalid JSON"
    assert failures[0].details["input"] == "Extract the number 7."
    assert "expected" not in failures[0].details


def test_nested_json_uses_numeric_tolerance_and_case_normalization():
    sample = EvalSample(sample_id="s", prompt="p", expected={"items": [{"total": 100, "name": "Alex"}]})
    assert generic_evaluate_sample(sample, '{"items":[{"total":100.005,"name":"ALEX"}]}').passed


@pytest.mark.parametrize("got", ["false", 0, None])
def test_boolean_fields_require_boolean_values(got):
    sample = EvalSample(sample_id="s", prompt="p", expected={"flag": False})
    assert generic_evaluate_sample(sample, json.dumps({"flag": got})).score == 0.0


def test_missing_null_field_does_not_match():
    sample = EvalSample(sample_id="s", prompt="p", expected={"value": None})
    assert generic_evaluate_sample(sample, '{}').score == 0.0


@pytest.mark.parametrize("response", ['[]', 'null', '42'])
def test_demo_evaluator_handles_non_object_json(response):
    sample = EvalSample(sample_id="s", prompt="p", expected={})
    result = evaluate_sample(sample, response)
    assert result.score == 0.0 and "not a JSON object" in result.details["error"]


@pytest.mark.asyncio
async def test_remote_loop_can_improve_without_a_judge_service():
    class Backend:
        is_mock = False
        name = "remote"

        async def evaluate_base(self, payload):
            async def stream():
                yield {"done": True, "eval_records": [
                    {"input": "Extract the number 7.", "actual": '{"n":0}'},
                ]}
            return stream()

        async def train(self, run_id, iteration, plan, base_model_id, remote_payload=None):
            async def stream():
                yield {"done": True, "artifact": ArtifactMeta(
                    run_id=run_id, iteration=iteration, adapter_path="adapter",
                    checksum="test", score=0.0, is_mock=False,
                ), "eval_records": [
                    {"input": "Extract the number 7.", "actual": '{"n":7}'},
                ]}
            return stream()

    orch = make_orchestrator(LLMJudgeEvaluator(api_key=""))
    orch._config.training_backend = "remote"
    orch._config.max_iterations = 1
    orch._backend = Backend()
    events = [event async for event in await orch.run()]
    assert orch._record.status == RunStatus.COMPLETE
    assert orch._record.baseline_score == 0.0
    assert orch._record.iterations[0].score == 1.0
    assert orch._record.best_score == 1.0
    assert any("locked objective scoring" in event.message for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_planner", [False, True])
async def test_real_goal_run_never_falls_back_to_demo_evals(has_planner):
    class NoEvalsPlanner(MiniMaxPlanner):
        def __init__(self):
            super().__init__(api_key="test", model="test", base_url="http://localhost/v1")
            self.attempts = 0

        async def generate_evals(self, goal):
            self.attempts += 1
            return None

    class MustNotTrain:
        is_mock = False
        name = "local"

        async def train(self, *args, **kwargs):
            pytest.fail("No training should start without goal-specific evals")

    orch = make_orchestrator(None)
    orch._config.goal = "classify tickets"
    orch._backend = MustNotTrain()
    if has_planner:
        orch._planner = NoEvalsPlanner()
    _ = [event async for event in await orch.run()]
    assert orch._record.status == RunStatus.FAILED
    assert "Goal-specific" in orch._record.error
    assert not orch._record.iterations
    if has_planner:
        assert orch._planner.attempts == 3
