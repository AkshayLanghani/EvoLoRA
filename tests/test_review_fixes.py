"""Regressions for defects found in the whole-codebase review (S1)."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

from evolora.agent.tools import extract_training_payload
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.evaluation.evaluator import evaluate_sample, generic_evaluate_sample
from evolora.models.core import (
    AgentPlan,
    EvalSample,
    IterationResult,
    JudgeReport,
    RunConfig,
    RunRecord,
    RunStatus,
    StopReason,
)
from evolora.orchestration.orchestrator import Orchestrator
from evolora.orchestration.retrain_advisor import MiniMaxRetrainAdvisor
from evolora.persistence.store import InMemoryRunStore
from evolora.training.backends import MockTrainingBackend, RemoteTrainingBackend


def _record(scores: list[float | None], baseline: float = 0.0) -> RunRecord:
    rec = RunRecord(config=RunConfig(), baseline_score=baseline)
    for index, score in enumerate(scores, start=1):
        result = IterationResult(iteration=index, plan=AgentPlan())
        if score is None:
            result.error = "training failed"
        else:
            result.score = score
        rec.iterations.append(result)
    return rec


@pytest.mark.parametrize(
    ("scores", "baseline", "expected"),
    [
        ([0.3, 0.6, 0.6], 0.0, 1),
        ([0.1, 0.5, 0.9, 0.9], 0.0, 1),
        ([0.6, 0.6, 0.6], 0.6, 3),
        ([0.5], 0.5, 1),
        ([0.5, 0.9], 0.0, 0),
        ([0.5, None, 0.5], 0.0, 1),
    ],
)
def test_no_improvement_count_counts_only_trailing_flat_iterations(scores, baseline, expected):
    assert _record(scores, baseline).no_improvement_count() == expected


class _AlwaysFailingBackend(MockTrainingBackend):
    async def train(self, *args, **kwargs):
        raise RuntimeError("GPU exploded")


@pytest.mark.parametrize("max_iterations", [1, 2])
async def test_run_where_every_iteration_fails_is_reported_failed(max_iterations):
    orch = Orchestrator(
        config=RunConfig(max_iterations=max_iterations, target_score=0.99, patience=5),
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        run_store=InMemoryRunStore(),
        training_backend=_AlwaysFailingBackend(),
    )
    async for _ in await orch.run():
        pass
    rec = orch._record
    assert rec.status == RunStatus.FAILED
    assert rec.stop_reason == StopReason.TRAINING_FAILURE
    assert "GPU exploded" in (rec.error or "")


def _advisor_with_content(content: str | None) -> MiniMaxRetrainAdvisor:
    advisor = MiniMaxRetrainAdvisor(api_key="k", base_url="http://x", model="m", label="Local")

    async def create(**kwargs):
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    advisor._make_client = lambda: client
    return advisor


async def _decide(content: str | None):
    return await _advisor_with_content(content).decide(
        goal="g",
        rating=0.2,
        target_score=0.9,
        iteration=1,
        max_iterations=3,
        judge_report=JudgeReport(rating=0.2, summary="weak", source="t", is_mock=True),
    )


@pytest.mark.parametrize("content", [None, "", "{}", '{"retrain_recommended": "maybe"}'])
async def test_advisor_without_a_boolean_answer_falls_back_to_heuristic(content):
    decision = await _decide(content)
    assert decision.source.endswith("_decision_fallback")
    assert decision.retrain_recommended is True  # low rating, iterations left


async def test_advisor_reads_string_booleans_correctly():
    assert (await _decide('{"retrain_recommended": "false"}')).retrain_recommended is False
    assert (await _decide('{"retrain_recommended": "True"}')).retrain_recommended is True
    assert (await _decide('{"retrain_recommended": false}')).reason == "Local decision completed."


@pytest.mark.parametrize("sample", LOCKED_EVAL_SET.samples + ADAPTIVE_EVAL_SET.samples)
def test_demo_expected_answers_match_the_prompt_arithmetic(sample):
    customers = json.loads(re.search(r"Customers: (\[.*\])\. ", sample.prompt).group(1))
    totals = {c["name"]: round(sum(c["purchases"]), 2) for c in customers}
    top = max(totals, key=totals.get)
    assert sample.expected["top_customer"] == top
    assert sample.expected["top_customer_total"] == pytest.approx(totals[top])
    assert sample.expected["customer_count"] == len(customers)
    assert sample.expected["total_revenue"] == pytest.approx(sum(totals.values()))


@pytest.mark.parametrize("huge", ["1" * 400, "1e999"])
def test_evaluators_score_overflowing_numbers_as_wrong_instead_of_raising(huge):
    demo = LOCKED_EVAL_SET.samples[0]
    response = (
        f'{{"top_customer": "Carol", "top_customer_total": {huge}, '
        f'"customer_count": {huge}, "total_revenue": {huge}, "summary": "x"}}'
    )
    assert evaluate_sample(demo, response).passed is False

    generic = EvalSample(sample_id="g", prompt="p", expected={"year": 2024})
    assert generic_evaluate_sample(generic, f'{{"year": {huge}}}').passed is False


def test_object_completions_are_stored_as_json_not_python_repr():
    examples, _ = extract_training_payload(
        {"examples": [{"prompt": "Extract", "completion": {"ok": True, "n": None}}]}
    )
    assert json.loads(examples[0]["completion"]) == {"ok": True, "n": None}


def test_sql_wrapping_needs_a_sql_statement_not_the_word_query():
    examples, _ = extract_training_payload(
        {
            "examples": [
                {"prompt": "Classify this customer query: charged twice", "completion": "billing"},
                {"prompt": "Count the orders", "completion": "SELECT COUNT(*) FROM orders"},
                {"prompt": 42, "completion": "ignored"},
            ]
        }
    )
    assert [e["completion"] for e in examples] == [
        "billing",
        '{"sql": "SELECT COUNT(*) FROM orders"}',
    ]


async def test_remote_command_does_not_hang_when_ssh_connect_fails():
    def refuse():
        raise OSError("connection refused")

    backend = RemoteTrainingBackend(
        ssh_host="gpu.example.com",
        ssh_user="root",
        ssh_key_path="C:/keys/evolora",
        ssh_client_factory=refuse,
    )

    async def drain():
        async for _ in backend._exec_remote_command("true"):
            pass

    with pytest.raises(OSError, match="connection refused"):
        await asyncio.wait_for(drain(), timeout=5)


async def test_generation_passes_the_worker_an_absolute_adapter_path(tmp_path, monkeypatch):
    from pathlib import Path

    from evolora.training import local_backend

    seen = {}

    async def fake_worker(python, command, job, job_dir):
        seen.update(job)
        job_dir.mkdir(parents=True, exist_ok=True)
        Path(job["output_file"]).write_text('{"s1": "ok"}', encoding="utf-8")
        yield {"event": "done"}

    monkeypatch.setattr(local_backend, "run_worker", fake_worker)
    runner = local_backend.LocalModelRunner(python="py", workspace=tmp_path, base_model_id="m")
    relative = Path("adapters") / "some-adapter-it1"
    assert await runner.run_batch([{"sample_id": "s1", "prompt": "p"}], str(relative)) == {"s1": "ok"}
    assert Path(seen["adapter_dir"]).is_absolute()
    assert Path(seen["adapter_dir"]) == relative.resolve()

    await runner.run_batch([{"sample_id": "s1", "prompt": "p"}])
    assert seen["adapter_dir"] is None


_TRIAGE_GOAL = (
    "Triage a customer support email. Return only JSON with exactly these keys: category, "
    "priority, order_id, refund_requested. category: billing, shipping or other. priority: "
    "high if it mentions a deadline; low if it is only a question; otherwise medium. "
    "order_id: the ID exactly as written, or null. refund_requested: true or false."
)
_TRIAGE_ANSWER = '{"category": "billing", "priority": "high", "order_id": "123", "refund_requested": true}'


@pytest.mark.parametrize(
    "suffix",
    [
        '\n\ncategory: billing\npriority: high\norder_id: "123"\nrefund_requested: true',
        "\n\nCategory: billing\nPriority: high (mentions a deadline)\nOrder ID: 123\nRefund requested: true",
        "\n\nExpected: " + _TRIAGE_ANSWER,
        '\n\n{"category": "billing", "priority": "high"}',
    ],
)
def test_training_examples_whose_prompt_states_the_answer_are_dropped(suffix):
    from evolora.agent.tools import prompt_leaks_answer

    email = '\n\nEmail: "Refund order 123 by Friday please."'
    assert prompt_leaks_answer(_TRIAGE_GOAL + email + suffix, _TRIAGE_ANSWER)
    examples, _ = extract_training_payload(
        {"examples": [{"prompt": _TRIAGE_GOAL + email + suffix, "completion": _TRIAGE_ANSWER}]}
    )
    assert examples == []


def test_instruction_listing_the_options_is_not_a_leak():
    from evolora.agent.tools import prompt_leaks_answer

    prompt = _TRIAGE_GOAL + '\n\nEmail: "Refund order 123 by Friday please. Priority: high!"'
    assert not prompt_leaks_answer(prompt, _TRIAGE_ANSWER)
    examples, _ = extract_training_payload(
        {"examples": [{"prompt": prompt, "completion": _TRIAGE_ANSWER}]}
    )
    assert len(examples) == 1


async def _chat_with_adapter(tmp_path, monkeypatch, **request):
    import evolora.web.app as web

    adapter = tmp_path / "triage-it1"
    adapter.mkdir(exist_ok=True)
    (adapter / "evolora.json").write_text(
        json.dumps({"base_model": "base", "goal": "Return triage JSON."}), encoding="utf-8"
    )
    sent = {}

    async def run_batch(prompts, adapter_path=None):
        sent["prompt"] = prompts[0]["prompt"]
        return {"chat": "{}"}

    cfg = SimpleNamespace(is_local=True, local_adapters_dir=str(tmp_path),
                          default_base_model="base", base_models=["base"])
    monkeypatch.setattr(web, "get_config", lambda: cfg)
    monkeypatch.setattr(web, "local_training_issue", lambda cfg: None)
    monkeypatch.setattr(web, "build_resources", lambda *a, **kw: None)
    monkeypatch.setattr(web, "get_runner", lambda *a, **kw: SimpleNamespace(run_batch=run_batch))
    result = await web.chat(web.ChatRequest(adapter="triage-it1", **request))
    return sent["prompt"], result


async def test_adapter_chat_adds_the_training_instruction(tmp_path, monkeypatch):
    prompt, result = await _chat_with_adapter(tmp_path, monkeypatch, prompt="My blender broke")
    assert prompt == "Return triage JSON.\n\nInput:\nMy blender broke"
    assert result["goal_applied"] is True


async def test_adapter_chat_does_not_repeat_or_force_the_instruction(tmp_path, monkeypatch):
    already = "Return triage JSON.\n\nInput:\nMy blender broke"
    prompt, result = await _chat_with_adapter(tmp_path, monkeypatch, prompt=already)
    assert prompt == already and result["goal_applied"] is False
    prompt, result = await _chat_with_adapter(
        tmp_path, monkeypatch, prompt="hello", apply_goal=False
    )
    assert prompt == "hello" and result["goal_applied"] is False


def test_leak_check_is_driven_by_the_answer_keys_not_by_one_goal():
    from evolora.agent.tools import prompt_leaks_answer

    answer = '{"customer_name": "Maria", "party_size": 4, "time": "19:00", "is_cancellation": false}'
    message = "Extract the booking as JSON.\n\nMessage: hi its Maria, table for 4 at 7pm please"
    assert not prompt_leaks_answer(message, answer)
    assert prompt_leaks_answer(
        message + "\n\nCustomer name: Maria\nParty size: 4\nTime: 19:00\nIs cancellation: false",
        answer,
    )