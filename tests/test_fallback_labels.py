"""Fallbacks name the service that actually failed (local runs never mention clouds)."""

from evolora.evaluation.digitalocean_judge import HeuristicJudge, get_judge
from evolora.models.core import AgentPlan, LoraHyperparams, TrainingDataSpec
from evolora.orchestration.retrain_advisor import get_retrain_advisor

DEAD = "http://127.0.0.1:9/v1"  # nothing listens here: the call fails at once


def _plan() -> AgentPlan:
    return AgentPlan(
        hyperparams=LoraHyperparams(r=8, lora_alpha=16, learning_rate=2e-4, num_epochs=1, batch_size=1),
        data_spec=TrainingDataSpec(examples=[{"prompt": "p", "completion": "{}"}], rationale="r", max_examples=1),
        rationale="r",
        focus_areas=["json"],
    )


async def test_local_judge_fallback_is_labeled_local():
    judge = get_judge(api_key="local", base_url=DEAD, model="qwen3-4b-instruct", label="local LLM")
    report = await judge.judge(
        goal="g", task_name="t", base_model_id="b", iteration=1, score=0.0,
        adaptive_score=None, plan=_plan(), eval_results=[], responses={},
    )
    assert report.source == "local_fallback"
    assert "local LLM judge failed safely" in report.summary
    assert "DigitalOcean" not in report.summary


async def test_local_advisor_fallback_is_labeled_local():
    report = await HeuristicJudge().judge(
        goal="g", task_name="t", base_model_id="b", iteration=1, score=0.1,
        adaptive_score=None, plan=_plan(), eval_results=[], responses={},
    )
    advisor = get_retrain_advisor(api_key="local", base_url=DEAD, model="qwen3-4b-instruct", label="local LLM")
    decision = await advisor.decide(
        goal="g", rating=0.1, target_score=0.85, iteration=1, max_iterations=3, judge_report=report,
    )
    assert decision.source == "local_decision_fallback"
    assert "MiniMax" not in decision.reason
