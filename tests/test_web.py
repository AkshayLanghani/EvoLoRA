"""Tests for the browser-facing HTTP API."""

import json

import pytest
from fastapi.testclient import TestClient

from evolora.config import get_config
from evolora.demo.task import LOCKED_EVAL_SET
from evolora.web import app as web

_SERVICE_KEYS = (
    "EVOLORA_PROFILE",
    "EVOLORA_DB_PATH",
    "MINIMAX_API_KEY",
    "DIGITAL_OCEAN_MODEL_ACCESS_KEY",
    "MONGODB_URI",
    "TRAINING_BACKEND",
    "MODEL_RUNNER",
)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Mock-mode app with no live services and a fresh in-memory store."""
    for key in _SERVICE_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    get_config.cache_clear()
    monkeypatch.setattr(web, "_store", None)
    monkeypatch.setattr(web, "_active_runs", 0)
    monkeypatch.setattr(web, "_limiter", web._RateLimiter(100, 3600))
    yield TestClient(web.app)
    get_config.cache_clear()


def _stream_events(client, payload):
    with client.stream("POST", "/api/runs", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in resp.iter_lines() if line]


def test_status_reports_fallbacks_without_keys(client):
    body = client.get("/api/status").json()
    assert body["planner"] == "heuristic"
    assert body["judge"] == "heuristic"
    assert body["persistence"] == "memory"
    assert body["training_backend"] == "mock"
    assert body["planner_model"] is None
    assert body["default_base_model"] in {m["id"] for m in body["base_models"]}


def test_mock_run_streams_full_loop(client):
    events = _stream_events(
        client,
        {"goal": "summarize customer spending as strict JSON", "training_samples": 4},
    )
    kinds = [e["kind"] for e in events]

    assert kinds[0] == "run_started"
    assert "eval_set_locked" in kinds
    assert "baseline_complete" in kinds
    assert "training_progress" in kinds
    assert "judge_complete" in kinds
    assert kinds[-2] in {"run_complete", "run_failed", "run_cancelled"}
    assert kinds[-1] == "run_summary"

    locked = next(e for e in events if e["kind"] == "eval_set_locked")
    assert locked["data"]["hash"] == LOCKED_EVAL_SET.hash

    summary = events[-1]["data"]
    assert summary["run_id"] == events[0]["run_id"]
    assert summary["eval_set_hash"] == LOCKED_EVAL_SET.hash
    assert summary["adapter_name"].startswith("summarize-customer-spending")
    assert 1 <= len(summary["iteration_scores"]) <= 3

    runs = client.get("/api/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [summary["run_id"]]


def test_run_rejects_unknown_base_model(client):
    resp = client.post("/api/runs", json={"base_model_id": "gpt2"})
    assert resp.status_code == 422


def test_run_rejects_too_many_samples(client):
    resp = client.post("/api/runs", json={"training_samples": web.WEB_MAX_TRAINING_SAMPLES + 1})
    assert resp.status_code == 422


def test_run_is_rate_limited(client, monkeypatch):
    monkeypatch.setattr(web, "_limiter", web._RateLimiter(0, 3600))
    resp = client.post("/api/runs", json={})
    assert resp.status_code == 429


def test_snap_moves_values_onto_choice_sets(client):
    resp = client.post(
        "/api/tools/snap",
        json={"args": {"learning_rate": 0.0003, "lora_rank_r": 48, "num_train_epochs": 9}},
    )
    snapped = resp.json()["snapped"]
    assert snapped["learning_rate"] == 2e-4
    assert snapped["r"] in {32, 64}
    assert snapped["num_epochs"] == 6


def test_verify_evals_detects_tampering(client):
    text = client.get("/api/tools/eval-set").json()["samples_json"]
    untouched = client.post("/api/tools/verify-evals", json={"samples_json": text}).json()
    assert untouched["match"] is True

    tampered = text.replace('"top_customer_total": 550.0', '"top_customer_total": 450.0')
    assert tampered != text
    result = client.post("/api/tools/verify-evals", json={"samples_json": tampered}).json()
    assert result["match"] is False
    assert result["locked_hash"] == LOCKED_EVAL_SET.hash

    invalid = client.post("/api/tools/verify-evals", json={"samples_json": "[{}]"})
    assert invalid.status_code == 422


def test_score_uses_objective_evaluator(client):
    expected = LOCKED_EVAL_SET.samples[0].expected
    perfect = client.post(
        "/api/tools/score", json={"sample_id": "lock-001", "response": json.dumps(expected)}
    ).json()
    assert perfect["score"] == 1.0
    assert perfect["passed"] is True

    not_object = client.post(
        "/api/tools/score", json={"sample_id": "lock-001", "response": "[1, 2]"}
    ).json()
    assert not_object["score"] == 0.0


class _FakeLiveRun:
    def __init__(self):
        self.answers = []
        self.cancelled = False

    def submit_retrain_approval(self, approved):
        self.answers.append(approved)

    def cancel(self):
        self.cancelled = True


def test_approval_and_cancel_reach_the_live_run(client, monkeypatch):
    live = _FakeLiveRun()
    monkeypatch.setattr(web, "_live_runs", {"run-1": live})

    assert client.post("/api/runs/run-1/approval", json={"approved": False}).status_code == 200
    assert client.post("/api/runs/run-1/cancel").status_code == 200
    assert live.answers == [False]
    assert live.cancelled is True


def test_approval_for_unknown_run_is_404(client):
    assert client.post("/api/runs/nope/approval", json={"approved": True}).status_code == 404


def test_hosted_runs_stay_autonomous(client, monkeypatch):
    captured = {}
    real_build = web._build_orchestrator

    def spy(req):
        orch = real_build(req)
        captured["approval"] = orch._config.require_retrain_approval
        return orch

    monkeypatch.setattr(web, "_build_orchestrator", spy)
    _stream_events(client, {"training_samples": 2, "autonomous": False})
    assert captured["approval"] is False  # only the local agent can pause for answers


def test_adapter_name_strips_generic_lead_in(client):
    resp = client.post(
        "/api/tools/adapter-name",
        json={"goal": "Make a model that specializes in generating strict JSON", "run_id": "1a2b3c4d"},
    )
    assert resp.json()["name"] == "strict-json-1a2b3c"
