"""Local planning retains accepted batches when later tool output is truncated."""

import json
from types import SimpleNamespace

import pytest

from evolora.agent.planner import MiniMaxPlanner
from evolora.agent.tools import TOOLS


def response(name, args, finish="tool_calls"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish,
        message=SimpleNamespace(tool_calls=[SimpleNamespace(
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )]),
    )])


class Client:
    def __init__(self, truncate_batch=None, duplicate=False):
        self.calls = []
        self.batch = 0
        self.number = 0
        self.truncate_batch = truncate_batch
        self.duplicate = duplicate
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["tool_choice"] == "required"
        assert len(kwargs["tools"]) == 1
        assert len(kwargs["messages"]) == 2
        name = kwargs["tools"][0]["function"]["name"]
        if name == "create_evals":
            return response(name, {"criteria": ["field accuracy"]})
        if name == "start_training_model":
            return response(name, {"lora_rank_r": 8, "per_device_train_batch_size": 1})
        self.batch += 1
        if self.batch == self.truncate_batch:
            return response(name, {}, "length")
        props = kwargs["tools"][0]["function"]["parameters"]["properties"]
        count = props["training_json"]["properties"]["examples"]["maxItems"]
        examples = []
        for _ in range(count):
            self.number += 1
            number = 1 if self.duplicate else self.number
            examples.append({"prompt": f"Extract n from record {number}; return JSON with key n.",
                             "completion": json.dumps({"n": number})})
        return response(name, {"training_json": {"examples": examples}})


def planner(client):
    obj = MiniMaxPlanner("test", "local-model", "http://localhost/v1",
                         named_tool_choice=False, example_batch_size=8)
    obj._make_client = lambda: client
    return obj


@pytest.mark.asyncio
async def test_large_local_dataset_is_split_into_small_isolated_batches():
    original = json.dumps(TOOLS)
    client = Client()
    plan, fallback = await planner(client).plan(1, 0, 0, [], 60, "extract records")
    assert not fallback
    assert len(plan.data_spec.examples) == 60
    assert client.batch == 20
    assert len(client.calls) == 22  # criteria, 20 batches, hyperparameters
    assert json.dumps(TOOLS) == original  # global cloud schemas unchanged
    assert all("Extract n from record" not in call["messages"][1]["content"]
               for call in client.calls)  # prior synthetic examples never accumulate in context


@pytest.mark.asyncio
async def test_truncation_retries_only_failed_batch_and_keeps_accepted_examples():
    client = Client(truncate_batch=2)
    plan, fallback = await planner(client).plan(1, 0, 0, [], 6, "extract records")
    assert not fallback and len(plan.data_spec.examples) == 6
    sizes = [call["tools"][0]["function"]["parameters"]["properties"]
             ["training_json"]["properties"]["examples"]["maxItems"]
             for call in client.calls if call["tools"][0]["function"]["name"] == "add_training_examples"]
    assert sizes == [3, 3, 1, 1, 1]
    assert client.calls[0]["tools"][0]["function"]["name"] == "create_evals"
    assert sum(call["tools"][0]["function"]["name"] == "create_evals"
               for call in client.calls) == 1
    assert plan.data_spec.examples[0]["completion"] == '{"n": 1}'


class FlakyClient(Client):
    """Truncates every listed add_training_examples call, whatever its size."""

    def __init__(self, truncated: set[int]):
        super().__init__()
        self.truncated = truncated

    async def create(self, **kwargs):
        name = kwargs["tools"][0]["function"]["name"]
        if name == "add_training_examples" and self.batch + 1 in self.truncated:
            self.batch += 1
            return response(name, {}, "length")
        return await super().create(**kwargs)


@pytest.mark.asyncio
async def test_truncated_single_example_call_is_retried_without_losing_accepted_batches():
    client = FlakyClient({2, 3, 4})  # size 3 fails, then two single-example calls fail
    plan, fallback = await planner(client).plan(1, 0, 0, [], 6, "extract records")
    assert not fallback and len(plan.data_spec.examples) == 6
    assert sum(call["tools"][0]["function"]["name"] == "create_evals"
               for call in client.calls) == 1  # the plan was never restarted


@pytest.mark.asyncio
async def test_three_single_example_failures_in_a_row_stop_the_attempt():
    client = FlakyClient(set(range(2, 40)))
    _, fallback = await planner(client).plan(1, 0, 0, [], 6, "extract records")
    assert fallback
    assert client.batch <= 12  # bounded: two attempts, a handful of calls each


@pytest.mark.asyncio
async def test_repeated_duplicates_cannot_fill_requested_dataset_or_loop_forever():
    client = Client(duplicate=True)
    _, fallback = await planner(client).plan(1, 0, 0, [], 4, "extract records")
    assert fallback
    assert client.batch == 24  # two attempts, each bounded by target + eight


@pytest.mark.asyncio
async def test_generated_eval_prompts_include_goal_rules_when_model_returns_bare_records():
    class EvalClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            return response("create_evals", {"eval_examples": [
                {"prompt": "Invoice #12, total $10", "expected_output": {"invoice_number": "12"}},
            ]})

    goal = "Return only JSON with key invoice_number; use null when missing."
    samples = await planner(EvalClient()).generate_evals(goal)
    assert samples[0]["prompt"] == goal + "\n\nInput:\nInvoice #12, total $10"
    assert "expected_output" not in samples[0]["prompt"]
