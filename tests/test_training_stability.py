"""Corrupt training must fail before it can be archived as successful."""

from types import SimpleNamespace

import pytest

from evolora.local.worker import (
    training_rows,
    validate_trainable_tensors,
    validate_training_metrics,
)


@pytest.mark.parametrize("key", ["loss", "grad_norm", "entropy"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_training_metrics_fail(key, value):
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_training_metrics({key: value})


def test_exploding_gradients_fail_even_with_finite_loss():
    with pytest.raises(RuntimeError, match="exploding gradients"):
        validate_training_metrics({"loss": 2.1, "grad_norm": 3.7e17})
    validate_training_metrics({"loss": 2.1, "grad_norm": 3.7, "entropy": 1.2})


@pytest.mark.parametrize("gradients", [False, True])
def test_invalid_trainable_weights_or_gradients_fail(gradients):
    trainable = SimpleNamespace(requires_grad=True, grad="invalid")
    frozen = SimpleNamespace(requires_grad=False)
    model = SimpleNamespace(named_parameters=lambda: [("base", frozen), ("lora", trainable)])
    checked = []

    def isfinite(tensor):
        checked.append(tensor)
        return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: False))

    with pytest.raises(RuntimeError, match="non-finite"):
        validate_trainable_tensors(model, SimpleNamespace(isfinite=isfinite), gradients=gradients)
    assert checked == (["invalid"] if gradients else [trainable])


class ChatTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **kwargs):
        text = "USER:" + messages[0]["content"] + "\nASSISTANT:"
        return text if len(messages) == 1 else text + messages[1]["content"] + "!\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}


def test_training_masks_prompt_and_preserves_complete_answer_and_eos():
    rows = training_rows(ChatTokenizer(), [{"prompt": "invoice", "completion": '{"total":null}'}],
                         100)
    row = rows[0]
    answer = bytes(t for t, mask in zip(row["input_ids"], row["completion_mask"], strict=True)
                   if mask).decode()
    assert answer == '{"total":null}!'
    assert row["completion_mask"][0] == 0
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["completion_mask"])


def test_training_rejects_sequence_overflow_instead_of_truncating_json():
    with pytest.raises(ValueError, match="Answer truncation refused"):
        training_rows(ChatTokenizer(), [{"prompt": "invoice", "completion": '{"total":null}'}], 10)


class _FakeTensor:
    """Just enough tensor arithmetic to run composed_softmax without torch installed."""

    def __init__(self, rows):
        self.rows = rows

    def amax(self, dim, keepdim):
        assert dim == -1 and keepdim
        return _FakeTensor([[max(row)] for row in self.rows])

    def detach(self):
        return self

    def __sub__(self, other):
        return _FakeTensor([[v - o[0] for v in row] for row, o in zip(self.rows, other.rows)])

    def exp(self):
        import math

        return _FakeTensor([[math.exp(v) for v in row] for row in self.rows])

    def sum(self, dim, keepdim):
        return _FakeTensor([[sum(row)] for row in self.rows])

    def __truediv__(self, other):
        return _FakeTensor([[v / o[0] for v in row] for row, o in zip(self.rows, other.rows)])


def test_composed_softmax_matches_softmax_and_survives_masked_entries():
    from evolora.local.worker import composed_softmax

    out = composed_softmax(_FakeTensor([[1.0, 2.0, 3.0], [1000.0, -3.4e38, 1000.0]])).rows
    assert out[0] == pytest.approx([0.09003057, 0.24472847, 0.66524096])
    assert out[1] == pytest.approx([0.5, 0.0, 0.5])  # no overflow, masked entry gets zero


def test_xpu_training_replaces_the_native_softmax(monkeypatch):
    import sys

    from evolora.local import worker

    functional = SimpleNamespace(softmax=object())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(functional=functional)))
    worker.use_composed_softmax()
    assert functional.softmax is worker.composed_softmax
