"""Exercise both VM inference scripts without loading GPU libraries or models."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", [
    "src/virtual_machine_code/evaluate.py", "scripts/vm/baseline_evaluate.py",
])
def test_vm_decodes_only_generated_tokens_and_preserves_response_markers(
    script, monkeypatch, tmp_path,
):
    answer = '{"text":"literal ### Response: inside the answer"}'

    class Tokens:
        shape = (1, 12)

        def __getitem__(self, key):
            assert key == (slice(None), slice(12, None))
            return "generated tokens only"

    class Encoding(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, texts, return_tensors):
            return Encoding(input_ids=Tokens())

        def batch_decode(self, tokens, skip_special_tokens):
            assert tokens == "generated tokens only"
            return [answer]

    class Model:
        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 512
            assert kwargs["do_sample"] is False
            return Tokens()

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return Model(), Tokenizer()

        @staticmethod
        def for_inference(model):
            pass

    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=FastLanguageModel))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    evals = tmp_path / "data" / "evals.json"
    evals.write_text(json.dumps([{"input": "extract text", "expected": answer}]))
    runpy.run_path(str(ROOT / script))
    assert json.loads(evals.read_text())[0]["actual"] == answer
