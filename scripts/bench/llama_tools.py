"""Test llama-server speed and real EvoLoRA tool schemas over repeated calls."""

import argparse
import importlib.util
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from jsonschema import Draft7Validator

TOOLS_PATH = Path(__file__).resolve().parents[2] / "src/evolora/agent/tools.py"
spec = importlib.util.spec_from_file_location("evolora_bench_tools", TOOLS_PATH)
assert spec and spec.loader
tools_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools_module)
TOOLS = tools_module.TOOLS


GOALS = [
    "extract a person's name and city from plain text into JSON keys name and city",
    "classify a support ticket as billing, account, or technical in a JSON category field",
    "extract an invoice total and currency into JSON keys total and currency",
    "convert a short date phrase into JSON keys year, month, and day",
    "identify the language of a short sentence in a JSON language field",
]


def request(base_url: str, payload: dict) -> tuple[dict, float]:
    req = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urlopen(req, timeout=240) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="local")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--mode", choices=("forced", "auto", "required", "both"),
                        default="forced")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    schema = next(t["function"]["parameters"] for t in TOOLS
                  if t["function"]["name"] == "create_evals")
    records = []
    modes = ("forced", "auto") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for index in range(args.trials):
            goal = GOALS[index % len(GOALS)]
            prompt = (
                f"Create evaluation criteria and two concrete examples for a model that must {goal}. "
                f"Call create_evals now. Trial {index + 1}."
            )
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": "Use the tools supplied. Output no prose."},
                    {"role": "user", "content": prompt},
                ],
                "tools": [tool for tool in TOOLS
                          if tool["function"]["name"] == "create_evals"]
                if mode == "required" else TOOLS,
                "tool_choice": (
                    {"type": "function", "function": {"name": "create_evals"}}
                    if mode == "forced" else mode
                ),
                "temperature": 0.3 if mode == "auto" else 0.9,
                "max_tokens": 1024,
            }
            record = {"mode": mode, "trial": index + 1}
            try:
                response, elapsed = request(args.base_url, payload)
                record["elapsed_s"] = elapsed
                record["usage"] = response.get("usage", {})
                record["timings"] = response.get("timings", {})
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                record["tool_names"] = [c.get("function", {}).get("name") for c in calls]
                matching = [c for c in calls if c.get("function", {}).get("name") == "create_evals"]
                if len(calls) != 1 or len(matching) != 1:
                    record["error"] = f"expected only create_evals, found {record['tool_names']}"
                else:
                    parameters = json.loads(matching[0]["function"]["arguments"])
                    Draft7Validator(schema).validate(parameters)
                    record["valid"] = True
                    record["arguments"] = parameters
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "arguments"}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
