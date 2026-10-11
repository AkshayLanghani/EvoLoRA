"""Summarize line-oriented tool-call trials and preserve raw records."""

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
               if line.startswith("{")]
    summary = {}
    for mode in sorted({record["mode"] for record in records}):
        batch = [record for record in records if record["mode"] == mode]
        speeds = [record["timings"]["predicted_per_second"] for record in batch
                  if record.get("timings", {}).get("predicted_per_second")]
        summary[mode] = {
            "trials": len(batch),
            "valid": sum(bool(record.get("valid")) for record in batch),
            "mean_generated_tokens_per_s": statistics.mean(speeds) if speeds else None,
            "median_generated_tokens_per_s": statistics.median(speeds) if speeds else None,
            "min_generated_tokens_per_s": min(speeds) if speeds else None,
            "max_generated_tokens_per_s": max(speeds) if speeds else None,
            "errors": [record.get("error") for record in batch if record.get("error")],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
