"""Expose the existing evaluator's train-split C0/C1 screen for this pilot.

Offline by default. --execute is for an explicitly authorized later live stage.
This wrapper is needed because evaluate_mut_mappo's CLI omits split=train;
the evaluator API already supports it. No task, reward or evaluator is copied.
"""
import argparse
import json
from pathlib import Path

from evaluation.evaluate_mut_mappo import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    report = evaluate(root / "base_config.json", team="A", split="train",
                      conditions=("C0", "C1"), validate_only=not args.execute,
                      output_dir=root / "red_screen_train")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
