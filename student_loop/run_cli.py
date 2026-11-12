#!/usr/bin/env python3
"""CLI entry for the student research loop (JSON out)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from student_loop.graph import get_status, run_simple_loop  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Student research loop (LangGraph skeleton)")
    p.add_argument("--status", action="store_true", help="Print graph readiness and exit")
    p.add_argument(
        "--dataset",
        default=str(REPO / "datasets" / "Titanic-Dataset.csv"),
    )
    p.add_argument("--target", default="Survived")
    p.add_argument("--target-score", type=float, default=0.8)
    p.add_argument("--max-cycles", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.status:
        print(json.dumps(get_status(), indent=2))
        return 0
    result = run_simple_loop(
        dataset_path=args.dataset,
        target_column=args.target,
        target_score=args.target_score,
        max_cycles=args.max_cycles,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
