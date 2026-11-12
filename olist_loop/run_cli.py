"""Run the Olist research loop from the command line (prints one line per agent).

Inside the gateway container (has Postgres access + deps):
    docker compose -f infra/docker-compose.yml --env-file .env exec gateway \
        python -m olist_loop.run_cli --max-cycles 3
"""

from __future__ import annotations

import argparse
import json

from olist_loop.graph import get_status, stream_olist_loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-score", type=float, default=0.5, help="validation F1 to accept")
    ap.add_argument("--max-cycles", type=int, default=3)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full table")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None, help="LLM id (default: OPENROUTER_MODEL)")
    ap.add_argument("--status", action="store_true", help="only print graph readiness")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(get_status(), indent=2))
        return
    for ev in stream_olist_loop(target_score=a.target_score, max_cycles=a.max_cycles,
                                max_rows=a.max_rows, seed=a.seed, model=a.model):
        t = ev["type"]
        if t == "node_completed":
            print(f"[cycle {ev.get('cycle') or 0}] {ev['node']:<20} {ev.get('note') or ''}")
        elif t == "router":
            print(f"{'':>10}router → {ev['route']} ({ev['action']})")
        elif t == "error":
            print("ERROR", ev["message"])
        elif t == "run_completed":
            r = ev["result"]
            print("\nHOLDOUT:", json.dumps({k: v for k, v in (r.get("holdout_metrics") or {}).items()
                                            if k != "airflow_baselines"}, default=str))


if __name__ == "__main__":
    main()
