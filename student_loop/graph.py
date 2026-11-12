"""STUDENT LAB — fill node bodies + wire build_graph().

LLM / train helpers are imported from loop_runtime.py (instructor-maintained).
Copy node bodies and wiring from mini_loop/graph.py or the Guide (/guide → LangGraph).

Flow:
  START → data_scientist → hypothesis → ml_engineer → judge
            ↑__________________________________|
            |   (revise: memory → context → hypothesis again)
            └── accept/stop → holdout_eval → END
"""

from __future__ import annotations

import json
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import (  # noqa: E402
    DEFAULT_MODEL,
    FAMILIES,
    _SPEC_SCHEMA_HINT,
    _coerce_experiment_spec,
    _fit_score,
    _heuristic_plan,
    _heuristic_profile,
    _heuristic_spec,
    _heuristic_verdict,
    _repair_heuristic,
    _resolve_model,
    _validate_spec,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)

# ---------------------------------------------------------------------------
# STATE
# LangGraph keeps one shared dict ("state") that every node reads/writes.
# Each node returns a *patch* — only the keys it wants to update.
# Annotated[list, operator.add] means "append to steps" instead of replace.
# ---------------------------------------------------------------------------
class SimpleLoopState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    target_score: float
    max_cycles: int
    cycle: int
    seed: int
    holdout_frac: float
    model: str  # OpenRouter / Cursor model id from the UI / env
    profile: dict
    brief: dict
    hypothesis: dict  # conceptual plan from hypothesis_generator (not HPT numbers)
    experiment_spec: dict  # concrete HPT config from ml_engineer
    execution_result: dict
    verdict: dict
    memory: dict
    best_spec: dict
    best_val_f1: float
    holdout_metrics: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# NODES
# A node is a plain function: state in → dict patch out.
# LangGraph calls it, merges the patch into state, then follows an edge.
# ---------------------------------------------------------------------------

def node_data_scientist(state: SimpleLoopState) -> dict:
    """Profile CSV via LLM; heuristic fallback if key/parse fails."""
    raise NotImplementedError("data_scientist")


def node_hypothesis(state: SimpleLoopState) -> dict:
    """Conceptual next experiment — no concrete HPT numbers."""
    raise NotImplementedError("hypothesis_generator")


def node_ml_engineer(state: SimpleLoopState) -> dict:
    """Turn conceptual hypothesis into concrete HPT ExperimentSpec, then train/val."""
    raise NotImplementedError("ml_engineer")


def node_judge(state: SimpleLoopState) -> dict:
    """Accept vs revise from validation metrics (+ optional LLM observation)."""
    raise NotImplementedError("judge")


def node_memory_writer(state: SimpleLoopState) -> dict:
    """Append this cycle's lesson so the next hypothesis can see history."""
    raise NotImplementedError("memory_writer")


def node_context_builder(state: SimpleLoopState) -> dict:
    """Build a short brief for the next hypothesis from recent memory."""
    raise NotImplementedError("context_builder")


def node_holdout(state: SimpleLoopState) -> dict:
    """Score the best spec ONCE on the sealed holdout (same seed as ml_engineer)."""
    raise NotImplementedError("holdout_eval")


def route_after_judge(state: SimpleLoopState) -> str:
    """Router = plain code (not an LLM). Return a key from conditional_edges.

    accept              → holdout
    revise + budget left → memory (loop)
    otherwise           → holdout
    """
    raise NotImplementedError("router")


# ---------------------------------------------------------------------------
# GRAPH WIRING
# StateGraph = the canvas.
# add_node(name, fn) = put a box on the canvas.
# add_edge(a, b)     = always go a → b after a finishes.
# add_conditional_edges = ask a router function which named edge to take.
# compile()          = freeze the graph into a runnable app (invoke / stream).
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(SimpleLoopState)

    # TODO: paste add_node / add_edge / add_conditional_edges / return g.compile()
    # from mini_loop/graph.py (identical block) or the Guide wiring snippet.
    raise NotImplementedError("wire nodes and edges, then return g.compile()")


REQUIRED_NODES = frozenset(
    {
        "data_scientist",
        "hypothesis_generator",
        "ml_engineer",
        "judge",
        "memory_writer",
        "context_builder",
        "holdout_eval",
    }
)


def scan_declared_nodes(source: str) -> list[str]:
    """Names passed to g.add_node(...) in source (order preserved, unique).

    Used so the UI can show boxes one-by-one before the graph compiles.
    """
    import re

    found: list[str] = []
    seen: set[str] = set()
    for name in re.findall(r"""\.add_node\(\s*['\"]([^'\"]+)['\"]""", source):
        if name not in seen:
            seen.add(name)
            found.append(name)
    return found


def _edge_endpoint(token: str) -> str:
    t = token.strip()
    if t in {"START", "END"}:
        return "__start__" if t == "START" else "__end__"
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return t[1:-1]
    return t


def scan_declared_edges(source: str) -> list[dict[str, str]]:
    """Pairs from g.add_edge(...) and judge conditional map in source."""
    import re

    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(src: str, dst: str) -> None:
        key = (src, dst)
        if key in seen or not src or not dst:
            return
        seen.add(key)
        edges.append({"source": src, "target": dst})

    for m in re.finditer(r"\.add_edge\(\s*([^,]+)\s*,\s*([^)]+)\)", source):
        add(_edge_endpoint(m.group(1)), _edge_endpoint(m.group(2)))

    # add_conditional_edges("judge", ..., {"memory": "memory_writer", "holdout": "holdout_eval"})
    if re.search(r"\.add_conditional_edges\(", source):
        if re.search(r"""['\"]memory['\"]\s*:\s*['\"]memory_writer['\"]""", source):
            add("judge", "memory_writer")
        if re.search(r"""['\"]holdout['\"]\s*:\s*['\"]holdout_eval['\"]""", source):
            add("judge", "holdout_eval")

    return edges


def partition_drawable_edges(
    nodes: list[str], edges: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[str]]:
    """Keep edges reachable from START; report the rest (not drawn on the UI)."""
    declared = set(nodes)
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["source"], []).append(e["target"])

    reachable: set[str] = {"__start__"}
    stack = ["__start__"]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)

    ok: list[dict[str, str]] = []
    problems: list[str] = []
    for e in edges:
        s, t = e["source"], e["target"]
        if s != "__start__" and s not in reachable:
            problems.append(f"{s} → {t} (unreachable from START — not drawn)")
            continue
        if t not in declared and t != "__end__":
            problems.append(f"{s} → {t} ({t} is not a declared node — not drawn)")
            continue
        ok.append(e)
    return ok, problems


def assess_graph_ready(app) -> dict:
    """Return readiness dict from a compiled LangGraph app."""
    raw = app.get_graph().nodes
    present = {n for n in raw if n not in {START, END, "__start__", "__end__"}}
    missing = sorted(REQUIRED_NODES - present)
    return {
        "ready": not missing,
        "nodes": sorted(present),
        "missing_nodes": missing,
        "error": None if not missing else f"missing nodes: {', '.join(missing)}",
    }


def get_status() -> dict:
    """Status for /student-loop: partial nodes/edges for the canvas; ready when fully wired."""
    import inspect

    try:
        src = inspect.getsource(build_graph)
    except (OSError, TypeError):
        src = ""
    declared = scan_declared_nodes(src)
    declared_edges = scan_declared_edges(src)
    drawable, edge_problems = partition_drawable_edges(declared, declared_edges)
    missing = sorted(REQUIRED_NODES - set(declared))

    def _error(*parts: str | None) -> str | None:
        msgs = [p for p in parts if p]
        if edge_problems:
            msgs.append("dead edges: " + "; ".join(edge_problems))
        return " · ".join(msgs) if msgs else None

    try:
        app = build_graph()
    except NotImplementedError as exc:
        return {
            "ready": False,
            "nodes": declared,
            "edges": drawable,
            "missing_nodes": missing if missing else sorted(REQUIRED_NODES),
            "error": _error(str(exc) or "build_graph not implemented"),
        }
    except Exception as exc:  # noqa: BLE001
        # Partial wiring often fails compile — still show declared nodes/edges.
        return {
            "ready": False,
            "nodes": declared,
            "edges": drawable,
            "missing_nodes": missing,
            "error": _error(str(exc)),
        }

    assessed = assess_graph_ready(app)
    nodes = assessed["nodes"] or declared
    return {
        "ready": assessed["ready"],
        "nodes": nodes,
        "edges": drawable,
        "missing_nodes": assessed["missing_nodes"],
        "error": _error(assessed["error"]),
    }
def _initial_state(
    *,
    dataset_path: str,
    target_column: str,
    target_score: float,
    max_cycles: int,
    seed: int,
    holdout_frac: float,
    model: str | None = None,
) -> SimpleLoopState:
    state: SimpleLoopState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "target_score": target_score,
        "max_cycles": max_cycles,
        "cycle": 0,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "best_val_f1": -1.0,
        "steps": [],
        "memory": {"history": [], "findings": [], "lessons": []},
        "brief": {},
    }
    chosen = (model or "").strip() or DEFAULT_MODEL
    state["model"] = chosen
    return state


def _finalize_payload(final: dict) -> dict:
    return {
        "dataset_path": final.get("dataset_path"),
        "target_column": final.get("target_column"),
        "cycles": final.get("cycle"),
        "best_val_f1": final.get("best_val_f1"),
        "best_spec": final.get("best_spec"),
        "holdout_metrics": final.get("holdout_metrics"),
        "verdict": final.get("verdict"),
        "stop_reason": final.get("stop_reason"),
        "profile": final.get("profile"),
        "memory": final.get("memory"),
        "model": final.get("model"),
        "steps": final.get("steps") or [],
    }


def run_simple_loop(
    *,
    dataset_path: str,
    target_column: str = "Survived",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
) -> dict:
    """Run the whole graph to completion; return a summary dict."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    final = app.invoke(initial, config={"recursion_limit": 8 + max_cycles * 6})
    return _finalize_payload(final)


def stream_simple_loop(
    *,
    dataset_path: str,
    target_column: str = "Survived",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
):
    """Yield UI events as each node starts/finishes (used by the gateway SSE)."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    yield {
        "type": "run_started",
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "model": initial.get("model"),
    }

    merged: dict = dict(initial)
    last_node: str | None = None
    try:
        for update in app.stream(
            initial, stream_mode="updates", config={"recursion_limit": 8 + max_cycles * 6}
        ):
            if not isinstance(update, dict):
                continue
            for node_name, patch in update.items():
                if last_node == "judge" and node_name in {"memory_writer", "holdout_eval"}:
                    action = str((merged.get("verdict") or {}).get("action") or "stop")
                    yield {
                        "type": "router",
                        "node": "router",
                        "status": "completed",
                        "action": action,
                        "route": "memory" if node_name == "memory_writer" else "holdout",
                    }
                    if node_name == "memory_writer":
                        yield {
                            "type": "node_completed",
                            "node": "research_memory",
                            "status": "completed",
                            "note": "history + lessons filed",
                        }

                yield {"type": "node_started", "node": node_name, "status": "running"}
                if isinstance(patch, dict):
                    steps = list(patch.get("steps") or [])
                    # Mirror Annotated[list, operator.add] — stream patches omit prior steps.
                    merged.update({k: v for k, v in patch.items() if k != "steps"})
                    if steps:
                        merged["steps"] = list(merged.get("steps") or []) + steps
                    note = steps[-1].get("note") if steps else None
                    yield {
                        "type": "node_completed",
                        "node": node_name,
                        "status": "completed",
                        "note": note,
                        "step": steps[-1] if steps else None,
                        "cycle": merged.get("cycle"),
                        "best_val_f1": merged.get("best_val_f1"),
                        "verdict": merged.get("verdict"),
                    }
                else:
                    yield {"type": "node_completed", "node": node_name, "status": "completed"}
                last_node = node_name
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "run_completed", "result": _finalize_payload(merged)}


if __name__ == "__main__":
    ds = REPO / "datasets" / "Titanic-Dataset.csv"
    result = run_simple_loop(dataset_path=str(ds), target_column="Survived", max_cycles=3)
    print(json.dumps(result, indent=2, default=str))
