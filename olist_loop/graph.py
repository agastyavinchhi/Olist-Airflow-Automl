"""Olist research loop — the mini_loop template adapted to Part 1's warehouse table.

Question: predict ``is_bad_review`` (review score <= 2) for delivered Olist orders,
using the cleaned table the Airflow DAG ``olist_pipeline`` loads into Postgres.

Flow (changes vs mini_loop marked):
  START → warehouse_loader (NEW) → data_scientist → hypothesis_generator
        → feature_engineer (NEW) → ml_engineer → evaluator (NEW) → judge
                ↑                                                    │
                └──────── research_memory (MERGED writer+context) ◄── revise
                                                                     │
                                         accept / out of cycles → holdout_eval → END

Agents and who owns what:
  warehouse_loader   code     read olist.orders_clean from Postgres (CSV fallback) + Airflow baselines
  data_scientist     LLM      profile: imbalance, skew, leakage — no modelling advice
  hypothesis_gen     LLM      conceptual direction: features + model family + imbalance handling
  feature_engineer   LLM+code pick recipes from a fixed menu; code applies them (LLM never edits data)
  ml_engineer        LLM+code concrete hyperparameters + threshold strategy; trains on train split
  evaluator          code     full validation report vs majority + Airflow baselines
  judge              LLM      accept / revise from the evaluator report only
  research_memory    code     file the lesson + build the next brief (was 2 nodes)
  holdout_eval       code     refit best config, score the sealed holdout once, log to Postgres

Large data never lives in graph state: each run gets a cache folder and agents pass
file paths (pickle / joblib / npy). State stays small JSON so it can stream to the UI.
"""

from __future__ import annotations

import json
import operator
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
# Repo root locally; /app in the gateway container (loop_runtime.py is mounted there).
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import (  # noqa: E402  — shared LLM plumbing from the template
    DEFAULT_MODEL,
    _resolve_model,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TARGET = "is_bad_review"
TABLE = "olist.orders_clean"
WAREHOUSE_URL = os.getenv(
    "WAREHOUSE_URL", "postgresql+psycopg2://warehouse:warehouse@localhost:5433/warehouse"
)
CSV_FALLBACKS = [
    Path("/bundled/airflow/olist_orders_latest.csv"),  # gateway container mount
    REPO / "datasets" / "airflow" / "olist_orders_latest.csv",
]
CACHE_ROOT = Path(os.getenv("OLIST_LOOP_CACHE", Path(tempfile.gettempdir()) / "olist_loop"))

NUMERIC_FEATURES = [
    "delivery_days", "estimated_days", "days_late", "is_late", "approval_hours",
    "carrier_days", "purchase_month", "purchase_dayofweek", "purchase_hour",
    "n_items", "n_products", "n_sellers", "total_price", "total_freight",
    "freight_ratio", "avg_product_weight_g", "avg_product_volume_cm3",
    "avg_product_photos", "payment_total", "n_payments", "max_installments",
    "used_voucher",
]
CATEGORICAL_FEATURES = ["customer_state", "main_payment_type", "main_category"]

FAMILIES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
THRESHOLD_STRATEGIES = ("default_0.5", "tune_f1")

# The feature engineer may ONLY choose from this menu (all row-wise → no leakage).
RECIPES: dict[str, str] = {
    "log_money": "log1p of price, freight, payment, weight and volume (tames heavy right skew)",
    "lateness_buckets": "categorical bucket of days_late + severely_late flag (> 7 days late)",
    "delivery_speed_ratio": "delivery_days / estimated_days (how badly the promise was missed)",
    "multi_seller_flag": "multi_seller flag + late_x_multi_seller interaction (split shipments)",
    "basket_complexity": "multi_item flag + items_per_product",
    "drop_calendar": "drop purchase_month / dayofweek / hour (likely noise)",
    "drop_state": "drop customer_state (27-level one-hot)",
}

HOLDOUT_FRAC = 0.2   # sealed until holdout_eval
VAL_FRAC = 0.25      # of the remaining 80% → 60 / 20 / 20 train / val / holdout


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class OlistLoopState(TypedDict, total=False):
    run_id: str
    cache_dir: str
    target_score: float          # validation F1 the judge needs to accept
    max_cycles: int
    max_rows: int                # 0 = full table
    cycle: int
    seed: int
    model: str                   # LLM id (OpenRouter / Cursor)
    data: dict                   # loader output: path, source, n_rows, baselines
    profile: dict
    hypothesis: dict
    features: dict               # recipes applied + feature list + path
    experiment_spec: dict
    execution_result: dict
    evaluation: dict
    verdict: dict
    memory: dict
    brief: dict
    best: dict                   # best cycle so far (spec + recipes + threshold + val metrics)
    best_val_f1: float
    holdout_metrics: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# SHARED HELPERS (code, not agents)
# ---------------------------------------------------------------------------
def _engine():
    from sqlalchemy import create_engine

    return create_engine(WAREHOUSE_URL, pool_pre_ping=True)


def _split(df: pd.DataFrame, seed: int):
    """Deterministic train / val / holdout indices — identical in every node."""
    y = df[TARGET]
    rest, hold = train_test_split(df.index, test_size=HOLDOUT_FRAC, random_state=seed, stratify=y)
    train, val = train_test_split(rest, test_size=VAL_FRAC, random_state=seed, stratify=y.loc[rest])
    return train, val, hold


def apply_recipes(df: pd.DataFrame, recipes: list[str]) -> pd.DataFrame:
    """Row-wise feature transforms from the fixed menu. Unknown names are ignored."""
    out = df.copy()
    for r in recipes:
        if r == "log_money":
            for c in ["total_price", "total_freight", "payment_total",
                      "avg_product_weight_g", "avg_product_volume_cm3"]:
                out[f"log_{c}"] = np.log1p(out[c].clip(lower=0))
                out = out.drop(columns=c)
        elif r == "lateness_buckets":
            out["lateness_bucket"] = pd.cut(
                out.days_late, [-np.inf, -14, -7, 0, 7, np.inf],
                labels=["14+_early", "7-14_early", "0-7_early", "1-7_late", "7+_late"],
            ).astype(str)
            out["severely_late"] = (out.days_late > 7).astype(int)
        elif r == "delivery_speed_ratio":
            out["delivery_speed_ratio"] = out.delivery_days / out.estimated_days.clip(lower=1)
        elif r == "multi_seller_flag":
            out["multi_seller"] = (out.n_sellers > 1).astype(int)
            out["late_x_multi_seller"] = out.is_late * out.multi_seller
        elif r == "basket_complexity":
            out["multi_item"] = (out.n_items > 1).astype(int)
            out["items_per_product"] = out.n_items / out.n_products.clip(lower=1)
        elif r == "drop_calendar":
            out = out.drop(columns=["purchase_month", "purchase_dayofweek", "purchase_hour"],
                           errors="ignore")
        elif r == "drop_state":
            out = out.drop(columns=["customer_state"], errors="ignore")
    return out


def _build_estimator(family: str, hp: dict):
    if family == "logistic_regression":
        return LogisticRegression(C=float(hp.get("C", 1.0)), max_iter=2000, class_weight="balanced")
    if family == "random_forest":
        depth = hp.get("max_depth")
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_depth=None if depth in (None, "None", 0) else int(depth),
            min_samples_leaf=int(hp.get("min_samples_leaf", 5)),
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        )
    return HistGradientBoostingClassifier(
        learning_rate=float(hp.get("learning_rate", 0.1)),
        max_iter=int(hp.get("max_iter", 200)),
        max_leaf_nodes=int(hp.get("max_leaf_nodes", 31)),
        l2_regularization=float(hp.get("l2_regularization", 0.0)),
        class_weight="balanced" if hp.get("balanced", True) else None,
        random_state=42,
    )


def _pipeline(X: pd.DataFrame, family: str, hp: dict) -> Pipeline:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                               min_frequency=50, sparse_output=False))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", _build_estimator(family, hp))])


def _clamp_hp(family: str, hp: dict) -> dict:
    """Keep LLM-proposed numbers inside safe ranges (cost + validity)."""
    def num(key, lo, hi, default, cast=float):
        try:
            return cast(min(hi, max(lo, cast(hp.get(key, default)))))
        except (TypeError, ValueError):
            return default
    if family == "logistic_regression":
        return {"C": num("C", 0.01, 100.0, 1.0)}
    if family == "random_forest":
        depth = hp.get("max_depth")
        return {
            "n_estimators": num("n_estimators", 50, 400, 200, int),
            "max_depth": None if depth in (None, "None", 0) else num("max_depth", 3, 30, 14, int),
            "min_samples_leaf": num("min_samples_leaf", 1, 50, 5, int),
        }
    return {
        "learning_rate": num("learning_rate", 0.01, 0.5, 0.1),
        "max_iter": num("max_iter", 50, 600, 200, int),
        "max_leaf_nodes": num("max_leaf_nodes", 8, 128, 31, int),
        "l2_regularization": num("l2_regularization", 0.0, 10.0, 0.0),
        "balanced": bool(hp.get("balanced", True)),
    }


def _metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "threshold": round(float(threshold), 3),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _llm_json(state, system: str, user: str, max_tokens: int = 600, temperature: float = 0.2):
    """(parsed_json | None, note, used_llm, model) — every LLM agent goes through here."""
    model = _resolve_model(state)
    if not llm_configured(model):
        return None, "no LLM key; heuristic", False, model
    content, err = openrouter_chat(system, user, model=model, max_tokens=max_tokens,
                                   temperature=temperature, json_object=True)
    parsed = extract_json_object(content or "") if content else None
    if parsed is None:
        return None, f"{err or 'llm unparsable'}; heuristic", False, model
    return parsed, f"llm · {model}", True, model


def _history(state) -> list:
    return list((state.get("memory") or {}).get("history") or [])


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def node_warehouse_loader(state: OlistLoopState) -> dict:
    """NEW. Read Part 1's cleaned table from Postgres + the Airflow baseline metrics."""
    cache = Path(state["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    cols = ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    source, baselines, note = "postgres", {}, ""
    try:
        eng = _engine()
        df = pd.read_sql(f"SELECT {cols} FROM {TABLE} ORDER BY order_id", eng)
        try:
            b = pd.read_sql(
                "SELECT model, accuracy, precision, recall, f1, roc_auc FROM olist.model_runs "
                "WHERE run_ts = (SELECT max(run_ts) FROM olist.model_runs)", eng)
            baselines = {r.model: {k: float(getattr(r, k)) for k in
                                   ("accuracy", "precision", "recall", "f1", "roc_auc")}
                         for r in b.itertuples()}
        except Exception:  # noqa: BLE001 — table may not exist yet
            baselines = {}
        note = f"read {len(df):,} rows from {TABLE}"
    except Exception as exc:  # noqa: BLE001
        csv = next((p for p in CSV_FALLBACKS if p.is_file()), None)
        if csv is None:
            raise RuntimeError(f"Postgres unreachable ({exc}) and no CSV fallback") from exc
        df = pd.read_csv(csv)
        source = "csv"
        note = f"Postgres unreachable → CSV fallback {csv.name} ({len(df):,} rows)"

    max_rows = int(state.get("max_rows") or 0)
    if max_rows and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, stratify=df[TARGET],
                                 random_state=int(state.get("seed") or 42))
        note += f" → stratified sample {max_rows:,}"
    df = df.reset_index(drop=True)
    path = cache / "base.pkl"
    df.to_pickle(path)
    train, val, hold = _split(df, int(state.get("seed") or 42))
    data = {
        "path": str(path), "source": source, "table": TABLE if source == "postgres" else None,
        "n_rows": int(len(df)), "n_train": len(train), "n_val": len(val), "n_holdout": len(hold),
        "target_rate": round(float(df[TARGET].mean()), 4),
        "baselines": baselines,
    }
    return {"data": data, "steps": [{"node": "warehouse_loader", "note": note, "output": data}]}


def node_data_scientist(state: OlistLoopState) -> dict:
    """Profile only — imbalance, skew, leakage. No model or feature choices."""
    df = pd.read_pickle(state["data"]["path"])
    num = [c for c in NUMERIC_FEATURES if c in df]
    skew = df[num].skew().sort_values(ascending=False)
    profile = {
        "n_rows": int(len(df)),
        "target": TARGET,
        "target_rate": round(float(df[TARGET].mean()), 4),
        "numeric_features": num,
        "categorical_features": [c for c in CATEGORICAL_FEATURES if c in df],
        "highly_skewed": [c for c, s in skew.items() if s > 3],
        "top_target_correlations": {
            c: round(float(v), 3) for c, v in
            df[num + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
            .head(6).items()},
        "leakage_suspects": [],
        "notes": [
            f"Imbalanced target: {df[TARGET].mean():.1%} positive → accuracy is misleading; optimise F1.",
            "review_score / review dates were excluded upstream (they define the label).",
        ],
    }
    system = (
        "You are the Data Scientist only. Profile the dataset and flag risks "
        "(imbalance, skew, leakage, weak signals). Do NOT choose models, hyperparameters "
        "or feature recipes. Return ONLY JSON with keys: notes (string[] ≤5 short bullets), "
        "leakage_suspects (string[] — only real column names), key_signals (string[])."
    )
    user = (f"target={TARGET} (1 = review score <= 2)\nstats={json.dumps(profile, default=str)}\n"
            f"describe={df[num].describe().round(2).to_dict()}")
    parsed, note, used, model = _llm_json(state, system, user, max_tokens=700)
    if parsed:
        cols = set(df.columns)
        profile["notes"] = [str(n) for n in (parsed.get("notes") or profile["notes"])][:5]
        profile["leakage_suspects"] = [c for c in parsed.get("leakage_suspects") or [] if c in cols]
        profile["key_signals"] = [str(s) for s in parsed.get("key_signals") or []][:5]
    return {"profile": profile, "steps": [{"node": "data_scientist", "note": note,
                                           "used_llm": used, "model": model, "output": profile}]}


def _heuristic_hypothesis(history: list) -> dict:
    n = len(history)
    ladder = [
        ("Start with a regularised linear baseline plus lateness buckets.",
         "baseline_then_iterate", ["lateness_buckets"], "logistic_regression"),
        ("Move to a tree ensemble with delivery-ratio and multi-seller features; tune the threshold.",
         "capacity", ["lateness_buckets", "delivery_speed_ratio", "multi_seller_flag"], "random_forest"),
        ("Switch to gradient boosting with the full recipe set and a tuned threshold.",
         "model_family_switch",
         ["lateness_buckets", "delivery_speed_ratio", "multi_seller_flag", "log_money", "basket_complexity"],
         "hist_gradient_boosting"),
    ]
    d, focus, recipes, fam = ladder[min(n, len(ladder) - 1)]
    return {"direction": d, "why": "heuristic ladder", "focus": focus,
            "feature_ideas": recipes, "model_family_hint": fam,
            "imbalance": "tune decision threshold on validation"}


def node_hypothesis(state: OlistLoopState) -> dict:
    """Conceptual next experiment: feature ideas + model family + imbalance handling."""
    history = _history(state)
    plan = _heuristic_hypothesis(history)
    system = (
        "You are the Hypothesis Generator. Propose ONE conceptual next experiment for "
        "predicting bad reviews. You may suggest feature recipes (by name, from the menu), "
        "a model family, and how to handle class imbalance. Do NOT give hyperparameter numbers. "
        "Return ONLY JSON: direction (1-2 sentences), why (cite profile / judge feedback), "
        f"focus (baseline_then_iterate|capacity|model_family_switch|feature_work), "
        f"feature_ideas (subset of {list(RECIPES)}), model_family_hint (one of {list(FAMILIES)}), "
        "imbalance (short phrase). Avoid repeating a failed (family, features) combination."
    )
    user = (f"profile={json.dumps(state.get('profile'), default=str)}\n"
            f"recipe_menu={json.dumps(RECIPES)}\n"
            f"latest_verdict={json.dumps(state.get('verdict'), default=str)}\n"
            f"brief={json.dumps(state.get('brief'), default=str)}")
    parsed, note, used, model = _llm_json(state, system, user, max_tokens=500, temperature=0.4)
    if parsed and parsed.get("direction"):
        plan = {
            "direction": str(parsed["direction"]),
            "why": str(parsed.get("why") or ""),
            "focus": str(parsed.get("focus") or plan["focus"]),
            "feature_ideas": [r for r in parsed.get("feature_ideas") or [] if r in RECIPES]
            or plan["feature_ideas"],
            "model_family_hint": parsed.get("model_family_hint")
            if parsed.get("model_family_hint") in FAMILIES else plan["model_family_hint"],
            "imbalance": str(parsed.get("imbalance") or plan["imbalance"]),
        }
    return {"hypothesis": plan, "steps": [{"node": "hypothesis_generator", "note": note,
                                           "used_llm": used, "model": model, "output": plan}]}


def node_feature_engineer(state: OlistLoopState) -> dict:
    """NEW. Choose recipes from the fixed menu (LLM) → apply them (code)."""
    plan = state.get("hypothesis") or {}
    recipes = list(plan.get("feature_ideas") or [])
    system = (
        "You are the Feature Engineer. Pick the feature recipes to apply this cycle, ONLY from "
        "the menu. You cannot write code or invent recipes. Return ONLY JSON: "
        "recipes (string[] subset of menu), rationale (one sentence)."
    )
    user = (f"hypothesis={json.dumps(plan, default=str)}\nmenu={json.dumps(RECIPES)}\n"
            f"profile_signals={json.dumps((state.get('profile') or {}).get('key_signals'))}\n"
            f"recent_history={json.dumps(_history(state)[-3:], default=str)}")
    parsed, note, used, model = _llm_json(state, system, user, max_tokens=300)
    rationale = "applied hypothesis feature ideas"
    if parsed and isinstance(parsed.get("recipes"), list):
        picked = [r for r in parsed["recipes"] if r in RECIPES]
        if picked or parsed["recipes"] == []:
            recipes, rationale = picked, str(parsed.get("rationale") or rationale)

    base = pd.read_pickle(state["data"]["path"])
    feat = apply_recipes(base, recipes)
    cycle = int(state.get("cycle") or 0) + 1
    path = Path(state["cache_dir"]) / f"features_c{cycle}.pkl"
    feat.to_pickle(path)
    added = sorted(set(feat.columns) - set(base.columns))
    removed = sorted(set(base.columns) - set(feat.columns))
    out = {"recipes": recipes, "rationale": rationale, "path": str(path),
           "n_features": int(feat.shape[1] - 1), "added": added, "removed": removed}
    return {"features": out, "steps": [{
        "node": "feature_engineer", "used_llm": used, "model": model, "output": out,
        "note": f"{note} · {len(recipes)} recipes · +{len(added)}/−{len(removed)} cols"}]}


def node_ml_engineer(state: OlistLoopState) -> dict:
    """Concrete hyperparameters + threshold strategy (LLM) → train on the train split (code)."""
    plan = state.get("hypothesis") or {}
    history = _history(state)
    fam = plan.get("model_family_hint") or "random_forest"
    defaults = {
        "logistic_regression": {"C": 1.0},
        "random_forest": {"n_estimators": 250, "max_depth": 14, "min_samples_leaf": 5},
        "hist_gradient_boosting": {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31,
                                   "l2_regularization": 1.0, "balanced": True},
    }
    spec = {"model_family": fam, "hyperparameters": defaults[fam],
            "threshold_strategy": "default_0.5" if not history else "tune_f1",
            "rationale": "heuristic defaults for hinted family"}
    system = (
        "You are the ML Engineer. Turn the hypothesis into a concrete training config. "
        "Return ONLY JSON: model_family (one of "
        f"{list(FAMILIES)}), hyperparameters (object: logistic_regression → C; random_forest → "
        "n_estimators ≤400, max_depth, min_samples_leaf; hist_gradient_boosting → learning_rate, "
        "max_iter ≤600, max_leaf_nodes, l2_regularization, balanced bool), threshold_strategy "
        f"(one of {list(THRESHOLD_STRATEGIES)}), rationale (one sentence)."
    )
    user = (f"hypothesis={json.dumps(plan, default=str)}\n"
            f"features={json.dumps({k: (state.get('features') or {}).get(k) for k in ('recipes', 'n_features')})}\n"
            f"recent_history={json.dumps(history[-3:], default=str)}")
    parsed, note, used, model = _llm_json(state, system, user, max_tokens=400)
    if parsed and parsed.get("model_family") in FAMILIES:
        spec = {"model_family": parsed["model_family"],
                "hyperparameters": parsed.get("hyperparameters") or {},
                "threshold_strategy": parsed.get("threshold_strategy")
                if parsed.get("threshold_strategy") in THRESHOLD_STRATEGIES else "tune_f1",
                "rationale": str(parsed.get("rationale") or "")}
    spec["hyperparameters"] = _clamp_hp(spec["model_family"], dict(spec["hyperparameters"]))
    cycle = int(state.get("cycle") or 0) + 1
    spec["experiment_id"] = f"exp-{cycle}"

    df = pd.read_pickle(state["features"]["path"])
    train, val, _ = _split(df, int(state.get("seed") or 42))
    X, y = df.drop(columns=TARGET), df[TARGET]
    t0 = time.time()
    try:
        pipe = _pipeline(X, spec["model_family"], spec["hyperparameters"]).fit(X.loc[train], y.loc[train])
    except Exception as exc:  # noqa: BLE001 — self-repair: fall back to safe defaults once
        fam = spec["model_family"]
        spec["hyperparameters"] = _clamp_hp(fam, defaults[fam])
        note += f" · repaired after {type(exc).__name__}"
        pipe = _pipeline(X, fam, spec["hyperparameters"]).fit(X.loc[train], y.loc[train])
    secs = round(time.time() - t0, 2)
    cache = Path(state["cache_dir"])
    proba = pipe.predict_proba(X.loc[val])[:, 1]
    np.save(cache / f"val_proba_c{cycle}.npy", proba)
    result = {"status": "succeeded", "experiment_id": spec["experiment_id"],
              "model_family": spec["model_family"], "train_seconds": secs,
              "n_train": len(train), "n_val": len(val),
              "val_proba_path": str(cache / f"val_proba_c{cycle}.npy")}
    return {"experiment_spec": spec, "execution_result": result, "cycle": cycle,
            "steps": [{"node": "ml_engineer", "used_llm": used, "model": model,
                       "experiment_id": spec["experiment_id"], "model_family": spec["model_family"],
                       "spec": spec, "output": {"experiment_spec": spec, "execution_result": result},
                       "note": f"{note} · {spec['model_family']} trained in {secs}s"}]}


def node_evaluator(state: OlistLoopState) -> dict:
    """NEW. Measure the validation predictions: full metric set + baseline comparison."""
    df = pd.read_pickle(state["features"]["path"])
    _, val, _ = _split(df, int(state.get("seed") or 42))
    y = df[TARGET].loc[val].to_numpy()
    proba = np.load(state["execution_result"]["val_proba_path"])
    spec = state["experiment_spec"]
    threshold = 0.5
    if spec.get("threshold_strategy") == "tune_f1":
        p, r, t = precision_recall_curve(y, proba)
        f1s = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
        threshold = float(t[int(np.argmax(f1s))])
    m = _metrics(y, proba, threshold)
    m_default = _metrics(y, proba, 0.5)
    airflow = (state.get("data") or {}).get("baselines") or {}
    rf = airflow.get("random_forest") or {}
    report = {
        "experiment_id": spec.get("experiment_id"),
        "model_family": spec.get("model_family"),
        "recipes": (state.get("features") or {}).get("recipes"),
        "validation": m,
        "f1_at_0.5": m_default["f1"],
        "majority_baseline_accuracy": round(1 - float(y.mean()), 4),
        "airflow_rf": {k: rf.get(k) for k in ("accuracy", "f1", "roc_auc")} if rf else None,
        "beats_airflow_f1": bool(rf) and m["f1"] > rf.get("f1", 1),
    }
    patch: dict = {"evaluation": report}
    if m["f1"] > float(state.get("best_val_f1") or -1):
        patch["best_val_f1"] = m["f1"]
        patch["best"] = {"spec": spec, "recipes": report["recipes"], "threshold": threshold,
                         "validation": m, "cycle": state.get("cycle")}
    note = (f"val F1 {m['f1']:.3f} (thr {threshold:.2f}) · acc {m['accuracy']:.3f} · "
            f"AUC {m['roc_auc']:.3f}" + (f" · Airflow RF F1 {rf['f1']:.3f}" if rf else ""))
    patch["steps"] = [{"node": "evaluator", "note": note, "f1": m["f1"],
                       "validation_metrics": m, "output": report}]
    return patch


def node_judge(state: OlistLoopState) -> dict:
    """Accept / revise from the evaluator report only (never re-scores, never edits configs)."""
    ev = state.get("evaluation") or {}
    f1 = float((ev.get("validation") or {}).get("f1") or 0)
    target = float(state.get("target_score") or 0.5)
    verdict = ({"action": "accept", "observation": f"val F1 {f1:.3f} ≥ target {target:.2f}",
                "next_action": "finalize"} if f1 >= target else
               {"action": "revise", "observation": f"val F1 {f1:.3f} < target {target:.2f}",
                "next_action": "try different features or a stronger model family"})
    system = (
        "You are the Judge. Decide from the evaluator report only. accept ONLY if validation "
        "f1 >= target_score; otherwise revise (or reject if the run is broken). Return ONLY JSON: "
        "action (accept|revise|reject), observation (1-2 sentences citing F1 and AUC), "
        "next_action (conceptual advice for the hypothesis agent, no numbers)."
    )
    user = f"target_score={target}\nreport={json.dumps(ev, default=str)}"
    parsed, note, used, model = _llm_json(state, system, user, max_tokens=400, temperature=0.1)
    if parsed and parsed.get("action") in {"accept", "revise", "reject"}:
        action = parsed["action"]
        if action == "accept" and f1 < target:  # guardrail: the rule outranks the LLM
            action = "revise"
        verdict = {"action": action,
                   "observation": str(parsed.get("observation") or verdict["observation"]),
                   "next_action": str(parsed.get("next_action") or verdict["next_action"])}
    return {"verdict": verdict, "steps": [{"node": "judge", "note": note, "used_llm": used,
                                           "model": model, "verdict": verdict}]}


def node_research_memory(state: OlistLoopState) -> dict:
    """MERGED memory_writer + context_builder: file the lesson, build the next brief."""
    mem = dict(state.get("memory") or {"history": [], "lessons": []})
    ev = state.get("evaluation") or {}
    hist = list(mem.get("history") or [])
    hist.append({
        "cycle": state.get("cycle"),
        "model_family": ev.get("model_family"),
        "recipes": ev.get("recipes"),
        "hyperparameters": (state.get("experiment_spec") or {}).get("hyperparameters"),
        "val_f1": (ev.get("validation") or {}).get("f1"),
        "val_auc": (ev.get("validation") or {}).get("roc_auc"),
        "verdict": (state.get("verdict") or {}).get("action"),
    })
    lessons = list(mem.get("lessons") or []) + [str((state.get("verdict") or {}).get("observation"))]
    mem = {"history": hist, "lessons": lessons}
    brief = {"tried": hist, "lessons": lessons[-3:], "best_val_f1": state.get("best_val_f1"),
             "next_action": (state.get("verdict") or {}).get("next_action")}
    return {"memory": mem, "brief": brief, "steps": [{
        "node": "research_memory", "note": f"filed cycle {state.get('cycle')} · brief ready",
        "output": brief}]}


def node_holdout(state: OlistLoopState) -> dict:
    """Refit the best config on train+val, score the sealed holdout ONCE, log to Postgres."""
    best = state.get("best") or {}
    spec = best.get("spec") or state.get("experiment_spec") or {}
    recipes = best.get("recipes") or []
    threshold = float(best.get("threshold") or 0.5)
    base = pd.read_pickle(state["data"]["path"])
    df = apply_recipes(base, recipes)
    train, val, hold = _split(df, int(state.get("seed") or 42))
    fit_idx = train.append(val)
    X, y = df.drop(columns=TARGET), df[TARGET]
    pipe = _pipeline(X, spec["model_family"], spec["hyperparameters"]).fit(X.loc[fit_idx], y.loc[fit_idx])
    proba = pipe.predict_proba(X.loc[hold])[:, 1]
    m = _metrics(y.loc[hold].to_numpy(), proba, threshold)
    m["n_holdout"] = len(hold)
    m["majority_baseline_accuracy"] = round(1 - float(y.loc[hold].mean()), 4)
    m["model_family"] = spec["model_family"]
    m["recipes"] = recipes
    m["hyperparameters"] = spec["hyperparameters"]
    m["airflow_baselines"] = (state.get("data") or {}).get("baselines")
    note = (f"holdout F1 {m['f1']:.3f} · acc {m['accuracy']:.3f} · AUC {m['roc_auc']:.3f} · "
            f"thr {threshold:.2f}")
    if (state.get("data") or {}).get("source") == "postgres":
        try:
            row = {"run_ts": datetime.now(timezone.utc).isoformat(), "run_id": state.get("run_id"),
                   "llm_model": state.get("model"), "cycles": state.get("cycle"),
                   "model_family": spec["model_family"], "recipes": ",".join(recipes),
                   "hyperparameters": json.dumps(spec["hyperparameters"]),
                   **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc",
                                         "pr_auc", "threshold", "n_holdout")}}
            pd.DataFrame([row]).to_sql("agent_runs", _engine(), schema="olist",
                                       if_exists="append", index=False)
            note += " · logged to olist.agent_runs"
        except Exception as exc:  # noqa: BLE001
            note += f" · (db log skipped: {type(exc).__name__})"
    return {"holdout_metrics": m, "stop_reason": "finalize",
            "steps": [{"node": "holdout_eval", "note": note, "metrics": m}]}


def route_after_judge(state: OlistLoopState) -> str:
    """Router = plain code (not an LLM)."""
    action = str((state.get("verdict") or {}).get("action") or "stop")
    if action == "revise" and int(state.get("cycle") or 0) < int(state.get("max_cycles") or 3):
        return "memory"
    return "holdout"


# ---------------------------------------------------------------------------
# GRAPH WIRING
# ---------------------------------------------------------------------------
NODES = (
    "warehouse_loader", "data_scientist", "hypothesis_generator", "feature_engineer",
    "ml_engineer", "evaluator", "judge", "research_memory", "holdout_eval",
)


def build_graph():
    g = StateGraph(OlistLoopState)
    g.add_node("warehouse_loader", node_warehouse_loader)
    g.add_node("data_scientist", node_data_scientist)
    g.add_node("hypothesis_generator", node_hypothesis)
    g.add_node("feature_engineer", node_feature_engineer)
    g.add_node("ml_engineer", node_ml_engineer)
    g.add_node("evaluator", node_evaluator)
    g.add_node("judge", node_judge)
    g.add_node("research_memory", node_research_memory)
    g.add_node("holdout_eval", node_holdout)

    g.add_edge(START, "warehouse_loader")
    g.add_edge("warehouse_loader", "data_scientist")
    g.add_edge("data_scientist", "hypothesis_generator")
    g.add_edge("hypothesis_generator", "feature_engineer")
    g.add_edge("feature_engineer", "ml_engineer")
    g.add_edge("ml_engineer", "evaluator")
    g.add_edge("evaluator", "judge")
    g.add_conditional_edges("judge", route_after_judge,
                            {"memory": "research_memory", "holdout": "holdout_eval"})
    g.add_edge("research_memory", "hypothesis_generator")  # loop back
    g.add_edge("holdout_eval", END)
    return g.compile()


def get_status() -> dict:
    try:
        app = build_graph()
        nodes = sorted(n for n in app.get_graph().nodes if not n.startswith("__"))
        return {"ready": True, "nodes": nodes, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "nodes": [], "error": str(exc)}


def _initial_state(*, target_score: float, max_cycles: int, max_rows: int, seed: int,
                   model: str | None) -> OlistLoopState:
    run_id = f"olr_{uuid.uuid4().hex[:10]}"
    return {
        "run_id": run_id, "cache_dir": str(CACHE_ROOT / run_id),
        "target_score": target_score, "max_cycles": max_cycles, "max_rows": max_rows,
        "cycle": 0, "seed": seed, "model": (model or "").strip() or DEFAULT_MODEL,
        "best_val_f1": -1.0, "steps": [], "memory": {"history": [], "lessons": []}, "brief": {},
    }


def _finalize_payload(final: dict) -> dict:
    return {k: final.get(k) for k in (
        "run_id", "model", "cycle", "data", "profile", "best", "best_val_f1",
        "holdout_metrics", "verdict", "memory", "stop_reason", "steps")} | {
        "cycles": final.get("cycle"), "target_column": TARGET,
        "dataset_path": (final.get("data") or {}).get("table") or (final.get("data") or {}).get("path")}


def run_olist_loop(*, target_score: float = 0.5, max_cycles: int = 3, max_rows: int = 0,
                   seed: int = 42, model: str | None = None) -> dict:
    app = build_graph()
    init = _initial_state(target_score=target_score, max_cycles=max_cycles,
                          max_rows=max_rows, seed=seed, model=model)
    final = app.invoke(init, config={"recursion_limit": 10 + max_cycles * 8})
    return _finalize_payload(final)


def stream_olist_loop(*, target_score: float = 0.5, max_cycles: int = 3, max_rows: int = 0,
                      seed: int = 42, model: str | None = None):
    """Yield UI events as each node finishes (consumed by the gateway SSE endpoint)."""
    app = build_graph()
    init = _initial_state(target_score=target_score, max_cycles=max_cycles,
                          max_rows=max_rows, seed=seed, model=model)
    yield {"type": "run_started", "run_id": init["run_id"], "model": init["model"],
           "dataset_path": TABLE, "target_column": TARGET}
    merged: dict = dict(init)
    yield {"type": "node_started", "node": "warehouse_loader", "status": "running"}
    try:
        for update in app.stream(init, stream_mode="updates",
                                 config={"recursion_limit": 10 + max_cycles * 8}):
            for node, patch in (update or {}).items():
                steps = list((patch or {}).get("steps") or [])
                merged.update({k: v for k, v in (patch or {}).items() if k != "steps"})
                merged["steps"] = list(merged.get("steps") or []) + steps
                yield {"type": "node_completed", "node": node, "status": "completed",
                       "note": steps[-1].get("note") if steps else None,
                       "step": steps[-1] if steps else None, "cycle": merged.get("cycle"),
                       "best_val_f1": merged.get("best_val_f1"), "verdict": merged.get("verdict")}
                # Announce the next node so the UI can light it while it works.
                nxt = {"warehouse_loader": "data_scientist", "data_scientist": "hypothesis_generator",
                       "hypothesis_generator": "feature_engineer", "feature_engineer": "ml_engineer",
                       "ml_engineer": "evaluator", "evaluator": "judge",
                       "research_memory": "hypothesis_generator"}.get(node)
                if node == "judge":
                    route = route_after_judge(merged)  # same function LangGraph calls
                    yield {"type": "router", "node": "router", "status": "completed",
                           "action": (merged.get("verdict") or {}).get("action"), "route": route}
                    nxt = "research_memory" if route == "memory" else "holdout_eval"
                if nxt:
                    yield {"type": "node_started", "node": nxt, "status": "running"}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return
    yield {"type": "run_completed", "result": _finalize_payload(merged)}
