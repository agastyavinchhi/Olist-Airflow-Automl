"""Olist bad-review agent loop — diagnosis-routed LangGraph (Part 2).

Problem: flag delivered orders likely to get a bad review (score <= 2) so customer support
can intervene. Data: ``olist.orders_clean`` written by the Airflow DAG ``olist_pipeline``
(Olist e-commerce tables + IBGE city statistics).

Graph (our design — not the class revise loop):

  START → warehouse_loader → data_scientist → strategist → feature_engineer → ml_engineer
                                                 ↑              ↑                  ↑     │
                                                 │ rethink      │ features   model │     ▼
                                                 └────── research_memory ◄─ revise ─ judge ◄─ evaluator
                                                                                 │
                                                       finish (target met · plateau · budget)
                                                                                 ▼
                                                                   holdout_eval → END

The evaluator DIAGNOSES every cycle, and the route depends on the diagnosis:
  * weak_ranking   (ROC-AUC not clearly above the Airflow baseline) → feature_engineer
                   (the model can't separate classes → it needs better inputs)
  * cutoff_or_capacity (ranking is good but F1 is low)               → ml_engineer
                   (same features; change threshold strategy / capacity / family)
  * stuck          (the same fix failed to improve twice in a row)  → strategist (new plan)
  * target met / no improvement for 2 cycles / budget spent         → holdout_eval
The judge (LLM) may pick any *allowed* route; code computes the allowed set and the
recommendation, so a bad LLM reply can never break the loop.

Agents:
  warehouse_loader  code      read the Postgres table + Airflow baseline metrics
  data_scientist    LLM       profile risks (imbalance, skew, leakage, city coverage)
  strategist        LLM       initial plan + full re-plan on "rethink"
  feature_engineer  LLM+code  choose recipes from a fixed menu; code applies them
  ml_engineer       LLM+code  model family, hyperparameters, threshold strategy; trains
  evaluator         code      metrics + diagnosis + recommended route
  judge             LLM       choose the next route from the allowed set
  research_memory   code      log the cycle, build the brief, dispatch the route
  holdout_eval      code      refit best config, score the sealed holdout once, log to Postgres
"""

from __future__ import annotations

import json
import operator
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

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

from olist_loop.llm import DEFAULT_MODEL, chat_json

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
TARGET = "is_bad_review"
TABLE = "olist.orders_clean"
WAREHOUSE_URL = os.getenv(
    "WAREHOUSE_URL", "postgresql+psycopg2://warehouse:warehouse@localhost:5433/warehouse")
CSV_FALLBACKS = [Path("/bundled/airflow/olist_orders_latest.csv"),
                 REPO / "datasets" / "airflow" / "olist_orders_latest.csv"]
CACHE_ROOT = Path(os.getenv("OLIST_LOOP_CACHE", Path(tempfile.gettempdir()) / "olist_loop"))

NUMERIC_FEATURES = [
    "delivery_days", "estimated_days", "days_late", "is_late", "approval_hours",
    "carrier_days", "purchase_month", "purchase_dayofweek", "purchase_hour",
    "n_items", "n_products", "n_sellers", "total_price", "total_freight",
    "freight_ratio", "avg_product_weight_g", "avg_product_volume_cm3",
    "avg_product_photos", "payment_total", "n_payments", "max_installments",
    "used_voucher",
    "city_matched", "city_population_log", "city_hdi", "city_hdi_income",
    "city_hdi_education", "city_gdp_per_capita", "city_is_capital", "city_distance_to_sp_km",
]
CATEGORICAL_FEATURES = ["customer_state", "main_payment_type", "main_category", "city_rural_urban"]
CITY_COLUMNS = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c.startswith("city_")]

FAMILIES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
THRESHOLDS = ("default_0.5", "tune_f1")
ROUTES = ("features", "model", "rethink", "finish")

# Feature recipes: the ONLY transforms the feature engineer can request (row-wise → no leakage).
RECIPES: dict[str, str] = {
    "lateness_buckets": "categorical days_late bucket + severely_late flag (> 7 days late)",
    "delivery_speed_ratio": "delivery_days / estimated_days (how badly the promise was missed)",
    "remote_city_x_late": "lateness × distance-from-São-Paulo and × rural/remote city (IBGE)",
    "multi_seller_flag": "multi_seller flag + late_x_multi_seller interaction (split shipments)",
    "log_money": "log1p of price, freight, payment, weight, volume (tames right skew)",
    "basket_complexity": "multi_item flag + items_per_product",
    "distance_bands": "categorical distance-from-São-Paulo bands (IBGE coordinates)",
    "drop_calendar": "drop purchase month / weekday / hour (likely noise)",
    "drop_city_context": "ablation: drop every IBGE city column",
}
FEATURE_PRIORITY = ["lateness_buckets", "delivery_speed_ratio", "remote_city_x_late",
                    "multi_seller_flag", "log_money", "basket_complexity", "distance_bands"]

HOLDOUT_FRAC, VAL_FRAC = 0.2, 0.25          # → 60 / 20 / 20 train / val / sealed holdout
PLATEAU_EPS, PLATEAU_CYCLES = 0.002, 2       # stop after 2 cycles without +0.002 F1
AUC_MARGIN = 0.005                          # "good ranking" = AUC ≥ Airflow RF AUC + 0.005


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class OlistState(TypedDict, total=False):
    run_id: str
    cache_dir: str
    model: str
    seed: int
    target_score: float
    max_cycles: int
    max_rows: int
    cycle: int
    data: dict
    profile: dict
    plan: dict
    features: dict
    spec: dict
    result: dict
    evaluation: dict
    decision: dict          # judge output: route + reasons
    route: str
    no_improve: int
    route_log: list
    history: list
    brief: dict
    best: dict
    best_val_f1: float
    holdout_metrics: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# HELPERS (plain code, not agents)
# ---------------------------------------------------------------------------
def _engine():
    from sqlalchemy import create_engine

    return create_engine(WAREHOUSE_URL, pool_pre_ping=True)


def _split(df: pd.DataFrame, seed: int):
    y = df[TARGET]
    rest, hold = train_test_split(df.index, test_size=HOLDOUT_FRAC, random_state=seed, stratify=y)
    train, val = train_test_split(rest, test_size=VAL_FRAC, random_state=seed, stratify=y.loc[rest])
    return train, val, hold


def apply_recipes(df: pd.DataFrame, recipes: list[str]) -> pd.DataFrame:
    out = df.copy()
    for r in recipes:
        if r == "lateness_buckets":
            out["lateness_bucket"] = pd.cut(
                out.days_late, [-np.inf, -14, -7, 0, 7, np.inf],
                labels=["14+_early", "7-14_early", "0-7_early", "1-7_late", "7+_late"]).astype(str)
            out["severely_late"] = (out.days_late > 7).astype(int)
        elif r == "delivery_speed_ratio":
            out["delivery_speed_ratio"] = out.delivery_days / out.estimated_days.clip(lower=1)
        elif r == "remote_city_x_late" and "city_distance_to_sp_km" in out:
            out["late_x_distance_1000km"] = out.is_late * out.city_distance_to_sp_km / 1000
            out["late_x_remote_city"] = out.is_late * out.city_rural_urban.str.endswith("remote").astype(int)
        elif r == "multi_seller_flag":
            out["multi_seller"] = (out.n_sellers > 1).astype(int)
            out["late_x_multi_seller"] = out.is_late * out.multi_seller
        elif r == "log_money":
            for c in ["total_price", "total_freight", "payment_total",
                      "avg_product_weight_g", "avg_product_volume_cm3"]:
                out[f"log_{c}"] = np.log1p(out[c].clip(lower=0))
                out = out.drop(columns=c)
        elif r == "basket_complexity":
            out["multi_item"] = (out.n_items > 1).astype(int)
            out["items_per_product"] = out.n_items / out.n_products.clip(lower=1)
        elif r == "distance_bands" and "city_distance_to_sp_km" in out:
            out["distance_band"] = pd.cut(out.city_distance_to_sp_km, [-1, 100, 500, 1000, 2000, np.inf],
                                          labels=["<100", "100-500", "500-1k", "1k-2k", "2k+"]).astype(str)
        elif r == "drop_calendar":
            out = out.drop(columns=["purchase_month", "purchase_dayofweek", "purchase_hour"], errors="ignore")
        elif r == "drop_city_context":
            out = out.drop(columns=[c for c in out.columns if c.startswith("city_")], errors="ignore")
    return out


def _estimator(family: str, hp: dict):
    if family == "logistic_regression":
        return LogisticRegression(C=float(hp.get("C", 1.0)), max_iter=2000, class_weight="balanced")
    if family == "random_forest":
        d = hp.get("max_depth")
        return RandomForestClassifier(
            n_estimators=int(hp.get("n_estimators", 250)),
            max_depth=None if d in (None, "None", 0) else int(d),
            min_samples_leaf=int(hp.get("min_samples_leaf", 5)),
            class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    return HistGradientBoostingClassifier(
        learning_rate=float(hp.get("learning_rate", 0.08)), max_iter=int(hp.get("max_iter", 300)),
        max_leaf_nodes=int(hp.get("max_leaf_nodes", 31)),
        l2_regularization=float(hp.get("l2_regularization", 1.0)),
        class_weight="balanced" if hp.get("balanced", True) else None, random_state=42)


DEFAULT_HP = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {"n_estimators": 250, "max_depth": 14, "min_samples_leaf": 5},
    "hist_gradient_boosting": {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31,
                               "l2_regularization": 1.0, "balanced": True},
}


def _clamp(family: str, hp: dict) -> dict:
    """Keep LLM-proposed hyperparameters inside safe ranges."""
    base = {**DEFAULT_HP[family], **(hp or {})}

    def num(k, lo, hi, cast=float):
        try:
            v = cast(min(hi, max(lo, cast(base[k]))))
            return round(v, 4) if cast is float else v
        except (TypeError, ValueError):
            return DEFAULT_HP[family][k]
    if family == "logistic_regression":
        return {"C": num("C", 0.01, 100.0)}
    if family == "random_forest":
        d = base.get("max_depth")
        return {"n_estimators": num("n_estimators", 50, 400, int),
                "max_depth": None if d in (None, "None", 0) else num("max_depth", 3, 30, int),
                "min_samples_leaf": num("min_samples_leaf", 1, 50, int)}
    return {"learning_rate": num("learning_rate", 0.01, 0.5), "max_iter": num("max_iter", 50, 800, int),
            "max_leaf_nodes": num("max_leaf_nodes", 8, 128, int),
            "l2_regularization": num("l2_regularization", 0.0, 10.0), "balanced": bool(base.get("balanced", True))}


def _pipeline(X: pd.DataFrame, family: str, hp: dict) -> Pipeline:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                               sparse_output=False))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", _estimator(family, hp))])


def _metrics(y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "threshold": round(float(thr), 3),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def _step(node: str, note: str, **kw) -> dict:
    return {"steps": [{"node": node, "note": note, **kw}]}


def _brief(state) -> str:
    return json.dumps({"history": (state.get("history") or [])[-4:],
                       "best_val_f1": state.get("best_val_f1"),
                       "last_decision": state.get("decision")}, default=str)


# ---------------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------------
def warehouse_loader(state: OlistState) -> dict:
    cache = Path(state["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    seed = int(state.get("seed") or 42)
    cols = ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    source, baselines = "postgres", {}
    try:
        eng = _engine()
        df = pd.read_sql(f"SELECT {cols} FROM {TABLE} ORDER BY order_id", eng)
        try:
            b = pd.read_sql("SELECT model, accuracy, precision, recall, f1, roc_auc FROM olist.model_runs "
                            "WHERE run_ts = (SELECT max(run_ts) FROM olist.model_runs)", eng)
            baselines = {r.model: {k: float(getattr(r, k)) for k in
                                   ("accuracy", "precision", "recall", "f1", "roc_auc")} for r in b.itertuples()}
        except Exception:  # noqa: BLE001
            baselines = {}
        note = f"read {len(df):,} rows from {TABLE}"
    except Exception as exc:  # noqa: BLE001 — resilience: fall back to the CSV the DAG publishes
        csv = next((p for p in CSV_FALLBACKS if p.is_file()), None)
        if csv is None:
            raise RuntimeError(f"Postgres unreachable ({exc}) and no CSV fallback") from exc
        df, source = pd.read_csv(csv), "csv"
        note = f"Postgres unreachable → CSV fallback ({len(df):,} rows)"
    if int(state.get("max_rows") or 0) and len(df) > int(state["max_rows"]):
        df, _ = train_test_split(df, train_size=int(state["max_rows"]), stratify=df[TARGET], random_state=seed)
        note += f" → stratified sample {int(state['max_rows']):,}"
    df = df.reset_index(drop=True)
    df.to_pickle(cache / "base.pkl")
    tr, va, ho = _split(df, seed)
    data = {"path": str(cache / "base.pkl"), "source": source, "n_rows": len(df),
            "n_train": len(tr), "n_val": len(va), "n_holdout": len(ho),
            "target_rate": round(float(df[TARGET].mean()), 4), "baselines": baselines,
            "city_match_rate": round(float(df["city_matched"].mean()), 4) if "city_matched" in df else None}
    return {"data": data, "cycle": 0, "no_improve": 0, "route_log": [], "history": [],
            "best_val_f1": -1.0, **_step("warehouse_loader", note, output=data)}


def data_scientist(state: OlistState) -> dict:
    df = pd.read_pickle(state["data"]["path"])
    num = [c for c in NUMERIC_FEATURES if c in df]
    corr = df[num + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    city_corr = {c: round(float(corr[c]), 3) for c in corr.index if c.startswith("city_")}
    profile = {
        "n_rows": len(df), "target_rate": state["data"]["target_rate"],
        "top_correlations": {c: round(float(v), 3) for c, v in corr.head(6).items()},
        "city_signal": dict(sorted(city_corr.items(), key=lambda kv: -kv[1])[:4]),
        "highly_skewed": [c for c, s in df[num].skew().items() if s > 3],
        "city_match_rate": state["data"].get("city_match_rate"),
        "notes": [f"Imbalanced target ({state['data']['target_rate']:.1%} positive): optimise F1, not accuracy.",
                  "Lateness features dominate; IBGE city context is a weaker, secondary signal."],
    }
    parsed, note = chat_json(
        "You are the Data Scientist. Profile risks only (imbalance, skew, leakage, weak signals, "
        "coverage of the IBGE city join). Do NOT choose models or features. JSON keys: "
        "notes (≤5 short strings), key_signals (≤5 column names that matter most).",
        f"profile={json.dumps(profile, default=str)}", model=state.get("model"), max_tokens=500)
    if parsed:
        profile["notes"] = [str(n) for n in parsed.get("notes") or profile["notes"]][:5]
        profile["key_signals"] = [c for c in parsed.get("key_signals") or [] if c in df.columns][:5]
    return {"profile": profile, **_step("data_scientist", note, used_llm=parsed is not None, output=profile)}


def strategist(state: OlistState) -> dict:
    """Initial plan, or a completely new plan when the router says 'rethink'."""
    history = state.get("history") or []
    tried = {(h.get("family"), tuple(h.get("recipes") or [])) for h in history}
    used_fams = [h.get("family") for h in history]
    fam = next((f for f in ("logistic_regression", "hist_gradient_boosting", "random_forest")
                if f not in used_fams), "hist_gradient_boosting")
    plan = {"model_family": fam,
            "recipes": ["lateness_buckets"] if not history else FEATURE_PRIORITY[:4],
            "threshold": "default_0.5" if not history else "tune_f1",
            "rationale": "start simple and let the evaluator's diagnosis drive changes" if not history
            else "previous direction stalled: switch family and use a broader feature set"}
    parsed, note = chat_json(
        "You are the Strategist for a bad-review classifier. Propose a plan: model_family "
        f"(one of {list(FAMILIES)}), recipes (subset of {list(RECIPES)}), threshold "
        f"(one of {list(THRESHOLDS)}), rationale (one sentence). If history exists this is a "
        "RETHINK: choose a direction clearly different from what was tried.",
        f"profile={json.dumps(state.get('profile'), default=str)}\nrecipes={json.dumps(RECIPES)}\n"
        f"brief={_brief(state)}", model=state.get("model"), max_tokens=400, temperature=0.4)
    if parsed and parsed.get("model_family") in FAMILIES:
        cand = {"model_family": parsed["model_family"],
                "recipes": [r for r in parsed.get("recipes") or [] if r in RECIPES],
                "threshold": parsed.get("threshold") if parsed.get("threshold") in THRESHOLDS else plan["threshold"],
                "rationale": str(parsed.get("rationale") or "")}
        if (cand["model_family"], tuple(cand["recipes"])) not in tried:
            plan = cand
        else:
            note += " · repeated a tried combo → kept heuristic plan"
    plan["kind"] = "rethink" if history else "initial"
    return {"plan": plan, **_step("strategist", f"{note} · {plan['kind']} plan: {plan['model_family']}",
                                  used_llm="llm" in note, output=plan)}


def feature_engineer(state: OlistState) -> dict:
    """Recipes come from the plan (after strategist) or are extended (route = features)."""
    route = state.get("route")
    current = list((state.get("features") or {}).get("recipes") or [])
    if route == "features":
        nxt = [r for r in FEATURE_PRIORITY if r not in current][:2]
        recipes = current + nxt
        why = f"diagnosis {state['evaluation'].get('diagnosis')}: add {', '.join(nxt) or 'nothing new'}"
    else:  # fresh plan from the strategist
        recipes = list(state["plan"]["recipes"])
        why = "apply strategist plan"
    parsed, note = chat_json(
        "You are the Feature Engineer. Choose recipes ONLY from the menu (you cannot write code). "
        "JSON keys: recipes (string array), rationale (one sentence).",
        f"route={route}\ncurrent={current}\nsuggested={recipes}\nmenu={json.dumps(RECIPES)}\n"
        f"last_evaluation={json.dumps(state.get('evaluation'), default=str)[:1500]}",
        model=state.get("model"), max_tokens=300)
    if parsed and isinstance(parsed.get("recipes"), list):
        picked = [r for r in parsed["recipes"] if r in RECIPES]
        if picked and picked != current:   # must actually change something
            recipes, why = picked, str(parsed.get("rationale") or why)
    base = pd.read_pickle(state["data"]["path"])
    feat = apply_recipes(base, recipes)
    path = Path(state["cache_dir"]) / f"features_{uuid.uuid4().hex[:6]}.pkl"
    feat.to_pickle(path)
    added, removed = sorted(set(feat) - set(base)), sorted(set(base) - set(feat))
    out = {"recipes": recipes, "rationale": why, "path": str(path),
           "n_features": feat.shape[1] - 1, "added": added, "removed": removed}
    return {"features": out, **_step("feature_engineer",
                                     f"{note} · {len(recipes)} recipes · +{len(added)}/−{len(removed)} cols",
                                     used_llm=parsed is not None, output=out)}


def ml_engineer(state: OlistState) -> dict:
    """Pick family / hyperparameters / threshold strategy, then train on the train split."""
    route, prev = state.get("route"), dict(state.get("spec") or {})
    if route == "model" and prev:          # same features: change the model side only
        fam, hp, thr = prev["model_family"], dict(prev["hyperparameters"]), prev["threshold"]
        if thr == "default_0.5":
            thr, why = "tune_f1", "ranking is fine → tune the decision threshold for F1"
        elif fam == "logistic_regression":
            fam, hp, why = "hist_gradient_boosting", {}, "linear model capped → gradient boosting"
        elif fam == "hist_gradient_boosting":
            hp = {**hp, "max_iter": int(hp.get("max_iter", 300)) + 200,
                  "learning_rate": max(0.03, float(hp.get("learning_rate", 0.08)) * 0.7)}
            why = "more, smaller boosting steps"
        else:
            hp = {**hp, "n_estimators": 400, "min_samples_leaf": max(2, int(hp.get("min_samples_leaf", 5)) - 2)}
            why = "more trees, finer leaves"
    else:
        plan = state.get("plan") or {}
        fam = prev.get("model_family") if route == "features" and prev else plan.get("model_family", "logistic_regression")
        hp = prev.get("hyperparameters", {}) if route == "features" and prev else {}
        thr = prev.get("threshold") if route == "features" and prev else plan.get("threshold", "default_0.5")
        why = "keep model, retrain on new features" if route == "features" else "follow strategist plan"
    parsed, note = chat_json(
        "You are the ML Engineer. Return the training config. JSON keys: model_family "
        f"(one of {list(FAMILIES)}), hyperparameters (object), threshold ({list(THRESHOLDS)}), "
        "rationale. If route=model you must change something on the model side; if route=features "
        "keep the model and let the new features speak.",
        f"route={route}\nproposal={json.dumps({'model_family': fam, 'hyperparameters': hp, 'threshold': thr})}\n"
        f"brief={_brief(state)}", model=state.get("model"), max_tokens=400)
    if parsed and parsed.get("model_family") in FAMILIES:
        fam = parsed["model_family"]
        hp = parsed.get("hyperparameters") if isinstance(parsed.get("hyperparameters"), dict) else hp
        thr = parsed.get("threshold") if parsed.get("threshold") in THRESHOLDS else thr
        why = str(parsed.get("rationale") or why)
    hp = _clamp(fam, hp if fam == prev.get("model_family") or parsed else {})
    cycle = int(state.get("cycle") or 0) + 1
    spec = {"experiment_id": f"exp-{cycle}", "model_family": fam, "hyperparameters": hp,
            "threshold": thr, "rationale": why}

    df = pd.read_pickle(state["features"]["path"])
    tr, va, _ = _split(df, int(state.get("seed") or 42))
    X, y = df.drop(columns=TARGET), df[TARGET]
    t0 = time.time()
    retried = False
    try:
        pipe = _pipeline(X, fam, hp).fit(X.loc[tr], y.loc[tr])
    except Exception:  # noqa: BLE001 — retry once with safe defaults
        retried, spec["hyperparameters"] = True, dict(DEFAULT_HP[fam])
        pipe = _pipeline(X, fam, spec["hyperparameters"]).fit(X.loc[tr], y.loc[tr])
    proba_path = Path(state["cache_dir"]) / f"val_proba_c{cycle}.npy"
    np.save(proba_path, pipe.predict_proba(X.loc[va])[:, 1])
    result = {"status": "succeeded", "train_seconds": round(time.time() - t0, 2),
              "retried": retried, "proba_path": str(proba_path)}
    return {"spec": spec, "result": result, "cycle": cycle,
            **_step("ml_engineer", f"{note} · {fam} ({thr}) trained in {result['train_seconds']}s"
                    + (" · retried with defaults" if retried else ""),
                    used_llm=parsed is not None, spec=spec, model_family=fam, experiment_id=spec["experiment_id"])}


def evaluator(state: OlistState) -> dict:
    """Metrics + DIAGNOSIS + recommended route (the core of the adaptive routing)."""
    df = pd.read_pickle(state["features"]["path"])
    _, va, _ = _split(df, int(state.get("seed") or 42))
    y = df[TARGET].loc[va].to_numpy()
    proba = np.load(state["result"]["proba_path"])
    spec = state["spec"]
    thr = 0.5
    if spec["threshold"] == "tune_f1":
        p, r, t = precision_recall_curve(y, proba)
        f1s = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
        thr = float(t[int(np.argmax(f1s))])
    m = _metrics(y, proba, thr)

    target = float(state.get("target_score") or 0.5)
    best_before = float(state.get("best_val_f1") or -1)
    improved = m["f1"] > best_before + PLATEAU_EPS
    no_improve = 0 if improved else int(state.get("no_improve") or 0) + 1
    rf_auc = ((state["data"].get("baselines") or {}).get("random_forest") or {}).get("roc_auc", 0.75)
    weak_ranking = m["roc_auc"] < rf_auc + AUC_MARGIN
    diagnosis = ("target_met" if m["f1"] >= target else
                 "weak_ranking" if weak_ranking else "cutoff_or_capacity")
    last_route = (state.get("route_log") or [None])[-1]
    if m["f1"] >= target:
        allowed, rec, why = ["finish"], "finish", "target met"
    elif int(state["cycle"]) >= int(state.get("max_cycles") or 5):
        allowed, rec, why = ["finish"], "finish", "budget spent"
    elif no_improve >= PLATEAU_CYCLES:
        allowed, rec, why = ["finish"], "finish", f"plateau: {no_improve} cycles without +{PLATEAU_EPS} F1"
    else:
        fix = "features" if weak_ranking else "model"
        stuck = (not improved) and last_route == fix
        rec = "rethink" if stuck else fix
        why = (f"same fix ({fix}) did not help → re-plan" if stuck else
               f"AUC {m['roc_auc']:.3f} vs Airflow RF {rf_auc:.3f} → " +
               ("improve inputs" if fix == "features" else "ranking OK, fix cutoff/capacity"))
        allowed = ["features", "model", "rethink"]
    report = {"experiment_id": spec["experiment_id"], "model_family": spec["model_family"],
              "recipes": state["features"]["recipes"], "validation": m, "improved": improved,
              "no_improve": no_improve, "diagnosis": diagnosis, "recommended_route": rec,
              "allowed_routes": allowed, "reason": why, "airflow_rf_auc": rf_auc}
    patch = {"evaluation": report, "no_improve": no_improve}
    if improved or best_before < 0:
        patch["best_val_f1"] = m["f1"]
        patch["best"] = {"spec": spec, "recipes": state["features"]["recipes"], "threshold": thr,
                         "validation": m, "cycle": state["cycle"]}
    note = (f"F1 {m['f1']:.3f} · AUC {m['roc_auc']:.3f} · thr {thr:.2f} → {diagnosis} "
            f"→ recommend {rec}")
    return {**patch, **_step("evaluator", note, f1=m["f1"], validation_metrics=m, output=report)}


def judge(state: OlistState) -> dict:
    ev = state["evaluation"]
    allowed, route = ev["allowed_routes"], ev["recommended_route"]
    reason, note = ev["reason"], "rule"
    if len(allowed) > 1:
        parsed, note = chat_json(
            "You are the Judge. Pick the next route for the research loop from allowed_routes only: "
            "features = engineer better inputs; model = keep inputs, change threshold/capacity/family; "
            "rethink = whole new plan. JSON keys: route, reason (one sentence).",
            f"allowed_routes={allowed}\nrecommended={route}\nreport={json.dumps(ev, default=str)}\n"
            f"brief={_brief(state)}", model=state.get("model"), max_tokens=300, temperature=0.1)
        if parsed and parsed.get("route") in allowed:
            route, reason = parsed["route"], str(parsed.get("reason") or reason)
        elif parsed:
            note += f" · invalid route {parsed.get('route')!r} → recommendation"
    stop = None
    if route == "finish":
        stop = "target_met" if ev["diagnosis"] == "target_met" else (
            "plateau" if ev["no_improve"] >= PLATEAU_CYCLES else "budget")
    decision = {"route": route, "reason": reason, "diagnosis": ev["diagnosis"], "stop": stop,
                "action": "accept" if stop == "target_met" else ("stop" if stop else "revise")}
    log = list(state.get("route_log") or []) + [route]
    return {"decision": decision, "route": route, "route_log": log,
            **({"stop_reason": stop} if stop else {}),
            **_step("judge", f"{note} · route → {route}" + (f" ({stop})" if stop else ""),
                    used_llm="llm" in note, verdict=decision)}


def research_memory(state: OlistState) -> dict:
    ev, d = state["evaluation"], state["decision"]
    hist = list(state.get("history") or []) + [{
        "cycle": state["cycle"], "family": ev["model_family"], "recipes": ev["recipes"],
        "threshold": state["spec"]["threshold"], "val_f1": ev["validation"]["f1"],
        "val_auc": ev["validation"]["roc_auc"], "diagnosis": ev["diagnosis"], "route": d["route"]}]
    target = {"features": "feature_engineer", "model": "ml_engineer", "rethink": "strategist"}[d["route"]]
    brief = {"tried": hist[-4:], "next": target, "why": d["reason"]}
    return {"history": hist, "brief": brief,
            **_step("research_memory", f"filed cycle {state['cycle']} → dispatch to {target}", output=brief)}


def holdout_eval(state: OlistState) -> dict:
    best = state.get("best") or {}
    spec, recipes, thr = best["spec"], best["recipes"], float(best["threshold"])
    df = apply_recipes(pd.read_pickle(state["data"]["path"]), recipes)
    tr, va, ho = _split(df, int(state.get("seed") or 42))
    X, y = df.drop(columns=TARGET), df[TARGET]
    pipe = _pipeline(X, spec["model_family"], spec["hyperparameters"]).fit(X.loc[tr.append(va)], y.loc[tr.append(va)])
    m = _metrics(y.loc[ho].to_numpy(), pipe.predict_proba(X.loc[ho])[:, 1], thr)
    m.update({"n_holdout": len(ho), "majority_baseline_accuracy": round(1 - float(y.loc[ho].mean()), 4),
              "model_family": spec["model_family"], "recipes": recipes,
              "hyperparameters": spec["hyperparameters"], "best_cycle": best.get("cycle"),
              "route_log": state.get("route_log"), "stop_reason": state.get("stop_reason"),
              "airflow_baselines": state["data"].get("baselines")})
    note = f"holdout F1 {m['f1']:.3f} · acc {m['accuracy']:.3f} · AUC {m['roc_auc']:.3f} · thr {thr:.2f}"
    if state["data"]["source"] == "postgres":
        try:
            row = {"run_ts": datetime.now(timezone.utc).isoformat(), "run_id": state["run_id"],
                   "llm_model": state.get("model"), "cycles": state["cycle"],
                   "stop_reason": state.get("stop_reason"), "route_log": ">".join(state.get("route_log") or []),
                   "model_family": spec["model_family"], "recipes": ",".join(recipes),
                   "hyperparameters": json.dumps(spec["hyperparameters"]),
                   **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                         "threshold", "n_holdout")}}
            pd.DataFrame([row]).to_sql("agent_runs", _engine(), schema="olist", if_exists="append", index=False)
            note += " · logged to olist.agent_runs"
        except Exception as exc:  # noqa: BLE001
            note += f" · (db log skipped: {type(exc).__name__})"
    return {"holdout_metrics": m, **_step("holdout_eval", note, metrics=m)}


# ---------------------------------------------------------------------------
# ROUTING + GRAPH
# ---------------------------------------------------------------------------
def after_judge(state: OlistState) -> str:
    return "holdout_eval" if state.get("route") == "finish" else "research_memory"


def after_memory(state: OlistState) -> str:
    return {"features": "feature_engineer", "model": "ml_engineer", "rethink": "strategist"}[state["route"]]


AGENTS = {
    "warehouse_loader": warehouse_loader, "data_scientist": data_scientist, "strategist": strategist,
    "feature_engineer": feature_engineer, "ml_engineer": ml_engineer, "evaluator": evaluator,
    "judge": judge, "research_memory": research_memory, "holdout_eval": holdout_eval,
}


def build_graph():
    g = StateGraph(OlistState)
    for name, fn in AGENTS.items():
        g.add_node(name, fn)
    g.add_edge(START, "warehouse_loader")
    g.add_edge("warehouse_loader", "data_scientist")
    g.add_edge("data_scientist", "strategist")
    g.add_edge("strategist", "feature_engineer")
    g.add_edge("feature_engineer", "ml_engineer")
    g.add_edge("ml_engineer", "evaluator")
    g.add_edge("evaluator", "judge")
    g.add_conditional_edges("judge", after_judge, ["research_memory", "holdout_eval"])
    g.add_conditional_edges("research_memory", after_memory, ["feature_engineer", "ml_engineer", "strategist"])
    g.add_edge("holdout_eval", END)
    return g.compile()


def get_status() -> dict:
    try:
        nodes = sorted(n for n in build_graph().get_graph().nodes if not n.startswith("__"))
        return {"ready": True, "nodes": nodes, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "nodes": [], "error": str(exc)}


def _init(*, target_score, max_cycles, max_rows, seed, model) -> OlistState:
    run_id = f"olr_{uuid.uuid4().hex[:10]}"
    return {"run_id": run_id, "cache_dir": str(CACHE_ROOT / run_id), "model": (model or "").strip() or DEFAULT_MODEL,
            "seed": seed, "target_score": target_score, "max_cycles": max_cycles, "max_rows": max_rows,
            "steps": []}


def _payload(s: dict) -> dict:
    keys = ("run_id", "model", "cycle", "data", "profile", "best", "best_val_f1", "holdout_metrics",
            "decision", "history", "route_log", "stop_reason", "steps")
    return {k: s.get(k) for k in keys} | {"cycles": s.get("cycle"), "target_column": TARGET, "dataset_path": TABLE}


def run_olist_loop(*, target_score=0.5, max_cycles=5, max_rows=0, seed=42, model=None) -> dict:
    final = build_graph().invoke(_init(target_score=target_score, max_cycles=max_cycles, max_rows=max_rows,
                                       seed=seed, model=model), config={"recursion_limit": 12 + max_cycles * 8})
    return _payload(final)


NEXT = {"warehouse_loader": "data_scientist", "data_scientist": "strategist", "strategist": "feature_engineer",
        "feature_engineer": "ml_engineer", "ml_engineer": "evaluator", "evaluator": "judge"}


def stream_olist_loop(*, target_score=0.5, max_cycles=5, max_rows=0, seed=42, model=None):
    """Yield UI events (node_started / node_completed / router / run_completed)."""
    init = _init(target_score=target_score, max_cycles=max_cycles, max_rows=max_rows, seed=seed, model=model)
    yield {"type": "run_started", "run_id": init["run_id"], "model": init["model"],
           "dataset_path": TABLE, "target_column": TARGET}
    merged: dict = dict(init)
    yield {"type": "node_started", "node": "warehouse_loader", "status": "running"}
    try:
        for update in build_graph().stream(init, stream_mode="updates",
                                           config={"recursion_limit": 12 + max_cycles * 8}):
            for node, patch in (update or {}).items():
                steps = list((patch or {}).get("steps") or [])
                merged.update({k: v for k, v in (patch or {}).items() if k != "steps"})
                merged["steps"] = list(merged.get("steps") or []) + steps
                yield {"type": "node_completed", "node": node, "status": "completed",
                       "note": steps[-1].get("note") if steps else None, "step": steps[-1] if steps else None,
                       "cycle": merged.get("cycle"), "best_val_f1": merged.get("best_val_f1")}
                nxt = NEXT.get(node)
                if node == "judge":
                    d = merged.get("decision") or {}
                    yield {"type": "router", "node": "router", "status": "completed",
                           "route": d.get("route"), "action": d.get("action"), "reason": d.get("reason")}
                    nxt = after_judge(merged)
                elif node == "research_memory":
                    nxt = after_memory(merged)
                if nxt:
                    yield {"type": "node_started", "node": nxt, "status": "running"}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        return
    yield {"type": "run_completed", "result": _payload(merged)}
