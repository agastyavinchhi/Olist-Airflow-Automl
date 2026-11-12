"""Olist + IBGE pipeline: 2 independent datasets → cleaned + merged → Postgres → analysis.

Problem: flag delivered orders that are likely to receive a bad review (score <= 2) so
customer support can reach out first. Inputs:
  * Olist Brazilian e-commerce (7 related tables: orders, customers, reviews, payments,
    items, products, category translation)
  * IBGE Brazilian Cities (independent public statistics: population, HDI, GDP per
    capita, urban/rural type, coordinates) — joined to the customer's city

Graph (TaskGroups in [brackets]; everything inside a group with no arrow between
it runs in parallel on the LocalExecutor):

    [ingest_and_clean]                                 [transform]                 [load]
      orders     : extract → clean  ───────────────┐
      customers  : extract → clean  ───────────────┤
      cities     : extract → clean  (IBGE) ────────┤
      reviews    : extract → clean  ───────────────┼──────────→ merge_and_engineer → load_orders_clean → quality_checks
      payments   : extract → clean  ───────────────┤                 ↑
      items      : extract → clean  ──┐            │                 │
      products   : extract_products ┐ ├→ order_item_features ────────┘
                   extract_translation ┴→ clean ┘

                                    [analysis]  (all three read from Postgres, in parallel)
    quality_checks ──→  make_charts │ train_models │ publish_for_agents  ──→ report

Data between tasks moves as parquet files in a per-run folder; XCom only carries
small dicts (paths + row counts). Postgres is reached through the Airflow
Connection ``warehouse_db`` (set via AIRFLOW_CONN_WAREHOUSE_DB in docker-compose).
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.task_group import TaskGroup

# ---------------------------------------------------------------------------
# Paths inside the Airflow containers (/repo is the mounted project root)
# ---------------------------------------------------------------------------
REPO = Path("/repo")
RAW = REPO / "datasets" / "olist" / "raw"
CITIES_RAW = REPO / "datasets" / "brazil_cities" / "raw"
OUT = REPO / "artifacts" / "airflow" / "olist"  # one sub-folder per run + latest/
AGENT_CSV = REPO / "datasets" / "airflow" / "olist_orders_latest.csv"  # Part 2 input

CONN_ID = "warehouse_db"
SCHEMA = "olist"
TABLE = "orders_clean"
TARGET = "is_bad_review"
AGENT_SAMPLE_ROWS = 5000

# raw file + the columns each extract must find (schema contract)
SOURCES = {
    "orders": (
        "olist_orders_dataset.csv",
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
         "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date"],
    ),
    "customers": (
        "olist_customers_dataset.csv",
        ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
         "customer_city", "customer_state"],
    ),
    "reviews": (
        "olist_order_reviews_dataset.csv",
        ["review_id", "order_id", "review_score", "review_answer_timestamp"],
    ),
    "payments": (
        "olist_order_payments_dataset.csv",
        ["order_id", "payment_sequential", "payment_type",
         "payment_installments", "payment_value"],
    ),
    "items": (
        "olist_order_items_dataset.csv",
        ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
    ),
    "products": (
        "olist_products_dataset.csv",
        ["product_id", "product_category_name", "product_photos_qty",
         "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    ),
    "translation": (
        "product_category_name_translation.csv",
        ["product_category_name", "product_category_name_english"],
    ),
    # Second, independent dataset (IBGE via Kaggle crisparada/brazilian-cities)
    "cities": (
        "BRAZIL_CITIES_REV2022.CSV",
        ["CITY", "STATE", "CAPITAL", "IBGE_RES_POP", "IDHM", "IDHM_Renda", "IDHM_Educacao",
         "GDP_CAPITA", "RURAL_URBAN", "LAT", "LONG"],
    ),
}
SAO_PAULO = (-23.5505, -46.6333)  # most Olist sellers ship from SP state

# Model inputs — no review_score / review dates / ids (that would leak the label)
NUMERIC_FEATURES = [
    "delivery_days", "estimated_days", "days_late", "is_late", "approval_hours",
    "carrier_days", "purchase_month", "purchase_dayofweek", "purchase_hour",
    "n_items", "n_products", "n_sellers", "total_price", "total_freight",
    "freight_ratio", "avg_product_weight_g", "avg_product_volume_cm3",
    "avg_product_photos", "payment_total", "n_payments", "max_installments",
    "used_voucher",
    # from the IBGE cities dataset
    "city_matched", "city_population_log", "city_hdi", "city_hdi_income",
    "city_hdi_education", "city_gdp_per_capita", "city_is_capital", "city_distance_to_sp_km",
]
CATEGORICAL_FEATURES = ["customer_state", "main_payment_type", "main_category", "city_rural_urban"]
CITY_NUMERIC = ["city_population", "city_hdi", "city_hdi_income", "city_hdi_education",
                "city_gdp_per_capita", "city_distance_to_sp_km"]

# Chart styling (light surface, one sequential blue for single-series bars)
INK, INK_2, MUTED, GRID, SURFACE, BLUE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6",
)


def city_key(name) -> str:
    """'São Paulo' and 'sao paulo' → 'sao paulo' (strip accents, punctuation, case)."""
    txt = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", txt).split())


def _haversine_km(lat, lon, lat0, lon0):
    import numpy as np

    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _run_dir(context) -> Path:
    """artifacts/airflow/olist/<logical-ts>/ — unique per DAG run."""
    d = OUT / context["ts_nodash"]
    (d / "staging").mkdir(parents=True, exist_ok=True)
    return d


def _engine():
    return PostgresHook(postgres_conn_id=CONN_ID).get_sqlalchemy_engine()


def _save(df: pd.DataFrame, context, name: str, rows_in: int | None = None) -> dict:
    path = _run_dir(context) / "staging" / f"{name}.parquet"
    df.to_parquet(path, index=False)
    meta = {"path": str(path), "rows": len(df)}
    if rows_in is not None:
        meta["rows_in"] = rows_in
        meta["rows_dropped"] = rows_in - len(df)
    return meta


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------
@dag(
    dag_id="olist_pipeline",
    description="Olist (7 tables) + IBGE cities → clean (parallel) → merge → Postgres → charts + models",
    # Daily at 06:00 UTC. catchup=False → unpausing runs only the latest interval,
    # not one run per day since start_date. max_active_runs=1 → runs never overlap
    # (they'd both replace olist.orders_clean).
    schedule="0 6 * * *",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["olist", "ibge", "assignment", "etl", "ml"],
)
def olist_pipeline():
    # ------------------------------------------------------------------ extract
    @task
    def extract(table: str, **context) -> dict:
        """Read one raw CSV, enforce its column contract, stage it as parquet."""
        filename, required = SOURCES[table]
        path = (CITIES_RAW if table == "cities" else RAW) / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} — see datasets/olist/ATTRIBUTION.md")
        df = pd.read_csv(path, encoding="utf-8-sig")  # translation file has a BOM
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{filename} missing columns: {missing}")
        return _save(df, context, f"raw_{table}")

    # -------------------------------------------------------------------- clean
    @task
    def clean_orders(raw: dict, **context) -> dict:
        """Delivered orders only, typed timestamps, delivery-timing features."""
        df = pd.read_parquet(raw["path"])
        n = len(df)
        ts_cols = [c for c in df.columns if c.startswith("order_") and c not in
                   ("order_id", "order_status")]
        for c in ts_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.drop_duplicates("order_id")
        df = df[df.order_status == "delivered"]
        df = df.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date"])
        df = df[df.order_delivered_customer_date >= df.order_purchase_timestamp]
        df["order_approved_at"] = df.order_approved_at.fillna(df.order_purchase_timestamp)

        day = pd.Timedelta(days=1)
        buy = df.order_purchase_timestamp
        df["delivery_days"] = (df.order_delivered_customer_date - buy) / day
        df["estimated_days"] = (df.order_estimated_delivery_date - buy) / day
        df["days_late"] = (df.order_delivered_customer_date - df.order_estimated_delivery_date) / day
        df["is_late"] = (df.days_late > 0).astype(int)
        df["approval_hours"] = (df.order_approved_at - buy) / pd.Timedelta(hours=1)
        df["carrier_days"] = ((df.order_delivered_carrier_date - df.order_approved_at) / day).clip(lower=0)
        df["carrier_days"] = df.carrier_days.fillna(df.carrier_days.median())  # 1 missing carrier date
        df["purchase_month"] = buy.dt.month
        df["purchase_dayofweek"] = buy.dt.dayofweek
        df["purchase_hour"] = buy.dt.hour
        keep = ["order_id", "customer_id", "order_purchase_timestamp",
                "order_delivered_customer_date", "order_estimated_delivery_date",
                "delivery_days", "estimated_days", "days_late", "is_late",
                "approval_hours", "carrier_days", "purchase_month",
                "purchase_dayofweek", "purchase_hour"]
        return _save(df[keep], context, "clean_orders", rows_in=n)

    @task
    def clean_customers(raw: dict, **context) -> dict:
        """One row per customer_id, normalized city/state, zero-padded zip."""
        df = pd.read_parquet(raw["path"])
        n = len(df)
        df = df.drop_duplicates("customer_id")
        df["customer_state"] = df.customer_state.str.strip().str.upper()
        df["customer_city"] = df.customer_city.str.strip().str.lower()
        df["customer_zip_code_prefix"] = df.customer_zip_code_prefix.astype(str).str.zfill(5)
        df["city_key"] = df.customer_city.map(city_key)
        keep = ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                "customer_city", "customer_state", "city_key"]
        return _save(df[keep], context, "clean_customers", rows_in=n)

    @task
    def clean_cities(raw: dict, **context) -> dict:
        """IBGE cities: normalized join key, 0-as-missing → NaN, dedupe, distance to São Paulo."""
        import numpy as np

        df = pd.read_parquet(raw["path"])
        n = len(df)
        df["city_key"] = df.CITY.map(city_key)
        df["customer_state"] = df.STATE.str.strip().str.upper()
        for c in ["IBGE_RES_POP", "IDHM", "IDHM_Renda", "IDHM_Educacao", "GDP_CAPITA", "LAT", "LONG"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan)  # 0 = missing
        df = (df.sort_values("IBGE_RES_POP", ascending=False)  # 4 duplicate keys → keep larger city
                .drop_duplicates(["city_key", "customer_state"]))
        rural = {"Urbano": "urban", "Intermediário Adjacente": "intermediate_adjacent",
                 "Intermediário Remoto": "intermediate_remote", "Rural Adjacente": "rural_adjacent",
                 "Rural Remoto": "rural_remote"}
        out = pd.DataFrame({
            "city_key": df.city_key,
            "customer_state": df.customer_state,
            "city_population": df.IBGE_RES_POP,
            "city_hdi": df.IDHM,
            "city_hdi_income": df.IDHM_Renda,
            "city_hdi_education": df.IDHM_Educacao,
            "city_gdp_per_capita": df.GDP_CAPITA,
            "city_is_capital": pd.to_numeric(df.CAPITAL, errors="coerce").fillna(0).astype(int),
            "city_rural_urban": df.RURAL_URBAN.map(rural).fillna("unknown"),
            "city_distance_to_sp_km": _haversine_km(df.LAT, df.LONG, *SAO_PAULO),
        })
        return _save(out, context, "clean_cities", rows_in=n)

    @task
    def clean_reviews(raw: dict, **context) -> dict:
        """Valid 1–5 scores, latest review per order (551 orders have >1)."""
        df = pd.read_parquet(raw["path"])
        n = len(df)
        df["review_answer_timestamp"] = pd.to_datetime(df.review_answer_timestamp, errors="coerce")
        df = df[df.review_score.between(1, 5)]
        df = (df.sort_values("review_answer_timestamp")
                .drop_duplicates("order_id", keep="last"))
        df["review_score"] = df.review_score.astype(int)
        return _save(df[["order_id", "review_score"]], context, "clean_reviews", rows_in=n)

    @task
    def clean_payments(raw: dict, **context) -> dict:
        """Drop 'not_defined' payments, then aggregate to one row per order."""
        df = pd.read_parquet(raw["path"])
        n = len(df)
        df = df[(df.payment_type != "not_defined") & (df.payment_value >= 0)]
        by_type = df.groupby(["order_id", "payment_type"]).payment_value.sum().reset_index()
        main_type = (by_type.sort_values("payment_value")
                            .drop_duplicates("order_id", keep="last")
                            .set_index("order_id").payment_type)
        agg = df.groupby("order_id").agg(
            payment_total=("payment_value", "sum"),
            n_payments=("payment_sequential", "count"),
            max_installments=("payment_installments", "max"),
            used_voucher=("payment_type", lambda s: int((s == "voucher").any())),
        )
        agg["main_payment_type"] = main_type
        return _save(agg.reset_index(), context, "clean_payments", rows_in=n)

    @task
    def clean_items(raw: dict, **context) -> dict:
        """Drop duplicate lines and non-positive prices."""
        df = pd.read_parquet(raw["path"])
        n = len(df)
        df = df.drop_duplicates(["order_id", "order_item_id"])
        df = df[(df.price > 0) & (df.freight_value >= 0)]
        keep = ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"]
        return _save(df[keep], context, "clean_items", rows_in=n)

    @task
    def clean_products(raw: dict, translation: dict, **context) -> dict:
        """English category names, 'unknown' for 610 missing, median-impute sizes."""
        df = pd.read_parquet(raw["path"])
        n = len(df)
        tr = pd.read_parquet(translation["path"])
        df = df.drop_duplicates("product_id").merge(tr, on="product_category_name", how="left")
        df["category"] = (df.product_category_name_english
                          .fillna(df.product_category_name)  # 2 cats lack a translation
                          .fillna("unknown"))
        for c in ["product_photos_qty", "product_weight_g", "product_length_cm",
                  "product_height_cm", "product_width_cm"]:
            df[c] = df[c].fillna(df[c].median())
        df["product_volume_cm3"] = df.product_length_cm * df.product_height_cm * df.product_width_cm
        keep = ["product_id", "category", "product_photos_qty", "product_weight_g",
                "product_volume_cm3"]
        return _save(df[keep], context, "clean_products", rows_in=n)

    # ---------------------------------------------------------------- transform
    @task
    def order_item_features(items: dict, products: dict, **context) -> dict:
        """items ⨝ products → one row per order (counts, totals, main category)."""
        it = pd.read_parquet(items["path"]).merge(
            pd.read_parquet(products["path"]), on="product_id", how="left")
        it["category"] = it.category.fillna("unknown")
        main_cat = (it.sort_values("price").drop_duplicates("order_id", keep="last")
                      .set_index("order_id").category)
        agg = it.groupby("order_id").agg(
            n_items=("order_item_id", "count"),
            n_products=("product_id", "nunique"),
            n_sellers=("seller_id", "nunique"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            avg_product_weight_g=("product_weight_g", "mean"),
            avg_product_volume_cm3=("product_volume_cm3", "mean"),
            avg_product_photos=("product_photos_qty", "mean"),
        )
        agg["main_category"] = main_cat
        return _save(agg.reset_index(), context, "order_item_features")

    @task
    def merge_and_engineer(orders: dict, customers: dict, cities: dict, reviews: dict,
                           payments: dict, item_feats: dict, **context) -> dict:
        """orders ⨝ customers ⨝ IBGE cities ⨝ items ⨝ payments ⨝ reviews → one row per order."""
        import numpy as np

        df = pd.read_parquet(orders["path"])
        n = len(df)
        cty = pd.read_parquet(cities["path"])
        df = (df.merge(pd.read_parquet(customers["path"]), on="customer_id", how="inner")
                .merge(cty, on=["city_key", "customer_state"], how="left", indicator="_city")
                .merge(pd.read_parquet(item_feats["path"]), on="order_id", how="inner")
                .merge(pd.read_parquet(payments["path"]), on="order_id", how="left")
                .merge(pd.read_parquet(reviews["path"]), on="order_id", how="inner"))
        # 1 delivered order has no payment row → fall back to item totals
        df["payment_total"] = df.payment_total.fillna(df.total_price + df.total_freight)
        df["n_payments"] = df.n_payments.fillna(1)
        df["max_installments"] = df.max_installments.fillna(1)
        df["used_voucher"] = df.used_voucher.fillna(0).astype(int)
        df["main_payment_type"] = df.main_payment_type.fillna("unknown")
        df["freight_ratio"] = df.total_freight / df.total_price
        # IBGE city context: ~0.6% of customers don't match (misspellings / districts)
        # → fall back to the median of their state, then the national median.
        df["city_matched"] = (df.pop("_city") == "both").astype(int)
        state_med = cty.groupby("customer_state")[CITY_NUMERIC].median()
        for c in CITY_NUMERIC:
            df[c] = df[c].fillna(df.customer_state.map(state_med[c])).fillna(cty[c].median())
        df["city_population_log"] = np.log1p(df.city_population)
        df["city_is_capital"] = df.city_is_capital.fillna(0).astype(int)
        df["city_rural_urban"] = df.city_rural_urban.fillna("unknown")
        df[TARGET] = (df.review_score <= 2).astype(int)
        if df.order_id.duplicated().any():
            raise ValueError("merge produced duplicate order_id rows")
        return _save(df, context, "orders_clean", rows_in=n)

    # --------------------------------------------------------------------- load
    @task
    def load_orders_clean(merged: dict) -> dict:
        """Replace olist.orders_clean, then add a primary key + index."""
        df = pd.read_parquet(merged["path"])
        eng = _engine()
        with eng.begin() as conn:
            conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        df.to_sql(TABLE, eng, schema=SCHEMA, if_exists="replace", index=False,
                  method="multi", chunksize=5000)
        with eng.begin() as conn:
            conn.exec_driver_sql(f"ALTER TABLE {SCHEMA}.{TABLE} ADD PRIMARY KEY (order_id)")
            conn.exec_driver_sql(
                f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_state ON {SCHEMA}.{TABLE} (customer_state)")
        return {"table": f"{SCHEMA}.{TABLE}", "rows": len(df)}

    @task
    def quality_checks(loaded: dict) -> dict:
        """Fail the run if the loaded table looks wrong (row count, nulls, label rate)."""
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        any_null = " OR ".join(f"{c} IS NULL" for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        rows, null_target, bad_rate, null_feat, city_match = hook.get_first(f"""
            SELECT count(*),
                   count(*) FILTER (WHERE {TARGET} IS NULL),
                   avg({TARGET}::float),
                   count(*) FILTER (WHERE {any_null}),
                   avg(city_matched::float)
            FROM {SCHEMA}.{TABLE}""")
        problems = []
        if rows != loaded["rows"]:
            problems.append(f"row count {rows} != staged {loaded['rows']}")
        if rows < 50_000:
            problems.append(f"only {rows} rows (expected ~95k)")
        if null_target or null_feat:
            problems.append(f"nulls: target={null_target} features={null_feat}")
        if not 0.02 <= bad_rate <= 0.5:
            problems.append(f"bad-review rate {bad_rate:.3f} outside [0.02, 0.5]")
        if city_match < 0.95:
            problems.append(f"only {city_match:.1%} of orders matched an IBGE city (expected ≥ 95%)")
        if problems:
            raise ValueError("; ".join(problems))
        return {"rows": rows, "bad_review_rate": round(bad_rate, 4),
                "city_match_rate": round(city_match, 4)}

    # ----------------------------------------------------------------- analysis
    @task
    def make_charts(dq: dict, **context) -> dict:
        """Aggregate in SQL (read from Postgres), render two PNG bar charts."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eng = _engine()
        overall = dq["bad_review_rate"]
        timing = pd.read_sql(f"""
            SELECT CASE
                     WHEN days_late <= -14 THEN 1 WHEN days_late <= -7 THEN 2
                     WHEN days_late <= 0   THEN 3 WHEN days_late <= 7  THEN 4
                     WHEN days_late <= 14  THEN 5 ELSE 6 END AS bucket,
                   count(*) AS orders, avg({TARGET}::float) AS bad_rate
            FROM {SCHEMA}.{TABLE} GROUP BY 1 ORDER BY 1""", eng)
        labels = {1: "14+ days\nearly", 2: "7–14\nearly", 3: "0–7\nearly",
                  4: "1–7\nlate", 5: "8–14\nlate", 6: "15+ days\nlate"}
        timing["label"] = [f"{labels[b]}\nn={n:,}" for b, n in zip(timing.bucket, timing.orders)]
        states = pd.read_sql(f"""
            SELECT customer_state, count(*) AS orders, avg({TARGET}::float) AS bad_rate
            FROM {SCHEMA}.{TABLE} GROUP BY 1 HAVING count(*) >= 500
            ORDER BY bad_rate""", eng)

        def style(ax, title, subtitle):
            ax.set_facecolor(SURFACE)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#c3c2b7")
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            ax.set_title(title, loc="left", fontsize=13, color=INK, pad=22, fontweight="bold")
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)

        out = _run_dir(context)
        paths = {}

        # 1) bad-review rate vs delivery timing
        fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
        ax.bar(timing.label, timing.bad_rate, color=BLUE, width=0.6, zorder=2)
        ax.axhline(overall, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(-0.45, overall + 0.025, f"all orders {overall:.1%}", ha="left",
                va="bottom", fontsize=8.5, color=INK_2)
        for x, rate in enumerate(timing.bad_rate):
            ax.text(x, rate + 0.012, f"{rate:.0%}", ha="center", fontsize=10, color=INK)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_ylim(0, max(timing.bad_rate) * 1.15)
        style(ax, "Late deliveries drive bad reviews",
              "Share of delivered orders with review score ≤ 2, by delivery vs. estimated date")
        fig.tight_layout()
        paths["timing"] = str(out / "bad_review_rate_by_delivery_timing.png")
        fig.savefig(paths["timing"], dpi=150)
        plt.close(fig)

        # 2) bad-review rate by customer state (states with >= 500 orders)
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(states) + 1.4), facecolor=SURFACE)
        ax.barh(states.customer_state, states.bad_rate, color=BLUE, height=0.62, zorder=2)
        ax.axvline(overall, color=MUTED, lw=1, ls="--", zorder=1)
        for y, (rate, n) in enumerate(zip(states.bad_rate, states.orders)):
            ax.text(rate + 0.002, y, f"{rate:.1%}  ({n:,})", va="center", fontsize=8, color=INK_2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(0, states.bad_rate.max() * 1.25)
        style(ax, "Bad-review rate by customer state",
              f"States with ≥ 500 orders · dashed line = all orders ({overall:.1%}) · (orders)")
        fig.tight_layout()
        paths["states"] = str(out / "bad_review_rate_by_state.png")
        fig.savefig(paths["states"], dpi=150)
        plt.close(fig)

        # 3) bad-review rate by distance from São Paulo (IBGE coordinates)
        dist = pd.read_sql(f"""
            SELECT CASE WHEN city_distance_to_sp_km < 100 THEN 1
                        WHEN city_distance_to_sp_km < 500 THEN 2
                        WHEN city_distance_to_sp_km < 1000 THEN 3
                        WHEN city_distance_to_sp_km < 2000 THEN 4 ELSE 5 END AS band,
                   count(*) AS orders, avg({TARGET}::float) AS bad_rate,
                   avg(is_late::float) AS late_rate
            FROM {SCHEMA}.{TABLE} GROUP BY 1 ORDER BY 1""", eng)
        band_lbl = {1: "< 100 km", 2: "100–500", 3: "500–1,000", 4: "1,000–2,000", 5: "2,000+ km"}
        dist["label"] = [f"{band_lbl[b]}\nn={n:,}" for b, n in zip(dist.band, dist.orders)]
        fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
        ax.bar(dist.label, dist.bad_rate, color=BLUE, width=0.6, zorder=2)
        ax.axhline(overall, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(-0.45, overall + 0.006, f"all orders {overall:.1%}", ha="left", va="bottom",
                fontsize=8.5, color=INK_2)
        for x, (rate, late) in enumerate(zip(dist.bad_rate, dist.late_rate)):
            ax.text(x, rate + 0.004, f"{rate:.1%}", ha="center", fontsize=10, color=INK)
            ax.text(x, rate / 2, f"late {late:.0%}", ha="center", fontsize=8, color="white")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_ylim(0, dist.bad_rate.max() * 1.18)
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        style(ax, "Customers 1,000+ km from São Paulo leave more bad reviews",
              "Bad-review rate by customer city's distance from São Paulo (IBGE coordinates) · "
              "white = share delivered late")
        fig.tight_layout()
        paths["distance"] = str(out / "bad_review_rate_by_distance.png")
        fig.savefig(paths["distance"], dpi=150)
        plt.close(fig)

        return {"charts": paths,
                "late_bad_rate": float(timing[timing.bucket >= 4].eval("bad_rate*orders").sum()
                                       / timing[timing.bucket >= 4].orders.sum()),
                "on_time_bad_rate": float(timing[timing.bucket <= 3].eval("bad_rate*orders").sum()
                                          / timing[timing.bucket <= 3].orders.sum())}

    @task
    def train_models(dq: dict, **context) -> dict:
        """Read the table from Postgres, train 2 classifiers, log metrics to olist.model_runs."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, roc_auc_score)
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        eng = _engine()
        cols = ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
        df = pd.read_sql(f"SELECT {cols} FROM {SCHEMA}.{TABLE}", eng)
        X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        def prep():
            return ColumnTransformer([
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=200), CATEGORICAL_FEATURES),
            ])

        models = {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=300, max_depth=14, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=4, random_state=42),
        }
        results = {"majority_baseline": {
            "accuracy": round(float(1 - y_te.mean()), 4),
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}}
        fitted = {}
        for name, clf in models.items():
            pipe = Pipeline([("prep", prep()), ("clf", clf)]).fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            results[name] = {
                "accuracy": round(accuracy_score(y_te, pred), 4),
                "precision": round(precision_score(y_te, pred, zero_division=0), 4),
                "recall": round(recall_score(y_te, pred), 4),
                "f1": round(f1_score(y_te, pred), 4),
                "roc_auc": round(roc_auc_score(y_te, proba), 4),
            }
            fitted[name] = pipe

        # Top random-forest feature importances → chart
        rf = fitted["random_forest"]
        names = [n.split("__", 1)[1] for n in rf.named_steps["prep"].get_feature_names_out()]
        imp = (pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
                 .sort_values().tail(12))
        out = _run_dir(context)
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.barh(imp.index, imp.values, color=BLUE, height=0.62, zorder=2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_title("What the random forest relies on", loc="left", fontsize=13,
                     color=INK, pad=22, fontweight="bold")
        ax.text(0, 1.02, f"Top 12 feature importances · test ROC-AUC "
                f"{results['random_forest']['roc_auc']:.3f}",
                transform=ax.transAxes, fontsize=9, color=INK_2)
        fig.tight_layout()
        imp_path = out / "random_forest_feature_importance.png"
        fig.savefig(imp_path, dpi=150)
        plt.close(fig)

        # Append to a history table so every scheduled run is comparable
        run_rows = pd.DataFrame([
            {"run_ts": context["ts"], "model": m, **v, "n_train": len(X_tr), "n_test": len(X_te)}
            for m, v in results.items()])
        run_rows.to_sql("model_runs", eng, schema=SCHEMA, if_exists="append", index=False)

        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        return {"metrics": results, "n_train": len(X_tr), "n_test": len(X_te),
                "top_features": list(imp.index[::-1][:5]), "importance_chart": str(imp_path)}

    @task
    def publish_for_agents(dq: dict) -> dict:
        """Stratified 5k-row sample (features + target) for the Part 2 agent loop."""
        cols = ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
        df = pd.read_sql(f"SELECT {cols} FROM {SCHEMA}.{TABLE}", _engine())
        frac = min(1.0, AGENT_SAMPLE_ROWS / len(df))
        sample = (df.groupby(TARGET, group_keys=False)
                    .apply(lambda g: g.sample(frac=frac, random_state=42))
                    .sample(frac=1, random_state=42))
        AGENT_CSV.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(AGENT_CSV, index=False)
        return {"path": str(AGENT_CSV), "rows": len(sample),
                "bad_review_rate": round(float(sample[TARGET].mean()), 4)}

    @task
    def report(merged: dict, dq: dict, charts: dict, model: dict, published: dict,
               **context) -> dict:
        """Write run_summary.json and mirror this run's outputs to olist/latest/."""
        out = _run_dir(context)
        summary = {
            "run_ts": context["ts"],
            "merge": {"delivered_orders_in": merged["rows_in"], "rows_out": merged["rows"]},
            "warehouse": {"table": f"{SCHEMA}.{TABLE}", **dq},  # includes city_match_rate
            "bad_review_rate_on_time": round(charts["on_time_bad_rate"], 4),
            "bad_review_rate_late": round(charts["late_bad_rate"], 4),
            "models": model["metrics"],
            "top_features": model["top_features"],
            "agent_dataset": published,
        }
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        latest = OUT / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.*"):  # png + json (not staging/)
            shutil.copy2(f, latest / f.name)
        return summary

    # ------------------------------------------------------------------ wiring
    # One sub-group per table: extract → clean. The sub-groups share no edges,
    # so the scheduler runs all of them at the same time.
    cleaners = {"orders": clean_orders, "customers": clean_customers,
                "cities": clean_cities, "reviews": clean_reviews,
                "payments": clean_payments, "items": clean_items}
    with TaskGroup("ingest_and_clean", tooltip="Extract + clean each table in parallel"):
        cleaned = {}
        for table, clean_fn in cleaners.items():
            with TaskGroup(table):
                raw = extract.override(task_id="extract")(table)
                cleaned[table] = clean_fn.override(task_id="clean")(raw)
        # products needs the category translation table → two parallel extracts, one clean
        with TaskGroup("products", tooltip="products ⨝ category translation"):
            raw_products = extract.override(task_id="extract_products")("products")
            raw_translation = extract.override(task_id="extract_translation")("translation")
            cleaned["products"] = clean_products.override(task_id="clean")(
                raw_products, raw_translation)

    with TaskGroup("transform"):
        feats = order_item_features(cleaned["items"], cleaned["products"])
        merged = merge_and_engineer(cleaned["orders"], cleaned["customers"], cleaned["cities"],
                                    cleaned["reviews"], cleaned["payments"], feats)

    with TaskGroup("load"):
        dq = quality_checks(load_orders_clean(merged))

    with TaskGroup("analysis", tooltip="Reads olist.orders_clean from Postgres"):
        charts = make_charts(dq)
        model = train_models(dq)
        published = publish_for_agents(dq)

    report(merged, dq, charts, model, published)


olist_pipeline()
