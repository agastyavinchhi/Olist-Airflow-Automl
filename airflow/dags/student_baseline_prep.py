"""STUDENT LAB — identical to simple_loop_baseline_prep.py; fill empty task bodies.

Copy each @task body + the wiring block from simple_loop_baseline_prep.py.
Only differences vs the reference: this docstring, dag_id/tags, empty bodies, commented wiring.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
from airflow.decorators import dag, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths inside the Airflow containers (/repo is the mounted project root)
# ---------------------------------------------------------------------------
REPO = Path("/repo")
CUSTOMERS = REPO / "datasets" / "airflow" / "customers.csv"
TRANSACTIONS = REPO / "datasets" / "airflow" / "transactions.csv"
RUNS = REPO / "artifacts" / "airflow" / "runs"
PUBLISH = REPO / "datasets" / "airflow" / "merged_latest.csv"
TARGET = "is_fraud"
JOIN_KEY = "customer_id"
SUBSET_ROWS = 2000


# @dag turns the function below into an Airflow DAG (a named pipeline).
# - dag_id: unique name you see in the Airflow UI
# - start_date: earliest logical date Airflow will consider (required)
# - schedule=None: manual Trigger only (no cron) — perfect for class
# - catchup=False: do not backfill old dates when the DAG is unpaused
# - tags: filters in the UI
#
# Calling student_baseline_prep() at the bottom REGISTERS this DAG
# with the scheduler. Without that call, Airflow never sees it.
@dag(
    dag_id="student_baseline_prep",
    description="STUDENT LAB: Merge → subset → baseline → publish (wire me)",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["student", "baseline", "prep", "classroom"],
)
def student_baseline_prep():
    """Define tasks, then wire them. Airflow builds the Graph from the wiring."""

    # @task turns a normal Python function into one node on the Graph.
    # - Airflow runs each task in its own worker process
    # - Whatever you return is stored in XCom (Airflow's small data bus)
    # - Downstream tasks receive that return value as a normal argument
    # - If the function raises, the task fails (red) in the UI
    @task
    def validate_customers():
        """Check the customers CSV exists and has the join key."""
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("validate_customers")

    @task
    def validate_transactions():
        """Check the transactions CSV exists and has join key + target label."""
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("validate_transactions")

    @task
    def merge_csvs(left_meta: dict, right_meta: dict, **context):
        """Join transactions ⟕ customers. One output row per transaction.

        left_meta / right_meta come from upstream @task return values (XCom).
        **context is Airflow runtime info — we use ts_nodash as a unique run folder.
        """
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("merge_csvs")

    @task
    def make_subset(merge_meta: dict, **context):
        """Downsample so baseline + the later simple-loop demo stay fast in class."""
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("make_subset")

    @task
    def profile_light(subset_meta: dict):
        """List columns that look leaky / id-like so training can drop them."""
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("profile_light")

    @task
    def train_baseline(subset_meta: dict, profile_meta: dict):
        """Tiny numeric-only LogisticRegression — a classroom A-baseline, not production ML."""
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("train_baseline")

    @task
    def publish_merged(subset_meta: dict, baseline: dict):
        """Copy the subset to a stable path the simple-loop UI can pick up."""
        # TODO: paste body from simple_loop_baseline_prep.py
        raise NotImplementedError("publish_merged")

    # -----------------------------------------------------------------------
    # WIRING = the dependency graph
    #
    # Calling a @task function here does NOT run the Python immediately.
    # It creates a node and an edge: "this task needs that task's XCom output".
    # Airflow's Graph tab is literally this block drawn as boxes + arrows.
    #
    #   validate_customers ─┐
    #                       ├─→ merge_csvs → make_subset → profile_light ─┐
    #   validate_transactions ┘                    └──────────────────────┼─→ train → publish
    # -----------------------------------------------------------------------
    # TODO: uncomment (identical to simple_loop_baseline_prep.py):
    # left = validate_customers()
    # right = validate_transactions()
    # merged = merge_csvs(left, right)  # waits for BOTH validates
    # demo = make_subset(merged)
    # prof = profile_light(demo)
    # base = train_baseline(demo, prof)  # needs subset path + drop list
    # publish_merged(demo, base)


# Actually register the DAG with Airflow (decorators alone are not enough).
student_baseline_prep()
