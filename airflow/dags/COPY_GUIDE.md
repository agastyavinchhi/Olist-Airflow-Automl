# Student copy guide — Airflow baseline prep

Paste into [`student_baseline_prep.py`](student_baseline_prep.py) **one task at a time**.  
Filled reference (same code + comments): [`simple_loop_baseline_prep.py`](simple_loop_baseline_prep.py).

The skeleton already has path constants, `@dag` / `@task` comments, and each function’s docstring.  
Paste **over the `raise NotImplementedError`** (keep the docstring above it).

In-app copy buttons: `/guide` → Airflow tab.

---

## Reference comments already in the skeleton (do not delete)

```python
# ---------------------------------------------------------------------------
# Paths inside the Airflow containers (/repo is the mounted project root)
# ---------------------------------------------------------------------------

# @dag turns the function below into an Airflow DAG (a named pipeline).
# - dag_id: unique name you see in the Airflow UI
# - start_date: earliest logical date Airflow will consider (required)
# - schedule=None: manual Trigger only (no cron) — perfect for class
# - catchup=False: do not backfill old dates when the DAG is unpaused
# - tags: filters in the UI
#
# Calling student_baseline_prep() at the bottom REGISTERS this DAG
# with the scheduler. Without that call, Airflow never sees it.

# @task turns a normal Python function into one node on the Graph.
# - Airflow runs each task in its own worker process
# - Whatever you return is stored in XCom (Airflow's small data bus)
# - Downstream tasks receive that return value as a normal argument
# - If the function raises, the task fails (red) in the UI

# Actually register the DAG with Airflow (decorators alone are not enough).
```

---

## Step 1 — `validate_customers`

Docstring on stub: `"""Check the customers CSV exists and has the join key."""`

```python
if not CUSTOMERS.is_file():
    raise FileNotFoundError(CUSTOMERS)
# nrows=0 → headers only (fast)
cols = list(pd.read_csv(CUSTOMERS, nrows=0).columns)
if JOIN_KEY not in cols:
    raise ValueError(f"missing {JOIN_KEY}")
# Returned dict becomes XCom → next task's left_meta
return {"path": str(CUSTOMERS)}
```

---

## Step 2 — `validate_transactions`

Docstring: `"""Check the transactions CSV exists and has join key + target label."""`

```python
if not TRANSACTIONS.is_file():
    raise FileNotFoundError(TRANSACTIONS)
cols = list(pd.read_csv(TRANSACTIONS, nrows=0).columns)
for need in (JOIN_KEY, TARGET):
    if need not in cols:
        raise ValueError(f"missing {need}")
return {"path": str(TRANSACTIONS)}
```

---

## Step 3 — `merge_csvs`

Docstring already explains XCom + `**context`. Paste body:

```python
run_id = context["ts_nodash"]  # e.g. 20261104T145012
out = RUNS / run_id / "merged.csv"
out.parent.mkdir(parents=True, exist_ok=True)

left = pd.read_csv(left_meta["path"])
right = pd.read_csv(right_meta["path"])
# fact table on the left of the merge → keep one row per transaction
merged = right.merge(left, on=JOIN_KEY, how="left", suffixes=("", "_customer"))
merged.to_csv(out, index=False)
return {"merged_path": str(out), "n_merged": len(merged)}
```

---

## Step 4 — `make_subset`

Docstring: `"""Downsample so baseline + the later simple-loop demo stay fast in class."""`

```python
run_id = context["ts_nodash"]
out = RUNS / run_id / "subset.csv"
df = pd.read_csv(merge_meta["merged_path"])
n = min(SUBSET_ROWS, len(df))
subset = df.sample(n=n, random_state=42) if n < len(df) else df
subset.to_csv(out, index=False)
# Keep key name merged_path so later tasks stay uniform
return {"merged_path": str(out), "n_subset": len(subset)}
```

---

## Step 5 — `profile_light`

Docstring: `"""List columns that look leaky / id-like so training can drop them."""`

```python
df = pd.read_csv(subset_meta["merged_path"])
drops = [
    c
    for c in df.columns
    if c != TARGET
    and (c.endswith("_id") or c in {"timestamp", "fraud_probability", "fraud_history"})
]
return {"recommended_drops": drops, "n_rows": len(df)}
```

---

## Step 6 — `train_baseline`

Docstring: `"""Tiny numeric-only LogisticRegression — a classroom A-baseline, not production ML."""`

```python
df = pd.read_csv(subset_meta["merged_path"])
drops = list(profile_meta.get("recommended_drops") or [])
y = df[TARGET]
X = (
    df.drop(columns=[TARGET, *drops], errors="ignore")
    .select_dtypes("number")
    .fillna(0)
)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() > 1 else None,
)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(X_train, y_train)
f1 = float(f1_score(y_val, clf.predict(X_val), zero_division=0))
return {"f1": f1, "n_train": len(X_train), "n_val": len(X_val)}
```

---

## Step 7 — `publish_merged`

Docstring: `"""Copy the subset to a stable path the simple-loop UI can pick up."""`

```python
PUBLISH.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(subset_meta["merged_path"], PUBLISH)
return {
    "published_latest": str(PUBLISH),
    "merged_dataset_path": subset_meta["merged_path"],
    "baseline_val_f1": baseline["f1"],
    "target_column": TARGET,
}
```

---

## Step 8 — Wiring (fills the Graph tab)

Replace the commented wiring block with this (same header comments as the reference):

```python
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
left = validate_customers()
right = validate_transactions()
merged = merge_csvs(left, right)  # waits for BOTH validates
demo = make_subset(merged)
prof = profile_light(demo)
base = train_baseline(demo, prof)  # needs subset path + drop list
publish_merged(demo, base)
```

---

## Step 9 — Run (terminal, from repo root)

```bash
docker compose -f infra/docker-compose.yml --env-file .env --profile airflow exec airflow-scheduler \
  airflow dags unpause student_baseline_prep

docker compose -f infra/docker-compose.yml --env-file .env --profile airflow exec airflow-scheduler \
  airflow dags trigger student_baseline_prep

docker compose -f infra/docker-compose.yml --env-file .env --profile airflow exec airflow-scheduler \
  airflow dags list-runs -d student_baseline_prep --state success --limit 5

ls -la datasets/airflow/merged_latest.csv
ls -la artifacts/airflow/runs/
ls -la artifacts/airflow/runs/$(ls -t artifacts/airflow/runs | head -1)/
```

Or UI: http://localhost:8088 → unpause → Trigger → Graph all green.
