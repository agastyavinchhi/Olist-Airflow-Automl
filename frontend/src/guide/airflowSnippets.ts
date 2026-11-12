// Source of truth: airflow/dags/simple_loop_baseline_prep.py
// Student file is identical except empty bodies + commented wiring.
// Snippets include the same inline comments / wiring header as the reference.

export interface Snippet {
  id: string
  title: string
  hint: string
  code: string
}

/** From repo root, with Airflow profile already up. */
export const UNPAUSE_CMD =
  'docker compose -f infra/docker-compose.yml --env-file .env --profile airflow exec airflow-scheduler \\\n  airflow dags unpause student_baseline_prep'

export const TRIGGER_CMD =
  'docker compose -f infra/docker-compose.yml --env-file .env --profile airflow exec airflow-scheduler \\\n  airflow dags trigger student_baseline_prep'

export const LIST_RUNS_CMD =
  'docker compose -f infra/docker-compose.yml --env-file .env --profile airflow exec airflow-scheduler \\\n  airflow dags list-runs -d student_baseline_prep --state success --limit 5'

export const CONFIRM_OUTPUT_CMD = `ls -la datasets/airflow/merged_latest.csv
ls -la artifacts/airflow/runs/
# optional: peek the newest run folder
ls -la artifacts/airflow/runs/$(ls -t artifacts/airflow/runs | head -1)/`

export const TASK_SNIPPETS: Snippet[] = [
  {
    id: 'validate_customers',
    title: 'validate_customers',
    hint: 'Paste over the raise (docstring already on the stub). Same comments as simple_loop_baseline_prep.py.',
    code: `if not CUSTOMERS.is_file():
    raise FileNotFoundError(CUSTOMERS)
# nrows=0 → headers only (fast)
cols = list(pd.read_csv(CUSTOMERS, nrows=0).columns)
if JOIN_KEY not in cols:
    raise ValueError(f"missing {JOIN_KEY}")
# Returned dict becomes XCom → next task's left_meta
return {"path": str(CUSTOMERS)}`,
  },
  {
    id: 'validate_transactions',
    title: 'validate_transactions',
    hint: 'Paste over the raise — identical to simple_loop_baseline_prep.py.',
    code: `if not TRANSACTIONS.is_file():
    raise FileNotFoundError(TRANSACTIONS)
cols = list(pd.read_csv(TRANSACTIONS, nrows=0).columns)
for need in (JOIN_KEY, TARGET):
    if need not in cols:
        raise ValueError(f"missing {need}")
return {"path": str(TRANSACTIONS)}`,
  },
  {
    id: 'merge_csvs',
    title: 'merge_csvs',
    hint: 'Paste over the raise — keep the multi-line docstring already on the stub.',
    code: `run_id = context["ts_nodash"]  # e.g. 20261104T145012
out = RUNS / run_id / "merged.csv"
out.parent.mkdir(parents=True, exist_ok=True)

left = pd.read_csv(left_meta["path"])
right = pd.read_csv(right_meta["path"])
# fact table on the left of the merge → keep one row per transaction
merged = right.merge(left, on=JOIN_KEY, how="left", suffixes=("", "_customer"))
merged.to_csv(out, index=False)
return {"merged_path": str(out), "n_merged": len(merged)}`,
  },
  {
    id: 'make_subset',
    title: 'make_subset',
    hint: 'Paste over the raise — identical comments as the reference.',
    code: `run_id = context["ts_nodash"]
out = RUNS / run_id / "subset.csv"
df = pd.read_csv(merge_meta["merged_path"])
n = min(SUBSET_ROWS, len(df))
subset = df.sample(n=n, random_state=42) if n < len(df) else df
subset.to_csv(out, index=False)
# Keep key name merged_path so later tasks stay uniform
return {"merged_path": str(out), "n_subset": len(subset)}`,
  },
  {
    id: 'profile_light',
    title: 'profile_light',
    hint: 'Paste over the raise — identical to simple_loop_baseline_prep.py.',
    code: `df = pd.read_csv(subset_meta["merged_path"])
drops = [
    c
    for c in df.columns
    if c != TARGET
    and (c.endswith("_id") or c in {"timestamp", "fraud_probability", "fraud_history"})
]
return {"recommended_drops": drops, "n_rows": len(df)}`,
  },
  {
    id: 'train_baseline',
    title: 'train_baseline',
    hint: 'Paste over the raise — classroom numeric-only LogisticRegression.',
    code: `df = pd.read_csv(subset_meta["merged_path"])
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
return {"f1": f1, "n_train": len(X_train), "n_val": len(X_val)}`,
  },
  {
    id: 'publish_merged',
    title: 'publish_merged',
    hint: 'Paste over the raise — publishes merged_latest.csv for the simple-loop UI.',
    code: `PUBLISH.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(subset_meta["merged_path"], PUBLISH)
return {
    "published_latest": str(PUBLISH),
    "merged_dataset_path": subset_meta["merged_path"],
    "baseline_val_f1": baseline["f1"],
    "target_column": TARGET,
}`,
  },
]

/** Full wiring block including the same header comments as the reference DAG. */
export const WIRING_SNIPPET = `# -----------------------------------------------------------------------
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
publish_merged(demo, base)`

export const REFERENCE_WIRING_SNIPPET = WIRING_SNIPPET
