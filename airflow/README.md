# Airflow (Docker-only) → simple research loop

Classroom sequence:

1. **Student DAG lab** — edit [`dags/student_baseline_prep.py`](dags/student_baseline_prep.py): fill task bodies, then wire deps. Graph tab stays **empty until wired**.
2. **Instructor reveal (DAG 1)** — `simple_loop_baseline_prep`: merge → **subset** → train baseline → publish subset
3. **LangGraph Build Loop** — `/student-loop` on the published CSV (or Titanic)
4. **Instructor reveal** — `/simple-loop` (filled `mini_loop`)
5. **Reveal DAG 2** — same prep + stub that will later trigger the loop from Airflow

Everything runs in **Docker**. Do not install Airflow on the host.

## Data

Vendored under [`datasets/airflow/`](../datasets/airflow/) from the Financial Transaction Intelligence Dataset 2026 zip:

- `customers.csv` ⟕ `transactions.csv` on `customer_id`
- Target: `is_fraud`  
- See [`datasets/airflow/ATTRIBUTION.md`](../datasets/airflow/ATTRIBUTION.md)

## Start Airflow (compose profile)

From the repo root:

```bash
docker compose -f infra/docker-compose.yml --env-file .env --profile airflow up --build -d
```

UI: http://localhost:8088 — login `admin` / `admin` (override with `AIRFLOW_ADMIN_*`).

DAGs:

| DAG id | What it does |
| --- | --- |
| `student_baseline_prep` | **Student lab** — stubs + empty Graph until wiring is uncommented |
| `simple_loop_baseline_prep` | Instructor: Merge → **make_subset** (default 2000 rows) → baseline → publish → **stop** |
| `simple_loop_baseline_and_trigger` | Same prep → **`trigger_simple_loop` stub** (no HTTP yet) |

Student path: implement `@task` bodies in `student_baseline_prep.py` (copy from [`dags/COPY_GUIDE.md`](dags/COPY_GUIDE.md) or `simple_loop_baseline_prep.py`), uncomment the wiring block, refresh the Graph tab, then Trigger.

Instructor path: unpause `simple_loop_baseline_prep` and Trigger.

### One-shot prep without the UI (still Docker)

```bash
docker compose -f infra/docker-compose.yml --env-file .env --profile airflow run --rm airflow-prep-once
```

Writes:

- `artifacts/airflow/runs/<run_id>/merged.csv` (full join)
- `artifacts/airflow/runs/<run_id>/subset.csv` (demo table used for training)
- `datasets/airflow/merged_latest.csv` (published subset for the simple-loop UI)

Override size with DAG param / conf `subset_rows` (default **2000**).

## Then: simple loop standalone

With the main stack up (`gateway` + frontend):

1. Open `/simple-loop`
2. Pick **`merged_latest.csv`**
3. Target column **`is_fraud`**
4. Run

## DAG 2 reveal (later)

`trigger_simple_loop` currently logs a stub payload pointing at `http://gateway:8000`.  
When you reveal the connection, replace that stub in [`dags/simple_loop_baseline_and_trigger.py`](dags/simple_loop_baseline_and_trigger.py) with a real `POST /simple-loop/run`.

Each DAG under `dags/` is **one file**: pandas/sklearn logic sits inside `@task` bodies. No helper modules, no `mini_loop` imports.

## Stop

```bash
docker compose -f infra/docker-compose.yml --profile airflow down
```
