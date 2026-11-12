# Olist Bad-Review Prediction: Airflow Pipeline + Agent-Based ML Loop

**Question:** which delivered e-commerce orders end up with a **bad review (score ≤ 2)**?

- **Part 1:** a scheduled **Airflow** DAG ingests 7 related tables from the Olist Brazilian
  e-commerce dataset. It cleans each table in parallel TaskGroups, merges them, loads the
  result into **PostgreSQL**, then reads it back to make charts and train two models.
- **Part 2:** the in-class **LangGraph** agent loop, adapted to read that Postgres table and to
  train and evaluate models. It has 3 new agents, 2 agents merged into 1, 5 modified agents,
  and an updated live agent graph in the web UI.

> Built on top of the class template
> [Kedar-V/automl-research-lab](https://github.com/Kedar-V/automl-research-lab). The template's
> original setup guide is preserved at [docs/TEMPLATE_README.md](docs/TEMPLATE_README.md).
> Everything described below is new work for this assignment.

---

## Assignment checklist

| Requirement | Where it is | Evidence |
|---|---|---|
| **Part 1.1** Deploy Airflow, scheduled DAG | `infra/docker-compose.yml` (`--profile airflow`), [`airflow/dags/olist_pipeline.py`](airflow/dags/olist_pipeline.py): `schedule="0 6 * * *"` | [DAG list](docs/images/olist/olist-dag-list.png) |
| Orchestrate ingestion → analysis | 21 tasks: extract → clean → transform → load → quality checks → analysis → report | [Graph view](docs/images/olist/olist-graph.png) |
| Independent tasks run in parallel | 7 extracts + 6 cleans run together; 3 analysis tasks run together (LocalExecutor) | [Gantt](docs/images/olist/olist-gantt.png) |
| **Part 1.2** ≥ 2 related datasets | 7 Olist tables joined on `order_id` / `customer_id` / `product_id` | [Attribution](datasets/olist/ATTRIBUTION.md) |
| Clean + transform each, TaskGroups | 10 TaskGroups (`ingest_and_clean` with one sub-group per table, `transform`, `load`, `analysis`) | [Grid](docs/images/olist/olist-grid.png) |
| Merge + load into a database | 5-way merge → `olist.orders_clean` in a dedicated Postgres (`warehouse-db`) | [§ Part 1](#load-into-postgresql) |
| **Part 1.3** Read cleaned data from the DB | All analysis tasks query Postgres (SQL aggregates + `SELECT` for training) | `make_charts`, `train_models` |
| Analysis: visualization / model | 3 charts + logistic regression and random forest, logged to `olist.model_runs` | [Charts](#analysis-results) |
| **Part 1.4** Document process, scheduling, parallelism | This README | [§ Part 1](#part-1--airflow-data-pipeline) |
| **Part 2** Adapt the in-class agent workflow to Part 1's cleaned data | [`olist_loop/graph.py`](olist_loop/graph.py) reads `olist.orders_clean` | [§ Part 2](#part-2--agent-based-ml-workflow) |
| Add / remove / modify agents | +3 new, 2→1 merged, 5 modified | [Agent changes](#agent-changes-vs-the-in-class-template) |
| Update the live agent graph | [`frontend/src/graph/olistLoopConfig.ts`](frontend/src/graph/olistLoopConfig.ts), page `/olist-loop` | [Live graph](docs/images/olist/olist-agent-graph-running.png) |
| Report accuracy / appropriate metric | Accuracy, precision, recall, **F1**, ROC-AUC, PR-AUC, confusion matrix | [Final evaluation](docs/images/olist/olist-agent-final-eval.png) |
| **Submission** Repo with DAG, agent code, run instructions | This repo | [§ How to run](#how-to-run-both-workflows) |
| Containerized dev setup | `infra/Dockerfile.airflow`, `infra/Dockerfile.service`, `infra/docker-compose.yml`, [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) | [§ Containers](#containerized-setup) |
| Screenshots: DAG run, DAG graph, agent graph, evaluation | [`docs/images/olist/`](docs/images/olist/) | [§ Screenshots](#screenshot-index) |
| Summary of pipeline, agent changes, findings | This README | [§ Key findings](#key-findings) |

---

## How to run both workflows

### Prerequisites

Docker Desktop (running), Git, Make. For the web UI you also need Node 20+. Airflow and every
Python dependency run inside containers, so nothing else needs installing on the host.
Alternatively, open the repo in the dev container (see [Containerized setup](#containerized-setup)).

### 0. Clone and configure

```bash
git clone <this-repo-url> && cd automl-research-lab
cp .env.example .env
```

`.env` works as-is. Optionally add `OPENROUTER_API_KEY=...` (free at openrouter.ai) so the Part 2
agents call a real LLM. Without a key, each agent falls back to a built-in heuristic and the loop
still runs end to end.

The raw Olist CSVs (~120 MB) are committed in `datasets/olist/raw/`, so no Kaggle download is needed.

### 1. Part 1: Airflow pipeline

```bash
make airflow        # builds the Airflow image, starts Airflow + its metadata DB + warehouse-db
```

1. Wait ~1–2 minutes, then open **http://localhost:8088** and log in with `admin` / `admin`.
2. Find **`olist_pipeline`** (tag `assignment`) and switch the toggle **on**.
   - Because `catchup=False`, the scheduler immediately runs the most recent 06:00 interval.
   - It then runs every day at 06:00 UTC.
   - You can also press ▶ to trigger a manual run.
3. A run takes about **45 seconds**. Every task turns green.

Check the output:

```bash
# cleaned table + model metrics in Postgres (warehouse-db is exposed on host port 5433)
docker compose -f infra/docker-compose.yml --env-file .env exec warehouse-db \
  psql -U warehouse -d warehouse -c "SELECT count(*), avg(is_bad_review) FROM olist.orders_clean;" \
  -c "SELECT run_ts, model, accuracy, f1, roc_auc FROM olist.model_runs ORDER BY run_ts DESC LIMIT 3;"

ls artifacts/airflow/olist/latest/    # charts, metrics.json, run_summary.json
```

### 2. Part 2: agent loop

The agent loop needs `olist.orders_clean`, so run Part 1 first.

```bash
make start-mac       # macOS/Linux: backend containers + web UI (use make start-windows in Git Bash)
```

Open **http://127.0.0.1:5173/olist-loop**, keep the defaults (target F1 0.50, 3 cycles, all rows),
and click **Run loop**.

- The live agent graph highlights each agent as it works.
- The *Experiments* table fills one row per cycle.
- *Final evaluation* appears after the sealed-holdout step.
- A run takes about **1 minute** with an LLM, or about 25 seconds with heuristics only.

Command-line alternative, inside the gateway container:

```bash
docker compose -f infra/docker-compose.yml --env-file .env exec -w / gateway \
  python -m olist_loop.run_cli --max-cycles 3 --target-score 0.5
```

### Stop everything

```bash
make down-mac        # or: make down-windows
```

| Service | URL |
|---|---|
| Airflow UI | http://localhost:8088 |
| Agent loop UI | http://127.0.0.1:5173/olist-loop |
| Gateway API | http://127.0.0.1:8000 (`/olist-loop/info`, `/olist-loop/stream`, `/olist-loop/runs`) |
| Warehouse Postgres | `postgresql://warehouse:warehouse@localhost:5433/warehouse` |

---

## Part 1 — Airflow data pipeline

### Data sources

The [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0) contains about 99k real, anonymized orders from 2016-09 to 2018-10. The pipeline
uses **7 related tables**:

| Table | Rows | Joins on | Role |
|---|---|---|---|
| `orders` | 99,441 | `order_id`, `customer_id` | fact table: status + 5 timestamps |
| `customers` | 99,441 | `customer_id` | customer city / state |
| `order_items` | 112,650 | `order_id`, `product_id` | 1–21 items per order: price, freight, seller |
| `order_payments` | 103,886 | `order_id` | 1–n payments per order: type, installments, value |
| `order_reviews` | 99,224 | `order_id` | **label source**: review score 1–5 |
| `products` | 32,951 | `product_id` | category, weight, dimensions |
| `product_category_name_translation` | 71 | category name | Portuguese → English category names |

**Target:** `is_bad_review = review_score <= 2`, which is **12.8%** of delivered orders.

### Transformations

Each table has its own `extract` task and `clean` task:

- `extract` reads the raw CSV, enforces a column contract (it fails fast if a required column is
  missing), and stages the data as parquet.
- `clean` applies the table-specific fixes below.

| Table | Problems found while profiling | Cleaning / transformation |
|---|---|---|
| orders | 8 statuses; 2,965 missing delivery dates | keep `delivered` orders with a delivery date; parse timestamps; derive `delivery_days`, `estimated_days`, **`days_late`**, `is_late`, `approval_hours`, `carrier_days`, purchase month / weekday / hour |
| customers | inconsistent text | dedupe; upper-case state; lower-case city; zero-pad zip |
| reviews | 551 orders with > 1 review | keep the **latest** review per order; valid scores 1–5 only |
| payments | 3 `not_defined` types; several rows per order | drop invalid rows; **aggregate per order**: total, count, max installments, voucher flag, main payment type |
| items | several rows per order | drop duplicates and non-positive prices |
| products | 610 missing categories / sizes | translate category to English; fill `unknown`; median-impute sizes; derive volume |

Then the `transform` TaskGroup:

- `order_item_features`: items ⨝ products → one row per order (item / product / seller counts,
  totals, average weight / volume, category of the most expensive item).
- `merge_and_engineer`: orders ⨝ customers ⨝ item features ⨝ payments ⨝ reviews.
  - Adds `freight_ratio` and the target.
  - Asserts there are no duplicate orders.
  - Result: **95,824 rows × 35 columns**, one row per delivered, reviewed order.

### Pipeline structure

![Airflow graph view: olist_pipeline with every TaskGroup expanded](docs/images/olist/olist-graph.png)

```
[ingest_and_clean]  (all sub-groups in parallel)          [transform]
  orders    : extract → clean ─────────────────────────────┐
  customers : extract → clean ─────────────────────────────┤
  reviews   : extract → clean ─────────────────────────────┼──→ merge_and_engineer
  payments  : extract → clean ─────────────────────────────┤            │
  items     : extract → clean ───────┐                     │            │
  products  : extract_products ──┐   ├──→ order_item_features ┘           ▼
              extract_translation┴→ clean ┘                  [load]  load_orders_clean → quality_checks
                                                                                              │
                                        [analysis]  (parallel; each reads Postgres)           ▼
                                        make_charts  │  train_models  │  publish_for_agents ──→ report
```

- **21 tasks** in **10 TaskGroups**, written with the TaskFlow API (`@task`).
- Only small dicts (file paths and row counts) travel through XCom. The data itself moves as
  parquet files in a per-run folder: `artifacts/airflow/olist/<run>/staging/`.
- Postgres is reached through the Airflow Connection **`warehouse_db`**, defined in
  docker-compose as `AIRFLOW_CONN_WAREHOUSE_DB` and used with `PostgresHook`.
- Each task retries once (`retries=1`, 1-minute delay).

### Scheduling

```python
@dag(schedule="0 6 * * *", start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1, ...)
```

- **Daily at 06:00 UTC.** The DAG list shows the cron expression and the next run time.
- **`catchup=False`:** turning the DAG on runs only the latest interval, not one run for every day
  since `start_date`. Our first run was the scheduler's `scheduled__2026-11-11T06:00` run, started
  automatically when the DAG was turned on.
- **`max_active_runs=1`:** two runs never replace `olist.orders_clean` at the same time.
- **History:** `olist.model_runs` is append-only, so each scheduled run adds a row per model and
  the metrics can be compared over time.

### Parallel processing

The DAG runs on Airflow's **LocalExecutor**, which allows up to 32 tasks at once and up to 16 per
DAG. Tasks with no dependency between them run at the same time:

- All **7 extract tasks** start together at the beginning of the run.
- Each table's `clean` starts as soon as its own `extract` finishes.
- `make_charts`, `train_models` and `publish_for_agents` all depend only on `quality_checks`, so
  they run together.

![Gantt chart: parallel extracts/cleans and parallel analysis tasks](docs/images/olist/olist-gantt.png)

The whole run takes **≈ 45 s**. Most of that is two steps: the Postgres load (~20 s, 96k rows) and
the random-forest fit (~20 s, which overlaps with the other analysis tasks).

### Load into PostgreSQL

A dedicated `warehouse-db` service (Postgres 16) holds the analytics data. It is separate from
`airflow-db`, which only stores Airflow's own metadata.

| Table | Written by | Contents |
|---|---|---|
| `olist.orders_clean` | `load.load_orders_clean` (replace each run, then `PRIMARY KEY (order_id)` + index on state) | the merged table, 95,824 rows |
| `olist.model_runs` | `analysis.train_models` (append) | per-run, per-model accuracy / precision / recall / F1 / ROC-AUC |
| `olist.agent_runs` | Part 2 `holdout_eval` (append) | agent loop holdout metrics |

`load.quality_checks` fails the run if any of these checks fail:

- the row count doesn't match the staged count;
- there are fewer than 50k rows;
- the target or any model feature contains nulls;
- the bad-review rate falls outside [2%, 50%].

### Analysis results

Every analysis task **reads from Postgres**. The charts are built from SQL aggregates, and model
training runs a `SELECT` over the feature columns.

![Bad-review rate by delivery timing](docs/images/olist/bad_review_rate_by_delivery_timing.png)

- **Late deliveries drive bad reviews.**
  - Orders delivered on time or early get a bad review **9.2%** of the time.
  - Late orders get one **54.1%** of the time, and orders more than a week late about **78–79%**.

![Bad-review rate by customer state](docs/images/olist/bad_review_rate_by_state.png)

- **Geography matters too.**
  - São Paulo is at 10.7%.
  - Rio de Janeiro (18.3%) and the northeast states, Maranhão at 19.9%, are much higher. Those
    states are farther from most sellers, which means longer and riskier deliveries.

**Models** (stratified 80/20 split, scored on the 20% test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority class (always "good") | 0.872 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression (balanced) | 0.811 | 0.349 | 0.550 | 0.427 | 0.754 |
| **Random forest (balanced)** | **0.860** | **0.456** | **0.487** | **0.471** | **0.758** |

**Accuracy alone is misleading here.** Always predicting "good review" scores 87% accuracy while
catching zero bad reviews. That's why we report **F1 and ROC-AUC**, and why F1 is the target metric
in Part 2.

![Random forest feature importances](docs/images/olist/random_forest_feature_importance.png)

The model's top features are `days_late`, `delivery_days` and `is_late`, which matches the chart.

---

## Part 2 — Agent-based ML workflow

The in-class template ([`mini_loop/graph.py`](mini_loop/graph.py)) is a LangGraph loop:

```
data_scientist → hypothesis_generator → ml_engineer → judge ─ revise → memory_writer → context_builder ─┐
                         ↑_______________________________________________________________________________┘
                                                             └ accept → holdout_eval
```

That template takes any CSV, trains on it, and judges by F1 alone. We adapted it into
[`olist_loop/graph.py`](olist_loop/graph.py), which works on **Part 1's Postgres table**:

```
olist.orders_clean ─SQL→ warehouse_loader → data_scientist → hypothesis_generator → feature_engineer → ml_engineer
                                                                ↑                                            │ val proba
                                                         research_memory ←revise─ router ← judge ←report─ evaluator
                                                                                    │ accept / out of budget
                                                                                    ▼
                                                              best_candidate → holdout_eval ─INSERT→ olist.agent_runs
```

The original `mini_loop` and its `/simple-loop` page are unchanged, so the two can be compared
side by side.

### Agent changes vs. the in-class template

| Agent | Change | Type | Responsibility |
|---|---|---|---|
| **warehouse_loader** | **added** | code | Reads `olist.orders_clean` **from Postgres** (`ORDER BY order_id`, so splits are reproducible). Also loads the latest Airflow metrics from `olist.model_runs` as baselines. Falls back to the CSV Airflow publishes if the database is unreachable. |
| data_scientist | modified | LLM | Olist-specific profile: class imbalance, skewed money columns, top correlations, leakage check. Explicitly forbidden from choosing models or features. |
| hypothesis_generator | modified | LLM | Proposes the next experiment as **feature ideas + model family + imbalance strategy**, using the memory brief. |
| **feature_engineer** | **added** | LLM + code | Picks recipes from a **fixed menu of 7 row-wise transforms**: `log_money`, `lateness_buckets`, `delivery_speed_ratio`, `multi_seller_flag`, `basket_complexity`, `drop_calendar`, `drop_state`. Code applies them, so the LLM never edits the data and can't introduce leakage. |
| ml_engineer | modified | LLM + code | Concrete hyperparameters for logistic regression / random forest / histogram gradient boosting, plus a **threshold strategy** (`default_0.5` or `tune_f1`). Values are clamped to safe ranges; if training fails, it falls back to default hyperparameters and retries once. |
| **evaluator** | **added** | code | Splits measuring from training. Computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix on validation. Tunes the decision threshold when requested, and compares against the majority baseline and Airflow's random forest. |
| judge | modified | LLM | Decides from the evaluator's report only. A coded guardrail overrides an LLM "accept" when F1 is below target. |
| memory_writer + context_builder | **merged → `research_memory`** | code | One node files the cycle's lesson and builds the next brief. The template's two bookkeeping nodes never needed to be separate. |
| holdout_eval | modified | code | Refits the best configuration on train + validation, scores the **sealed 20% holdout exactly once** with the tuned threshold, and **writes the result to `olist.agent_runs`**. |

**How the data is split:** a fixed seed divides the data into 60% train, 20% validation and 20%
sealed holdout, and every agent recomputes the same split. Large data never sits in graph state.
Agents pass file paths (pickle / npy) in a per-run cache folder, which keeps the state small enough
to stream to the browser.

### Updated live agent graph

The template's React Flow canvas was hardwired to the simple loop. We changed
[`LiveSimpleLoopGraph.tsx`](frontend/src/components/LiveSimpleLoopGraph.tsx) to accept any loop's
config (the default is unchanged, so `/simple-loop` still works).

The Olist graph is defined in [`olistLoopConfig.ts`](frontend/src/graph/olistLoopConfig.ts) and
rendered on the new page [`OlistLoopPage.tsx`](frontend/src/pages/OlistLoopPage.tsx) at
**`/olist-loop`**. It has 13 nodes: the 9 agents, the router, the best-candidate state, and the two
Postgres tables.

New gateway endpoints in [`services/gateway/app/main.py`](services/gateway/app/main.py):

- `/olist-loop/stream` streams each agent start, finish and router decision as Server-Sent Events.
- `/olist-loop/runs` returns run history.

The page shows the active agent, animates the edge into it, and marks completed agents green.

![Live agent graph mid-run: Hypothesis Generator running in cycle 2, brief edge from Research memory animated](docs/images/olist/olist-agent-graph-running.png)

### Evaluation results

These are from the run shown in the screenshots, using the free OpenRouter LLM
`nvidia/nemotron-3.5-lightning:free`. The target was validation F1 ≥ 0.50, with a budget of 3 cycles.

![Experiments table: one row per cycle](docs/images/olist/olist-agent-experiments.png)

| Cycle | Agents chose | Val F1 | Val accuracy | Val ROC-AUC | Threshold |
|---|---|---|---|---|---|
| 1 | logistic regression + log_money, lateness_buckets, delivery_speed_ratio | 0.440 | 0.821 | 0.759 | 0.50 |
| 2 | random forest + basket_complexity, multi_seller_flag | 0.491 | 0.884 | 0.765 | 0.52 |
| 3 | hist. gradient boosting + all 5 additive recipes | **0.497** | 0.881 | 0.768 | 0.65 |

![Final evaluation on the sealed holdout vs Part 1 baselines](docs/images/olist/olist-agent-final-eval.png)

**Final evaluation on the sealed holdout** (19,165 orders), compared with Part 1:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority class | 0.872 | 0.000 | 0.000 | 0.000 | 0.500 |
| Airflow logistic regression (Part 1) | 0.811 | 0.349 | 0.550 | 0.427 | 0.754 |
| Airflow random forest (Part 1) | 0.860 | 0.456 | 0.487 | 0.471 | 0.758 |
| **Agent loop: hist. gradient boosting** | **0.874** | **0.507** | 0.442 | **0.472** | **0.771** |

Confusion matrix on the holdout: 1,084 bad reviews caught, 1,370 missed, 1,052 false alarms out of
16,711 good orders. PR-AUC is 0.476.

The Airflow rows come from Part 1's own 20% test split; the agent row comes from the loop's sealed
20% holdout. Both splits are the same size and stratified, but they aren't the same rows.

Across the 4 agent runs we did, holdout F1 ranged from 0.461 to 0.472 and ROC-AUC from 0.758 to
0.771. Every run is logged in `olist.agent_runs`.

---

## Key findings

1. **Delivery performance is the main driver of bad reviews.**
   - On-time or early orders: 9.2% bad reviews. Late orders: 54.1%. More than a week late: ~78%.
   - Every model's top features are lateness features.
2. **Accuracy is the wrong metric for this problem.** A model that always predicts "good" scores
   87.2% accuracy and is useless. F1 and ROC-AUC show the real differences between models.
3. **The agent loop matched Part 1 on F1 and beat it on accuracy and ranking.**
   - Holdout F1 0.472 vs 0.471; accuracy 0.874 vs 0.860; ROC-AUC 0.771 vs 0.758.
   - The two biggest gains came from the new agents: the **evaluator's threshold tuning** (+0.04 F1
     in cycle 3, from 0.457 at a 0.5 cutoff to 0.497) and the **feature engineer's recipes** plus
     the switch to gradient boosting.
4. **F1 levels off around 0.47–0.50.** No run reached the 0.50 target on holdout, so the judge
   never accepted and the loop ran out of cycles each time. The data explains *delivery-caused*
   dissatisfaction well, but product quality and expectations aren't in the data.
5. **The agents were built to stay safe.**
   - The LLM chooses from fixed menus: recipes, model families, clamped hyperparameter ranges.
   - Code does every data operation, and the holdout set is scored only once.
   - So a weak or failing free LLM can only produce a worse experiment, not a broken or leaky one.

---

## Containerized setup

Every component runs in Docker via [`infra/docker-compose.yml`](infra/docker-compose.yml):

| Service | Image / Dockerfile | Purpose |
|---|---|---|
| `airflow-webserver`, `airflow-scheduler`, `airflow-init` | [`infra/Dockerfile.airflow`](infra/Dockerfile.airflow): Airflow 2.10.4 + pandas < 2.2, scikit-learn, matplotlib | Part 1 (profile `airflow`) |
| `airflow-db` | postgres:16-alpine | Airflow metadata only |
| `warehouse-db` | postgres:16-alpine, host port 5433 | `olist.*` analytics tables (no profile → shared by both parts) |
| `gateway` | [`infra/Dockerfile.service`](infra/Dockerfile.service) + LangGraph, SQLAlchemy, psycopg2 | runs the agent loop and streams SSE |
| other template services | `infra/Dockerfile.service` | unchanged template stack |

pandas is pinned below 2.2 in the Airflow image: pandas 2.2+ needs SQLAlchemy 2 for `to_sql`, but
Airflow 2.10 pins SQLAlchemy 1.4.

**Dev container:** [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) provides
Python 3.12, Node 20, the Docker CLI (using the host's Docker daemon), `make` and `psql`. It forwards
ports 8088 / 8000 / 5173 / 5433 and creates `.env` automatically.

1. Open the repo in VS Code and choose **Reopen in Container**.
2. Run `make airflow` and `make start-mac` as above.

---

## What's new in this repo

| Path | What |
|---|---|
| [`airflow/dags/olist_pipeline.py`](airflow/dags/olist_pipeline.py) | Part 1 DAG |
| [`datasets/olist/`](datasets/olist/) | raw Olist CSVs + [attribution and data-quality notes](datasets/olist/ATTRIBUTION.md) |
| [`olist_loop/`](olist_loop/) | Part 2 LangGraph loop (`graph.py`) + CLI (`run_cli.py`) |
| [`frontend/src/pages/OlistLoopPage.tsx`](frontend/src/pages/OlistLoopPage.tsx), [`frontend/src/graph/olistLoopConfig.ts`](frontend/src/graph/olistLoopConfig.ts) | Part 2 page + live graph config |
| `frontend/src/components/LiveSimpleLoopGraph.tsx`, `.../graph/SimpleLoopFlowNode.tsx` | graph canvas made reusable for any loop; Postgres node style |
| `services/gateway/app/main.py`, `services/gateway/requirements.txt` | `/olist-loop/*` endpoints + DB drivers |
| `infra/docker-compose.yml`, `infra/Dockerfile.airflow` | `warehouse-db`, Airflow connection, gateway wiring, image deps |
| [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) | dev container |
| [`docs/images/olist/`](docs/images/olist/) | all screenshots and charts |
| `datasets/airflow/olist_orders_latest.csv` | 5k-row sample published by the DAG; the agent loop's fallback input |

## Screenshot index

| Requirement | File |
|---|---|
| Successful DAG execution: runs + task states | [olist-grid.png](docs/images/olist/olist-grid.png) |
| Schedule + run summary | [olist-dag-list.png](docs/images/olist/olist-dag-list.png) |
| DAG graph view | [olist-graph.png](docs/images/olist/olist-graph.png) |
| Parallel execution timeline | [olist-gantt.png](docs/images/olist/olist-gantt.png) |
| Analysis charts | [delivery timing](docs/images/olist/bad_review_rate_by_delivery_timing.png) · [state](docs/images/olist/bad_review_rate_by_state.png) · [feature importance](docs/images/olist/random_forest_feature_importance.png) |
| Agent graph (running / finished) | [running](docs/images/olist/olist-agent-graph-running.png) · [finished](docs/images/olist/olist-agent-graph-done.png) |
| Model evaluation results | [final evaluation](docs/images/olist/olist-agent-final-eval.png) · [experiments](docs/images/olist/olist-agent-experiments.png) · [full page](docs/images/olist/olist-agent-results-page.png) |

## Credits

- Template: [Kedar-V/automl-research-lab](https://github.com/Kedar-V/automl-research-lab).
  This repo uses its Docker stack, Airflow setup, LangGraph `mini_loop`, LLM helpers
  (`loop_runtime.py`) and React Flow UI. The template's own guide is in
  [docs/TEMPLATE_README.md](docs/TEMPLATE_README.md).
- Data: Olist, *Brazilian E-Commerce Public Dataset* (Kaggle, CC BY-NC-SA 4.0).
