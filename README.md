# Bad-Review Risk for Brazilian E-commerce: Airflow Pipeline + Diagnosis-Routed Agents

## Problem statement

Bad reviews (1–2★) hurt a marketplace's sellers and ranking. Once one is posted, it's too late.
This project **flags delivered orders that are likely to receive a bad review**, so a customer
support team can contact those customers first: apologize for a late delivery, offer a voucher,
fix the issue.

It combines **two independent datasets**:

| Dataset | Publisher | What it adds |
|---|---|---|
| **Olist Brazilian E-Commerce** (7 related tables) | Olist, via Kaggle | orders, delivery timestamps, items, payments, products, **reviews (label)** |
| **Brazilian Cities** (IBGE statistics) | IBGE / Atlas Brasil, via Kaggle | for the customer's city: population, human development index, GDP per capita, urban/rural type, coordinates → **distance from São Paulo** |

- **Part 1:** an **Airflow** DAG, built from scratch. It runs on a daily schedule, ingests both
  datasets, cleans 7 tables in parallel TaskGroups, joins them, loads the result into
  **PostgreSQL**, then reads it back to make charts and train two models.
- **Part 2:** a **fresh multi-agent architecture** that I designed and built from scratch in
  LangGraph: my own graph, agents, routing and LLM client. It doesn't reuse the class loop's code
  or structure. An evaluator agent **diagnoses** each experiment, and the diagnosis decides which agent the work
  goes back to: feature engineer, ML engineer, or strategist. It can also stop early. A new UI
  page shows the live graph, the route taken each cycle, a learning curve and the final
  evaluation.

---

## Assignment checklist

| Requirement | Where | Evidence |
|---|---|---|
| **1.1** Deploy Airflow, scheduled DAG | `infra/docker-compose.yml` (`--profile airflow`), [`airflow/dags/olist_pipeline.py`](airflow/dags/olist_pipeline.py): `schedule="0 6 * * *"` | [DAG list](docs/images/olist/olist-dag-list.png) |
| Orchestrate ingestion → analysis | 23 tasks: extract → clean → transform → load → quality checks → analysis → report | [Graph view](docs/images/olist/olist-graph.png) |
| Independent tasks in parallel | 8 extracts + 7 cleans run together; 3 analysis tasks run together | [Gantt](docs/images/olist/olist-gantt.png) |
| **1.2** At least 2 related datasets | Olist (7 tables) **+ IBGE Brazilian Cities** (independent source) | [Olist](datasets/olist/ATTRIBUTION.md) · [IBGE](datasets/brazil_cities/ATTRIBUTION.md) |
| Clean + transform each, TaskGroups | 11 TaskGroups: `ingest_and_clean` with one sub-group per table, then `transform`, `load`, `analysis` | [Grid](docs/images/olist/olist-grid.png) |
| Merge + load into a database | 6-way join → `olist.orders_clean` (95,824 × 46) in a dedicated Postgres | [§ Load](#load-into-postgresql) |
| **1.3** Read cleaned data from the DB | every analysis task queries Postgres | `make_charts`, `train_models` |
| Analysis | 4 charts + logistic regression / random forest, logged to `olist.model_runs` | [§ Results](#analysis-results) |
| **1.4** Document process, scheduling, parallelism | this README | [§ Part 1](#part-1--airflow-pipeline) |
| **2** Agent workflow trained/evaluated on Part 1's data | [`olist_loop/`](olist_loop/) reads `olist.orders_clean` | [§ Part 2](#part-2--diagnosis-routed-agent-loop) |
| Add / remove / modify agents | **Fresh architecture designed from scratch:** 9 agents with their own roles and diagnosis-based routing, replacing the class loop rather than modifying it | [§ Agents](#agents) |
| Update the live agent graph | [`olistLoopConfig.ts`](frontend/src/graph/olistLoopConfig.ts), page `/olist-loop` | [Live graph](docs/images/olist/olist-agent-graph-running.png) |
| Report accuracy / appropriate metric | accuracy, precision, recall, **F1**, ROC-AUC, PR-AUC, confusion matrix | [Final evaluation](docs/images/olist/olist-agent-final-eval.png) |
| **Submission** repo + run instructions | this repo | [§ How to run](#how-to-run) |
| Containerized dev setup | Dockerfiles + `docker-compose.yml` + [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) | [§ Containers](#containerized-setup) |
| Screenshots: DAG run, DAG graph, agent graph, evaluation | [`docs/images/olist/`](docs/images/olist/) | [§ Screenshots](#screenshot-index) |

---

## How to run

### Prerequisites
Docker Desktop (running), Git, Make, and Node 20+ for the web UI. Airflow and all Python
dependencies run inside containers. Alternatively, open the repo in the
[dev container](#containerized-setup).

### 0. Clone and configure
```bash
git clone https://github.com/agastyavinchhi/Olist-Airflow-Automl.git && cd Olist-Airflow-Automl
cp .env.example .env
```
Optional: add `OPENROUTER_API_KEY=...` (free at openrouter.ai) so the agents call an LLM.
Without a key, or by choosing **"Heuristics only (no LLM)"** in the UI, every agent uses its
coded fallback. The routing logic is identical either way.

Both datasets are committed in `datasets/`, so no Kaggle download is needed.

### 1. Part 1: Airflow
```bash
make airflow
```
1. Wait ~1–2 minutes, then open **http://localhost:8088** (`admin` / `admin`).
2. Switch **`olist_pipeline`** on.
   - With `catchup=False`, the scheduler immediately runs the most recent 06:00 interval, then runs
     daily at 06:00 UTC.
   - You can also press ▶ to trigger a manual run.
3. A run takes about **50 seconds**.

Check the output:
```bash
docker compose -f infra/docker-compose.yml --env-file .env exec warehouse-db \
  psql -U warehouse -d warehouse -c "SELECT count(*), avg(is_bad_review), avg(city_matched) FROM olist.orders_clean;"
```

### 2. Part 2: agent loop (needs Part 1's table)
```bash
make start-mac        # make start-windows in Git Bash
```
Open **http://127.0.0.1:5173/olist-loop**, keep the defaults (target F1 0.50, 5 cycles, all rows),
and click **Run loop**. A run takes 40 s with heuristics only, or 1–3 minutes with the free LLM.

Command-line alternative:
```bash
docker compose -f infra/docker-compose.yml --env-file .env exec -w / gateway \
  python -m olist_loop.run_cli --max-cycles 5 --model none   # omit --model to use the LLM
```

### Stop
```bash
make down-mac         # or make down-windows
```

| Service | URL |
|---|---|
| Airflow | http://localhost:8088 |
| Agent loop UI | http://127.0.0.1:5173/olist-loop |
| In-class loop (baseline) | http://127.0.0.1:5173/simple-loop |
| Gateway API | http://127.0.0.1:8000/olist-loop/info |
| Warehouse Postgres | `postgresql://warehouse:warehouse@localhost:5433/warehouse` |

---

## Part 1 — Airflow pipeline

### Data sources

| Table | Source | Rows | Join key | Role |
|---|---|---|---|---|
| `orders` | Olist | 99,441 | `order_id`, `customer_id` | fact: status + 5 timestamps |
| `customers` | Olist | 99,441 | `customer_id` → (city, state) | customer location |
| `order_items` | Olist | 112,650 | `order_id`, `product_id` | price, freight, seller |
| `order_payments` | Olist | 103,886 | `order_id` | payment type, installments |
| `order_reviews` | Olist | 99,224 | `order_id` | **label**: review score |
| `products` + `category_translation` | Olist | 32,951 / 71 | `product_id` | category, size, weight |
| **`BRAZIL_CITIES_REV2022`** | **IBGE** | 5,578 cities | (city, state) | population, HDI, GDP per capita, urban/rural type, coordinates |

**Target:** `is_bad_review = review_score <= 2`, which is **12.8%** of delivered orders.

### Transformations

Each table has an `extract` task, which enforces a column contract and stages the data as
parquet, and a `clean` task:

| Table | Issues found while profiling | Cleaning / transformation |
|---|---|---|
| orders | 8 statuses; 2,965 missing delivery dates | keep delivered orders; parse timestamps; derive `days_late`, `is_late`, `delivery_days`, `estimated_days`, `approval_hours`, `carrier_days`, purchase month / weekday / hour |
| customers | free-text cities without accents | dedupe; normalize state; build `city_key` (accents stripped, lower-case) |
| **cities (IBGE)** | accented names; 4 duplicate keys; missing values coded as `0`; 5 unclassified | same `city_key` normalization; keep the larger city on duplicates; `0` → missing; English urban/rural labels; **haversine distance from São Paulo** |
| reviews | 551 orders with > 1 review | latest review per order |
| payments | `not_defined` types; many rows per order | drop invalid; aggregate per order: total, count, installments, voucher flag, main type |
| items | many rows per order | dedupe; drop non-positive prices |
| products | 610 missing categories / sizes | English categories; `unknown`; median-impute sizes; volume |

Then, in `transform`:
- `order_item_features`: items ⨝ products → one row per order.
- `merge_and_engineer`: orders ⨝ customers ⨝ **IBGE cities** ⨝ item features ⨝ payments ⨝ reviews.
  - The 0.6% of customers whose city doesn't match IBGE get their state's median values, and are
    flagged with `city_matched = 0`.
  - Result: **95,824 rows × 46 columns**.

### Pipeline structure

![Airflow graph view with every TaskGroup expanded](docs/images/olist/olist-graph.png)

```
[ingest_and_clean]  (all sub-groups in parallel)                [transform]
  orders    : extract → clean ──────────────────────────────────┐
  customers : extract → clean ──────────────────────────────────┤
  cities    : extract → clean   (IBGE) ─────────────────────────┤
  reviews   : extract → clean ──────────────────────────────────┼──→ merge_and_engineer
  payments  : extract → clean ──────────────────────────────────┤            │
  items     : extract → clean ───────┐                          │            ▼
  products  : extract_products ──┐   ├──→ order_item_features ──┘   [load] load_orders_clean → quality_checks
              extract_translation┴→ clean ┘                                                       │
                                     [analysis] (parallel; each reads Postgres)                   ▼
                                     make_charts │ train_models │ publish_for_agents ──→ report
```

- Written with the TaskFlow API.
- XCom carries only file paths and row counts; the data moves as parquet in a per-run folder.
- Postgres is reached through the Airflow Connection **`warehouse_db`** (`PostgresHook`).
- Each task retries once.

### Scheduling

```python
@dag(schedule="0 6 * * *", start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1, ...)
```
- Runs **daily at 06:00 UTC**.
- **`catchup=False`:** turning the DAG on runs only the latest interval. The first run was
  started by the scheduler on its own (`scheduled__2026-11-11T06:00`).
- **`max_active_runs=1`:** two runs never replace the table at the same time.
- **History:** `olist.model_runs` is append-only, so each scheduled run's metrics are kept for
  comparison.

### Parallel processing

The DAG runs on Airflow's **LocalExecutor** (up to 32 tasks at once, 16 per DAG):

- All **8 extract tasks start together**.
- Each table's `clean` starts as soon as its own extract finishes, so the 7 table pipelines run
  side by side.
- `make_charts`, `train_models` and `publish_for_agents` depend only on `quality_checks`, so they
  also run together.

![Gantt chart: parallel ingest and parallel analysis](docs/images/olist/olist-gantt.png)

A run takes **≈ 50 s**. Most of that is the Postgres load (~20 s) and the random-forest fit
(~20 s, which overlaps with the other analysis tasks).

### Load into PostgreSQL

A dedicated `warehouse-db` service (Postgres 16), separate from Airflow's own metadata database:

| Table | Written by | Contents |
|---|---|---|
| `olist.orders_clean` | `load_orders_clean` (replace + primary key + index) | merged table, 95,824 × 46 |
| `olist.model_runs` | `train_models` (append) | per run, per model metrics |
| `olist.agent_runs` | Part 2 `holdout_eval` (append) | holdout metrics, route path, stop reason |

`quality_checks` fails the run if any of these fail:
- the row count doesn't match the staged data;
- there are fewer than 50k rows;
- the target or any model feature is null;
- the bad-review rate falls outside 2–50%;
- **fewer than 95% of orders matched an IBGE city.**

### Analysis results

Every analysis task **reads from Postgres**.

![Bad-review rate by delivery timing](docs/images/olist/bad_review_rate_by_delivery_timing.png)

**Late delivery is the main driver.** Orders delivered on time or early get a bad review **9.2%**
of the time. Late orders get one **54.1%** of the time, and orders more than a week late about
**78–79%**.

![Bad-review rate by distance from São Paulo](docs/images/olist/bad_review_rate_by_distance.png)

**The IBGE data adds geography.**
- Customers within 100 km of São Paulo leave **11.0%** bad reviews; those 1,000–2,000 km away
  **15.4%**, and 2,000+ km **16.3%**.
- The share of late deliveries roughly doubles with distance: **6% → 13%**.
- So distance raises the risk of a bad review mainly by making late delivery more likely.

![Bad-review rate by customer state](docs/images/olist/bad_review_rate_by_state.png)

**Models** (stratified 80/20 split, scored on the 20% test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority class (always "good") | 0.872 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression (balanced) | 0.809 | 0.347 | 0.559 | 0.428 | 0.755 |
| **Random forest (balanced)** | **0.866** | **0.477** | **0.472** | **0.475** | **0.760** |

- **Accuracy is misleading:** always predicting "good" scores 87.2% accuracy and catches zero bad
  reviews. That's why I use **F1** and **ROC-AUC**.
- Adding the IBGE features lifted the random forest from F1 0.471 / AUC 0.758 to 0.475 / 0.760.
- The top features are `days_late`, `is_late`, `delivery_days`, `n_items` and `carrier_days`.

![Feature importances](docs/images/olist/random_forest_feature_importance.png)

---

## Part 2 — Diagnosis-routed agent loop

### A fresh architecture, not a modified class loop

I designed this agent system from scratch for the bad-review problem. It doesn't import or adapt
the class loop's code, and its graph has a different structure. The in-class loop is kept
unchanged in the repo **only as a comparison baseline** (`/simple-loop`). It always runs the same
cycle: hypothesis → train → judge → *revise* → hypothesis, so every failure gets the same
treatment.

| | In-class loop (baseline) | My architecture |
|---|---|---|
| Loop structure | one fixed revise path back to the hypothesis agent | **3 different loop-back targets** (feature engineer, ML engineer, strategist) + early stop |
| Who decides the route | judge accept / revise | **evaluator diagnosis** (code) → judge picks from the allowed routes |
| Data source | any uploaded CSV | **Postgres table written by my Airflow DAG**, plus the Airflow models' scores as baselines |
| Agents | data scientist, hypothesis generator, ML engineer, judge, memory writer, context builder, holdout | **warehouse loader, data scientist, strategist, feature engineer, ML engineer, evaluator, judge, research memory, holdout** |
| Features | none (raw columns) | **feature engineer** with a fixed recipe menu, including IBGE city features |
| Imbalance handling | class weights only | class weights + **F1-tuned decision threshold** |
| Stopping | target met or budget spent | target met, budget spent, or **plateau early-stop** |
| LLM client | shared template helper | **my own client** with JSON extraction, retries, a quota circuit breaker and a heuristics-only mode |
| Output | in-memory result | holdout metrics + route path **logged to Postgres** (`olist.agent_runs`) |

My problem gives a clearer signal about **why** a model falls short:
- If **ROC-AUC** isn't better than the baseline, the model can't separate good from bad orders. It
  needs **better inputs**.
- If ROC-AUC is good but **F1** is low, the ranking is fine. The **decision cutoff or model
  capacity** is the issue.

So my evaluator diagnoses each result, and the router sends the work to the agent that can fix it.

```
olist.orders_clean ─SQL→ warehouse_loader → data_scientist → strategist → feature_engineer → ml_engineer → evaluator
                                                               ↑                 ↑                 ↑            │ diagnosis
                                                               │ rethink         │ features   model│            ▼
                                                               └──────── research_memory ◄─ revise ─ router ◄─ judge
                                                                                                      │ finish (target · plateau · budget)
                                                                                                      ▼
                                                                         best_candidate → holdout_eval ─INSERT→ olist.agent_runs
```

### Routing rules

The evaluator computes them in code. The judge (an LLM) picks a route, but **only from the allowed
set**.

| Condition after a cycle | Diagnosis | Route → agent |
|---|---|---|
| validation F1 ≥ target | target met | **finish** → holdout |
| cycle budget spent | — | **finish** |
| 2 cycles without +0.002 F1 | plateau | **finish** (early stop) |
| ROC-AUC < Airflow RF AUC + 0.005 | weak ranking | **features** → feature engineer (keep the model, add inputs) |
| ROC-AUC good, F1 below target | cutoff or capacity | **model** → ML engineer (keep the features; tune the threshold, then capacity, then family) |
| the same fix didn't improve twice in a row | stuck | **rethink** → strategist (whole new plan: new family + feature set) |

### Agents

All 9 agents were written from scratch for this architecture.

| Agent | Type | Responsibility |
|---|---|---|
| `warehouse_loader` | code | Reads `olist.orders_clean` from Postgres plus the latest Airflow metrics (used as baselines); falls back to the CSV the DAG publishes. Sets a 60/20/20 train/validation/**sealed holdout** split. |
| `data_scientist` | LLM | Profiles risks: imbalance, skew, leakage, IBGE coverage, which city features carry signal. |
| `strategist` | LLM | Writes the initial plan, and a **completely new plan** on `rethink` (it's never allowed to repeat a tried family + feature combination). |
| `feature_engineer` | LLM + code | Picks recipes from a **fixed menu of 9**, including IBGE ones (`remote_city_x_late`, `distance_bands`). Code applies them, so the LLM can't corrupt data or leak the label. |
| `ml_engineer` | LLM + code | Model family, hyperparameters (clamped to safe ranges) and threshold strategy. On a `model` route it may only change the model side. If training fails, it retries once with safe defaults. |
| `evaluator` | code | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, F1-tuned threshold → **diagnosis + allowed routes + recommendation**. |
| `judge` | LLM | Chooses the route from the allowed set, with a reason. Invalid answers fall back to the recommendation. |
| `research_memory` | code | Logs the cycle (family, recipes, scores, diagnosis, route) and **dispatches** to the next agent. |
| `holdout_eval` | code | Refits the best configuration on train + validation, scores the **sealed holdout once**, and writes metrics + route path + stop reason to `olist.agent_runs`. |

**Reliability features:**
- Own OpenRouter client ([`olist_loop/llm.py`](olist_loop/llm.py)) with JSON extraction and retries
  on rate limits.
- A circuit breaker that stops calling the LLM once the daily free quota is used up.
- A heuristics-only mode.
- The agents always finish a run, even if the LLM is unavailable.

### Live agent graph and UI

The page at **`/olist-loop`** ([`OlistLoopPage.tsx`](frontend/src/pages/OlistLoopPage.tsx)) shows:
- the 13-node graph, which highlights the running agent and animates the active route;
- a **learning curve**: validation F1 per cycle, labeled with the route taken, against the target
  and Airflow's F1;
- an **experiments table** with each cycle's diagnosis and route;
- the **final evaluation** against the Part 1 baselines, with a confusion matrix.

![Live graph mid-run: research memory dispatching back to the ML engineer on a "model" route](docs/images/olist/olist-agent-graph-running.png)

![Learning curve with routes](docs/images/olist/olist-agent-learning-curve.png)

![Experiments: diagnosis and route per cycle](docs/images/olist/olist-agent-experiments.png)

### Evaluation results

These screenshots and numbers come from a **heuristics-only run** (`--model none`), because the
free OpenRouter quota (50 LLM requests per day) was used up during development. Every agent then
uses its coded fallback; the diagnosis and routing logic are exactly the same as with an LLM.
All runs are logged in `olist.agent_runs`.

| Cycle | What the agents did | Val F1 | Val AUC | Diagnosis → route |
|---|---|---|---|---|
| 1 | logistic regression + lateness buckets | 0.440 | 0.760 | weak ranking → **features** |
| 2 | + delivery-speed ratio, remote-city × late (IBGE) | 0.441 | 0.760 | no gain from the same fix → **rethink** |
| 3 | strategist re-plans: gradient boosting, 4 recipes, tuned cutoff | 0.495 | 0.769 | ranking OK → **model** |
| 4 | ML engineer: more, smaller boosting steps (500 iterations, lower learning rate) | **0.499** | 0.768 | ranking OK → **model** |
| 5 | ML engineer: same again (700 iterations) | 0.496 | 0.769 | budget spent → **finish** |

![Final evaluation on the sealed holdout vs Part 1 baselines](docs/images/olist/olist-agent-final-eval.png)

**Sealed holdout** (19,165 orders; best configuration = cycle 4):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority class | 0.872 | 0.000 | 0.000 | 0.000 | 0.500 |
| Airflow logistic regression (Part 1) | 0.809 | 0.347 | 0.559 | 0.428 | 0.755 |
| Airflow random forest (Part 1) | 0.866 | 0.477 | 0.472 | **0.475** | 0.760 |
| **Agent loop: gradient boosting** | **0.879** | **0.535** | 0.421 | 0.471 | **0.772** |

Confusion matrix on the holdout: **1,034** bad reviews caught, 1,420 missed, 900 false alarms out
of 16,711 good orders.

**How to read this for the support use case:** among orders the model flags, **53.5% really do get
a bad review**, versus 12.8% for an order picked at random. That's about a 4× better hit rate for
the support team's outreach.

The Airflow rows come from Part 1's own 20% split; the agent row from the loop's sealed 20%. Both
are stratified and the same size, but they aren't the same rows.

---

## Key findings

1. **Delivery performance explains bad reviews.** On-time orders: 9.2% bad reviews; late: 54.1%;
   more than a week late: about 78%.
2. **Geography (IBGE) matters through delivery.** Customers more than 1,000 km from São Paulo leave
   15–16% bad reviews versus 11% nearby, and their orders are late twice as often.
3. **Accuracy is the wrong metric.** "Always good" scores 87% accuracy. F1 and ROC-AUC separate
   the models.
4. **The diagnosis routing found the improvements.**
   - Adding features didn't help the linear model (cycles 1–2), so the router triggered a
     **rethink**.
   - The re-plan to gradient boosting with a tuned cutoff jumped validation F1 from 0.441 to 0.495.
   - From there, **model**-route tuning added a little more.
   - The final model has the best ROC-AUC (0.772) and precision (0.535) of all models, while F1
     ties the Part 1 random forest (0.471 vs 0.475).
5. **There's a ceiling around F1 ≈ 0.5.** The data explains delivery-driven dissatisfaction well,
   but product quality and buyer expectations aren't observed. The plateau rule stops the loop
   when it stops improving instead of wasting cycles.

---

## Containerized setup

| Service | Image | Purpose |
|---|---|---|
| `airflow-webserver` / `-scheduler` / `-init` | [`infra/Dockerfile.airflow`](infra/Dockerfile.airflow): Airflow 2.10.4 + pandas < 2.2, scikit-learn, matplotlib | Part 1 (`--profile airflow`) |
| `airflow-db` | postgres:16-alpine | Airflow metadata only |
| `warehouse-db` | postgres:16-alpine, host port 5433 | `olist.*` tables, shared by both parts |
| `gateway` | [`infra/Dockerfile.service`](infra/Dockerfile.service) + LangGraph, SQLAlchemy, psycopg2 | runs the agent loop, streams events to the UI |
| other services | `infra/Dockerfile.service` | template backend services the gateway depends on |

**Dev container:** [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) provides
Python 3.12, Node 20, the Docker CLI (using the host's Docker daemon), `make` and `psql`. It
forwards ports 8088 / 8000 / 5173 / 5433 and creates `.env`. Open the repo in VS Code →
**Reopen in Container** → `make airflow`, then `make start-mac`.

---

## What I built vs. what came from the class template

| Mine (new for this assignment) | Reused from the class template |
|---|---|
| [`airflow/dags/olist_pipeline.py`](airflow/dags/olist_pipeline.py): the whole pipeline | Docker Compose stack, Airflow image base, Makefile targets |
| [`datasets/olist/`](datasets/olist/), [`datasets/brazil_cities/`](datasets/brazil_cities/): both datasets + profiling notes | React Flow canvas component (I made it accept any graph config) |
| [`olist_loop/`](olist_loop/): **fresh agent architecture built from scratch**: graph, 9 agents, diagnosis routing, own LLM client, CLI | gateway service skeleton (I added `/olist-loop/*` endpoints) |
| `warehouse-db`, the `warehouse_db` Airflow connection, quality checks | `mini_loop/` + `/simple-loop`: the in-class loop, **kept unchanged as the comparison baseline** |
| [`frontend/src/pages/OlistLoopPage.tsx`](frontend/src/pages/OlistLoopPage.tsx), [`olistLoopConfig.ts`](frontend/src/graph/olistLoopConfig.ts) | template backend services the stack starts |

The class's own exercises (student DAGs, fraud and Titanic datasets, student loop, guide pages,
design docs) were removed from this repo.

## Screenshot index

| Requirement | File |
|---|---|
| DAG execution (runs + task states) | [olist-grid.png](docs/images/olist/olist-grid.png) |
| Schedule | [olist-dag-list.png](docs/images/olist/olist-dag-list.png) |
| DAG graph | [olist-graph.png](docs/images/olist/olist-graph.png) |
| Parallel timeline | [olist-gantt.png](docs/images/olist/olist-gantt.png) |
| Analysis charts | [timing](docs/images/olist/bad_review_rate_by_delivery_timing.png) · [distance (IBGE)](docs/images/olist/bad_review_rate_by_distance.png) · [state](docs/images/olist/bad_review_rate_by_state.png) · [importances](docs/images/olist/random_forest_feature_importance.png) |
| Agent graph | [running](docs/images/olist/olist-agent-graph-running.png) · [finished](docs/images/olist/olist-agent-graph-done.png) |
| Model evaluation | [final evaluation](docs/images/olist/olist-agent-final-eval.png) · [learning curve](docs/images/olist/olist-agent-learning-curve.png) · [experiments](docs/images/olist/olist-agent-experiments.png) · [full page](docs/images/olist/olist-agent-results-page.png) |

## Credits
- Class template: [Kedar-V/automl-research-lab](https://github.com/Kedar-V/automl-research-lab)
  (infrastructure and UI canvas; see the table above).
- Data: Olist, *Brazilian E-Commerce Public Dataset* (Kaggle, CC BY-NC-SA 4.0); Cristiana Parada,
  *Brazilian Cities* (Kaggle), compiled from IBGE and Atlas Brasil statistics.
