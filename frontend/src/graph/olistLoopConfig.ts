/** Live graph for the Olist bad-review loop (olist_loop/graph.py). */
import type { LoopGraphConfig } from '../components/LiveSimpleLoopGraph'

export const OLIST_LOOP_CONFIG: LoopGraphConfig = {
  subtitle: 'Olist loop · olist.orders_clean → agents → holdout · status from SSE',
  nodes: [
    { id: 'warehouse', title: 'olist.orders_clean', role: 'Postgres table from Airflow', kind: 'db', service: 'warehouse-db' },
    { id: 'warehouse_loader', title: 'Warehouse loader', role: 'NEW · read table + Airflow baselines', kind: 'code', service: 'node script' },
    { id: 'data_scientist', title: 'Data Scientist', role: 'Imbalance · skew · leakage', kind: 'llm', service: 'node script' },
    { id: 'hypothesis_generator', title: 'Hypothesis Generator', role: 'Features + family + imbalance plan', kind: 'llm', service: 'node script' },
    { id: 'feature_engineer', title: 'Feature Engineer', role: 'NEW · recipes from fixed menu', kind: 'code_llm', service: 'node script' },
    { id: 'ml_engineer', title: 'ML Engineer', role: 'HPT + threshold strategy · train', kind: 'code_llm', service: 'node script' },
    { id: 'evaluator', title: 'Evaluator', role: 'NEW · F1 · AUC · vs baselines', kind: 'code', service: 'node script' },
    { id: 'judge', title: 'Judge', role: 'Accept / revise from report', kind: 'llm', service: 'node script' },
    { id: 'router', title: 'Router', role: 'accept · revise · stop', kind: 'code', service: 'langgraph edge' },
    { id: 'research_memory', title: 'Research memory', role: 'MERGED · lesson + next brief', kind: 'store', service: 'node script' },
    { id: 'best_candidate', title: 'Best candidate', role: 'Best val F1 config', kind: 'store', service: 'state' },
    { id: 'holdout_eval', title: 'Final holdout', role: 'Refit · score sealed 20% once', kind: 'code', service: 'node script' },
    { id: 'agent_runs', title: 'olist.agent_runs', role: 'Holdout metrics logged', kind: 'db', service: 'warehouse-db' },
  ],
  edges: [
    { id: 'e-wh-load', source: 'warehouse', target: 'warehouse_loader', kind: 'data', label: 'SQL', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-load-ds', source: 'warehouse_loader', target: 'data_scientist', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-ds-hyp', source: 'data_scientist', target: 'hypothesis_generator', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-hyp-fe', source: 'hypothesis_generator', target: 'feature_engineer', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-fe-ml', source: 'feature_engineer', target: 'ml_engineer', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-ml-eval', source: 'ml_engineer', target: 'evaluator', kind: 'control', label: 'val proba', sourceHandle: 'bottom-out', targetHandle: 'top-in' },
    { id: 'e-eval-judge', source: 'evaluator', target: 'judge', kind: 'control', label: 'report', sourceHandle: 'left-out', targetHandle: 'right-in' },
    { id: 'e-judge-router', source: 'judge', target: 'router', kind: 'control', sourceHandle: 'left-out', targetHandle: 'right-in' },
    { id: 'e-router-mem', source: 'router', target: 'research_memory', kind: 'control', label: 'revise', sourceHandle: 'left-out', targetHandle: 'right-in' },
    { id: 'e-mem-hyp', source: 'research_memory', target: 'hypothesis_generator', kind: 'data', label: 'brief', sourceHandle: 'top-out', targetHandle: 'bottom-in' },
    { id: 'e-router-best', source: 'router', target: 'best_candidate', kind: 'control', label: 'accept / out of budget', sourceHandle: 'bottom-out', targetHandle: 'top-in' },
    { id: 'e-best-hold', source: 'best_candidate', target: 'holdout_eval', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-hold-db', source: 'holdout_eval', target: 'agent_runs', kind: 'data', label: 'INSERT', sourceHandle: 'right-out', targetHandle: 'left-in' },
  ],
  /*
   * top:    warehouse · loader · ds · hyp · fe · ml
   * mid:                   memory ·  router · judge · evaluator
   * bottom:                          best · holdout · agent_runs
   */
  positions: {
    warehouse: { x: 0, y: 40 },
    warehouse_loader: { x: 260, y: 40 },
    data_scientist: { x: 520, y: 40 },
    hypothesis_generator: { x: 780, y: 40 },
    feature_engineer: { x: 1040, y: 40 },
    ml_engineer: { x: 1300, y: 40 },
    research_memory: { x: 520, y: 260 },
    router: { x: 780, y: 260 },
    judge: { x: 1040, y: 260 },
    evaluator: { x: 1300, y: 260 },
    best_candidate: { x: 780, y: 480 },
    holdout_eval: { x: 1040, y: 480 },
    agent_runs: { x: 1300, y: 480 },
  },
}

export const OLIST_AGENT_NODES = [
  'warehouse_loader',
  'data_scientist',
  'hypothesis_generator',
  'feature_engineer',
  'ml_engineer',
  'evaluator',
  'judge',
  'research_memory',
  'holdout_eval',
] as const
