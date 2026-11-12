/** Live graph for the diagnosis-routed Olist loop (olist_loop/graph.py). */
import type { LoopGraphConfig } from '../components/LiveSimpleLoopGraph'

export const OLIST_LOOP_CONFIG: LoopGraphConfig = {
  subtitle: 'Olist + IBGE · evaluator diagnosis picks the route · status from SSE',
  nodes: [
    { id: 'warehouse', title: 'olist.orders_clean', role: 'Olist + IBGE cities (Airflow)', kind: 'db', service: 'warehouse-db' },
    { id: 'warehouse_loader', title: 'Warehouse loader', role: 'Read table + Airflow baselines', kind: 'code', service: 'agent' },
    { id: 'data_scientist', title: 'Data Scientist', role: 'Imbalance · skew · city coverage', kind: 'llm', service: 'agent' },
    { id: 'strategist', title: 'Strategist', role: 'Initial plan · re-plan on rethink', kind: 'llm', service: 'agent' },
    { id: 'feature_engineer', title: 'Feature Engineer', role: 'Recipes from fixed menu', kind: 'code_llm', service: 'agent' },
    { id: 'ml_engineer', title: 'ML Engineer', role: 'Family · HPT · threshold · train', kind: 'code_llm', service: 'agent' },
    { id: 'evaluator', title: 'Evaluator', role: 'Metrics → diagnosis → route', kind: 'code', service: 'agent' },
    { id: 'judge', title: 'Judge', role: 'Pick route from allowed set', kind: 'llm', service: 'agent' },
    { id: 'router', title: 'Router', role: 'features · model · rethink · finish', kind: 'code', service: 'conditional edge' },
    { id: 'research_memory', title: 'Research memory', role: 'Log cycle · dispatch route', kind: 'store', service: 'conditional edge' },
    { id: 'best_candidate', title: 'Best candidate', role: 'Best val F1 config', kind: 'store', service: 'state' },
    { id: 'holdout_eval', title: 'Final holdout', role: 'Score sealed 20% once', kind: 'code', service: 'agent' },
    { id: 'agent_runs', title: 'olist.agent_runs', role: 'Holdout metrics + route log', kind: 'db', service: 'warehouse-db' },
  ],
  edges: [
    { id: 'e-wh-load', source: 'warehouse', target: 'warehouse_loader', kind: 'data', label: 'SQL', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-load-ds', source: 'warehouse_loader', target: 'data_scientist', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-ds-st', source: 'data_scientist', target: 'strategist', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-st-fe', source: 'strategist', target: 'feature_engineer', kind: 'control', label: 'plan', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-fe-ml', source: 'feature_engineer', target: 'ml_engineer', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-ml-ev', source: 'ml_engineer', target: 'evaluator', kind: 'control', label: 'val proba', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-ev-judge', source: 'evaluator', target: 'judge', kind: 'control', label: 'diagnosis', sourceHandle: 'bottom-out', targetHandle: 'top-in' },
    { id: 'e-judge-router', source: 'judge', target: 'router', kind: 'control', sourceHandle: 'left-out', targetHandle: 'right-in' },
    { id: 'e-router-mem', source: 'router', target: 'research_memory', kind: 'control', label: 'revise', sourceHandle: 'left-out', targetHandle: 'right-in' },
    { id: 'e-mem-fe', source: 'research_memory', target: 'feature_engineer', kind: 'control', label: 'features', sourceHandle: 'top-out', targetHandle: 'bottom-in' },
    { id: 'e-mem-ml', source: 'research_memory', target: 'ml_engineer', kind: 'control', label: 'model', sourceHandle: 'top-out', targetHandle: 'bottom-in' },
    { id: 'e-mem-st', source: 'research_memory', target: 'strategist', kind: 'control', label: 'rethink', sourceHandle: 'left-out', targetHandle: 'bottom-in' },
    { id: 'e-router-best', source: 'router', target: 'best_candidate', kind: 'control', label: 'finish', sourceHandle: 'bottom-out', targetHandle: 'top-in' },
    { id: 'e-best-hold', source: 'best_candidate', target: 'holdout_eval', kind: 'control', sourceHandle: 'right-out', targetHandle: 'left-in' },
    { id: 'e-hold-db', source: 'holdout_eval', target: 'agent_runs', kind: 'data', label: 'INSERT', sourceHandle: 'right-out', targetHandle: 'left-in' },
  ],
  /*
   * top:    warehouse · loader · ds · strategist · features · ml · evaluator
   * mid:                                 memory  ·  router  · judge
   * bottom:                                         best    · holdout · agent_runs
   */
  positions: {
    warehouse: { x: 0, y: 40 },
    warehouse_loader: { x: 260, y: 40 },
    data_scientist: { x: 520, y: 40 },
    strategist: { x: 780, y: 40 },
    feature_engineer: { x: 1040, y: 40 },
    ml_engineer: { x: 1300, y: 40 },
    evaluator: { x: 1560, y: 40 },
    research_memory: { x: 1040, y: 300 },
    router: { x: 1300, y: 300 },
    judge: { x: 1560, y: 300 },
    best_candidate: { x: 1300, y: 540 },
    holdout_eval: { x: 1560, y: 540 },
    agent_runs: { x: 1820, y: 540 },
  },
}

/** Route → the agent research_memory dispatches to. */
export const ROUTE_TARGET: Record<string, string> = {
  features: 'feature_engineer',
  model: 'ml_engineer',
  rethink: 'strategist',
}
