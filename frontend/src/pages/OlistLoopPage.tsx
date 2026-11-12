import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  listOlistLoopRuns,
  listSimpleLoopModels,
  type OlistLoopRunRecord,
  type SimpleLoopModel,
} from '../api/gateway'
import {
  LiveSimpleLoopGraph,
  type SimpleLoopRuntime,
  type SimpleNodeStatus,
} from '../components/LiveSimpleLoopGraph'
import { OLIST_LOOP_CONFIG } from '../graph/olistLoopConfig'

// ---------------------------------------------------------------------------
// Types mirrored from olist_loop/graph.py step payloads
// ---------------------------------------------------------------------------
type Metrics = {
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  pr_auc?: number
  threshold?: number
  confusion?: { tn: number; fp: number; fn: number; tp: number }
}
type Baselines = Record<string, { accuracy: number; precision?: number; recall?: number; f1: number; roc_auc: number }>
type Holdout = Metrics & {
  n_holdout?: number
  majority_baseline_accuracy?: number
  model_family?: string
  recipes?: string[]
  hyperparameters?: Record<string, unknown>
  airflow_baselines?: Baselines | null
}
type Step = {
  node?: string
  note?: string
  used_llm?: boolean
  output?: Record<string, unknown>
  verdict?: { action?: string; observation?: string; next_action?: string }
  metrics?: Holdout
  spec?: { model_family?: string; hyperparameters?: Record<string, unknown>; threshold_strategy?: string }
}
type StreamEvent = {
  type: string
  node?: string
  note?: string | null
  action?: string
  route?: string
  message?: string
  step?: Step
  cycle?: number
  best_val_f1?: number
  result?: { steps?: Step[]; holdout_metrics?: Holdout; best_val_f1?: number; cycles?: number }
}
type ExperimentRow = {
  cycle: number
  family?: string
  recipes?: string[]
  hp?: Record<string, unknown>
  validation?: Metrics
  f1At05?: number
  verdict?: string
  observation?: string
}
type LogLine = { t: string; node: string; note: string; llm?: boolean }

const ROWS_OPTIONS = [
  { value: 0, label: 'All rows (~96k)' },
  { value: 30000, label: '30k sample' },
  { value: 10000, label: '10k sample' },
]

const fmt = (v?: number | null, d = 3) => (typeof v === 'number' ? v.toFixed(d) : '—')
const now = () => new Date().toLocaleTimeString()

function setStatus(prev: SimpleLoopRuntime, id: string, status: SimpleNodeStatus, note?: string) {
  return { ...prev, [id]: { status, note: note ?? prev[id]?.note } }
}

/** Rebuild the experiments table from a finished run's step list (history replay). */
function rowsFromSteps(steps: Step[] = []): ExperimentRow[] {
  const rows: ExperimentRow[] = []
  let cycle = 0
  for (const s of steps) {
    if (s.node === 'ml_engineer') {
      cycle += 1
      rows.push({ cycle, family: s.spec?.model_family, hp: s.spec?.hyperparameters })
    } else if (s.node === 'evaluator' && rows.length) {
      const o = (s.output ?? {}) as { validation?: Metrics; recipes?: string[]; 'f1_at_0.5'?: number }
      Object.assign(rows[rows.length - 1], {
        validation: o.validation,
        recipes: o.recipes,
        f1At05: o['f1_at_0.5'],
      })
    } else if (s.node === 'judge' && rows.length) {
      Object.assign(rows[rows.length - 1], {
        verdict: s.verdict?.action,
        observation: s.verdict?.observation,
      })
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------
function Panel({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="border border-line bg-panel/60">
      <div className="border-b border-line px-4 py-2">
        <h2 className="font-display text-lg text-paper">{title}</h2>
        {sub && <p className="text-xs text-mist">{sub}</p>}
      </div>
      <div className="p-4">{children}</div>
    </section>
  )
}

function ExperimentsTable({ rows, target, airflowF1 }: { rows: ExperimentRow[]; target: number; airflowF1?: number }) {
  if (rows.length === 0) return <p className="text-sm text-mist">No experiments yet — run the loop.</p>
  const best = Math.max(...rows.map((r) => r.validation?.f1 ?? -1))
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="font-mono uppercase tracking-wide text-mist">
          <tr className="border-b border-line">
            <th className="py-1 pr-2">#</th>
            <th className="pr-2">Model · features</th>
            <th className="pr-2 text-right">Val F1</th>
            <th className="pr-2 text-right">Acc</th>
            <th className="pr-2 text-right">AUC</th>
            <th className="pr-2 text-right">Thr</th>
            <th>Judge</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const f1 = r.validation?.f1
            const isBest = f1 != null && f1 === best
            return (
              <tr key={r.cycle} className="border-b border-line/60 align-top">
                <td className="py-1.5 pr-2 font-mono text-mist">{r.cycle}</td>
                <td className="pr-2">
                  <p className="text-paper">{r.family?.replace(/_/g, ' ') ?? '…'}</p>
                  <p className="text-[10px] text-mist">{(r.recipes ?? []).join(', ') || 'no recipes'}</p>
                </td>
                <td className={`pr-2 text-right font-mono ${isBest ? 'font-semibold text-teal' : 'text-paper'}`}>
                  {fmt(f1)}
                  {typeof r.f1At05 === 'number' && r.validation?.threshold !== 0.5 && (
                    <span className="block text-[10px] font-normal text-mist">@0.5: {fmt(r.f1At05)}</span>
                  )}
                </td>
                <td className="pr-2 text-right font-mono text-paper">{fmt(r.validation?.accuracy)}</td>
                <td className="pr-2 text-right font-mono text-paper">{fmt(r.validation?.roc_auc)}</td>
                <td className="pr-2 text-right font-mono text-mist">{fmt(r.validation?.threshold, 2)}</td>
                <td>
                  <span
                    className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase ${
                      r.verdict === 'accept' ? 'border-ok/40 text-ok' : 'border-line text-mist'
                    }`}
                  >
                    {r.verdict ?? '…'}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-mist">
        Target val F1 {target.toFixed(2)}
        {airflowF1 != null ? ` · Airflow random forest F1 ${airflowF1.toFixed(3)}` : ''} · thresholds tuned on
        validation when the ML Engineer picks <span className="font-mono">tune_f1</span>
      </p>
    </div>
  )
}

function FinalEvaluation({ h }: { h: Holdout | null }) {
  if (!h) return <p className="text-sm text-mist">Appears after the Final holdout agent runs.</p>
  const af = h.airflow_baselines ?? {}
  const rows: { name: string; m: Partial<Metrics>; note?: string; strong?: boolean }[] = [
    { name: 'Majority class (always “good”)', m: { accuracy: h.majority_baseline_accuracy, precision: 0, recall: 0, f1: 0, roc_auc: 0.5 } },
    ...(af.logistic_regression ? [{ name: 'Airflow · logistic regression', m: af.logistic_regression, note: 'Part 1' }] : []),
    ...(af.random_forest ? [{ name: 'Airflow · random forest', m: af.random_forest, note: 'Part 1' }] : []),
    { name: `Agent loop · ${h.model_family?.replace(/_/g, ' ')}`, m: h, strong: true },
  ]
  const c = h.confusion
  return (
    <div className="space-y-4">
      <table className="w-full text-left text-xs">
        <thead className="font-mono uppercase tracking-wide text-mist">
          <tr className="border-b border-line">
            <th className="py-1">Model</th>
            <th className="text-right">Accuracy</th>
            <th className="text-right">Precision</th>
            <th className="text-right">Recall</th>
            <th className="text-right">F1</th>
            <th className="text-right">ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className={`border-b border-line/60 ${r.strong ? 'text-teal' : 'text-paper'}`}>
              <td className={`py-1.5 ${r.strong ? 'font-semibold' : ''}`}>
                {r.name}
                {r.note && <span className="ml-1 text-[10px] text-mist">({r.note})</span>}
              </td>
              <td className="text-right font-mono">{fmt(r.m.accuracy)}</td>
              <td className="text-right font-mono">{fmt(r.m.precision)}</td>
              <td className="text-right font-mono">{fmt(r.m.recall)}</td>
              <td className="text-right font-mono">{fmt(r.m.f1)}</td>
              <td className="text-right font-mono">{fmt(r.m.roc_auc)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
        {c && (
          <div>
            <p className="mb-1 font-mono text-[10px] uppercase tracking-wide text-mist">Confusion (holdout)</p>
            <table className="font-mono text-xs">
              <thead>
                <tr className="text-mist">
                  <th />
                  <th className="px-2">pred good</th>
                  <th className="px-2">pred bad</th>
                </tr>
              </thead>
              <tbody className="text-paper">
                <tr>
                  <td className="pr-2 text-mist">actual good</td>
                  <td className="border border-line px-2 py-1 text-right">{c.tn.toLocaleString()}</td>
                  <td className="border border-line px-2 py-1 text-right">{c.fp.toLocaleString()}</td>
                </tr>
                <tr>
                  <td className="pr-2 text-mist">actual bad</td>
                  <td className="border border-line px-2 py-1 text-right">{c.fn.toLocaleString()}</td>
                  <td className="border border-line px-2 py-1 text-right text-teal">{c.tp.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
        <div className="text-xs text-mist">
          <p>
            Sealed holdout: <span className="font-mono text-paper">{h.n_holdout?.toLocaleString()}</span> orders ·
            threshold <span className="font-mono text-paper">{fmt(h.threshold, 2)}</span> · PR-AUC{' '}
            <span className="font-mono text-paper">{fmt(h.pr_auc)}</span>
          </p>
          <p className="mt-1">Features: {(h.recipes ?? []).join(', ') || 'base columns only'}</p>
          <p className="mt-1 font-mono text-[10px]">{JSON.stringify(h.hyperparameters)}</p>
          <p className="mt-2">
            Airflow rows come from <span className="font-mono">olist.model_runs</span> (its own 20% test split);
            the agent row is scored once on the loop’s sealed 20% holdout and written to{' '}
            <span className="font-mono">olist.agent_runs</span>.
          </p>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------
export function OlistLoopPage() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [models, setModels] = useState<SimpleLoopModel[]>([])
  const [modelId, setModelId] = useState('')
  const [targetScore, setTargetScore] = useState(0.5)
  const [maxCycles, setMaxCycles] = useState(3)
  const [maxRows, setMaxRows] = useState(0)
  const [runtime, setRuntime] = useState<SimpleLoopRuntime>({})
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null)
  const [rows, setRows] = useState<ExperimentRow[]>([])
  const [holdout, setHoldout] = useState<Holdout | null>(null)
  const [airflowF1, setAirflowF1] = useState<number | undefined>()
  const [log, setLog] = useState<LogLine[]>([])
  const [cycle, setCycle] = useState(0)
  const [history, setHistory] = useState<OlistLoopRunRecord[]>([])
  const sourceRef = useRef<EventSource | null>(null)
  const runningRef = useRef(false)

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await listOlistLoopRuns())
    } catch {
      /* non-fatal */
    }
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        const cat = await listSimpleLoopModels()
        // Free OpenRouter models + Cursor SDK only — no accidental paid calls.
        const list = (cat.models ?? []).filter((m) => m.free || m.provider === 'cursor')
        setModels(list)
        setModelId(cat.default || list[0]?.id || '')
      } catch {
        /* gateway default model is used */
      }
    })()
    void refreshHistory()
    return () => sourceRef.current?.close()
  }, [refreshHistory])

  const reset = useCallback(() => {
    setRuntime({})
    setActiveNodeId(null)
    setRows([])
    setHoldout(null)
    setLog([])
    setCycle(0)
    setError(null)
  }, [])

  const openHistory = useCallback((run: OlistLoopRunRecord) => {
    sourceRef.current?.close()
    const res = (run.result ?? {}) as StreamEvent['result'] & { data?: { baselines?: Baselines } }
    const steps = res?.steps ?? []
    setRows(rowsFromSteps(steps))
    setHoldout(res?.holdout_metrics ?? null)
    setAirflowF1(res?.data?.baselines?.random_forest?.f1)
    setCycle(res?.cycles ?? 0)
    setError(run.error ?? null)
    setLog(steps.map((s) => ({ t: '', node: s.node ?? '?', note: s.note ?? '', llm: s.used_llm })))
    const rt: SimpleLoopRuntime = {}
    for (const n of OLIST_LOOP_CONFIG.nodes) rt[n.id] = { status: 'completed' }
    for (const s of steps) if (s.node) rt[s.node] = { status: 'completed', note: s.note }
    setRuntime(rt)
    setActiveNodeId(null)
  }, [])

  const onRun = useCallback(() => {
    sourceRef.current?.close()
    reset()
    setBusy(true)
    runningRef.current = true
    const params = new URLSearchParams({
      target_score: String(targetScore),
      max_cycles: String(maxCycles),
      max_rows: String(maxRows),
    })
    if (modelId) params.set('model', modelId)
    const source = new EventSource(`/api/olist-loop/stream?${params}`)
    sourceRef.current = source

    const finish = () => {
      runningRef.current = false
      setBusy(false)
      setActiveNodeId(null)
      source.close()
    }

    const handle = (msg: MessageEvent<string>) => {
      let ev: StreamEvent
      try {
        ev = JSON.parse(msg.data) as StreamEvent
      } catch {
        return
      }
      if (ev.type === 'run_started') {
        setRuntime((p) => setStatus(p, 'warehouse', 'completed', 'olist.orders_clean'))
        return
      }
      if (ev.type === 'node_started' && ev.node) {
        const node = ev.node
        setActiveNodeId(node)
        setRuntime((p) => setStatus(p, node, 'running'))
        if (node === 'holdout_eval') setRuntime((p) => setStatus(p, 'best_candidate', 'completed', 'best val F1 locked'))
        return
      }
      if (ev.type === 'router') {
        setRuntime((p) => setStatus(p, 'router', 'completed', `route → ${ev.route} (${ev.action})`))
        setLog((l) => [...l, { t: now(), node: 'router', note: `→ ${ev.route} (${ev.action})` }])
        return
      }
      if (ev.type === 'node_completed' && ev.node) {
        const node = ev.node
        const step = ev.step ?? {}
        setRuntime((p) => setStatus(p, node, 'completed', ev.note ?? undefined))
        setLog((l) => [...l, { t: now(), node, note: ev.note ?? '', llm: step.used_llm }])
        if (typeof ev.cycle === 'number') setCycle(ev.cycle)
        if (node === 'warehouse_loader') {
          const b = (step.output as { baselines?: Baselines } | undefined)?.baselines
          setAirflowF1(b?.random_forest?.f1)
        }
        if (node === 'ml_engineer') {
          setRows((r) => [
            ...r,
            { cycle: (ev.cycle as number) ?? r.length + 1, family: step.spec?.model_family, hp: step.spec?.hyperparameters },
          ])
        }
        if (node === 'evaluator') {
          const o = (step.output ?? {}) as { validation?: Metrics; recipes?: string[]; 'f1_at_0.5'?: number }
          setRows((r) =>
            r.map((row, i) =>
              i === r.length - 1 ? { ...row, validation: o.validation, recipes: o.recipes, f1At05: o['f1_at_0.5'] } : row,
            ),
          )
        }
        if (node === 'judge') {
          setRows((r) =>
            r.map((row, i) =>
              i === r.length - 1 ? { ...row, verdict: step.verdict?.action, observation: step.verdict?.observation } : row,
            ),
          )
        }
        if (node === 'holdout_eval' && step.metrics) {
          setHoldout(step.metrics)
          setRuntime((p) => setStatus(p, 'agent_runs', 'completed', 'row inserted'))
        }
        return
      }
      if (ev.type === 'run_completed') {
        if (ev.result?.holdout_metrics) setHoldout(ev.result.holdout_metrics)
        finish()
        void refreshHistory()
        return
      }
      if (ev.type === 'error') {
        setError(ev.message || 'stream error')
        finish()
        void refreshHistory()
      }
    }
    ;['run_started', 'node_started', 'node_completed', 'router', 'run_completed', 'error'].forEach((t) =>
      source.addEventListener(t, handle as EventListener),
    )
    source.onerror = () => {
      if (runningRef.current) {
        setError('Stream disconnected before the run finished')
        finish()
      }
    }
  }, [maxCycles, maxRows, modelId, refreshHistory, reset, targetScore])

  return (
    <div className="lab-grid flex min-h-screen flex-col">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-line px-6 py-4">
        <div className="min-w-0">
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-teal">Part 2 · agent workflow</p>
          <h1 className="font-display text-3xl font-semibold text-paper">Olist bad-review loop</h1>
          <p className="mt-1 text-sm text-mist">
            Predict review score ≤ 2 from <span className="font-mono">olist.orders_clean</span> (loaded by the Airflow
            DAG) · 9 agents · live SSE{cycle > 0 ? ` · cycle ${cycle}` : ''}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-mist">
            Model
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={busy}
              className="max-w-[16rem] border border-line bg-ink-2 px-2 py-1.5 font-mono text-xs text-paper"
            >
              {models.length === 0 && <option value="">gateway default</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={m.available === false}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-mist">
            Target F1
            <input
              type="number"
              step="0.01"
              min={0.1}
              max={0.95}
              value={targetScore}
              onChange={(e) => setTargetScore(Number(e.target.value))}
              disabled={busy}
              className="w-20 border border-line bg-ink-2 px-2 py-1.5 font-mono text-paper"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-mist">
            Cycles
            <input
              type="number"
              min={1}
              max={6}
              value={maxCycles}
              onChange={(e) => setMaxCycles(Number(e.target.value))}
              disabled={busy}
              className="w-16 border border-line bg-ink-2 px-2 py-1.5 font-mono text-paper"
            />
          </label>
          <label className="flex items-center gap-2 text-sm text-mist">
            Data
            <select
              value={maxRows}
              onChange={(e) => setMaxRows(Number(e.target.value))}
              disabled={busy}
              className="border border-line bg-ink-2 px-2 py-1.5 font-mono text-xs text-paper"
            >
              {ROWS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <span
            className={`rounded border px-2 py-1 font-mono text-[11px] uppercase tracking-wide ${
              busy ? 'border-teal/50 bg-teal/15 text-teal' : holdout ? 'border-line bg-panel text-paper' : 'border-line text-mist'
            }`}
          >
            {busy ? 'running' : holdout ? 'completed' : error ? 'error' : 'idle'}
          </span>
          <button
            type="button"
            onClick={onRun}
            disabled={busy}
            className="rounded border border-teal bg-teal px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy ? 'Running…' : 'Run loop'}
          </button>
        </div>
      </header>

      {error && <p className="border-b border-danger/40 bg-danger/10 px-6 py-2 text-sm text-danger">{error}</p>}

      <div className="mx-auto grid w-full max-w-[1500px] flex-1 gap-6 px-4 py-6 xl:grid-cols-[1.25fr_0.75fr]">
        <div className="flex min-w-0 flex-col gap-6">
          <LiveSimpleLoopGraph config={OLIST_LOOP_CONFIG} runtime={runtime} activeNodeId={activeNodeId} />
          <Panel title="Final evaluation" sub="Sealed holdout vs Part 1 baselines">
            <FinalEvaluation h={holdout} />
          </Panel>
        </div>
        <div className="flex min-w-0 flex-col gap-6">
          <Panel title="Experiments" sub="One row per cycle · validation split">
            <ExperimentsTable rows={rows} target={targetScore} airflowF1={airflowF1} />
          </Panel>
          <Panel title="Agent log" sub="What each agent reported (LLM calls marked)">
            {log.length === 0 ? (
              <p className="text-sm text-mist">Waiting for the first agent…</p>
            ) : (
              <ol className="max-h-[360px] space-y-1 overflow-y-auto font-mono text-[11px]">
                {log.map((l, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="w-16 shrink-0 text-mist">{l.t}</span>
                    <span className="w-36 shrink-0 text-paper">
                      {l.node}
                      {l.llm ? <span className="ml-1 text-teal">●</span> : null}
                    </span>
                    <span className="text-mist">{l.note}</span>
                  </li>
                ))}
              </ol>
            )}
          </Panel>
          <Panel title="Run history" sub="Saved by the gateway · click to replay">
            {history.length === 0 ? (
              <p className="text-sm text-mist">No runs yet.</p>
            ) : (
              <ul className="space-y-1 text-xs">
                {history.slice(0, 8).map((r) => (
                  <li key={r.run_id}>
                    <button
                      type="button"
                      onClick={() => openHistory(r)}
                      className="flex w-full justify-between gap-2 border border-line px-2 py-1 text-left hover:border-teal"
                    >
                      <span className="text-paper">{new Date(r.created_at).toLocaleString()}</span>
                      <span className="font-mono text-mist">
                        {r.error ? 'failed' : `holdout F1 ${fmt(r.holdout_f1)} · ${r.cycles ?? '—'} cycles`}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>

      <footer className="border-t border-line px-6 py-3 text-center text-xs text-mist">
        <Link to="/simple-loop" className="text-teal hover:underline">
          Original simple loop (template)
        </Link>
        {' · '}
        Graph code: <span className="font-mono">olist_loop/graph.py</span>
      </footer>
    </div>
  )
}
