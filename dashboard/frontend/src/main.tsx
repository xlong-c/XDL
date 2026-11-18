import { ComponentChildren, render } from 'preact'
import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { Activity, Archive, ChevronRight, CircleAlert, CirclePause, CirclePlay, Cpu, Database, FileText, Gauge, GitBranch, HardDrive, Laptop, Layers3, ListPlus, LoaderCircle, Pause, Play, Plus, RotateCcw, Save, Settings2, Square, Terminal, X } from 'lucide-preact'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import './styles.css'
import { api, DashboardEvent, formatBytes, formatDuration, MetricSample, Run, subscribe, SystemSnapshot, Template, TemplateField } from './api'
import { WorkflowEditor } from './workflow_editor'

const ACTIVE = new Set(['queued', 'validating', 'starting', 'running', 'pause_requested', 'paused', 'stopping', 'checkpoint_pausing'])
const LIVE = new Set(['running', 'pause_requested', 'paused', 'stopping', 'checkpoint_pausing'])

function App() {
  const [templates, setTemplates] = useState<Template[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [selectedId, setSelectedId] = useState<string>('')
  const [metrics, setMetrics] = useState<MetricSample[]>([])
  const [logs, setLogs] = useState<DashboardEvent[]>([])
  const [system, setSystem] = useState<SystemSnapshot>({})
  const [showComposer, setShowComposer] = useState(false)
  const [systemOnline, setSystemOnline] = useState(false)
  const [notice, setNotice] = useState('')
  const [view, setView] = useState<'overview' | 'workflow'>('overview')

  const selectedRun = runs.find((run) => run.id === selectedId) ?? runs[0]
  const selectedTemplate = templates.find((template) => template.id === selectedRun?.template_id)

  const refreshRuns = useCallback(async () => {
    try {
      const response = await api.runs()
      setRuns(response.runs)
      setSelectedId((current) => current || response.runs[0]?.id || '')
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Unable to load runs')
    }
  }, [])

  useEffect(() => {
    Promise.all([api.templates(), api.runs(), api.system()]).then(([templateResponse, runResponse, systemResponse]) => {
      setTemplates(templateResponse.templates)
      setRuns(runResponse.runs)
      setSelectedId(runResponse.runs[0]?.id ?? '')
      setSystem(systemResponse.sample)
      setSystemOnline(true)
    }).catch((error) => setNotice(error instanceof Error ? error.message : 'Unable to connect'))
    const timer = window.setInterval(refreshRuns, 3000)
    return () => window.clearInterval(timer)
  }, [refreshRuns])

  useEffect(() => {
    const source = subscribe('/api/system/stream', {
      system: (event) => {
        setSystem(event.payload as SystemSnapshot)
        setSystemOnline(true)
      },
    })
    source.onerror = () => setSystemOnline(false)
    return () => source.close()
  }, [])

  useEffect(() => {
    if (!selectedRun) return
    let mounted = true
    Promise.all([api.metrics(selectedRun.id), api.events(selectedRun.id)]).then(([metricResponse, eventResponse]) => {
      if (!mounted) return
      setMetrics(metricResponse.metrics)
      setLogs(eventResponse.events.filter((event) => event.type === 'log').slice(-250))
    }).catch(() => undefined)
    const source = subscribe(`/api/runs/${selectedRun.id}/stream`, {
      metrics: (event) => {
        const payload = event.payload
        const sample: MetricSample = {
          id: event.id,
          created_at: event.created_at,
          global_step: Number(payload.global_step ?? 0),
          epoch: Number(payload.epoch ?? 0),
          metrics: (payload.metrics ?? {}) as Record<string, number>,
        }
        setMetrics((current) => [...current.filter((item) => item.global_step !== sample.global_step), sample].slice(-2000))
      },
      log: (event) => setLogs((current) => [...current, event].slice(-250)),
      run_state: (event) => {
        setRuns((current) => current.map((run) => run.id === selectedRun.id ? { ...run, status: String(event.payload.state ?? run.status), updated_at: event.created_at } : run))
      },
      worker_exit: () => refreshRuns(),
      worker_error: () => refreshRuns(),
    })
    return () => {
      mounted = false
      source.close()
    }
  }, [selectedRun?.id, refreshRuns])

  const runAction = async (action: string) => {
    if (!selectedRun) return
    try {
      await api.action(selectedRun.id, action)
      await refreshRuns()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Action rejected')
    }
  }

  const onlineRuns = runs.filter((run) => ACTIVE.has(run.status)).length
  const gpuCount = system.gpus?.length ?? 0

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand-mark"><span className="brand-glyph">X</span><div><strong>XDL</strong><small>training observatory</small></div></div>
      <nav className="side-nav">
        <button className={`nav-item ${view === 'overview' ? 'active' : ''}`} onClick={() => setView('overview')}><Gauge size={16} /> Overview <span className="nav-count">{onlineRuns}</span></button>
        <button className={`nav-item ${view === 'workflow' ? 'active' : ''}`} onClick={() => setView('workflow')}><GitBranch size={16} /> Workflow editor <span className="nav-count muted">{view === 'workflow' ? 'ON' : ''}</span></button>
        <a className="nav-item" href="#runs"><Layers3 size={16} /> Run archive <span className="nav-count muted">{runs.length}</span></a>
        <a className="nav-item" href="#system"><Activity size={16} /> Host signals <span className={`signal-dot ${systemOnline ? 'online' : ''}`} /></a>
      </nav>
      <div className="sidebar-footer"><div className="tiny-label">CONTROL PLANE</div><div className="ssh-line"><span className="signal-dot online" /> loopback / SSH</div><div className="version">dashboard 0.1 / xdl runtime</div></div>
    </aside>

    <main className={`main-content ${view === 'workflow' ? 'workflow-view' : ''}`} id={view === 'workflow' ? 'workflow' : 'overview'}>
      <header className="topbar">
        <div>{view === 'workflow' ? <><div className="eyebrow">WORKFLOW BUILDER</div><h1>Blank canvas <span className="slash">/</span> training graph</h1></> : <><div className="eyebrow">LOCAL TRAINING CONTROL</div><h1>Run room <span className="slash">/</span> overview</h1></>}</div>
        <div className="top-actions"><div className="clock-chip"><span className="signal-dot online" />{systemOnline ? 'LIVE TELEMETRY' : 'CONNECTING'}</div><button className="button subtle mobile-view-toggle" onClick={() => setView(view === 'workflow' ? 'overview' : 'workflow')}>{view === 'workflow' ? <><Gauge size={15} /> Monitor</> : <><GitBranch size={15} /> Workflow</>}</button>{view === 'workflow' ? <button className="button subtle" onClick={() => setView('overview')}><Gauge size={15} /> Open monitor</button> : <button className="button primary" onClick={() => setShowComposer(true)}><Plus size={16} /> New run</button>}</div>
      </header>

      {notice && <div className="notice"><CircleAlert size={15} /> {notice}<button className="icon-button" title="Dismiss" onClick={() => setNotice('')}><X size={15} /></button></div>}

      {view === 'workflow' ? <WorkflowEditor gpus={system.gpus ?? []} onQueued={(run) => { setSelectedId(run.id); refreshRuns() }} onNotice={setNotice} /> : <>
        <section className="signal-strip" id="system">
          <SignalCard icon={<Cpu size={15} />} label="CPU" value={`${Math.round(system.cpu_percent ?? 0)}%`} progress={system.cpu_percent ?? 0} tone="orange" />
          <SignalCard icon={<Database size={15} />} label="MEMORY" value={`${Math.round(system.memory?.percent ?? 0)}%`} progress={system.memory?.percent ?? 0} tone="cyan" detail={system.memory ? `${formatBytes(system.memory.used)} / ${formatBytes(system.memory.total)}` : undefined} />
          <SignalCard icon={<HardDrive size={15} />} label="DISK" value={`${Math.round(system.disk?.percent ?? 0)}%`} progress={system.disk?.percent ?? 0} tone="ink" detail={system.disk ? formatBytes(system.disk.free) + ' free' : undefined} />
          <SignalCard icon={<Laptop size={15} />} label="PROCESSES" value={`${system.processes?.count ?? 0}`} progress={Math.min(100, system.processes?.cpu_percent ?? 0)} tone="green" detail={system.processes ? `${Math.round(system.processes.cpu_percent)}% cpu` : undefined} />
          <div className="gpu-signal"><div className="signal-heading"><span><Layers3 size={15} /> NVIDIA</span><span className="tiny-label">{gpuCount} DEVICES</span></div><div className="gpu-list">{system.gpus?.length ? system.gpus.map((gpu) => <span className="gpu-pill" key={gpu.index}><b>GPU {gpu.index}</b><i>{Math.round(gpu.utilization)}%</i></span>) : <span className="empty-signal">no sample</span>}</div></div>
        </section>
        <section className="workspace-grid">
          <RunRail runs={runs} selectedId={selectedRun?.id ?? ''} onSelect={setSelectedId} onNew={() => setShowComposer(true)} />
          <section className="run-detail">
            {selectedRun ? <>
              <RunHeader run={selectedRun} template={selectedTemplate} onAction={runAction} />
              <div className="chart-panel"><div className="panel-head"><div><div className="eyebrow">METRICS / STEP {selectedRun.last_step}</div><h2>Training pulse</h2></div><MetricLegend metrics={metrics} /></div><MetricChart metrics={metrics} /></div>
              <div className="bottom-grid"><LogPanel logs={logs} /><RunFacts run={selectedRun} template={selectedTemplate} /></div>
            </> : <EmptyState onNew={() => setShowComposer(true)} />}
          </section>
        </section>
      </>}
    </main>
    {showComposer && <RunComposer templates={templates} gpus={system.gpus ?? []} onClose={() => setShowComposer(false)} onCreated={(run) => { setShowComposer(false); setSelectedId(run.id); refreshRuns() }} onError={setNotice} />}
  </div>
}

function SignalCard({ icon, label, value, progress, tone, detail }: { icon: ComponentChildren; label: string; value: string; progress: number; tone: string; detail?: string }) {
  return <div className="signal-card"><div className="signal-heading"><span>{icon} {label}</span><strong>{value}</strong></div><div className="meter"><span className={`meter-fill ${tone}`} style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} /></div><small>{detail ?? 'awaiting sample'}</small></div>
}

function RunRail({ runs, selectedId, onSelect, onNew }: { runs: Run[]; selectedId: string; onSelect: (id: string) => void; onNew: () => void }) {
  return <aside className="run-rail" id="runs"><div className="rail-head"><div><div className="eyebrow">WORK QUEUE</div><h2>Runs</h2></div><button className="icon-button accent" title="Create a new run" onClick={onNew}><ListPlus size={17} /></button></div><div className="run-filter"><span>ALL RUNS</span><span>{runs.length.toString().padStart(2, '0')}</span></div><div className="run-list">{runs.length ? runs.map((run) => <button className={`run-item ${run.id === selectedId ? 'selected' : ''}`} key={run.id} onClick={() => onSelect(run.id)}><div className="run-item-top"><span className={`status-marker ${run.status}`} /><strong>{run.config?.workflow_name ?? run.config?.name ?? run.template_id}</strong><ChevronRight size={14} /></div><div className="run-item-meta"><span>{run.status.replaceAll('_', ' ')}</span><span>{(run.devices ?? []).length ? `GPU ${(run.devices ?? []).join(',')}` : 'CPU'}</span></div><div className="run-progress"><span style={{ width: `${run.status === 'succeeded' ? '100%' : LIVE.has(run.status) ? '48%' : '12%'}` }} /></div></button>) : <div className="empty-list">No runs yet</div>}</div><button className="rail-new" onClick={onNew}><Plus size={15} /> queue a run</button></aside>
}

function RunHeader({ run, template, onAction }: { run: Run; template?: Template; onAction: (action: string) => void }) {
  const live = LIVE.has(run.status)
  return <div className="run-header"><div className="run-title"><div className={`large-status ${run.status}`}><span className="status-marker" /></div><div><div className="eyebrow">{template?.label ?? run.config?.workflow_name ?? run.template_id} <span className="bullet">•</span> {run.id.slice(0, 12)}</div><h2>{run.config?.name ?? 'Untitled run'}</h2><div className="run-subline"><span className={`status-word ${run.status}`}>{run.status.replaceAll('_', ' ')}</span><span>{(run.devices ?? []).length ? `cuda / ${(run.devices ?? []).map((device) => `GPU ${device}`).join(', ')}` : 'cpu'}</span><span>epoch {run.last_epoch}</span></div></div></div><div className="run-controls">{run.status === 'paused' || run.status === 'pause_requested' ? <button className="icon-button control" title="Resume" onClick={() => onAction('resume')}><Play size={16} /></button> : live && <button className="icon-button control" title="Pause at batch boundary" onClick={() => onAction('pause')}><Pause size={16} /></button>}{live && <button className="icon-button control" title="Save checkpoint and pause" onClick={() => onAction('checkpoint-pause')}><Save size={16} /></button>}{live && <button className="icon-button danger" title="Stop safely" onClick={() => onAction('stop')}><Square size={15} /></button>}{ACTIVE.has(run.status) && run.status !== 'queued' && <button className="icon-button danger" title="Kill process" onClick={() => onAction('kill')}><X size={16} /></button>}{['checkpoint_paused', 'stopped', 'succeeded'].includes(run.status) && run.checkpoint_path && <button className="button subtle" onClick={() => onAction('resume-from-checkpoint')}><RotateCcw size={15} /> Resume</button>}</div></div>
}

function MetricLegend({ metrics }: { metrics: MetricSample[] }) {
  const names = metricNames(metrics).slice(0, 4)
  return <div className="metric-legend">{names.map((name, index) => <span key={name}><i className={`legend-dot dot-${index}`} />{name}</span>)}</div>
}

function MetricChart({ metrics }: { metrics: MetricSample[] }) {
  const target = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const names = metricNames(metrics).slice(0, 4)
  useEffect(() => {
    if (!target.current || !names.length) return
    const data: uPlot.AlignedData = [metrics.map((sample) => sample.global_step), ...names.map((name) => metrics.map((sample) => sample.metrics[name] ?? null))]
    const options: uPlot.Options = {
      width: target.current.clientWidth,
      height: 280,
      cursor: { drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [{ stroke: '#9d9a91', grid: { stroke: '#e2dfd7', width: 1 }, ticks: { stroke: '#d7d3ca' } }, { stroke: '#9d9a91', grid: { stroke: '#e2dfd7', width: 1 }, ticks: { stroke: '#d7d3ca' } }],
      series: [{ label: 'step' }, ...names.map((name, index) => ({ label: name, stroke: ['#e56b2f', '#1d9b9b', '#253238', '#b4a43d'][index], width: 2, points: { show: metrics.length < 20, size: 5 } }))],
    }
    plot.current?.destroy()
    plot.current = new uPlot(options, data, target.current)
    const observer = new ResizeObserver(() => { if (target.current && plot.current) plot.current.setSize({ width: target.current.clientWidth, height: 280 }) })
    observer.observe(target.current)
    return () => { observer.disconnect(); plot.current?.destroy(); plot.current = null }
  }, [metrics, names.join('|')])
  return names.length ? <div className="chart-wrap" ref={target} /> : <div className="chart-empty"><Activity size={22} /><span>Waiting for the first metric sample</span></div>
}

function LogPanel({ logs }: { logs: DashboardEvent[] }) {
  return <section className="log-panel"><div className="panel-head compact"><div><div className="eyebrow">STREAM / STDOUT + STDERR</div><h3>Worker log</h3></div><Terminal size={16} /></div><div className="log-body">{logs.length ? logs.map((log) => <div className="log-line" key={`${log.id}-${log.created_at}`}><time>{new Date(log.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time><span className={log.payload.stream === 'stderr' ? 'stderr' : ''}>{String(log.payload.message ?? '')}</span></div>) : <div className="log-empty">No worker output</div>}</div></section>
}

function RunFacts({ run, template }: { run: Run; template?: Template }) {
  return <section className="facts-panel"><div className="panel-head compact"><div><div className="eyebrow">RUN RECORD</div><h3>Facts</h3></div><Archive size={16} /></div><dl>{run.workflow_id ? <><div><dt>Workflow</dt><dd title={run.workflow_id}>{run.config?.workflow_name ?? run.workflow_id}</dd></div><div><dt>Snapshot</dt><dd title={run.workflow_snapshot_hash ?? ''}>{(run.workflow_snapshot_hash ?? '').slice(0, 12) || '—'}</dd></div><div><dt>Profile</dt><dd>{run.config?.execution_profile ?? '—'}</dd></div><div><dt>Compiler</dt><dd>{run.compiler_version ?? '—'}</dd></div></> : <div><dt>Template</dt><dd>{template?.label ?? run.template_id}</dd></div>}<div><dt>Duration</dt><dd>{formatDuration(run.started_at, run.finished_at)}</dd></div><div><dt>Launcher</dt><dd>{run.launcher}</dd></div><div><dt>Process</dt><dd>{run.process_id ?? '—'}</dd></div><div><dt>Output</dt><dd title={run.output_dir}>{run.output_dir.split('/').pop()}</dd></div></dl>{run.error_message && <div className="fact-error"><CircleAlert size={14} />{run.error_message}</div>}</section>
}

function EmptyState({ onNew }: { onNew: () => void }) {
  return <div className="empty-state"><div className="empty-orbit"><Activity size={28} /></div><div className="eyebrow">NO ACTIVE SELECTION</div><h2>Queue the first run</h2><button className="button primary" onClick={onNew}><Plus size={16} /> New run</button></div>
}

function RunComposer({ templates, gpus, onClose, onCreated, onError }: { templates: Template[]; gpus: NonNullable<SystemSnapshot['gpus']>; onClose: () => void; onCreated: (run: Run) => void; onError: (message: string) => void }) {
  const [templateId, setTemplateId] = useState(templates[0]?.id ?? '')
  const template = templates.find((item) => item.id === templateId)
  const [values, setValues] = useState<Record<string, unknown>>(() => defaultsFor(templates[0]))
  const [launcher, setLauncher] = useState('none')
  const [devices, setDevices] = useState<number[]>([])
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  useEffect(() => { setValues(defaultsFor(template)) }, [templateId])
  const setValue = (field: TemplateField, value: unknown) => setValues((current) => ({ ...current, [field.path]: field.type === 'integer' ? Number(value) : field.type === 'number' ? Number(value) : value }))
  const submit = async (event: Event) => {
    event.preventDefault()
    setSubmitting(true)
    try {
      const run = await api.submit({ template_id: templateId, name: name || undefined, parameters: values, launcher, devices })
      onCreated(run)
    } catch (error) {
      onError(error instanceof Error ? error.message : 'Submission rejected')
    } finally { setSubmitting(false) }
  }
  return <div className="modal-backdrop" onClick={(event) => { if (event.target === event.currentTarget) onClose() }}><section className="composer"><div className="composer-head"><div><div className="eyebrow">QUEUE / CONFIGURE</div><h2>New training run</h2></div><button className="icon-button" title="Close" onClick={onClose}><X size={17} /></button></div><form onSubmit={submit}><label className="field"><span>Template</span><select value={templateId} onChange={(event) => setTemplateId((event.target as HTMLSelectElement).value)}>{templates.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label><label className="field"><span>Run name</span><input value={name} onInput={(event) => setName((event.target as HTMLInputElement).value)} placeholder={template?.fields.find((field) => field.path === 'runtime.experiment_name')?.default?.toString() ?? 'optional'} /></label><div className="form-grid">{template?.fields.filter((field) => field.path !== 'runtime.experiment_name').map((field) => <TemplateInput key={field.path} field={field} value={values[field.path]} onChange={(value) => setValue(field, value)} />)}</div><div className="form-rule"><span>EXECUTION</span></div><div className="form-grid two"><label className="field"><span>Launcher</span><select value={launcher} onChange={(event) => setLauncher((event.target as HTMLSelectElement).value)}><option value="none">Single process</option><option value="torchrun">Torchrun</option><option value="accelerate">Accelerate</option></select></label><div className="field"><span>GPU IDs <small>(select manually)</small></span><div className="gpu-select">{gpus.length ? gpus.map((gpu) => <button type="button" key={gpu.index} className={`gpu-choice ${devices.includes(gpu.index) ? 'selected' : ''}`} aria-pressed={devices.includes(gpu.index)} onClick={() => setDevices((current) => current.includes(gpu.index) ? current.filter((item) => item !== gpu.index) : [...current, gpu.index].sort((a, b) => a - b))}><span>GPU {gpu.index}</span><b>{Math.round(gpu.utilization)}%</b></button>) : <span className="no-gpu">No NVIDIA devices reported</span>}</div></div></div><div className="composer-foot"><span className="composer-note"><Settings2 size={14} /> Validated against XDL schema on start</span><button className="button primary" disabled={submitting || !templateId}>{submitting ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />} Queue run</button></div></form></section></div>
}

function TemplateInput({ field, value, onChange }: { field: TemplateField; value: unknown; onChange: (value: unknown) => void }) {
  if (field.type === 'boolean') return <label className="field check-field"><span>{field.label}</span><button type="button" className={`toggle ${value ? 'on' : ''}`} aria-pressed={Boolean(value)} onClick={() => onChange(!value)}><i /></button></label>
  return <label className="field"><span>{field.label}{field.required && <em>*</em>}</span><input type={field.type === 'integer' || field.type === 'number' ? 'number' : 'text'} value={String(value ?? '')} min={field.min} max={field.max} step={field.type === 'number' ? 'any' : 1} required={field.required} onInput={(event) => onChange((event.target as HTMLInputElement).value)} placeholder={field.type === 'data_directory' ? 'relative to data/' : undefined} /></label>
}

function defaultsFor(template?: Template): Record<string, unknown> {
  return Object.fromEntries(template?.fields.map((field) => [field.path, field.default ?? (field.type === 'boolean' ? false : '')]) ?? [])
}

function metricNames(metrics: MetricSample[]): string[] {
  const names = new Set<string>()
  metrics.forEach((sample) => Object.keys(sample.metrics).forEach((name) => names.add(name)))
  return Array.from(names)
}

render(<App />, document.getElementById('app')!)
