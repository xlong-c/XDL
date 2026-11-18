import type { NodeCatalog } from './workflow/catalog'
import type {
  WorkflowIssue,
  WorkflowRecord,
  WorkflowSnapshot,
  ValidateResponse,
} from './workflow/types'

export type TemplateField = {
  path: string
  type: 'string' | 'integer' | 'number' | 'boolean' | 'data_directory'
  label: string
  default?: string | number | boolean
  min?: number
  max?: number
  required?: boolean
  options?: string[]
}

export type Template = {
  id: string
  label: string
  summary: string
  version: number
  requires_gpu: boolean
  fields: TemplateField[]
}

export type RunConfig = {
  submitted_parameters?: Record<string, unknown>
  resolved_parameters?: Record<string, unknown>
  name?: string
  workflow_id?: string
  workflow_name?: string
  workflow_snapshot_hash?: string
  execution_profile?: string
}

export type Run = {
  id: string
  parent_run_id?: string
  template_id: string
  status: string
  submitted_at: string
  updated_at: string
  started_at?: string
  finished_at?: string
  devices: number[]
  launcher: string
  process_id?: number
  last_step: number
  last_epoch: number
  checkpoint_path?: string
  output_dir: string
  error_message?: string
  config?: RunConfig
  workflow_id?: string
  workflow_snapshot_hash?: string
  compiler_version?: string
}

export type DashboardEvent = {
  id: number
  run_id: string
  type: string
  created_at: string
  payload: Record<string, any>
  reliable: boolean
}

export type MetricSample = {
  id: number
  created_at: string
  global_step: number
  epoch: number
  metrics: Record<string, number>
}

export type SystemSnapshot = {
  timestamp?: string
  cpu_percent?: number
  memory?: { total: number; used: number; available: number; percent: number }
  disk?: { total: number; used: number; free: number; percent: number }
  gpus?: Array<{ index: number; utilization: number; memory_used_mib: number; memory_total_mib: number; temperature_c: number; power_w: number }>
  processes?: { count: number; rss_bytes: number; cpu_percent: number }
}

/** 工作流请求失败时携带服务端返回的可定位问题. */
export class WorkflowRequestError extends Error {
  readonly issues: WorkflowIssue[]

  constructor(message: string, issues: WorkflowIssue[]) {
    super(message)
    this.name = 'WorkflowRequestError'
    this.issues = issues
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.error?.message ?? `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

async function requestWorkflow<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const message = payload?.error?.message ?? `Request failed: ${response.status}`
    throw new WorkflowRequestError(message, Array.isArray(payload?.issues) ? payload.issues : [])
  }
  return payload as T
}

export type WorkflowSubmitPayload = {
  name?: string
  launcher?: string
  devices?: number[]
  parent_run_id?: string
  resume_checkpoint?: string
}

export const api = {
  templates: () => request<{ templates: Template[] }>('/api/templates'),
  runs: () => request<{ runs: Run[] }>('/api/runs?limit=100'),
  system: () => request<{ sample: SystemSnapshot }>('/api/system'),
  run: (id: string) => request<Run>(`/api/runs/${encodeURIComponent(id)}`),
  metrics: (id: string) => request<{ metrics: MetricSample[] }>(`/api/runs/${encodeURIComponent(id)}/metrics`),
  events: (id: string) => request<{ events: DashboardEvent[] }>(`/api/runs/${encodeURIComponent(id)}/events`),
  submit: (payload: unknown) => request<Run>('/api/runs', { method: 'POST', body: JSON.stringify(payload) }),
  action: (id: string, action: string, payload?: unknown) => request<unknown>(`/api/runs/${encodeURIComponent(id)}/actions/${action}`, { method: 'POST', body: JSON.stringify(payload ?? {}) }),

  workflowCatalog: () => request<{ catalog: NodeCatalog }>('/api/workflows/catalog'),
  workflows: () => request<{ workflows: WorkflowRecord[] }>('/api/workflows?limit=50'),
  workflow: (id: string) => requestWorkflow<WorkflowRecord>(`/api/workflows/${encodeURIComponent(id)}`),
  createWorkflow: (snapshot: WorkflowSnapshot) =>
    requestWorkflow<WorkflowRecord>('/api/workflows', { method: 'POST', body: JSON.stringify(snapshot) }),
  validateWorkflow: (snapshot: WorkflowSnapshot) =>
    requestWorkflow<ValidateResponse>('/api/workflows/validate', { method: 'POST', body: JSON.stringify({ snapshot }) }),
  submitWorkflow: (id: string, payload: WorkflowSubmitPayload) =>
    requestWorkflow<Run>(`/api/workflows/${encodeURIComponent(id)}/runs`, { method: 'POST', body: JSON.stringify(payload) }),
}

export function subscribe(path: string, handlers: Record<string, (event: DashboardEvent) => void>): EventSource {
  const source = new EventSource(path)
  for (const [type, handler] of Object.entries(handlers)) {
    source.addEventListener(type, (event) => {
      try {
        handler(JSON.parse((event as MessageEvent).data) as DashboardEvent)
      } catch {
        // The server keeps SSE payloads JSON; ignore malformed client-side data.
      }
    })
  }
  return source
}

export function formatBytes(value: number | undefined): string {
  if (!value || value < 1024) return `${Math.round(value ?? 0)} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let amount = value
  let unit = -1
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${amount.toFixed(amount >= 10 ? 0 : 1)} ${units[unit]}`
}

export function formatDuration(start?: string, end?: string): string {
  if (!start) return '—'
  const seconds = Math.max(0, (new Date(end ?? Date.now()).getTime() - new Date(start).getTime()) / 1000)
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}
