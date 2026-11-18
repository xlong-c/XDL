/**
 * WorkflowSnapshot v1 的前端类型契约.
 *
 * 与 docs/md/architecture/dashboard-workflow-ir.md 以及 xdl/workflow/ir.py 保持一致.
 * 前端只负责编辑与本地结构提示; 语义校验由 /api/workflows/validate 的 Python
 * compiler 完成, 前端校验不是安全边界.
 */

export const SNAPSHOT_VERSION = 1

export const EXECUTION_PROFILES = ['schema-v1', 'dag-runtime'] as const
export type ExecutionProfile = (typeof EXECUTION_PROFILES)[number]

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

export type Params = Record<string, JsonValue>

export type PortRef = {
  node: string
  port: string
}

export type NodePosition = {
  x: number
  y: number
}

export type WorkflowNode = {
  id: string
  kind: string
  target: string
  params: Params
  position: NodePosition
  metadata?: Params
}

export type WorkflowEdge = {
  id: string
  from: PortRef
  to: PortRef
}

export type Graph = { nodes: WorkflowNode[]; edges: WorkflowEdge[] }

export type WorkflowSettings = {
  launcher?: string
  devices?: number[]
  runtime?: Params
}

export type WorkflowSnapshot = {
  version: number
  id?: string
  name: string
  execution_profile: ExecutionProfile
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  settings: WorkflowSettings
  step_graph?: Graph
}

export type IssueLevel = 'error' | 'warning'

export type WorkflowIssue = {
  code: string
  level: IssueLevel
  message: string
  path?: string
  node_id?: string
  edge_id?: string
}

export type WorkflowRecord = {
  id: string
  version: number
  name: string
  execution_profile: ExecutionProfile
  snapshot_hash: string
  created_at: string
  updated_at: string
  snapshot?: WorkflowSnapshot
}

export type ValidateResponse = {
  valid: boolean
  issues: WorkflowIssue[]
  snapshot_hash?: string
}

/** 编辑器持有的草稿状态; 与快照一一对应, 只是没有服务端 id. */
export type EditorState = {
  name: string
  executionProfile: ExecutionProfile
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  settings: WorkflowSettings
  stepGraph?: Graph
}
