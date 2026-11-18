/**
 * 草稿持久化与快照序列化.
 *
 * - localStorage 只保存未提交的草稿, 键名带版本, 解析失败时回退到空图.
 * - 提交给服务端的快照由 `toSnapshot` 生成: 服务端分配 id, 前端不携带 id.
 * - `canonicalSnapshotJson` 用于稳定比较与调试输出, 不作为服务端哈希来源.
 */

import type { CatalogIndex } from './catalog'
import { buildStarterGraph, defaultParams, indexStepCatalog, kindOf, targetOf } from './catalog'
import type {
  EditorState,
  ExecutionProfile,
  Graph,
  JsonValue,
  Params,
  WorkflowEdge,
  WorkflowNode,
  WorkflowSnapshot,
} from './types'
import { EXECUTION_PROFILES, SNAPSHOT_VERSION } from './types'

export const DRAFT_STORAGE_KEY = 'xdl.dashboard.workflow.v2'
export const DEFAULT_WORKFLOW_NAME = '新建训练流水线'

export function emptyState(): EditorState {
  return {
    name: DEFAULT_WORKFLOW_NAME,
    executionProfile: 'schema-v1',
    nodes: [],
    edges: [],
    settings: { launcher: 'none', devices: [], runtime: { seed: 42 } },
  }
}

export function starterState(index: CatalogIndex): EditorState {
  const graph = buildStarterGraph(index)
  return {
    ...emptyState(),
    name: '标准训练图',
    nodes: graph.nodes,
    edges: graph.edges,
  }
}

/** 把编辑器状态转换为待提交快照 (不含服务端 id). */
export function toSnapshot(state: EditorState): WorkflowSnapshot {
  return {
    version: SNAPSHOT_VERSION,
    name: state.name.trim() || DEFAULT_WORKFLOW_NAME,
    execution_profile: state.executionProfile,
    ...snapshotGraph(state),
    ...(state.stepGraph ? { step_graph: snapshotGraph(state.stepGraph) } : {}),
    settings: normalizeSettings(state.settings),
  }
}

function snapshotGraph(state: Graph): Graph {
  return {
    nodes: state.nodes.map((node) => ({
      id: node.id,
      kind: node.kind,
      target: node.target,
      params: node.params,
      position: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
      ...(node.metadata ? { metadata: node.metadata } : {}),
    })),
    edges: state.edges.map((edge) => ({ id: edge.id, from: edge.from, to: edge.to })),
  }
}

function normalizeSettings(settings: EditorState['settings']): EditorState['settings'] {
  const runtime: Params = {}
  const seed = settings.runtime?.seed
  if (typeof seed === 'number' && Number.isInteger(seed)) {
    runtime.seed = seed
  }
  return {
    launcher: settings.launcher ?? 'none',
    devices: [...(settings.devices ?? [])].sort((left, right) => left - right),
    ...(Object.keys(runtime).length ? { runtime } : {}),
  }
}

/** 稳定 JSON: 节点与边按 id 排序, 键按字典序, 用于比较和调试. */
export function canonicalSnapshotJson(snapshot: WorkflowSnapshot): string {
  const sorted: WorkflowSnapshot = {
    ...snapshot,
    nodes: [...snapshot.nodes].sort((left, right) => left.id.localeCompare(right.id)),
    edges: [...snapshot.edges].sort((left, right) => left.id.localeCompare(right.id)),
    ...(snapshot.step_graph ? { step_graph: {
      nodes: [...snapshot.step_graph.nodes].sort((left, right) => left.id.localeCompare(right.id)),
      edges: [...snapshot.step_graph.edges].sort((left, right) => left.id.localeCompare(right.id)),
    } } : {}),
  }
  return JSON.stringify(sortValue(sorted as unknown as JsonValue))
}

function sortValue(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(sortValue)
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, JsonValue>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortValue(item)] as const)
    return Object.fromEntries(entries) as JsonValue
  }
  return value
}

export function serializeDraft(state: EditorState): string {
  return JSON.stringify({
    version: SNAPSHOT_VERSION,
    name: state.name,
    execution_profile: state.executionProfile,
    nodes: state.nodes,
    edges: state.edges,
    settings: state.settings,
    step_graph: state.stepGraph,
  })
}

/**
 * 解析本地草稿.
 *
 * 任何字段与节点目录不一致的节点或边都会被丢弃, 而不是猜一个默认值:
 * 目录变化后旧草稿可能失效, 显式丢弃比静默修正更容易发现问题.
 */
export function parseDraft(raw: string | null, index: CatalogIndex): EditorState | null {
  if (!raw) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== 'object') return null
  const record = parsed as Record<string, unknown>
  if (record.version !== SNAPSHOT_VERSION) return null
  const profile = EXECUTION_PROFILES.includes(record.execution_profile as ExecutionProfile)
    ? (record.execution_profile as ExecutionProfile)
    : 'schema-v1'
  const step = record.step_graph
  const stepGraph = step && typeof step === 'object' && !Array.isArray(step)
    ? parseGraph(step as Record<string, unknown>, indexStepCatalog(index)) : undefined
  const settings = asSettings(record.settings)
  return {
    name: typeof record.name === 'string' && record.name.trim() ? record.name : DEFAULT_WORKFLOW_NAME,
    executionProfile: profile,
    ...parseGraph(record, index),
    stepGraph,
    settings,
  }
}

function parseGraph(record: Record<string, unknown>, index: CatalogIndex): Graph {
  const nodes: WorkflowNode[] = []
  const seen = new Set<string>()
  for (const candidate of Array.isArray(record.nodes) ? record.nodes : []) {
    const node = asNode(candidate, index)
    if (!node || seen.has(node.id)) continue
    seen.add(node.id)
    nodes.push(node)
  }
  const edges: WorkflowEdge[] = []
  const edgeIds = new Set<string>()
  for (const candidate of Array.isArray(record.edges) ? record.edges : []) {
    const edge = asEdge(candidate, seen)
    if (!edge || edgeIds.has(edge.id)) continue
    edgeIds.add(edge.id)
    edges.push(edge)
  }
  return { nodes, edges }
}

function asNode(candidate: unknown, index: CatalogIndex): WorkflowNode | null {
  if (!candidate || typeof candidate !== 'object') return null
  const record = candidate as Record<string, unknown>
  const id = typeof record.id === 'string' ? record.id : ''
  const kind = typeof record.kind === 'string' ? record.kind : ''
  const target = typeof record.target === 'string' ? record.target : ''
  if (!id || !kind || !target) return null
  const kindSpec = kindOf(index, kind)
  if (!kindSpec) return null
  const targetSpec = targetOf(index, kind, target)
  if (!targetSpec) return null
  const params = mergeParams(record.params, targetSpec.params, defaultParams(targetSpec))
  const position = asPosition(record.position)
  const metadata = asMetadata(record.metadata)
  return {
    id,
    kind,
    target,
    params,
    position,
    ...(metadata ? { metadata } : {}),
  }
}

function mergeParams(
  raw: unknown,
  specs: { name: string }[],
  defaults: Params,
): Params {
  const allowed = new Set(specs.map((spec) => spec.name))
  const params: Params = { ...defaults }
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    for (const [key, value] of Object.entries(raw as Record<string, JsonValue>)) {
      if (!allowed.has(key)) continue
      params[key] = value
    }
  }
  return params
}

function asPosition(raw: unknown): { x: number; y: number } {
  if (raw && typeof raw === 'object') {
    const record = raw as Record<string, unknown>
    const x = Number(record.x)
    const y = Number(record.y)
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return { x: Math.max(0, x), y: Math.max(0, y) }
    }
  }
  return { x: 40, y: 40 }
}

function asMetadata(raw: unknown): Params | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined
  const label = (raw as Record<string, unknown>).label
  return typeof label === 'string' ? { label } : undefined
}

function asEdge(candidate: unknown, nodeIds: Set<string>): WorkflowEdge | null {
  if (!candidate || typeof candidate !== 'object') return null
  const record = candidate as Record<string, unknown>
  const id = typeof record.id === 'string' ? record.id : ''
  const from = asPortRef(record.from)
  const to = asPortRef(record.to)
  if (!id || !from || !to) return null
  if (!nodeIds.has(from.node) || !nodeIds.has(to.node)) return null
  return { id, from, to }
}

function asPortRef(raw: unknown): { node: string; port: string } | null {
  if (!raw || typeof raw !== 'object') return null
  const record = raw as Record<string, unknown>
  if (typeof record.node !== 'string' || typeof record.port !== 'string') return null
  if (!record.node || !record.port) return null
  return { node: record.node, port: record.port }
}

function asSettings(raw: unknown): EditorState['settings'] {
  const settings: EditorState['settings'] = { launcher: 'none', devices: [], runtime: { seed: 42 } }
  if (!raw || typeof raw !== 'object') return settings
  const record = raw as Record<string, unknown>
  if (typeof record.launcher === 'string') settings.launcher = record.launcher
  if (Array.isArray(record.devices)) {
    settings.devices = record.devices
      .map((item) => Number(item))
      .filter((item) => Number.isInteger(item) && item >= 0)
  }
  if (record.runtime && typeof record.runtime === 'object') {
    const seed = (record.runtime as Record<string, unknown>).seed
    if (typeof seed === 'number' && Number.isInteger(seed)) {
      settings.runtime = { seed }
    }
  }
  return settings
}
