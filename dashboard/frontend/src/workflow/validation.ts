/**
 * 本地结构校验.
 *
 * 覆盖与 Go 结构校验, Python 语义校验同名的错误码, 用于即时交互反馈.
 * 真正的语义校验仍然由 `POST /api/workflows/validate` 的 Python compiler 给出.
 */

import type { CatalogIndex, ParamSpec } from './catalog'
import { checkParam, findPort, indexStepCatalog, kindOf, targetOf } from './catalog'
import type { EditorState, WorkflowEdge, WorkflowIssue, WorkflowNode } from './types'

const UNIQUE_ROLES = ['trainer', 'model', 'optimizer']
const REQUIRED_ROLES = ['trainer', 'model', 'optimizer', 'loss']
const OPTIONAL_UNIQUE_ROLES = ['scheduler', 'checkpoint']

export function validateLocally(state: EditorState, index: CatalogIndex): WorkflowIssue[] {
  const issues = validateGraph(state, index)
  if (state.stepGraph) {
    const step = { ...state, ...state.stepGraph, stepGraph: undefined }
    issues.push(...validateGraph(step, indexStepCatalog(index), true).map((problem) => ({
      ...problem, path: `step_graph/${problem.path ?? 'nodes'}`,
    })))
    if (state.executionProfile !== 'dag-runtime') {
      issues.push(issue('unsupported_profile', '训练步图需要 dag-runtime', 'step_graph'))
    }
    for (const node of step.nodes) {
      if (node.kind === 'Select' && ((node.params.index != null) === Boolean(node.params.key))) {
        issues.push(issue('invalid_param', '索引和字段名需要二选一', `step_graph/nodes/${node.id}/params`, node.id))
      }
      if (node.kind === 'LossCall' || node.kind === 'MetricCall') {
        const kind = node.kind === 'LossCall' ? 'loss' : 'metric'
        if (!state.nodes.some((component) => component.kind === kind && component.id === node.params.component)) {
          issues.push(issue('invalid_param', `请选择装配图中的 ${kind} 组件`, `step_graph/nodes/${node.id}/params/component`, node.id))
        }
      }
    }
  }
  return issues
}

function validateGraph(state: EditorState, index: CatalogIndex, step = false): WorkflowIssue[] {
  const issues: WorkflowIssue[] = []
  const limits = index.catalog.limits

  if (!step && !state.name.trim()) {
    issues.push(issue('invalid_snapshot', 'name 必须是非空字符串', 'name'))
  } else if (!step && state.name.length > limits.max_name_length) {
    issues.push(issue('limit_exceeded', `name 长度超过上限 ${limits.max_name_length}`, 'name'))
  }
  if (!state.nodes.length) {
    issues.push(issue('invalid_snapshot', step ? '训练步图为空' : '画布为空, 至少添加数据集, 模型, 损失函数和训练循环节点', 'nodes'))
  }
  if (state.nodes.length > limits.max_nodes) {
    issues.push(issue('limit_exceeded', `节点数超过上限 ${limits.max_nodes}`, 'nodes'))
  }
  if (state.edges.length > limits.max_edges) {
    issues.push(issue('limit_exceeded', `边数超过上限 ${limits.max_edges}`, 'edges'))
  }

  const nodeIds = new Set<string>()
  for (const node of state.nodes) {
    if (!node.id.trim()) {
      issues.push(issue('invalid_snapshot', '节点 id 必须是非空字符串', 'nodes'))
      continue
    }
    if (nodeIds.has(node.id)) {
      issues.push(issue('duplicate_id', `节点 id ${node.id} 重复`, 'nodes', node.id))
      continue
    }
    nodeIds.add(node.id)
    const kind = kindOf(index, node.kind)
    if (!kind) {
      issues.push(issue('unknown_node_kind', `未知节点类型 ${node.kind}`, `nodes/${node.id}/kind`, node.id))
      continue
    }
    const target = targetOf(index, node.kind, node.target)
    if (!target) {
      issues.push(issue('unknown_target', `节点 ${node.id} 的 target ${node.target} 不在目录内`, `nodes/${node.id}/target`, node.id))
      continue
    }
    if (!kind.profiles.includes(state.executionProfile)) {
      issues.push(
        issue(
          'unsupported_profile',
          `节点 ${node.id} 的类型 ${node.kind} 不支持 ${state.executionProfile}`,
          `nodes/${node.id}/kind`,
          node.id,
        ),
      )
    }
    issues.push(...validateParams(node, target.params))
  }

  issues.push(...validateEdges(state, index))
  issues.push(...validateRequiredInputs(state, index))
  if (step) {
    for (const role of ['BatchInput', 'StepOutput']) {
      const count = state.nodes.filter((node) => node.kind === role).length
      if (count !== 1) issues.push(issue(count ? 'duplicate_role' : 'missing_required_node', `需要恰好一个 ${role} 节点`, 'nodes'))
    }
  } else {
    issues.push(...validateRoles(state, index))
  }

  const cycle = detectCycle(state.nodes, state.edges)
  if (cycle.length) {
    issues.push(issue('cycle_detected', `节点图存在环: ${cycle.join(' -> ')}`, 'edges', cycle[0]))
  } else if (!step) {
    issues.push(...validateConnectivity(state, index))
    if (state.executionProfile === 'dag-runtime') {
      issues.push(...validateOutputsConsumed(state, index))
    }
  }
  return issues
}

function validateParams(node: WorkflowNode, specs: ParamSpec[]): WorkflowIssue[] {
  const issues: WorkflowIssue[] = []
  const declared = new Map(specs.map((spec) => [spec.name, spec]))
  for (const name of Object.keys(node.params)) {
    if (!declared.has(name)) {
      issues.push(
        issue('invalid_param', `${node.target} 不接受参数 ${name}`, `nodes/${node.id}/params/${name}`, node.id),
      )
    }
  }
  for (const spec of specs) {
    const value = node.params[spec.name]
    const problem = checkParam(spec, value ?? null)
    if (problem) {
      issues.push(
        issue('invalid_param', `参数 ${spec.name}: ${problem}`, `nodes/${node.id}/params/${spec.name}`, node.id),
      )
    }
  }
  return issues
}

function validateEdges(state: EditorState, index: CatalogIndex): WorkflowIssue[] {
  const issues: WorkflowIssue[] = []
  const nodes = new Map(state.nodes.map((node) => [node.id, node]))
  const edgeIds = new Set<string>()
  const pairs = new Set<string>()
  const occupied = new Map<string, string>()
  for (const edge of state.edges) {
    if (edgeIds.has(edge.id)) {
      issues.push(issue('duplicate_id', `边 id ${edge.id} 重复`, 'edges', undefined, edge.id))
      continue
    }
    edgeIds.add(edge.id)
    const source = nodes.get(edge.from.node)
    const target = nodes.get(edge.to.node)
    if (!source || !target) {
      issues.push(
        issue(
          'unknown_node',
          `边 ${edge.id} 引用了不存在的节点 ${source ? edge.to.node : edge.from.node}`,
          `edges/${edge.id}`,
          undefined,
          edge.id,
        ),
      )
      continue
    }
    if (edge.from.node === edge.to.node) {
      issues.push(issue('self_loop', `边 ${edge.id} 连接了节点自身`, `edges/${edge.id}`, edge.from.node, edge.id))
      continue
    }
    const sourcePort = findPort(index, source.kind, edge.from.port, 'out')
    const targetPort = findPort(index, target.kind, edge.to.port, 'in')
    if (!sourcePort) {
      issues.push(
        issue('unknown_port', `节点 ${source.id} 没有输出端口 ${edge.from.port}`, `edges/${edge.id}/from`, source.id, edge.id),
      )
      continue
    }
    if (!targetPort) {
      issues.push(
        issue('unknown_port', `节点 ${target.id} 没有输入端口 ${edge.to.port}`, `edges/${edge.id}/to`, target.id, edge.id),
      )
      continue
    }
    const pair = `${edge.from.node}:${edge.from.port}->${edge.to.node}:${edge.to.port}`
    if (pairs.has(pair)) {
      issues.push(issue('duplicate_edge', `边 ${edge.id} 与已有连线重复`, `edges/${edge.id}`, undefined, edge.id))
      continue
    }
    pairs.add(pair)
    const slot = `${edge.to.node}:${edge.to.port}`
    if (!targetPort.multi && occupied.has(slot)) {
      issues.push(
        issue(
          'port_conflict',
          `输入端口 ${target.id}.${edge.to.port} 已被 ${occupied.get(slot)} 占用`,
          `edges/${edge.id}/to`,
          target.id,
          edge.id,
        ),
      )
      continue
    }
    occupied.set(slot, edge.id)
    if (!typesCompatible(sourcePort.type, targetPort.type)) {
      issues.push(
        issue(
          'type_mismatch',
          `边 ${edge.id}: ${source.id}.${edge.from.port} (${sourcePort.type}) 不能连到 ${target.id}.${edge.to.port} (${targetPort.type})`,
          `edges/${edge.id}/to`,
          target.id,
          edge.id,
        ),
      )
    }
  }
  return issues
}

function validateRequiredInputs(state: EditorState, index: CatalogIndex): WorkflowIssue[] {
  const issues: WorkflowIssue[] = []
  const connected = new Set(state.edges.map((edge) => `${edge.to.node}:${edge.to.port}`))
  for (const node of state.nodes) {
    const kind = kindOf(index, node.kind)
    if (!kind) continue
    for (const port of kind.inputs) {
      if (port.required && !connected.has(`${node.id}:${port.name}`)) {
        issues.push(
          issue('missing_port', `节点 ${node.id} 的必需输入端口 ${port.name} 没有连接`, `nodes/${node.id}/inputs/${port.name}`, node.id),
        )
      }
    }
  }
  return issues
}

function validateRoles(state: EditorState, index: CatalogIndex): WorkflowIssue[] {
  const issues: WorkflowIssue[] = []
  const byKind = new Map<string, WorkflowNode[]>()
  for (const node of state.nodes) {
    const list = byKind.get(node.kind) ?? []
    list.push(node)
    byKind.set(node.kind, list)
  }
  for (const kind of REQUIRED_ROLES) {
    if (!(byKind.get(kind) ?? []).length) {
      issues.push(issue('missing_required_node', `图缺少必需的 ${kind} 节点`, 'nodes'))
    }
  }
  for (const kind of [...UNIQUE_ROLES, ...OPTIONAL_UNIQUE_ROLES]) {
    const list = byKind.get(kind) ?? []
    if (list.length > 1) {
      issues.push(issue('duplicate_role', `图中有 ${list.length} 个 ${kind} 节点, 最多允许 1 个`, 'nodes'))
    }
  }
  const trainers = byKind.get('trainer') ?? []
  if (trainers.length === 1) {
    const hasVal = state.edges.some((edge) => edge.to.node === trainers[0].id && edge.to.port === 'val_loader')
    if (!hasVal) {
      issues.push({
        code: 'missing_port',
        level: 'warning',
        message: `训练循环 ${trainers[0].id} 没有 val_loader 输入, 训练过程中不会评估`,
        path: `nodes/${trainers[0].id}/inputs/val_loader`,
        node_id: trainers[0].id,
      })
    }
  }
  return issues
}

function validateConnectivity(state: EditorState, index: CatalogIndex): WorkflowIssue[] {
  const trainers = state.nodes.filter((node) => node.kind === 'trainer')
  if (trainers.length !== 1) return []
  const adjacency = new Map<string, Set<string>>()
  for (const node of state.nodes) adjacency.set(node.id, new Set())
  for (const edge of state.edges) {
    adjacency.get(edge.from.node)?.add(edge.to.node)
    adjacency.get(edge.to.node)?.add(edge.from.node)
  }
  const seen = new Set<string>()
  const stack = [trainers[0].id]
  while (stack.length) {
    const current = stack.pop() as string
    if (seen.has(current)) continue
    seen.add(current)
    for (const next of adjacency.get(current) ?? []) stack.push(next)
  }
  const issues: WorkflowIssue[] = []
  for (const node of state.nodes) {
    if (seen.has(node.id)) continue
    const label = kindOf(index, node.kind)?.label ?? node.kind
    issues.push(issue('unreachable_node', `节点 ${node.id} (${label}) 与训练循环不连通, 不会影响训练`, `nodes/${node.id}`, node.id))
  }
  return issues
}

function validateOutputsConsumed(state: EditorState, index: CatalogIndex): WorkflowIssue[] {
  const consumed = new Set(state.edges.map((edge) => `${edge.from.node}:${edge.from.port}`))
  const trainers = new Set(state.nodes.filter((node) => node.kind === 'trainer').map((node) => node.id))
  const issues: WorkflowIssue[] = []
  for (const node of state.nodes) {
    const kind = kindOf(index, node.kind)
    if (!kind) continue
    for (const port of kind.outputs) {
      if (consumed.has(`${node.id}:${port.name}`)) continue
      if (trainers.has(node.id) && port.name === 'run') continue
      issues.push(
        issue(
          'dangling_output',
          `节点 ${node.id} 的输出端口 ${port.name} 没有消费方; dag-runtime 要求显式消费或接 sink 节点`,
          `nodes/${node.id}/outputs/${port.name}`,
          node.id,
        ),
      )
    }
  }
  return issues
}

/** 返回处于环上的节点 ID; 无环时返回空数组. */
export function detectCycle(nodes: WorkflowNode[], edges: WorkflowEdge[]): string[] {
  const indegree = new Map<string, number>()
  const outgoing = new Map<string, string[]>()
  for (const node of nodes) {
    indegree.set(node.id, 0)
    outgoing.set(node.id, [])
  }
  for (const edge of edges) {
    if (!indegree.has(edge.to.node) || !outgoing.has(edge.from.node)) continue
    indegree.set(edge.to.node, (indegree.get(edge.to.node) ?? 0) + 1)
    outgoing.get(edge.from.node)?.push(edge.to.node)
  }
  const ready = [...indegree.entries()].filter(([, degree]) => degree === 0).map(([id]) => id)
  const visited = new Set<string>()
  while (ready.length) {
    const current = ready.pop() as string
    if (visited.has(current)) continue
    visited.add(current)
    for (const next of outgoing.get(current) ?? []) {
      const degree = (indegree.get(next) ?? 0) - 1
      indegree.set(next, degree)
      if (degree === 0) ready.push(next)
    }
  }
  return nodes.map((node) => node.id).filter((id) => !visited.has(id))
}

/** 稳定拓扑序; 有环时把剩余节点按 ID 追加在末尾. */
export function topologicalLayers(nodes: WorkflowNode[], edges: WorkflowEdge[]): string[][] {
  const indegree = new Map<string, number>()
  const outgoing = new Map<string, string[]>()
  for (const node of nodes) {
    indegree.set(node.id, 0)
    outgoing.set(node.id, [])
  }
  for (const edge of edges) {
    if (!indegree.has(edge.to.node) || !outgoing.has(edge.from.node)) continue
    indegree.set(edge.to.node, (indegree.get(edge.to.node) ?? 0) + 1)
    outgoing.get(edge.from.node)?.push(edge.to.node)
  }
  let ready = [...indegree.entries()].filter(([, degree]) => degree === 0).map(([id]) => id).sort()
  const layers: string[][] = []
  const seen = new Set<string>()
  while (ready.length) {
    layers.push([...ready])
    const next: string[] = []
    for (const current of ready) {
      seen.add(current)
      for (const target of outgoing.get(current) ?? []) {
        const degree = (indegree.get(target) ?? 0) - 1
        indegree.set(target, degree)
        if (degree === 0) next.push(target)
      }
    }
    ready = next.sort()
  }
  const leftover = nodes.map((node) => node.id).filter((id) => !seen.has(id))
  if (leftover.length) layers.push(leftover.sort())
  return layers
}

export function issuesBySeverity(issues: WorkflowIssue[]): {
  errors: WorkflowIssue[]
  warnings: WorkflowIssue[]
} {
  return {
    errors: issues.filter((item) => item.level === 'error'),
    warnings: issues.filter((item) => item.level === 'warning'),
  }
}

export function nodesWithIssues(issues: WorkflowIssue[]): Set<string> {
  const ids = new Set<string>()
  for (const item of issues) {
    if (item.level === 'error' && item.node_id) ids.add(item.node_id)
  }
  return ids
}

function typesCompatible(source: string, target: string): boolean {
  return source === 'any' || target === 'any' || source === target
}

function issue(
  code: string,
  message: string,
  path?: string,
  nodeId?: string,
  edgeId?: string,
): WorkflowIssue {
  return {
    code,
    level: 'error',
    message,
    ...(path ? { path } : {}),
    ...(nodeId ? { node_id: nodeId } : {}),
    ...(edgeId ? { edge_id: edgeId } : {}),
  }
}
