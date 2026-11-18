/**
 * 节点目录: 前端唯一的节点语义来源.
 *
 * 目录由 `GET /api/workflows/catalog` 原样返回 (来自 xdl/workflow/node_catalog.json),
 * 前端不硬编码第二份 target 或端口列表.
 */

import type { Params, PortRef, WorkflowEdge, WorkflowNode, JsonValue } from './types'

export type ParamType =
  | 'integer'
  | 'number'
  | 'boolean'
  | 'string'
  | 'enum'
  | 'list_integer'
  | 'list_number'
  | 'data_path'

export type ParamSpec = {
  name: string
  type: ParamType
  label: string
  required: boolean
  default?: JsonValue
  min?: number | null
  max?: number | null
  options?: string[]
}

export type PortSpec = {
  name: string
  type: string
  required: boolean
  multi?: boolean
}

export type TargetSpec = {
  target: string
  label: string
  summary: string
  params: ParamSpec[]
}

export type KindSpec = {
  kind: string
  label: string
  group: string
  summary: string
  profiles: string[]
  inputs: PortSpec[]
  outputs: PortSpec[]
  targets: TargetSpec[]
}

export type CatalogLimits = {
  max_nodes: number
  max_edges: number
  max_params_depth: number
  max_params_bytes: number
  max_snapshot_bytes: number
  max_name_length: number
  max_id_length: number
}

export type NodeCatalog = {
  version: number
  catalog_version: string
  limits: CatalogLimits
  profiles: Record<string, string>
  port_types: Record<string, string>
  kinds: KindSpec[]
  step_kinds?: KindSpec[]
}

export type CatalogIndex = {
  catalog: NodeCatalog
  kinds: Map<string, KindSpec>
  kindNames: string[]
}

const KIND_COLORS: Record<string, string> = {
  dataset: '#3f6ad1',
  transform: '#2f8a9b',
  dataloader: '#607db8',
  model: '#6072d0',
  loss: '#c2772f',
  optimizer: '#a06bc0',
  scheduler: '#8b6bb0',
  metric: '#4a9b7a',
  trainer: '#5c83ad',
  checkpoint: '#718194',
  identity: '#7a6f5c',
  compose: '#9b7f4a',
  merge: '#6b8f5c',
  sink: '#6d7784',
}

const KIND_ICONS: Record<string, string> = {
  dataset: 'D',
  transform: 'T',
  dataloader: 'B',
  model: 'M',
  loss: 'L',
  optimizer: 'O',
  scheduler: 'S',
  metric: 'E',
  trainer: 'R',
  checkpoint: 'C',
  identity: 'I',
  compose: 'X',
  merge: 'G',
  sink: 'K',
  dataset_split: 'S', collate: 'C',
  BatchInput: 'B', Select: '⋯', Forward: 'F', LossCall: 'L',
  MetricCall: 'M', WeightedSum: 'Σ', StepOutput: '→',
}

export function indexCatalog(catalog: NodeCatalog): CatalogIndex {
  const kinds = new Map<string, KindSpec>()
  for (const kind of catalog.kinds) {
    kinds.set(kind.kind, kind)
  }
  return { catalog, kinds, kindNames: catalog.kinds.map((item) => item.kind) }
}

export function indexStepCatalog(index: CatalogIndex): CatalogIndex {
  return indexCatalog({ ...index.catalog, kinds: index.catalog.step_kinds ?? [] })
}

export function kindOf(index: CatalogIndex, kind: string): KindSpec | undefined {
  return index.kinds.get(kind)
}

export function targetOf(
  index: CatalogIndex,
  kind: string,
  target: string,
): TargetSpec | undefined {
  return kindOf(index, kind)?.targets.find((item) => item.target === target)
}

export function kindColor(kind: string): string {
  return KIND_COLORS[kind] ?? '#607080'
}

export function kindIcon(kind: string): string {
  return KIND_ICONS[kind] ?? '?'
}

export function nodeLabel(node: WorkflowNode, index: CatalogIndex): string {
  const label = node.metadata?.label
  if (typeof label === 'string' && label.trim()) return label
  const target = targetOf(index, node.kind, node.target)
  return target?.label ?? kindOf(index, node.kind)?.label ?? node.kind
}

export function inputPorts(index: CatalogIndex, kind: string): PortSpec[] {
  return kindOf(index, kind)?.inputs ?? []
}

export function outputPorts(index: CatalogIndex, kind: string): PortSpec[] {
  return kindOf(index, kind)?.outputs ?? []
}

export function findPort(
  index: CatalogIndex,
  kind: string,
  port: string,
  direction: 'in' | 'out',
): PortSpec | undefined {
  const ports = direction === 'in' ? inputPorts(index, kind) : outputPorts(index, kind)
  return ports.find((item) => item.name === port)
}

/** 按目录默认值构造节点参数, 只填充有默认值的声明. */
export function defaultParams(target: TargetSpec | undefined): Params {
  const params: Params = {}
  for (const spec of target?.params ?? []) {
    if (spec.default !== undefined && spec.default !== null) {
      params[spec.name] = spec.default
    }
  }
  return params
}

/** 参数值是否满足目录声明; 返回错误文案或 null. */
export function checkParam(spec: ParamSpec, value: JsonValue): string | null {
  if (value === null || value === undefined || value === '') {
    return spec.required ? '必填' : null
  }
  switch (spec.type) {
    case 'integer':
      if (typeof value !== 'number' || !Number.isInteger(value)) return '必须是整数'
      return checkRange(spec, value)
    case 'number':
      if (typeof value !== 'number' || !Number.isFinite(value)) return '必须是数字'
      return checkRange(spec, value)
    case 'boolean':
      return typeof value === 'boolean' ? null : '必须是布尔值'
    case 'string':
      return typeof value === 'string' ? null : '必须是字符串'
    case 'enum':
      return typeof value === 'string' && (spec.options ?? []).includes(value)
        ? null
        : `必须是 ${(spec.options ?? []).join(' / ')} 之一`
    case 'list_integer':
      return Array.isArray(value) && value.every((item) => typeof item === 'number' && Number.isInteger(item))
        ? null
        : '必须是整数数组'
    case 'list_number':
      return Array.isArray(value) && value.every((item) => typeof item === 'number' && Number.isFinite(item))
        ? null
        : '必须是数字数组'
    case 'data_path':
      if (typeof value !== 'string') return '必须是相对路径'
      if (value.startsWith('/') || /^[A-Za-z]:/.test(value) || value.split('/').includes('..')) {
        return '必须是 data 目录内的相对路径'
      }
      return null
    default:
      return '未知参数类型'
  }
}

function checkRange(spec: ParamSpec, value: number): string | null {
  if (typeof spec.min === 'number' && value < spec.min) return `必须 >= ${spec.min}`
  if (typeof spec.max === 'number' && value > spec.max) return `必须 <= ${spec.max}`
  return null
}

let idCounter = 0

export function newNodeId(kind: string): string {
  idCounter += 1
  return `${kind}-${Date.now().toString(36)}-${idCounter}`
}

export function newEdgeId(): string {
  idCounter += 1
  return `e-${Date.now().toString(36)}-${idCounter}`
}

export type StarterGraph = {
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
}

type StarterSpec = {
  id: string
  kind: string
  target: string
  params?: Params
  position: { x: number; y: number }
  label: string
}

/** 用目录默认值构造一张可直接训练的标准 schema-v1 图. */
export function buildStarterGraph(index: CatalogIndex): StarterGraph {
  const pick = (kind: string, preferred: string): TargetSpec | undefined => {
    const spec = kindOf(index, kind)
    if (!spec) return undefined
    return spec.targets.find((item) => item.target === preferred) ?? spec.targets[0]
  }
  const dataset = pick('dataset', 'torchvision.datasets:FakeData')
  const dataloader = pick('dataloader', 'builtin:DataLoader')
  const model = pick('model', 'registry:vgg11_bn')
  const loss = pick('loss', 'torch.nn:CrossEntropyLoss')
  const optimizer = pick('optimizer', 'torch.optim:SGD')
  const trainer = pick('trainer', 'builtin:Trainer')
  if (!dataset || !dataloader || !model || !loss || !optimizer || !trainer) {
    return { nodes: [], edges: [] }
  }
  const specs: StarterSpec[] = [
    { id: 'transform', kind: 'transform', target: 'torchvision.transforms:ToTensor', position: { x: 40, y: 70 }, label: '转张量' },
    { id: 'dataset', kind: 'dataset', target: dataset.target, params: defaultParams(dataset), position: { x: 300, y: 70 }, label: '数据集' },
    { id: 'loader', kind: 'dataloader', target: dataloader.target, params: defaultParams(dataloader), position: { x: 560, y: 70 }, label: '批次加载' },
    { id: 'model', kind: 'model', target: model.target, params: defaultParams(model), position: { x: 40, y: 250 }, label: '模型' },
    { id: 'loss', kind: 'loss', target: loss.target, params: defaultParams(loss), position: { x: 40, y: 430 }, label: '损失函数' },
    { id: 'optimizer', kind: 'optimizer', target: optimizer.target, params: defaultParams(optimizer), position: { x: 300, y: 250 }, label: '优化器' },
    { id: 'trainer', kind: 'trainer', target: trainer.target, params: defaultParams(trainer), position: { x: 840, y: 150 }, label: '训练循环' },
  ]
  const nodes: WorkflowNode[] = specs.map((spec) => ({
    id: spec.id,
    kind: spec.kind,
    target: spec.target,
    params: spec.params ?? {},
    position: spec.position,
    metadata: { label: spec.label },
  }))
  const edge = (id: string, from: PortRef, to: PortRef): WorkflowEdge => ({ id, from, to })
  const edges: WorkflowEdge[] = [
    edge('e-transform', { node: 'transform', port: 'pipeline' }, { node: 'dataset', port: 'transform' }),
    edge('e-dataset', { node: 'dataset', port: 'samples' }, { node: 'loader', port: 'dataset' }),
    edge('e-loader', { node: 'loader', port: 'batches' }, { node: 'trainer', port: 'train_loader' }),
    edge('e-model', { node: 'model', port: 'module' }, { node: 'trainer', port: 'model' }),
    edge('e-model-optimizer', { node: 'model', port: 'module' }, { node: 'optimizer', port: 'model' }),
    edge('e-optimizer', { node: 'optimizer', port: 'optimizer' }, { node: 'trainer', port: 'optimizer' }),
    edge('e-loss', { node: 'loss', port: 'loss' }, { node: 'trainer', port: 'loss' }),
  ]
  return { nodes, edges }
}
