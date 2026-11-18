import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks'
import {
  AlertCircle,
  Box,
  Check,
  CircleDot,
  GitBranch,
  Layers3,
  Play,
  Plus,
  Save,
  Search,
  Settings2,
  Trash2,
  WandSparkles,
  X,
} from 'lucide-preact'
import { api, Run, SystemSnapshot, WorkflowRequestError } from './api'
import {
  CatalogIndex,
  ParamSpec,
  checkParam,
  defaultParams,
  indexCatalog,
  indexStepCatalog,
  inputPorts,
  kindColor,
  kindIcon,
  kindOf,
  newEdgeId,
  newNodeId,
  nodeLabel,
  outputPorts,
  targetOf,
} from './workflow/catalog'
import { issuesBySeverity, nodesWithIssues, topologicalLayers, validateLocally } from './workflow/validation'
import {
  DRAFT_STORAGE_KEY,
  emptyState,
  parseDraft,
  serializeDraft,
  starterState,
  toSnapshot,
} from './workflow/serialization'
import type {
  EditorState,
  Graph,
  JsonValue,
  WorkflowIssue,
  WorkflowNode,
  WorkflowRecord,
} from './workflow/types'

type WorkflowEditorProps = {
  gpus: NonNullable<SystemSnapshot['gpus']>
  onQueued: (run: Run) => void
  onNotice: (message: string) => void
}

type Panel = 'library' | 'settings'
type Phase = 'draft' | 'checking' | 'queued' | 'failed'

type DragState = { id: string; offsetX: number; offsetY: number }
type LinkState = { node: string; port: string; x: number; y: number }

const NODE_WIDTH = 196
const HEAD_HEIGHT = 44
const PORT_ROW = 20
const PORT_TOP = HEAD_HEIGHT + 44
const CANVAS_PADDING = 60
const EMPTY_GRAPH: Graph = { nodes: [], edges: [] }
type GraphLayer = 'assembly' | 'step'

function portOffset(list: { name: string }[], name: string): number {
  const index = list.findIndex((item) => item.name === name)
  return PORT_TOP + (index < 0 ? 0 : index) * PORT_ROW
}

function nodeHeight(index: CatalogIndex, kind: string): number {
  const rows = Math.max(inputPorts(index, kind).length, outputPorts(index, kind).length, 1)
  return PORT_TOP + rows * PORT_ROW + 12
}

export function WorkflowEditor({ gpus, onQueued, onNotice }: WorkflowEditorProps) {
  const [assemblyIndex, setIndex] = useState<CatalogIndex | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [state, setState] = useState<EditorState>(emptyState)
  const [layer, setLayer] = useState<GraphLayer>('assembly')
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null)
  const [panel, setPanel] = useState<Panel>('library')
  const [query, setQuery] = useState('')
  const [phase, setPhase] = useState<Phase>('draft')
  const [preflightOpen, setPreflightOpen] = useState(true)
  const [serverIssues, setServerIssues] = useState<WorkflowIssue[] | null>(null)
  const [history, setHistory] = useState<WorkflowRecord[]>([])
  const [link, setLink] = useState<LinkState | null>(null)
  const drag = useRef<DragState | null>(null)
  const canvasRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    api.workflowCatalog()
      .then((response) => {
        if (cancelled) return
        const catalogIndex = indexCatalog(response.catalog)
        setIndex(catalogIndex)
        const draft = parseDraft(window.localStorage.getItem(DRAFT_STORAGE_KEY), catalogIndex)
        setState(draft ?? starterState(catalogIndex))
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setCatalogError(error instanceof Error ? error.message : '节点目录加载失败')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    api.workflows()
      .then((response) => setHistory(response.workflows))
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!assemblyIndex) return
    window.localStorage.setItem(DRAFT_STORAGE_KEY, serializeDraft(state))
  }, [assemblyIndex, state])

  const stepIndex = useMemo(() => assemblyIndex && indexStepCatalog(assemblyIndex), [assemblyIndex])
  const index = layer === 'step' ? stepIndex : assemblyIndex
  const graph = layer === 'step' ? state.stepGraph ?? EMPTY_GRAPH : state

  const switchLayer = useCallback((next: GraphLayer) => {
    setLayer(next)
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
    setQuery('')
    setLink(null)
    drag.current = null
  }, [])

  const localIssues = useMemo(() => assemblyIndex ? validateLocally(state, assemblyIndex) : [], [assemblyIndex, state])
  const issues = serverIssues ?? localIssues
  const { errors, warnings } = issuesBySeverity(issues)
  const flagged = nodesWithIssues(errors.filter((issue) => issue.path?.split("/")[0] === "step_graph" ? layer === "step" : layer === "assembly"))
  const selectedNode = graph.nodes.find((node) => node.id === selectedNodeId) ?? null
  const linkedCount = graph.edges.length

  const update = useCallback((updater: (current: EditorState) => EditorState) => {
    setState((current) => updater(current))
    setServerIssues(null)
  }, [])

  const updateGraph = useCallback((updater: (current: Graph) => Graph) => {
    update((current) => layer === 'step'
      ? { ...current, stepGraph: updater(current.stepGraph ?? EMPTY_GRAPH), executionProfile: 'dag-runtime' }
      : { ...current, ...updater({ nodes: current.nodes, edges: current.edges }) })
  }, [layer, update])

  const addNode = useCallback(
    (kind: string) => {
      if (!index) return
      const kindSpec = kindOf(index, kind)
      const target = kindSpec?.targets[0]
      if (!kindSpec || !target) return
      const node: WorkflowNode = {
        id: newNodeId(kind),
        kind,
        target: target.target,
        params: defaultParams(target),
        position: nextFreePosition(graph.nodes),
        metadata: { label: target.label },
      }
      updateGraph((current) => ({ ...current, nodes: [...current.nodes, node] }))
      setSelectedNodeId(node.id)
      setSelectedEdgeId(null)
      setPanel('settings')
      onNotice(`已添加 ${target.label} 节点`)
    },
    [index, onNotice, graph.nodes, updateGraph],
  )

  const deleteNode = useCallback(
    (id: string) => {
      updateGraph((current) => ({
        ...current,
        nodes: current.nodes.filter((node) => node.id !== id),
        edges: current.edges.filter((edge) => edge.from.node !== id && edge.to.node !== id),
      }))
      setSelectedNodeId(null)
      onNotice('节点已删除, 相关连线已清理')
    },
    [onNotice, updateGraph],
  )

  const duplicateNode = useCallback(
    (id: string) => {
      const source = graph.nodes.find((node) => node.id === id)
      if (!source) return
      const copy: WorkflowNode = {
        ...source,
        id: newNodeId(source.kind),
        params: { ...source.params },
        position: { x: source.position.x + 24, y: source.position.y + 24 },
        metadata: source.metadata ? { ...source.metadata } : undefined,
      }
      updateGraph((current) => ({ ...current, nodes: [...current.nodes, copy] }))
      setSelectedNodeId(copy.id)
      onNotice('节点已复制, 参数一并复制')
    },
    [onNotice, graph.nodes, updateGraph],
  )

  const patchNode = useCallback(
    (id: string, patch: Partial<WorkflowNode>) => {
      updateGraph((current) => ({
        ...current,
        nodes: current.nodes.map((node) => (node.id === id ? { ...node, ...patch } : node)),
      }))
    },
    [updateGraph],
  )

  const setParam = useCallback(
    (id: string, name: string, value: JsonValue) => {
      updateGraph((current) => ({
        ...current,
        nodes: current.nodes.map((node) =>
          node.id === id ? { ...node, params: changedParams(node, name, value) } : node,
        ),
      }))
    },
    [updateGraph],
  )

  const deleteEdge = useCallback(
    (id: string) => {
      updateGraph((current) => ({ ...current, edges: current.edges.filter((edge) => edge.id !== id) }))
      setSelectedEdgeId(null)
      onNotice('连线已删除')
    },
    [onNotice, updateGraph],
  )

  const connect = useCallback(
    (from: { node: string; port: string }, to: { node: string; port: string }) => {
      if (!index) return
      if (from.node === to.node) {
        onNotice('不能连接节点自身')
        return
      }
      const source = graph.nodes.find((node) => node.id === from.node)
      const target = graph.nodes.find((node) => node.id === to.node)
      if (!source || !target) return
      const sourcePort = outputPorts(index, source.kind).find((port) => port.name === from.port)
      const targetPort = inputPorts(index, target.kind).find((port) => port.name === to.port)
      if (!sourcePort || !targetPort) return
      if (!typesCompatible(sourcePort.type, targetPort.type)) {
        onNotice(`端口类型不兼容: ${sourcePort.type} 不能连到 ${targetPort.type}`)
        return
      }
      const occupied = graph.edges.find(
        (edge) => edge.to.node === to.node && edge.to.port === to.port,
      )
      if (occupied && !targetPort.multi) {
        onNotice(`输入端口 ${targetPort.name} 已有来源, 请先断开`)
        return
      }
      const duplicate = graph.edges.some(
        (edge) =>
          edge.from.node === from.node &&
          edge.from.port === from.port &&
          edge.to.node === to.node &&
          edge.to.port === to.port,
      )
      if (duplicate) return
      updateGraph((current) => ({
        ...current,
        edges: [...current.edges, { id: newEdgeId(), from, to }],
      }))
    },
    [index, onNotice, graph.edges, graph.nodes, updateGraph],
  )

  const autoLayout = useCallback(() => {
    if (!index) return
    const layers = topologicalLayers(graph.nodes, graph.edges)
    const positions = new Map<string, { x: number; y: number }>()
    layers.forEach((layer, column) => {
      layer.forEach((id, row) => {
        positions.set(id, {
          x: CANVAS_PADDING + column * (NODE_WIDTH + 90),
          y: CANVAS_PADDING + row * 150,
        })
      })
    })
    updateGraph((current) => ({
      ...current,
      nodes: current.nodes.map((node) => ({
        ...node,
        position: positions.get(node.id) ?? node.position,
      })),
    }))
    onNotice('已按拓扑顺序自动排布节点')
  }, [index, onNotice, graph.edges, graph.nodes, updateGraph])

  const clearCanvas = useCallback(() => {
    updateGraph((current) => ({ ...current, nodes: [], edges: [] }))
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
    setPreflightOpen(true)
    onNotice('画布已清空')
  }, [onNotice, updateGraph])

  const loadStarter = useCallback(() => {
    if (!assemblyIndex) return
    if (layer === 'step' && stepIndex) {
      updateGraph(() => starterStepGraph(stepIndex, state))
    } else {
      setState(starterState(assemblyIndex))
    }
    setSelectedNodeId(null)
    setSelectedEdgeId(null)
    setServerIssues(null)
    onNotice(layer === 'step' ? '已插入分类训练步' : '已插入标准训练图')
  }, [assemblyIndex, layer, stepIndex, state, updateGraph, onNotice])

  const loadSavedWorkflow = useCallback(
    async (id: string) => {
      if (!index) return
      try {
        const record = await api.workflow(id)
        if (!record.snapshot) {
          onNotice('该 workflow 记录没有快照内容')
          return
        }
        setState(fromSnapshot(record))
        switchLayer('assembly')
        setServerIssues(null)
        onNotice(`已载入 workflow ${record.name}`)
      } catch (error) {
        onNotice(error instanceof Error ? error.message : '载入 workflow 失败')
      }
    },
    [index, onNotice, switchLayer],
  )

  const checkAndQueue = useCallback(async () => {
    if (!index) return
    setPreflightOpen(true)
    setPhase('checking')
    const snapshot = toSnapshot(state)
    try {
      const validated = await api.validateWorkflow(snapshot)
      if (!validated.valid) {
        setServerIssues(validated.issues)
        setPhase('failed')
        onNotice('预检未通过, 请先处理阻断错误')
        return
      }
      setServerIssues(validated.issues)
      const record = await api.createWorkflow(snapshot)
      const run = await api.submitWorkflow(record.id, {
        name: snapshot.name,
        launcher: state.settings.launcher ?? 'none',
        devices: state.settings.devices ?? [],
      })
      setPhase('queued')
      setHistory((current) => [record, ...current].slice(0, 50))
      onQueued(run)
      onNotice('工作流已通过预检并加入训练队列')
    } catch (error) {
      setPhase('failed')
      if (error instanceof WorkflowRequestError) {
        setServerIssues(error.issues)
      }
      onNotice(error instanceof Error ? error.message : '工作流加入队列失败')
    }
  }, [index, onNotice, onQueued, state])

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setLink(null)
        return
      }
      if (event.key !== 'Delete' && event.key !== 'Backspace') return
      const target = event.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (selectedNodeId) {
        event.preventDefault()
        deleteNode(selectedNodeId)
        return
      }
      if (selectedEdgeId) {
        event.preventDefault()
        deleteEdge(selectedEdgeId)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [deleteEdge, deleteNode, selectedEdgeId, selectedNodeId])

  const onNodePointerDown = (event: PointerEvent, node: WorkflowNode) => {
    const target = event.target as HTMLElement
    if (target.closest('button, input, select, [data-port]')) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    drag.current = {
      id: node.id,
      offsetX: event.clientX - rect.left - node.position.x,
      offsetY: event.clientY - rect.top - node.position.y,
    }
    setSelectedNodeId(node.id)
    setSelectedEdgeId(null)
    ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
  }

  const onCanvasPointerMove = (event: PointerEvent) => {
    const current = drag.current
    const canvas = canvasRef.current
    if (current && canvas) {
      const rect = canvas.getBoundingClientRect()
      const x = Math.max(0, event.clientX - rect.left - current.offsetX)
      const y = Math.max(0, event.clientY - rect.top - current.offsetY)
      updateGraph((currentState) => ({
        ...currentState,
        nodes: currentState.nodes.map((node) =>
          node.id === current.id ? { ...node, position: { x, y } } : node,
        ),
      }))
      return
    }
    if (link && canvas) {
      const rect = canvas.getBoundingClientRect()
      setLink({ ...link, x: event.clientX - rect.left, y: event.clientY - rect.top })
    }
  }

  const onCanvasPointerUp = (event: PointerEvent) => {
    drag.current = null
    if (!link) return
    const pending = link
    setLink(null)
    const element = document.elementFromPoint(event.clientX, event.clientY) as HTMLElement | null
    const portElement = element?.closest('[data-port-direction="in"]') as HTMLElement | null
    if (!portElement) return
    const node = portElement.dataset.node
    const port = portElement.dataset.port
    if (!node || !port) return
    connect({ node: pending.node, port: pending.port }, { node, port })
  }

  const startLink = (event: PointerEvent, node: WorkflowNode, port: string) => {
    event.stopPropagation()
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    setLink({
      node: node.id,
      port,
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    })
  }

  const canvasSize = useMemo(() => {
    const maxX = graph.nodes.reduce((value, node) => Math.max(value, node.position.x), 0)
    const maxY = graph.nodes.reduce(
      (value, node) => Math.max(value, node.position.y + (index ? nodeHeight(index, node.kind) : 120)),
      0,
    )
    return {
      width: Math.max(1180, maxX + NODE_WIDTH + CANVAS_PADDING),
      height: Math.max(560, maxY + CANVAS_PADDING),
    }
  }, [index, graph.nodes])

  if (catalogError) {
    return (
      <section className="workflow-editor">
        <div className="workflow-stage">
          <div className="workflow-preflight error">
            <AlertCircle size={14} />
            <b>节点目录不可用</b>
            <span>{catalogError}</span>
          </div>
          <div className="workflow-empty">
            <div className="workflow-empty-mark">
              <AlertCircle size={22} />
            </div>
            <strong>无法编辑工作流</strong>
            <p>节点目录来自 xdl/workflow/node_catalog.json, 请确认 dashboard 后端已正确加载。</p>
          </div>
        </div>
      </section>
    )
  }

  if (!index) {
    return (
      <section className="workflow-editor">
        <div className="workflow-stage">
          <div className="workflow-preflight">
            <CircleDot size={14} />
            <b>加载节点目录…</b>
          </div>
        </div>
      </section>
    )
  }

  const visibleKinds = index.catalog.kinds.filter((kind) =>
    `${kind.label} ${kind.summary} ${kind.kind}`.toLowerCase().includes(query.toLowerCase()),
  )
  const starterLabel = layer === 'step' ? '插入分类训练步' : '插入标准训练图'
  const groups = Array.from(new Set(index.catalog.kinds.map((kind) => kind.group)))
  const phaseLabels: Record<Phase, string> = {
    draft: '草稿',
    checking: '预检中',
    queued: '已排队',
    failed: '预检失败',
  }

  return (
    <section className="workflow-editor">
      <div className="workflow-rail">
        <span className="workflow-logo">X</span>
        <button className="selected" type="button" title="节点画布">
          <GitBranch size={16} />
        </button>
        <button type="button" title="实验">
          <Layers3 size={16} />
        </button>
        <button type="button" title="产物">
          <Box size={16} />
        </button>
        <span className="workflow-rail-rule" />
        <button type="button" title="设置">
          <Settings2 size={16} />
        </button>
      </div>

      <aside className="workflow-sidebar">
        <div className="workflow-sidebar-head">
          <div>
            <h2>工作流构建器</h2>
            <small>
              {graph.nodes.length
                ? `${graph.nodes.length} nodes · ${linkedCount} edges`
                : 'empty canvas'}
            </small>
          </div>
          <button className="workflow-clear" type="button" onClick={clearCanvas}>
            清空
          </button>
        </div>
        <div className="workflow-tabs" role="group" aria-label="画布层">
          <button type="button" aria-pressed={layer === 'assembly'} onClick={() => switchLayer('assembly')}>
            装配图
          </button>
          <button type="button" aria-pressed={layer === 'step'} onClick={() => switchLayer('step')}
            disabled={!stepIndex?.kindNames.length}>
            训练步{state.stepGraph ? ` · ${state.stepGraph.nodes.length}` : ''}
          </button>
        </div>
        <div className="workflow-tabs">
          <button type="button" aria-pressed={panel === 'library'} onClick={() => setPanel('library')}>
            节点库
          </button>
          <button type="button" aria-pressed={panel === 'settings'} onClick={() => setPanel('settings')}>
            流程设置
          </button>
        </div>
        {panel === 'library' ? (
          <>
            <label className="workflow-search">
              <Search size={14} />
              <input
                placeholder="搜索节点类型"
                value={query}
                onInput={(event) => setQuery((event.currentTarget as HTMLInputElement).value)}
              />
            </label>
            {groups.map((group) => {
              const items = visibleKinds.filter((kind) => kind.group === group)
              if (!items.length) return null
              return (
                <section className="workflow-library-group" key={group}>
                  <span>{group}</span>
                  {items.map((kind) => (
                    <button
                      className="workflow-library-item"
                      type="button"
                      key={kind.kind}
                      style={{ '--node-color': kindColor(kind.kind) } as Record<string, string>}
                      title={`${kind.summary} (${kind.profiles.join(' / ')})`}
                      onClick={() => addNode(kind.kind)}
                    >
                      <i>{kindIcon(kind.kind)}</i>
                      <span>
                        <b>{kind.label}</b>
                        <small>{kind.summary}</small>
                      </span>
                      <Plus size={14} />
                    </button>
                  ))}
                </section>
              )
            })}
            <button className="workflow-library-action" type="button" onClick={loadStarter}>
              <WandSparkles size={14} /> {starterLabel}
            </button>
            <p className="workflow-hint">
              点击添加节点, 拖动节点排布. 从右侧输出端口拖到目标输入端口即可连线, 点击连线可删除.
            </p>
          </>
        ) : (
          <WorkflowSettings
            state={state}
            history={history}
            gpus={gpus}
            onPatch={(patch) => update((current) => ({ ...current, ...patch }))}
            onLoad={loadSavedWorkflow}
          />
        )}
      </aside>

      <main className="workflow-stage">
        <header className="workflow-toolbar">
          <div className="workflow-title">
            <input
              value={state.name}
              aria-label="流水线名称"
              onInput={(event) =>
                update((current) => ({
                  ...current,
                  name: (event.currentTarget as HTMLInputElement).value,
                }))
              }
            />
            <small>
              {phaseLabels[phase]} · {state.executionProfile}
            </small>
          </div>
          <div className="workflow-state-pills">
            <span>PROFILE</span>
            <button
              type="button"
              aria-pressed={state.executionProfile === 'schema-v1'}
              disabled={Boolean(state.stepGraph)}
              onClick={() => update((current) => ({ ...current, executionProfile: 'schema-v1' }))}
            >
              schema-v1
            </button>
            <button
              type="button"
              aria-pressed={state.executionProfile === 'dag-runtime'}
              onClick={() => update((current) => ({ ...current, executionProfile: 'dag-runtime' }))}
            >
              dag-runtime
            </button>
          </div>
          <button className="button subtle" type="button" onClick={autoLayout}>
            <WandSparkles size={14} /> 自动排布
          </button>
          <button
            className="button subtle"
            type="button"
            onClick={() => setPreflightOpen((value) => !value)}
          >
            <AlertCircle size={14} /> 预检 {issues.length || <Check size={13} />}
          </button>
          <button
            className="button primary"
            type="button"
            disabled={phase === 'checking'}
            onClick={() => void checkAndQueue()}
          >
            <Play size={14} /> {phase === 'checking' ? '预检中…' : '校验并排队'}
          </button>
        </header>

        <div
          className={`workflow-preflight ${errors.length ? 'error' : warnings.length ? 'warn' : 'ok'}`}
        >
          <span>PREFLIGHT</span>
          <b>
            {errors.length
              ? `${errors.length} 个错误`
              : warnings.length
                ? `${warnings.length} 个提醒`
                : '全部通过'}
          </b>
          <span>·</span>
          <span>
            {graph.nodes.length ? `${graph.nodes.length} 个节点 / ${linkedCount} 条连线` : '尚未开始搭建'}
          </span>
          <button type="button" onClick={() => setPreflightOpen((value) => !value)}>
            {preflightOpen ? '收起详情' : '查看详情'}
          </button>
        </div>

        {preflightOpen && (
          <div className="workflow-preflight-panel">
            {issues.length ? (
              issues.map((item, position) => (
                <button
                  className={`workflow-issue ${item.level}`}
                  type="button"
                  key={`${item.code}-${item.path ?? position}-${position}`}
                  onClick={() => {
                    const next = item.path?.split('/')[0] === 'step_graph' ? 'step' : 'assembly'
                    if (next !== layer) switchLayer(next)
                    if (item.node_id) {
                      setSelectedNodeId(item.node_id)
                      setSelectedEdgeId(null)
                    } else if (item.edge_id) {
                      setSelectedNodeId(null)
                      setSelectedEdgeId(item.edge_id)
                    }
                  }}
                >
                  <i>{item.level === 'error' ? '!' : 'i'}</i>
                  <span>{item.message}</span>
                  {item.path && <em>{item.path}</em>}
                </button>
              ))
            ) : (
              <div className="workflow-issue ok">
                <i>✓</i>
                <span>结构与语义预检通过, 可以加入训练队列.</span>
              </div>
            )}
          </div>
        )}

        {layer === 'step' && state.stepGraph && (
          <div className="workflow-preflight">
            <span>训练与验证复用此计算图</span>
            <button type="button" onClick={() => {
              update((current) => ({ ...current, stepGraph: undefined }))
              switchLayer('assembly')
            }}>停用训练步图</button>
          </div>
        )}
        <div className="workflow-main">
          <section className="workflow-canvas-panel">
            <div className="workflow-canvas-head">
              <div>
                <h3>{layer === 'step' ? '训练步 · 每批执行' : '装配图 · 运行前构建'}</h3>
                <span>
                  {graph.nodes.length} nodes · {linkedCount} edges · {state.executionProfile}
                </span>
              </div>
              <span className="workflow-zoom">拖端口连线 / Delete 删除</span>
            </div>
            <div className="workflow-canvas-scroll">
              <div
                className={`workflow-canvas ${graph.nodes.length ? '' : 'empty'}`}
                ref={canvasRef}
                style={{ width: canvasSize.width, height: canvasSize.height }}
                onPointerMove={onCanvasPointerMove}
                onPointerUp={onCanvasPointerUp}
                onPointerLeave={() => {
                  drag.current = null
                }}
              >
                <svg className="workflow-links" width={canvasSize.width} height={canvasSize.height}>
                  <defs>
                    <marker
                      id="workflow-arrow"
                      viewBox="0 0 8 8"
                      refX="7"
                      refY="4"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto"
                    >
                      <path d="M0 0 L8 4 L0 8 z" fill="#607080" />
                    </marker>
                  </defs>
                  {graph.edges.map((edge) => {
                    const source = graph.nodes.find((node) => node.id === edge.from.node)
                    const target = graph.nodes.find((node) => node.id === edge.to.node)
                    if (!source || !target) return null
                    const sourcePorts = outputPorts(index, source.kind)
                    const targetPorts = inputPorts(index, target.kind)
                    const y1 = source.position.y + portOffset(sourcePorts, edge.from.port)
                    const y2 = target.position.y + portOffset(targetPorts, edge.to.port)
                    const x1 = source.position.x + NODE_WIDTH
                    const x2 = target.position.x
                    const curve = Math.max(40, Math.abs(x2 - x1) * 0.4)
                    const selected = selectedEdgeId === edge.id
                    return (
                      <g key={edge.id}>
                        <path
                          className="workflow-link-hit"
                          d={`M${x1} ${y1} C${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`}
                          onClick={() => deleteEdge(edge.id)}
                        >
                          <title>{`${edge.from.node}.${edge.from.port} → ${edge.to.node}.${edge.to.port} (点击删除)`}</title>
                        </path>
                        <path
                          className={`workflow-link ${selected ? 'selected' : ''}`}
                          d={`M${x1} ${y1} C${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`}
                        />
                      </g>
                    )
                  })}
                  {link && (
                    <path
                      className="workflow-link pending"
                      d={(() => {
                        const source = graph.nodes.find((node) => node.id === link.node)
                        if (!source) return ''
                        const sourcePorts = outputPorts(index, source.kind)
                        const x1 = source.position.x + NODE_WIDTH
                        const y1 = source.position.y + portOffset(sourcePorts, link.port)
                        const curve = Math.max(40, Math.abs(link.x - x1) * 0.4)
                        return `M${x1} ${y1} C${x1 + curve} ${y1}, ${link.x - curve} ${link.y}, ${link.x} ${link.y}`
                      })()}
                    />
                  )}
                </svg>

                {graph.nodes.map((node) => {
                  const kindSpec = kindOf(index, node.kind)
                  if (!kindSpec) return null
                  const color = kindColor(node.kind)
                  const flaggedNode = flagged.has(node.id)
                  return (
                    <article
                      key={node.id}
                      className={`workflow-node ${selectedNodeId === node.id ? 'selected' : ''} ${flaggedNode ? 'invalid' : ''}`}
                      style={{
                        left: node.position.x,
                        top: node.position.y,
                        width: NODE_WIDTH,
                        height: nodeHeight(index, node.kind),
                        '--node-color': color,
                      } as Record<string, string | number>}
                      onClick={() => {
                        setSelectedNodeId(node.id)
                        setSelectedEdgeId(null)
                      }}
                      onPointerDown={(event) =>
                        onNodePointerDown(event as unknown as PointerEvent, node)
                      }
                    >
                      <div className="workflow-node-head">
                        <span className="workflow-node-icon">{kindIcon(node.kind)}</span>
                        <span>{kindSpec.label}</span>
                        <button
                          className="workflow-node-delete"
                          type="button"
                          aria-label={`删除${nodeLabel(node, index)}`}
                          onClick={(event) => {
                            event.stopPropagation()
                            deleteNode(node.id)
                          }}
                        >
                          <X size={13} />
                        </button>
                      </div>
                      <strong>{nodeLabel(node, index)}</strong>
                      <small>{node.target}</small>
                      {kindSpec.inputs.map((port) => {
                        const connected = graph.edges.find(
                          (edge) => edge.to.node === node.id && edge.to.port === port.name,
                        )
                        return (
                          <span
                            key={`in-${port.name}`}
                            className={`workflow-port-row in ${connected ? 'connected' : ''} ${port.required && !connected ? 'missing' : ''}`}
                            style={{ top: portOffset(kindSpec.inputs, port.name) - 7 }}
                          >
                            <i
                              className="workflow-port workflow-port-in"
                              data-port-direction="in"
                              data-node={node.id}
                              data-port={port.name}
                            />
                            <em>{port.name}</em>
                          </span>
                        )
                      })}
                      {kindSpec.outputs.map((port) => (
                        <span
                          key={`out-${port.name}`}
                          className="workflow-port-row out"
                          style={{ top: portOffset(kindSpec.outputs, port.name) - 7 }}
                        >
                          <em>{port.name}</em>
                          <i
                            className="workflow-port workflow-port-out"
                            data-port-direction="out"
                            data-node={node.id}
                            data-port={port.name}
                            onPointerDown={(event) =>
                              startLink(event as unknown as PointerEvent, node, port.name)
                            }
                          />
                        </span>
                      ))}
                    </article>
                  )
                })}

                {!graph.nodes.length && (
                  <div className="workflow-empty">
                    <div className="workflow-empty-mark">
                      <GitBranch size={22} />
                    </div>
                    <strong>空白画布</strong>
                    <p>{layer === 'step' ? '添加节点或插入分类模板。编辑即启用 dag-runtime。' : '从左侧添加节点, 再将输出端口连到下游输入端口。'}</p>
                    <button className="button primary" type="button" onClick={loadStarter}>
                      <WandSparkles size={14} /> {starterLabel}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </section>

          <aside className="workflow-inspector">
            {selectedNode ? (
              <NodeInspector
                key={`${layer}:${selectedNode.id}`}
                index={index}
                node={selectedNode}
                graph={graph}
                components={layer === 'step' ? componentOptions(state, selectedNode.kind) : []}
                onPatch={(patch) => patchNode(selectedNode.id, patch)}
                onParam={(name, value) => setParam(selectedNode.id, name, value)}
                onDelete={() => deleteNode(selectedNode.id)}
                onDuplicate={() => duplicateNode(selectedNode.id)}
                onDisconnect={(edgeId) => deleteEdge(edgeId)}
              />
            ) : (
              <div className="workflow-inspector-empty">
                <CircleDot size={30} />
                <h3>选择一个节点</h3>
                <p>点击画布中的节点, 在这里编辑显示名, 组件 target 和参数.</p>
                <div className="workflow-shortcuts">
                  <span>
                    <b>拖动</b>重新排布
                  </span>
                  <span>
                    <b>Delete</b>删除节点或连线
                  </span>
                  <span>
                    <b>Esc</b>取消连线
                  </span>
                </div>
              </div>
            )}
          </aside>
        </div>
      </main>
    </section>
  )
}

function NodeInspector({
  index,
  node,
  graph,
  components,
  onPatch,
  onParam,
  onDelete,
  onDuplicate,
  onDisconnect,
}: {
  index: CatalogIndex
  node: WorkflowNode
  graph: Graph
  components: WorkflowNode[]
  onPatch: (patch: Partial<WorkflowNode>) => void
  onParam: (name: string, value: JsonValue) => void
  onDelete: () => void
  onDuplicate: () => void
  onDisconnect: (edgeId: string) => void
}) {
  const kindSpec = kindOf(index, node.kind)
  const target = targetOf(index, node.kind, node.target)
  if (!kindSpec) return null
  const color = kindColor(node.kind)
  return (
    <>
      <div className="workflow-inspector-head" style={{ '--node-color': color } as Record<string, string>}>
        <span className="workflow-inspector-icon">{kindIcon(node.kind)}</span>
        <div>
          <small>{kindSpec.label}</small>
          <strong>{nodeLabel(node, index)}</strong>
        </div>
        <button className="workflow-delete" type="button" onClick={onDelete}>
          <Trash2 size={13} /> 删除
        </button>
      </div>
      <div className="workflow-inspector-body">
        <label className="workflow-field">
          <span>显示名</span>
          <input
            value={typeof node.metadata?.label === 'string' ? node.metadata.label : ''}
            placeholder={target?.label ?? kindSpec.label}
            onInput={(event) =>
              onPatch({
                metadata: { label: (event.currentTarget as HTMLInputElement).value },
              })
            }
          />
        </label>
        <label className="workflow-field">
          <span>组件 target</span>
          <select
            value={node.target}
            onChange={(event) => {
              const next = (event.currentTarget as HTMLSelectElement).value
              const nextTarget = targetOf(index, node.kind, next)
              onPatch({
                target: next,
                params: defaultParams(nextTarget),
                metadata: { label: nextTarget?.label ?? node.metadata?.label ?? '' },
              })
            }}
          >
            {kindSpec.targets.map((option) => (
              <option value={option.target} key={option.target}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        {target?.params.length ? (
          target.params.map((spec) => (
            spec.name === 'component' && (node.kind === 'LossCall' || node.kind === 'MetricCall') ? (
              <label className="workflow-field" key={`${node.id}:${node.target}:${spec.name}`}>
                <span>{node.kind === 'LossCall' ? '引用损失组件' : '引用指标组件'}</span>
                <select value={String(node.params.component ?? '')}
                  onChange={(event) => onParam('component', event.currentTarget.value)}>
                  <option value="">选择装配图中的节点</option>
                  {components.map((item) => (
                    <option value={item.id} key={item.id}>{item.metadata?.label ?? item.id} · {item.id}</option>
                  ))}
                </select>
              </label>
            ) : <ParamField
              key={`${node.id}:${node.target}:${spec.name}`}
              spec={spec}
              value={node.params[spec.name] ?? null}
              onChange={(value) => onParam(spec.name, value)}
            />
          ))
        ) : (
          <p className="workflow-hint">该 target 没有可配置参数.</p>
        )}

        <section className="workflow-inputs">
          <div>
            <strong>输入端口</strong>
            <small>从其他节点的输出端口拖到这里连线</small>
          </div>
          {kindSpec.inputs.length ? (
            kindSpec.inputs.map((port) => {
              const connections = graph.edges.filter(
                (item) => item.to.node === node.id && item.to.port === port.name,
              )
              return (
                <div className="workflow-port-status" key={port.name}>
                  <b>{port.name}</b>
                  <em title={index.catalog.port_types[port.type]}>{port.type}</em>
                  {connections.length ? (
                    <div className="workflow-port-connections">
                      {connections.map((edge) => (
                        <button key={edge.id} type="button" onClick={() => onDisconnect(edge.id)}>
                          {edge.from.node}.{edge.from.port}
                          <X size={11} />
                        </button>
                      ))}
                    </div>
                  ) : (
                    <span className={port.required ? 'missing' : ''}>
                      {port.required ? '必需, 未连接' : '可选'}
                    </span>
                  )}
                </div>
              )
            })
          ) : (
            <i>该节点没有输入端口</i>
          )}
        </section>

        <section className="workflow-inputs">
          <div>
            <strong>输出端口</strong>
            <small>可以连到多个下游节点</small>
          </div>
          {kindSpec.outputs.length ? (
            kindSpec.outputs.map((port) => {
              const count = graph.edges.filter(
                (item) => item.from.node === node.id && item.from.port === port.name,
              ).length
              return (
                <div className="workflow-port-status" key={port.name}>
                  <b>{port.name}</b>
                  <em title={index.catalog.port_types[port.type]}>{port.type}</em>
                  <span>{count ? `${count} 条下游` : '未连接'}</span>
                </div>
              )
            })
          ) : (
            <i>该节点是终端节点</i>
          )}
        </section>

        <div className="workflow-inspector-actions">
          <button className="button subtle" type="button" onClick={onDuplicate}>
            <Plus size={13} /> 复制节点
          </button>
        </div>
      </div>
    </>
  )
}

function ParamField({
  spec,
  value,
  onChange,
}: {
  spec: ParamSpec
  value: JsonValue
  onChange: (value: JsonValue) => void
}) {
  const [raw, setRaw] = useState(() => rawValue(value))
  useEffect(() => {
    if (value == null) setRaw('')
  }, [value])
  const problem = checkParam(spec, value)
  const isNumber = spec.type === 'integer' || spec.type === 'number'
  const isList = spec.type === 'list_integer' || spec.type === 'list_number'
  return (
    <label className={`workflow-field ${problem ? 'invalid' : ''}`}>
      <span>
        {spec.label}
        {spec.required && <em>*</em>}
      </span>
      {spec.type === 'boolean' ? (
        <button
          type="button"
          className={`toggle ${value ? 'on' : ''}`}
          aria-pressed={Boolean(value)}
          onClick={() => onChange(!value)}
        >
          <i />
        </button>
      ) : spec.type === 'enum' ? (
        <select
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => onChange((event.currentTarget as HTMLSelectElement).value)}
        >
          {(spec.options ?? []).map((option) => (
            <option value={option} key={option}>
              {option}
            </option>
          ))}
        </select>
      ) : (
        <input
          value={raw}
          inputMode={isNumber || isList ? 'numeric' : 'text'}
          placeholder={
            isList
              ? '逗号分隔, 例如 3,32,32'
              : spec.type === 'data_path'
                ? '相对于 data/ 的路径'
                : undefined
          }
          onInput={(event) => {
            const text = (event.currentTarget as HTMLInputElement).value
            setRaw(text)
            if (isNumber) {
              const parsed = Number(text)
              if (!text.trim()) onChange(null)
              else if (Number.isFinite(parsed)) {
                onChange(spec.type === 'integer' ? Math.trunc(parsed) : parsed)
              }
              return
            }
            if (isList) {
              const items = text
                .split(',')
                .map((item) => item.trim())
                .filter((item) => item !== '')
                .map(Number)
              if (items.every((item) => Number.isFinite(item))) {
                onChange(spec.type === 'list_integer' ? items.map(Math.trunc) : items)
              }
              return
            }
            onChange(text)
          }}
        />
      )}
      {problem ? <em className="workflow-field-error">{problem}</em> : null}
    </label>
  )
}

function rawValue(value: JsonValue): string {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) return value.join(',')
  return String(value)
}

function WorkflowSettings({
  state,
  history,
  gpus,
  onPatch,
  onLoad,
}: {
  state: EditorState
  history: WorkflowRecord[]
  gpus: NonNullable<SystemSnapshot['gpus']>
  onPatch: (patch: Partial<EditorState>) => void
  onLoad: (id: string) => void
}) {
  const launcher = state.settings.launcher ?? 'none'
  const devices = state.settings.devices ?? []
  const seed = typeof state.settings.runtime?.seed === 'number' ? state.settings.runtime.seed : 42
  return (
    <div className="workflow-settings">
      <div className="workflow-settings-note">
        <Save size={14} />
        <span>草稿只保存在当前浏览器. 点击"校验并排队"才会写入服务端快照.</span>
      </div>
      <label className="workflow-field">
        <span>执行 profile</span>
        <select
          value={state.executionProfile}
          onChange={(event) =>
            onPatch({
              executionProfile: (event.currentTarget as HTMLSelectElement).value as EditorState['executionProfile'],
            })
          }
        >
          <option value="schema-v1" disabled={Boolean(state.stepGraph)}>schema-v1</option>
          <option value="dag-runtime">dag-runtime</option>
        </select>
      </label>
      <label className="workflow-field">
        <span>Launcher</span>
        <select
          value={launcher}
          onChange={(event) =>
            onPatch({
              settings: {
                ...state.settings,
                launcher: (event.currentTarget as HTMLSelectElement).value,
              },
            })
          }
        >
          <option value="none">单进程</option>
          <option value="torchrun">torchrun</option>
          <option value="accelerate">accelerate</option>
        </select>
      </label>
      <div className="workflow-field">
        <span>GPU</span>
        <div className="gpu-select">
          {gpus.length ? (
            gpus.map((gpu) => (
              <button
                type="button"
                key={gpu.index}
                className={`gpu-choice ${devices.includes(gpu.index) ? 'selected' : ''}`}
                aria-pressed={devices.includes(gpu.index)}
                onClick={() =>
                  onPatch({
                    settings: {
                      ...state.settings,
                      devices: devices.includes(gpu.index)
                        ? devices.filter((item) => item !== gpu.index)
                        : [...devices, gpu.index].sort((left, right) => left - right),
                    },
                  })
                }
              >
                GPU {gpu.index}
              </button>
            ))
          ) : (
            <span className="workflow-empty-signal">未检测到 NVIDIA GPU, 将使用 CPU</span>
          )}
        </div>
      </div>
      <label className="workflow-field">
        <span>随机种子</span>
        <input
          value={String(seed)}
          inputMode="numeric"
          onInput={(event) => {
            const parsed = Number((event.currentTarget as HTMLInputElement).value)
            if (!Number.isFinite(parsed)) return
            onPatch({
              settings: { ...state.settings, runtime: { seed: Math.trunc(parsed) } },
            })
          }}
        />
      </label>
      <div className="workflow-settings-item">
        <span>节点 / 连线</span>
        <b>
          {state.nodes.length} / {state.edges.length}
        </b>
      </div>
      <div className="workflow-settings-item">
        <span>运行接口</span>
        <b>POST /api/workflows/:id/runs</b>
      </div>
      <section className="workflow-history">
        <div>
          <strong>已保存快照</strong>
          <small>点击可重新载入到画布</small>
        </div>
        {history.length ? (
          history.slice(0, 8).map((record) => (
            <button type="button" key={record.id} onClick={() => onLoad(record.id)}>
              <span>{record.name}</span>
              <em>{record.snapshot_hash.slice(0, 8)}</em>
            </button>
          ))
        ) : (
          <i>还没有保存过快照</i>
        )}
      </section>
    </div>
  )
}

function fromSnapshot(record: WorkflowRecord): EditorState {
  const snapshot = record.snapshot
  if (!snapshot) return emptyState()
  return {
    name: snapshot.name,
    executionProfile: snapshot.execution_profile,
    nodes: snapshot.nodes.map((node) => ({
      ...node,
      position: node.position ?? { x: 40, y: 40 },
    })),
    edges: snapshot.edges,
    ...(snapshot.step_graph ? { stepGraph: {
      nodes: snapshot.step_graph.nodes.map((node) => ({ ...node, position: node.position ?? { x: 40, y: 40 } })),
      edges: snapshot.step_graph.edges,
    } } : {}),
    settings: {
      launcher: snapshot.settings.launcher ?? 'none',
      devices: snapshot.settings.devices ?? [],
      runtime: snapshot.settings.runtime ?? { seed: 42 },
    },
  }
}

function nextFreePosition(nodes: WorkflowNode[]): { x: number; y: number } {
  const columns = 4
  const index = nodes.length
  return {
    x: CANVAS_PADDING + (index % columns) * (NODE_WIDTH + 60),
    y: CANVAS_PADDING + Math.floor(index / columns) * 150,
  }
}

function typesCompatible(source: string, target: string): boolean {
  return source === 'any' || target === 'any' || source === target
}

function changedParams(node: WorkflowNode, name: string, value: JsonValue): WorkflowNode['params'] {
  const params = { ...node.params }
  if (value === null || value === '') delete params[name]
  else params[name] = value
  if (node.kind === 'Select' && value !== null && value !== '') {
    if (name === 'key') delete params.index
    if (name === 'index') delete params.key
  }
  return params
}

function componentOptions(state: EditorState, kind: string): WorkflowNode[] {
  if (kind === 'MetricCall') return state.nodes.filter((node) => node.kind === 'metric')
  if (kind !== 'LossCall') return []
  const trainer = state.nodes.find((node) => node.kind === 'trainer')
  const lossIds = new Set(state.edges.filter((edge) =>
    edge.to.node === trainer?.id && edge.to.port === 'loss',
  ).map((edge) => edge.from.node))
  return state.nodes.filter((node) => node.kind === 'loss' && lossIds.has(node.id))
}

function starterStepGraph(index: CatalogIndex, state: EditorState): Graph {
  const [loss] = componentOptions(state, 'LossCall')
  const specs: [string, string, number, number, WorkflowNode['params']][] = [
    ['batch', 'BatchInput', 0, 0, {}],
    ['features', 'Select', 1, 0, { index: 0 }],
    ['labels', 'Select', 1, 1, { index: 1 }],
    ['forward', 'Forward', 2, 0, {}],
    ['loss', 'LossCall', 3, 0, loss ? { component: loss.id } : {}],
    ['output', 'StepOutput', 4, 0, {}],
  ]
  const nodes = specs.flatMap(([id, kind, column, row, params]) => {
    const target = kindOf(index, kind)?.targets[0]
    return target ? [{ id, kind, target: target.target, params,
      position: { x: 32 + column * 240, y: 48 + row * 160 } }] : []
  })
  if (nodes.length !== specs.length) return EMPTY_GRAPH
  const edges = [
    ['batch', 'batch', 'features', 'batch'],
    ['batch', 'batch', 'labels', 'batch'],
    ['features', 'value', 'forward', 'features'],
    ['forward', 'predictions', 'loss', 'predictions'],
    ['labels', 'value', 'loss', 'labels'],
    ['loss', 'loss', 'output', 'loss'],
  ].map(([source, output, target, input]) => ({
    id: newEdgeId(), from: { node: source, port: output }, to: { node: target, port: input },
  }))
  return { nodes, edges }
}
