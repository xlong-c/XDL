import { describe, expect, it } from 'vitest'

import {
  buildStarterGraph,
  checkParam,
  defaultParams,
  findPort,
  indexCatalog,
  kindOf,
  targetOf,
} from './catalog'
import { loadCatalog, loadIndex } from './fixtures'
import { validateLocally } from './validation'
import { issuesBySeverity } from './validation'

describe('node catalog', () => {
  it('indexes every declared kind', () => {
    const index = loadIndex()
    expect(index.kindNames.length).toBeGreaterThan(5)
    for (const name of index.kindNames) {
      expect(kindOf(index, name)?.kind).toBe(name)
    }
  })

  it('keeps dag-only kinds out of the schema-v1 profile', () => {
    const index = loadIndex()
    for (const kind of ['identity', 'compose', 'merge', 'sink']) {
      expect(kindOf(index, kind)?.profiles).toEqual(['dag-runtime'])
    }
  })

  it('resolves ports only when the kind declares them', () => {
    const index = loadIndex()
    expect(findPort(index, 'trainer', 'loss', 'in')?.multi).toBe(true)
    expect(findPort(index, 'trainer', 'loss', 'out')).toBeUndefined()
    expect(findPort(index, 'model', 'module', 'out')?.type).toBe('model')
  })

  it('reads defaults from the catalog instead of hardcoding them', () => {
    const index = loadIndex()
    const target = targetOf(index, 'trainer', 'builtin:Trainer')
    const params = defaultParams(target)
    expect(params.max_epochs).toBe(1)
    expect(params.batch_size).toBe(64)
    expect(params.precision).toBe('32')
  })
})

describe('param validation', () => {
  const index = loadIndex()
  const trainer = targetOf(index, 'trainer', 'builtin:Trainer')

  function spec(name: string) {
    const found = trainer?.params.find((item) => item.name === name)
    if (!found) throw new Error(`missing param spec ${name}`)
    return found
  }

  it('accepts values inside the declared range', () => {
    expect(checkParam(spec('max_epochs'), 3)).toBeNull()
    expect(checkParam(spec('precision'), 'bf16')).toBeNull()
  })

  it('rejects wrong types and out-of-range values', () => {
    expect(checkParam(spec('max_epochs'), 'many')).toContain('整数')
    expect(checkParam(spec('max_epochs'), 0)).toContain('>=')
    expect(checkParam(spec('max_epochs'), true)).toContain('整数')
    expect(checkParam(spec('precision'), 'float64')).toContain('之一')
  })

  it('rejects absolute and traversing data paths', () => {
    const dataset = targetOf(index, 'dataset', 'torchvision.datasets:CIFAR10')
    const root = dataset?.params.find((item) => item.name === 'root')
    if (!root) throw new Error('missing root spec')
    expect(checkParam(root, '/etc/passwd')).not.toBeNull()
    expect(checkParam(root, '../data')).not.toBeNull()
    expect(checkParam(root, 'cifar10')).toBeNull()
  })

  it('accepts numeric lists only', () => {
    const normalize = targetOf(index, 'transform', 'torchvision.transforms:Normalize')
    const mean = normalize?.params.find((item) => item.name === 'mean')
    if (!mean) throw new Error('missing mean spec')
    expect(checkParam(mean, [0.5, 0.5])).toBeNull()
    expect(checkParam(mean, ['a'])).toContain('数字数组')
  })
})

describe('starter graph', () => {
  it('only references kinds, targets and ports declared in the catalog', () => {
    const catalog = loadCatalog()
    const index = indexCatalog(catalog)
    const graph = buildStarterGraph(index)
    expect(graph.nodes.length).toBeGreaterThan(4)
    const byId = new Map(graph.nodes.map((item) => [item.id, item]))
    for (const item of graph.nodes) {
      const kind = kindOf(index, item.kind)
      expect(kind, `kind ${item.kind}`).toBeDefined()
      expect(targetOf(index, item.kind, item.target), `target ${item.target}`).toBeDefined()
    }
    for (const item of graph.edges) {
      const source = byId.get(item.from.node)
      const target = byId.get(item.to.node)
      expect(source, `edge ${item.id} source`).toBeDefined()
      expect(target, `edge ${item.id} target`).toBeDefined()
      expect(findPort(index, source!.kind, item.from.port, 'out'), `edge ${item.id} out`).toBeDefined()
      expect(findPort(index, target!.kind, item.to.port, 'in'), `edge ${item.id} in`).toBeDefined()
    }
  })

  it('produces a graph without blocking errors', () => {
    const index = loadIndex()
    const graph = buildStarterGraph(index)
    const issues = validateLocally(
      {
        name: 'starter',
        executionProfile: 'schema-v1',
        nodes: graph.nodes,
        edges: graph.edges,
        settings: { launcher: 'none', devices: [], runtime: { seed: 42 } },
      },
      index,
    )
    expect(issuesBySeverity(issues).errors).toEqual([])
  })
})
