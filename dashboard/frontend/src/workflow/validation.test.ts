import { describe, expect, it } from 'vitest'

import { buildStarterGraph, defaultParams, targetOf } from './catalog'
import { edge, loadIndex, node, state } from './fixtures'
import { detectCycle, issuesBySeverity, topologicalLayers, validateLocally } from './validation'

const index = loadIndex()

function codes(current: ReturnType<typeof state>): string[] {
  return validateLocally(current, index).map((item) => item.code)
}

function starter() {
  const graph = buildStarterGraph(index)
  return state(graph.nodes, graph.edges)
}

describe('local validation', () => {
  it('accepts the starter graph and only warns about the missing validation loader', () => {
    const issues = validateLocally(starter(), index)
    expect(issuesBySeverity(issues).errors).toEqual([])
    expect(issuesBySeverity(issues).warnings.map((item) => item.code)).toEqual(['missing_port'])
  })

  it('rejects an empty canvas', () => {
    expect(codes(state([]))).toContain('invalid_snapshot')
  })

  it('rejects unknown kinds and targets', () => {
    const unknownKind = state([node('a', 'teleporter', 'registry:vgg11_bn')])
    expect(codes(unknownKind)).toContain('unknown_node_kind')
    const unknownTarget = state([node('a', 'model', 'registry:nope')])
    expect(codes(unknownTarget)).toContain('unknown_target')
  })

  it('rejects duplicate node ids', () => {
    const duplicated = state([
      node('a', 'model', 'registry:vgg11_bn'),
      node('a', 'model', 'registry:vgg11_bn'),
    ])
    expect(codes(duplicated)).toContain('duplicate_id')
  })

  it('rejects a type mismatch between ports', () => {
    const broken = state(
      [node('m', 'model', 'registry:vgg11_bn'), node('t', 'trainer', 'builtin:Trainer')],
      [edge('e1', ['m', 'module'], ['t', 'loss'])],
    )
    expect(codes(broken)).toContain('type_mismatch')
  })

  it('rejects a second source on a single-input port', () => {
    const broken = state(
      [
        node('m1', 'model', 'registry:vgg11_bn'),
        node('m2', 'model', 'registry:vgg16_bn'),
        node('o', 'optimizer', 'torch.optim:SGD'),
      ],
      [edge('e1', ['m1', 'module'], ['o', 'model']), edge('e2', ['m2', 'module'], ['o', 'model'])],
    )
    expect(codes(broken)).toContain('port_conflict')
  })

  it('allows multiple sources on a multi port', () => {
    const graph = starter()
    const lossTarget = targetOf(index, 'loss', 'torch.nn:MSELoss')
    const extra = node('loss-2', 'loss', 'torch.nn:MSELoss', defaultParams(lossTarget))
    const withExtra = {
      ...graph,
      nodes: [...graph.nodes, extra],
      edges: [...graph.edges, edge('e-loss-2', ['loss-2', 'loss'], ['trainer', 'loss'])],
    }
    const issues = issuesBySeverity(validateLocally(withExtra, index)).errors
    expect(issues.map((item) => item.code)).not.toContain('port_conflict')
  })

  it('detects cycles before reporting connectivity', () => {
    const cyclic = state(
      [node('a', 'transform', 'torchvision.transforms:ToTensor'), node('b', 'transform', 'torchvision.transforms:ToTensor')],
      [edge('e1', ['a', 'pipeline'], ['b', 'pipeline']), edge('e2', ['b', 'pipeline'], ['a', 'pipeline'])],
    )
    expect(codes(cyclic)).toContain('cycle_detected')
    expect(detectCycle(cyclic.nodes, cyclic.edges).sort()).toEqual(['a', 'b'])
  })

  it('flags nodes that are disconnected from the trainer', () => {
    const graph = starter()
    const orphan = node('orphan', 'model', 'registry:resnet18')
    const issues = issuesBySeverity(
      validateLocally({ ...graph, nodes: [...graph.nodes, orphan] }, index),
    ).errors
    expect(issues.map((item) => item.code)).toContain('unreachable_node')
    expect(issues.find((item) => item.code === 'unreachable_node')?.node_id).toBe('orphan')
  })

  it('flags duplicate roles', () => {
    const graph = starter()
    const extraOptimizer = node('opt-2', 'optimizer', 'torch.optim:Adam')
    const issues = issuesBySeverity(
      validateLocally(
        {
          ...graph,
          nodes: [...graph.nodes, extraOptimizer],
          edges: [...graph.edges, edge('e-opt-2', ['model', 'module'], ['opt-2', 'model'])],
        },
        index,
      ),
    ).errors
    expect(issues.map((item) => item.code)).toContain('duplicate_role')
  })

  it('rejects dag-only nodes in the schema-v1 profile', () => {
    const graph = starter()
    const issues = issuesBySeverity(
      validateLocally(
        {
          ...graph,
          nodes: [...graph.nodes, node('ident', 'identity', 'builtin:Identity')],
          edges: [
            ...graph.edges,
            edge('e-ident', ['model', 'module'], ['ident', 'in']),
            edge('e-ident-out', ['ident', 'out'], ['optimizer', 'model']),
          ],
        },
        index,
      ),
    ).errors
    expect(issues.map((item) => item.code)).toContain('unsupported_profile')
  })

  it('requires consumed outputs for dag-runtime graphs', () => {
    const graph = buildStarterGraph(index)
    const dag = {
      name: 'dag',
      executionProfile: 'dag-runtime' as const,
      nodes: [...graph.nodes, node('ident', 'identity', 'builtin:Identity')],
      edges: [...graph.edges, edge('e-ident', ['model', 'module'], ['ident', 'in'])],
      settings: { launcher: 'none', devices: [], runtime: { seed: 42 } },
    }
    expect(codes(dag)).toContain('dangling_output')
  })

  it('reports invalid params with the node id attached', () => {
    const graph = starter()
    const patched = {
      ...graph,
      nodes: graph.nodes.map((item) =>
        item.id === 'trainer' ? { ...item, params: { ...item.params, max_epochs: 'many' } } : item,
      ),
    }
    const issues = validateLocally(patched, index).filter((item) => item.code === 'invalid_param')
    expect(issues).toHaveLength(1)
    expect(issues[0].node_id).toBe('trainer')
    expect(issues[0].path).toContain('max_epochs')
  })
})

describe('topological layers', () => {
  it('orders the starter graph by dependency depth', () => {
    const graph = buildStarterGraph(index)
    const layers = topologicalLayers(graph.nodes, graph.edges)
    const flat = layers.flat()
    expect(flat).toHaveLength(graph.nodes.length)
    expect(layers[0]).toContain('transform')
    const trainerLayer = layers.findIndex((layer) => layer.includes('trainer'))
    const loaderLayer = layers.findIndex((layer) => layer.includes('loader'))
    expect(trainerLayer).toBeGreaterThan(loaderLayer)
  })

  it('keeps cyclic nodes in the last layer instead of dropping them', () => {
    const layers = topologicalLayers(
      [node('a', 'transform', 'torchvision.transforms:ToTensor'), node('b', 'transform', 'torchvision.transforms:ToTensor')],
      [edge('e1', ['a', 'pipeline'], ['b', 'pipeline']), edge('e2', ['b', 'pipeline'], ['a', 'pipeline'])],
    )
    expect(layers.flat().sort()).toEqual(['a', 'b'])
  })
})
