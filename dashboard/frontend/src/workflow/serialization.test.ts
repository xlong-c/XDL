import { describe, expect, it } from 'vitest'

import { buildStarterGraph } from './catalog'
import { edge, loadIndex, node } from './fixtures'
import { validateLocally } from './validation'
import {
  DRAFT_STORAGE_KEY,
  canonicalSnapshotJson,
  emptyState,
  parseDraft,
  serializeDraft,
  starterState,
  toSnapshot,
} from './serialization'

const index = loadIndex()

describe('snapshot serialization', () => {
  it('omits the server id and keeps the declared profile', () => {
    const snapshot = toSnapshot(starterState(index))
    expect(snapshot.version).toBe(1)
    expect(snapshot.id).toBeUndefined()
    expect(snapshot.execution_profile).toBe('schema-v1')
    expect(snapshot.nodes.length).toBeGreaterThan(4)
  })

  it('rounds canvas positions to integers', () => {
    const base = starterState(index)
    const shifted = {
      ...base,
      nodes: base.nodes.map((item, position) => ({
        ...item,
        position: { x: 12.6 + position, y: 40.2 },
      })),
    }
    const snapshot = toSnapshot(shifted)
    for (const item of snapshot.nodes) {
      expect(Number.isInteger(item.position.x)).toBe(true)
      expect(Number.isInteger(item.position.y)).toBe(true)
    }
  })

  it('keeps settings to the declared shape', () => {
    const base = starterState(index)
    const snapshot = toSnapshot({
      ...base,
      settings: { launcher: 'torchrun', devices: [2, 0], runtime: { seed: 7, extra: 'dropped' } },
    })
    expect(snapshot.settings.launcher).toBe('torchrun')
    expect(snapshot.settings.devices).toEqual([0, 2])
    expect(snapshot.settings.runtime).toEqual({ seed: 7 })
  })

  it('produces a canonical JSON that ignores node and edge ordering', () => {
    const base = starterState(index)
    const forward = toSnapshot(base)
    const reversed = toSnapshot({
      ...base,
      nodes: [...base.nodes].reverse(),
      edges: [...base.edges].reverse(),
    })
    expect(canonicalSnapshotJson(forward)).toBe(canonicalSnapshotJson(reversed))
  })

  it('falls back to the default name when the draft name is blank', () => {
    const snapshot = toSnapshot({ ...emptyState(), name: '   ' })
    expect(snapshot.name.length).toBeGreaterThan(0)
  })
})

describe('draft persistence', () => {
  it('roundtrips through serialize and parse', () => {
    const original = starterState(index)
    const restored = parseDraft(serializeDraft(original), index)
    expect(restored).not.toBeNull()
    expect(restored?.nodes.length).toBe(original.nodes.length)
    expect(restored?.edges.length).toBe(original.edges.length)
    expect(restored?.executionProfile).toBe(original.executionProfile)
  })

  it('preserves independent node ids and locates step validation issues', () => {
    const base = starterState(index)
    const original = {
      ...base,
      executionProfile: 'dag-runtime' as const,
      stepGraph: {
        nodes: [node('batch', 'BatchInput', 'builtin:BatchInput'), node('loss', 'LossCall', 'builtin:LossCall')],
        edges: [edge('e-test', ['batch', 'batch'], ['loss', 'labels'])],
      },
    }
    const restored = parseDraft(serializeDraft(original), index)
    expect(restored?.nodes.find((item) => item.id === 'loss')?.kind).toBe('loss')
    expect(restored?.stepGraph).toEqual(original.stepGraph)
    expect(toSnapshot(restored!).step_graph).toEqual(original.stepGraph)
    expect(validateLocally(restored!, index)).toContainEqual(expect.objectContaining({
      code: 'type_mismatch', path: 'step_graph/edges/e-test/to', edge_id: 'e-test',
    }))
  })

  it('returns null for missing, malformed or mismatched drafts', () => {
    expect(parseDraft(null, index)).toBeNull()
    expect(parseDraft('not json', index)).toBeNull()
    expect(parseDraft(JSON.stringify({ version: 99 }), index)).toBeNull()
  })

  it('drops nodes whose kind or target left the catalog', () => {
    const raw = JSON.stringify({
      version: 1,
      name: 'legacy draft',
      execution_profile: 'schema-v1',
      nodes: [
        { id: 'good', kind: 'model', target: 'registry:vgg11_bn', params: { num_classes: 10 } },
        { id: 'bad-kind', kind: 'teleporter', target: 'registry:vgg11_bn' },
        { id: 'bad-target', kind: 'model', target: 'registry:gone' },
      ],
      edges: [
        { id: 'e1', from: { node: 'good', port: 'module' }, to: { node: 'bad-kind', port: 'model' } },
      ],
      settings: { launcher: 'none', devices: [] },
    })
    const restored = parseDraft(raw, index)
    expect(restored?.nodes.map((item) => item.id)).toEqual(['good'])
    expect(restored?.edges).toEqual([])
  })

  it('drops params that the target no longer declares', () => {
    const raw = JSON.stringify({
      version: 1,
      name: 'legacy params',
      execution_profile: 'schema-v1',
      nodes: [
        {
          id: 'model',
          kind: 'model',
          target: 'registry:vgg11_bn',
          params: { num_classes: 4, removed_param: 1 },
        },
      ],
      edges: [],
      settings: { launcher: 'none', devices: [] },
    })
    const restored = parseDraft(raw, index)
    expect(restored?.nodes[0].params.num_classes).toBe(4)
    expect(restored?.nodes[0].params.removed_param).toBeUndefined()
  })

  it('uses a versioned storage key', () => {
    expect(DRAFT_STORAGE_KEY).toBe('xdl.dashboard.workflow.v2')
  })

  it('builds an empty state that serializes without nodes', () => {
    const snapshot = toSnapshot(emptyState())
    expect(snapshot.nodes).toEqual([])
    expect(snapshot.edges).toEqual([])
    expect(buildStarterGraph(index).nodes.length).toBeGreaterThan(0)
  })
})
