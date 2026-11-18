/**
 * 共享测试夹具: 直接读取仓库里的真实节点目录.
 *
 * 前端目录类型必须和 xdl/workflow/node_catalog.json 保持一致, 因此测试不造假目录.
 */

import { readFileSync } from 'node:fs'

import { indexCatalog } from './catalog'
import type { CatalogIndex, NodeCatalog } from './catalog'
import type { EditorState, WorkflowNode } from './types'

export const CATALOG_PATH = new URL('../../../../xdl/workflow/node_catalog.json', import.meta.url)

export function loadCatalog(): NodeCatalog {
  return JSON.parse(readFileSync(CATALOG_PATH, 'utf8')) as NodeCatalog
}

export function loadIndex(): CatalogIndex {
  return indexCatalog(loadCatalog())
}

export function node(
  id: string,
  kind: string,
  target: string,
  params: Record<string, unknown> = {},
): WorkflowNode {
  return {
    id,
    kind,
    target,
    params: params as WorkflowNode['params'],
    position: { x: 40, y: 40 },
    metadata: { label: id },
  }
}

export function state(nodes: WorkflowNode[], edges: EditorState['edges'] = []): EditorState {
  return {
    name: 'test graph',
    executionProfile: 'schema-v1',
    nodes,
    edges,
    settings: { launcher: 'none', devices: [], runtime: { seed: 42 } },
  }
}

export function edge(id: string, from: [string, string], to: [string, string]) {
  return {
    id,
    from: { node: from[0], port: from[1] },
    to: { node: to[0], port: to[1] },
  }
}
