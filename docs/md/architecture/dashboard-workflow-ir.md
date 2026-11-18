# Dashboard Workflow IR 契约 (WorkflowSnapshot v1)

本文是 dashboard 工作流系统的跨语言契约事实源, 覆盖前端, Go dashboard, Python compiler 和 XDL 执行层. 修改节点目录, 快照字段, 错误码或 API 形状时, 先改本文与 `xdl/workflow/node_catalog.json`, 再改实现.

对应规划文档: [../../plans/2026-11-18-dashboard-arbitrary-dag-plan.md](../../plans/2026-11-18-dashboard-arbitrary-dag-plan.md).

## 目标与非目标

**目标:** 让 dashboard 的画布成为一个真实的一等对象: 有类型, 有版本, 可持久化, 可校验, 可编译, 可复现, 并且节点图的修改确实改变训练配置.

**非目标:**

- 不允许浏览器提交 Python 源码, import 路径或任意可执行字符串. 节点的 `target` 必须命中节点目录.
- 不在 Go 中复制 XDL registry, 组件构建逻辑或端口语义.
- 不改变 `xdl/config/schema.py` 与 `xdl/config/setup.py` 的稳定使用方式.

## 三层边界

| 层 | 负责 | 明确不负责 |
| --- | --- | --- |
| Preact 前端 | 编辑图, 渲染端口与参数表单, 本地结构提示, 保存草稿, 调用预检与排队 | 生成最终 YAML, 判断 target 是否真实存在 |
| Go dashboard | JSON envelope, 结构校验, 规模限制, 快照持久化与不可变 run 关联, 队列与 GPU 互斥, SSE, 转发节点目录 | 组件目录语义, 参数语义, 端口类型语义, XDL 组件构建 |
| Python compiler | 节点目录语义, 端口类型检查, target 与参数校验, 图可达性, schema-v1 lowering, dag-runtime manifest | 队列, 进程生命周期, GPU 资源 |
| XDL | `setup_from_yaml()` 稳定入口, `TrainSetup`, `Trainer` 生命周期, `xdl.workflow.executor` 的 dag-runtime 执行 | dashboard 的 HTTP 与持久化 |

Go 只从节点目录文件读取 `limits` 一个键, 其余字段原样转发给前端. 这是唯一一处 Go 读取目录内容的地方, 目的是让快照规模上限只有一份数字.

## WorkflowSnapshot v1

快照是不可变 JSON 对象. 权威结构模型在 `xdl/workflow/ir.py` 的 `WorkflowSnapshot`, 下面是等价 JSON 形状:

```json
{
  "version": 1,
  "id": "wf-7f3c1d2e",
  "name": "CIFAR-10 VGG-11 标准训练图",
  "execution_profile": "schema-v1",
  "nodes": [
    {
      "id": "model",
      "kind": "model",
      "target": "registry:vgg11_bn",
      "params": {"num_classes": 10, "dropout": 0.2},
      "position": {"x": 40, "y": 200},
      "metadata": {"label": "VGG-11 BN"}
    }
  ],
  "edges": [
    {
      "id": "e-model",
      "from": {"node": "model", "port": "module"},
      "to": {"node": "trainer", "port": "model"}
    }
  ],
  "settings": {"launcher": "none", "devices": [], "runtime": {"seed": 42}}
}
```

字段规则:

| 字段 | 规则 |
| --- | --- |
| `version` | 必须显式存在且等于 `1`. 缺失或未知版本一律拒绝, 不做兼容猜测. |
| `id` | 服务端创建快照时生成; 客户端提交时可以省略. |
| `name` | 非空字符串, 长度不超过 `limits.max_name_length`. |
| `execution_profile` | 必须显式声明, 取值 `schema-v1` 或 `dag-runtime`. 缺失即拒绝. |
| `nodes` | 非空数组, 数量不超过 `limits.max_nodes`. |
| `nodes[].id` | 非空字符串, 图内唯一, 长度不超过 `limits.max_id_length`. |
| `nodes[].kind` | 必须命中节点目录的 `kind`. |
| `nodes[].target` | 必须是该 `kind` 在目录中声明的 `target` 之一. |
| `nodes[].params` | JSON 对象, 只允许目录为该 target 声明的参数名; 深度与字节数受限. |
| `nodes[].position` | UI 元数据, 不参与训练语义, 但参与快照哈希. |
| `nodes[].metadata` | UI 元数据 (显示名等), 不参与训练语义, 但参与快照哈希. |
| `edges` | 数组, 数量不超过 `limits.max_edges`; 只引用已存在节点与已声明端口. |
| `edges[].id` | 非空字符串, 图内唯一; 省略时由服务端按索引生成 `edge-<n>`. |
| `edges[].from` / `to` | `{"node": <节点 id>, "port": <端口名>}`. 不允许自环, 不允许重复连线. 一个输入端口只允许一个来源, 除非目录把它声明为 `multi` (例如 `trainer.loss`). |
| `settings` | JSON 对象. `settings.runtime` 只允许 `seed`; `launcher` 与 `devices` 是排队默认值, run 请求可以覆盖. |
| `step_graph` | 可省略. 只支持 `dag-runtime`. 由独立的 `nodes` 和 `edges` 组成, 使用相同的结构限制及独立的 ID 命名空间, 并参与快照哈希. 子图节点取自目录的 `step_kinds`. |

`inputs: string[]` 这种旧表达不再被接受: 连接关系只能通过 `edges` 表达, 端口名必须来自节点目录.

### 端口类型

端口类型是封闭集合, 定义在节点目录的 `port_types`:

装配层使用 `dataset`, `transform`, `collate`, `loader`, `model`, `loss`, `optimizer`, `scheduler`, `metric`, `run`, `checkpoint`, `any`. 训练步层使用 `batch`, `tensor`, `scalar`, `metric_value`.

普通端口必须同类型. 装配层的 `any` 用于泛型节点; `identity` 与 `compose` 的输出类型根据上游确定, 不会把已知类型放宽为 `any`. 类型不一致报 `type_mismatch`.

### 节点目录

唯一实体来源是 `xdl/workflow/node_catalog.json`, 由 `xdl.workflow.load_catalog()` 读取, 通过 `GET /api/workflows/catalog` 原样转发给前端. 目录包含:

- `limits`: 快照规模上限 (Go 与 Python 共用).
- `profiles`: 执行 profile 说明.
- `port_types`: 端口类型说明.
- `kinds[]`: 每种节点的 `label`, `group`, `summary`, `profiles`, `inputs`, `outputs`, `targets`.
- `kinds[].targets[].params[]`: 参数名, 类型, 是否必需, 默认值, 取值范围, 枚举选项.

参数类型封闭集合: `integer`, `number`, `boolean`, `string`, `enum`, `list_integer`, `list_number`, `data_path`.

`data_path` 必须是相对路径, 不允许绝对路径, 不允许 `..`; 实际解析到 configured `data_root` 由 dashboard 侧完成.

端口声明里的 `multi: true` 表示该输入端口可以同时接入多个上游, 例如 `trainer.loss` 和 `StepOutput.metrics`. 其余输入端口只允许一个来源.

`target` 使用 `source:name` 格式:

- `registry:<name>`: XDL registry 名称, 例如 `registry:vgg11_bn`.
- `<module>:<attr>`: 可导入模块属性, 例如 `torch.optim:SGD`.
- `builtin:<name>`: dashboard 与 workflow runtime 内建节点, 例如 `builtin:Trainer`. `builtin` 不是 registry, 不能出现在用户自建的 registry 组件里.

### 标准训练图

标准 `schema-v1` 图必须包含且只包含一个 `trainer` 节点, 一个 `model` 节点, 一个 `optimizer` 节点, 至少一个 `loss` 节点, 至少一个接到 `trainer.train_loader` 的 `dataloader`. 可选 `val_loader`, `scheduler`, `checkpoint`, 多个 `metric`.

共享测试快照包括 [CIFAR-10 标准图](<../../../dashboard/templates/workflows/cifar10-demo.json>)、[CPU 冒烟图](<../../../dashboard/templates/workflows/synthetic-cpu-demo.json>)、[分支合并图](<../../../dashboard/templates/workflows/dag-runtime-demo.json>) 和 [CPU 双层训练图](<../../../dashboard/templates/workflows/step-graph-cpu-demo.json>).

### schema-v1 lowering 规则

| 节点 | 落点 |
| --- | --- |
| `transform` 链 (按端口串联) | `train_transforms` / `val_transforms`, 生成 `Compose(transforms=[...])` |
| `dataset` | `train_dataset` / `val_dataset`, `params.transform: ${train_transforms}` |
| `dataloader` | `train_dataloader` / `val_dataloader`, `dataset: ${train_dataset}`, `params` 来自节点参数 |
| `model` | `model: {target, params}` |
| `loss` (可多个) | `loss: [{target, params, weight}]` |
| `optimizer` | `optimization.optimizer: {target, params}` |
| `scheduler` | `optimization.scheduler: {target, params}` |
| `metric` (可多个) | `metrics: [{target, params}]` |
| `trainer` | `trainer: {max_epochs, batch_size, gradient_accumulation_steps, grad_clip_max_norm, precision}` |
| `checkpoint` | `checkpoint: {save_last, save_top_k, monitor, mode, every_n_epochs}` |
| `settings.runtime.seed` | `runtime.seed` |

`train` / `val` 的归属完全由 `trainer` 节点的 `train_loader` / `val_loader` 输入端口反推, 不靠节点顺序或命名猜测. 无法反推到 `train_loader` 的 dataset 分支报 `unreachable_node`.

### 执行 profile

| profile | 支持图 | 执行入口 |
| --- | --- | --- |
| `schema-v1` | 目录中 `profiles` 含 `schema-v1` 的节点, 且能按上表 lowering | 生成 YAML -> `setup_from_yaml()` -> `Trainer` |
| `dag-runtime` | 目录中 `profiles` 含 `dag-runtime` 的节点, 含分支, 多输入, 合并 | `xdl.workflow.executor` 构建 `TrainSetup` -> `Trainer.from_setup()` |

`identity`, `compose`, `merge`, `sink`, `dataset_split` 和 `collate` 只支持 `dag-runtime`. 图里出现这些节点但 `execution_profile` 是 `schema-v1` 时报 `unsupported_profile`.

### 装配节点的载荷

- `dataset_split.dataset -> train / val`: 输入可索引数据集, 输出两个不重叠的子集. `val_ratio` 控制比例, `seed` 可覆盖 `settings.runtime.seed`, 均未指定时用 42. 输入至少有两个样本.
- `collate.collate -> dataloader.collate`: 可选的批次整理函数. 默认整理可处理同形状张量, `DictCollate` 需要字典样本, `PadCollate` 面向变长张量. `dataloader.batch_size` 显式值优先于旧 `trainer.batch_size` 默认来源; `gradient_accumulation_steps` 是独立的优化设置.
- `identity` 原样保留上游类型; `compose` 按 `left` 后 `right` 串联 transform, 或对两个 loss 等权求和后归一化. `merge` 只拼接数据集; 已知样本结构或合成样本形状不一致时预检拒绝. `sink` 只消费输入, 不产生文件或指标.
- `dag-runtime` 检查沿透传链的端口类型、模型与优化器绑定、优化器与调度器绑定, 以及目录中可确认的样本和模型契约. 运行时张量形状不靠静态猜测.

### 训练步图

装配图构建数据和训练组件, 训练步图描述如何处理一个 batch. 训练与验证共用训练步图; 只有训练步调用 CoreModel 的现有手动优化接口. 梯度累积、检查点与控制仍走 XDL 生命周期. 调度器只在优化器实际更新后步进. 省略 `step_graph` 时沿用旧训练适配器.

| 节点 | 输入与输出 | 属性 |
| --- | --- | --- |
| `BatchInput` | 无输入, 输出 `batch` | 恰好一个 |
| `Select` | `batch` -> 一个 `tensor` | `index` (tuple/list, 从 0 开始) 和 `key` (字典) 二选一 |
| `Forward` | `features` -> `predictions` | 使用 Trainer 装配的模型 |
| `LossCall` | `predictions`, `labels` -> `loss` | `component` 为接入 Trainer 的外层损失节点 ID; 返回 0 维浮点张量 |
| `MetricCall` | `predictions`, `labels` -> `value` | `component` 为外层指标节点 ID, `name` 为唯一日志名; 不参与梯度 |
| `WeightedSum` | `left`, `right` -> `loss` | `left_weight` 和 `right_weight` 默认 1, 非负, 不归一化 |
| `StepOutput` | 必需 `loss`, 可选多输入 `metrics` | 恰好一个, 交回优化用标量和日志指标 |

端口的批次、张量、标量和指标类型必须匹配; 不接受表达式或任意组件导入. 预检问题的路径以 `step_graph/` 开头, 运行期异常包含出错节点 ID. 当前仅支持单张量模型输入和输出, 不自动解包嵌套结构.

## 错误码

错误码在 `xdl/workflow/errors.py` 的 `ERROR_CODES` 中封闭声明. 新增错误码必须同时更新本文与前端映射.

| 错误码 | 产生层 | 含义 |
| --- | --- | --- |
| `invalid_snapshot` | Go, Python | 快照不是合法 JSON 对象, 字段类型或形状错误 |
| `unsupported_version` | Go, Python | `version` 缺失或不等于 1 |
| `duplicate_id` | Go, Python | 节点 id 或边 id 重复 |
| `unknown_node` | Go, Python | 边引用了不存在的节点 |
| `unknown_port` | Python | 引用了节点未声明的端口 |
| `missing_port` | Python | 必需输入端口没有入边 |
| `duplicate_edge` | Go, Python | 同一对 (源节点, 源端口, 目标节点, 目标端口) 重复连线 |
| `port_conflict` | Python | 非 `multi` 输入端口已经有一个来源 |
| `dangling_output` | Python | dag-runtime 图的输出端口没有消费方 |
| `self_loop` | Go, Python | 边连接节点自身 |
| `cycle_detected` | Go, Python | 图存在环 |
| `limit_exceeded` | Go, Python | 超出节点数, 边数, 深度, 字节数或长度上限 |
| `invalid_param` | Python | 参数名未声明, 缺必需参数, 类型或范围错误 |
| `unknown_node_kind` | Python | `kind` 不在节点目录 |
| `unknown_target` | Python | `target` 不在该 kind 的目录项 |
| `unsupported_profile` | Python | 节点或图不支持声明的执行 profile |
| `type_mismatch` | Python | 边两端端口类型不兼容 |
| `unreachable_node` | Python | 节点不在任何到 `trainer` 的路径上 |
| `missing_required_node` | Python | 缺少 profile 必需的节点角色 |
| `duplicate_role` | Python | 同一角色出现多次 (例如两个 `model`) |
| `path_not_allowed` | Python | `data_path` 是绝对路径或包含 `..` |
| `unsupported_graph` | Python | 图合法但不被当前 profile 的 lowering 支持 |
| `compiler_unavailable` | Go | 语义预检子进程不可用, 超时或返回不可解析结果 |

问题对象形状:

```json
{
  "code": "type_mismatch",
  "level": "error",
  "message": "边 e1: model.module (model) 不能连到 trainer.loss (loss)",
  "path": "edges/0/to/port",
  "node_id": "trainer",
  "edge_id": "e1"
}
```

`level` 只取 `error` 或 `warning`. 只要有 `error`, 快照就不能排队, worker 不能被启动.

## Canonical JSON 与哈希

- Go 写入数据库和磁盘的快照是 canonical JSON: 对象键排序, 紧凑分隔符, `nodes` 与 `edges` 按 `id` 排序.
- `workflow_snapshot_hash` = 该 canonical JSON 字节的 SHA-256 (小写十六进制).
- 数字按提交时的字面量保留 (Go 侧使用 `json.Number`), 避免 `1` 与 `1.0` 产生不同哈希.
- Python 侧不重新实现 canonical 化: worker 直接对 Go 写出的文件字节计算 SHA-256, 与 manifest 中的 `workflow_snapshot_hash` 比对. 不一致时在训练启动前报错.
- `xdl.workflow.WorkflowSnapshot.sha256()` 只用于 Python 内部比对与测试, 不承诺与 Go 字节一致.

## HTTP API

所有端点只通过 loopback / SSH tunnel 使用. 错误响应沿用 `{"error": {"code", "message"}}`; 校验类失败额外带 `issues` 数组.

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/workflows/catalog` | 原样返回节点目录 JSON, 响应体为 `{"catalog": {...}}` |
| `POST` | `/api/workflows` | 请求体是快照; 结构校验 + 语义预检通过后创建, 返回 `201` 与 workflow 对象. `id` 一律由服务端分配, 请求体里的 `id` 不参与语义 |
| `GET` | `/api/workflows?limit=` | 返回 `{"workflows": [workflowSummary]}`, 不含 `snapshot` |
| `GET` | `/api/workflows/:id` | 返回 workflow 对象, 含 `snapshot` |
| `POST` | `/api/workflows/validate` | 请求体 `{"snapshot": {...}}`, 返回 `{"valid", "issues", "snapshot_hash"}`. 只有 `level: "error"` 的问题会让 `valid` 变成 `false` |
| `POST` | `/api/workflows/:id/runs` | 以不可变快照创建 run, 返回 `202` 与 run 对象 |

workflow 对象:

```json
{
  "id": "wf-7f3c1d2e",
  "name": "CIFAR-10 VGG-11 标准训练图",
  "version": 1,
  "execution_profile": "schema-v1",
  "snapshot_hash": "9f2c...",
  "created_at": "2026-11-18T01:00:00Z",
  "updated_at": "2026-11-18T01:00:00Z",
  "snapshot": { "...": "canonical snapshot" }
}
```

run 请求体:

```json
{
  "name": "cifar-run-1",
  "launcher": "none",
  "devices": [0],
  "parent_run_id": "",
  "resume_checkpoint": ""
}
```

省略 `launcher` / `devices` 时使用快照 `settings` 中的值. 其余队列规则 (GPU 互斥, 多卡必须 launcher, checkpoint 恢复沿用父 run) 与模板 run 完全一致.

模板 run 路径 `POST /api/runs` 保持不变, 旧模板不依赖 workflow.

## Python compiler 契约

模块: `dashboard.worker.workflow_compiler`. CLI: `python -m dashboard.worker.workflow_cli`, 从 stdin 读请求, 向 stdout 写结果, 不使用命令行参数解析库.

请求:

```json
{
  "action": "validate",
  "snapshot": { "...": "snapshot" },
  "options": {
    "data_root": "/abs/data",
    "project_root": "/abs/repo",
    "output_path": "/abs/run/config.yaml",
    "system_overrides": {"runtime.output_dir": "/abs/run"},
    "check_registry": true
  }
}
```

- `action` 取 `validate` 或 `compile`.
- `output_path` 与 `system_overrides` 只对 `compile` 生效.
- `check_registry` 为真时额外校验 target 在 XDL registry 中真实存在 (需要能 import xdl 与 torch).

响应:

```json
{
  "ok": true,
  "action": "validate",
  "compiler_version": "workflow-compiler/1",
  "execution_profile": "schema-v1",
  "issues": [],
  "warnings": [],
  "config": { "...": "lowered schema v1 mapping" },
  "config_path": "/abs/run/config.yaml",
  "config_hash": "ab12..."
}
```

- `ok` 为 `false` 时 `issues` 至少有一条 `error`, 且不产生 `config_path`.
- `config` 与 `config_path` / `config_hash` 只在 `action == "compile"` 时出现.
- 进程退出码: `0` 表示请求被正常处理 (无论 `ok`), `2` 表示请求本身不可解析.

## Worker manifest 契约

模板 run 的 manifest 保持现状 (`template_path`, `overrides`, `allowed_paths`, `system_overrides`). workflow run 的 manifest 新增:

```json
{
  "kind": "workflow",
  "workflow_path": "/abs/run/workflow.json",
  "workflow_snapshot_hash": "9f2c...",
  "execution_profile": "schema-v1",
  "config_path": "/abs/run/config.yaml",
  "control_path": "/abs/run/control.json",
  "data_root": "/abs/data",
  "project_root": "/abs/repo",
  "system_overrides": {"runtime.output_dir": "/abs/run"},
  "resume_checkpoint": null,
  "metrics_interval_s": 2.0,
  "metrics_every_n_steps": 1
}
```

Go 只在创建快照和排队时做结构校验; 语义预检在创建时调用 Python compiler 并缓存结果, 因为快照不可变. worker 仍会在启动训练前重新编译一次, 编译失败只产生 `run_error`, 不启动训练进程.

worker 处理顺序:

1. 读取 `workflow_path`, 校验字节哈希等于 `workflow_snapshot_hash`, 不一致立即 `run_error`, 不启动训练.
2. 调用 compiler 做语义校验与编译. 失败时发 `run_error`, 附带 `issues` 与字段路径.
3. `schema-v1` profile: 写 `config.yaml`, 发 `workflow_compiled` 事件, 走 `setup_from_yaml()`.
4. `dag-runtime` profile: 由 `xdl.workflow.executor` 构建 `TrainSetup`, 走 `Trainer.from_setup()`.
5. 训练生命周期, 控制文件, DashboardCallback 与模板 run 完全一致.

`workflow_compiled` 事件载荷: `{"compiler_version", "execution_profile", "config_path", "config_hash"}`. Go 侧记录到 `runs.compiler_version`, 供 run 详情展示.

## ADR

### ADR-1 Python compiler 是 XDL 语义事实源

Go 只做 envelope, 规模, 引用和环检测, 以及资源和进程管理. 节点目录, 端口类型, 参数语义, target 存在性和 lowering 只在 Python 实现一份. 代价是预检需要调用 Python 子进程, 因此需要超时, 临时目录和明确的错误映射; 拿不到 Python 时预检报错而不是放行.

### ADR-2 快照不直接存成可变 YAML

数据库保存 canonical JSON 快照与哈希, 每次 run 固化一份快照文件. YAML 只是编译产物. 这样 UI 布局, 端口关系和 profile 可以版本化, 可重放, 可比较.

### ADR-3 schema-v1 与 dag-runtime 并存

标准训练图继续走 `setup_from_yaml()`, 保持既有训练 API 和旧模板稳定. 真正超出固定训练槽位的图走新 runtime. 不把任意 DAG 塞进顶层 schema, 避免隐式转换和不可解释的特殊情况.

### ADR-4 第一阶段禁止任意 Python 节点

节点 `target` 必须来自节点目录, 不允许浏览器提交 import 路径或代码字符串. dashboard 走 loopback / SSH 不等于输入可信; 任意代码执行会扩大权限, 审计和复现风险.

## 验收

```bash
cd dashboard/frontend && pnpm run test && pnpm run build
cd ../backend && go test ./...
cd ../..
python -m pytest dashboard/worker tests/workflow tests/config -q
ruff check dashboard/worker xdl/workflow tests
ruff format --check dashboard/worker xdl/workflow tests
```

真实浏览器检查: 构建前端后启动 dashboard 服务, 在 1440 / 1280 / 768 / 390 px 下操作节点图, 观察 network, console, SSE 与生成 artifact, 确认无 console error 与页面级横向溢出.

## 相关源码

- [xdl/workflow/node_catalog.json](../../../xdl/workflow/node_catalog.json) - 节点目录唯一实体来源.
- [xdl/workflow/errors.py](../../../xdl/workflow/errors.py) - 错误码与问题对象.
- [xdl/workflow/ir.py](../../../xdl/workflow/ir.py) - 快照结构模型与结构校验.
- [xdl/workflow/catalog.py](../../../xdl/workflow/catalog.py) - 节点目录加载与参数校验.
- [xdl/workflow/executor.py](../../../xdl/workflow/executor.py) - dag-runtime 执行器.
- [dashboard/templates/workflows/](../../../dashboard/templates/workflows/) - 共享图 fixture.
- [dashboard/README.md](../../../dashboard/README.md) - dashboard 运行与 API 说明.
