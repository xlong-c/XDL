# Dashboard 任意 DAG 工作流实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 dashboard 的节点画布从本地 UI 草稿升级为可持久化,可校验,可编译并能真实启动训练的有类型 DAG 工作流系统.

**Architecture:** 前端编辑统一的 Workflow IR, Go dashboard 负责快照持久化,队列和资源调度, Python compiler 负责 XDL 语义校验和配置生成. 现有 schema v1 作为标准训练图的兼容后端; 超出固定训练角色的 DAG 进入独立的 XDL workflow runtime, 不修改 `setup_from_yaml()` 的稳定入口.

**Tech Stack:** Preact + TypeScript, Go `net/http` + SQLite, Python + OmegaConf, XDL `CoreModel + Trainer + Callback`.

---

## 1. 现状与必须解决的问题

当前 dashboard 已经具备真实训练监控和队列能力, 但工作流编辑器的排队逻辑只取第一个模板的默认参数. 节点的 `type`, `target`, 名称, 参数和连线没有进入后端, 也没有生成训练配置.

现有 XDL 配置入口是固定 schema v1. 它可以表达模型,数据集,变换,DataLoader,优化器,损失,指标和 Trainer, 但不能表达任意执行图. 因此"任意 DAG 一次到位"必须拆成两个可验证层次:

1. **通用 Workflow IR:** 支持节点,端口,边,参数,位置,版本和执行 profile.
2. **执行后端:** 标准训练 DAG 编译到现有 schema v1; 需要分支,多输入,合并或自定义执行语义的 DAG 编译到新增 workflow runtime.

不允许把任意 Python 源码直接从浏览器传入并执行. 自定义代码能力必须另立白名单和部署边界, 不纳入第一阶段.

## 2. 推荐目标架构

```text
Preact Workflow Editor
        |
        | WorkflowSnapshot v1
        v
Go API / Store
  - structural validation
  - snapshot persistence
  - run association
  - GPU/queue scheduling
        |
        | validate / compile request
        v
Python Workflow Compiler
  - node catalog and port contracts
  - cycle and type validation
  - semantic validation against XDL registry
  - lower to schema-v1 YAML or workflow runtime manifest
        |
        +--> schema-v1 YAML --> setup_from_yaml() --> Trainer
        |
        +--> workflow runtime manifest --> WorkflowExecutor --> Trainer/CoreModel
        |
        +--> DashboardCallback --> Go events / SQLite / SSE
```

### 2.1 责任边界

- **前端:** 编辑图,展示端口与错误,保存草稿,调用预检和提交. 前端校验只用于交互反馈, 不是安全边界.
- **Go:** 校验 JSON envelope,限制快照大小和字段,持久化不可变 snapshot,创建 run,管理 GPU 互斥和进程生命周期. Go 不重复实现 XDL component builder.
- **Python compiler:** 负责 DAG 语义的唯一事实源, 使用 OmegaConf 和 XDL registry, 输出规范化 IR,schema v1 YAML 或 runtime manifest.
- **XDL:** 保持 `setup_from_yaml()` 兼容. 新增 runtime 只通过明确的新入口接入, 不在 `TrainSetup` 中增加重复扁平字段.

## 3. WorkflowSnapshot v1 数据契约

建议快照结构如下:

```json
{
  "version": 1,
  "id": "workflow-uuid",
  "name": "cifar training graph",
  "execution_profile": "schema-v1",
  "nodes": [
    {
      "id": "model-1",
      "kind": "model",
      "target": "registry:vgg11_bn",
      "params": {"num_classes": 10},
      "position": {"x": 420, "y": 180},
      "metadata": {"label": "classifier"}
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "from": {"node": "dataset-1", "port": "samples"},
      "to": {"node": "loader-1", "port": "dataset"}
    }
  ],
  "settings": {
    "runtime": {},
    "launcher": "none",
    "devices": []
  }
}
```

约束:

- `version` 必须显式存在, 未知版本拒绝提交.
- `kind` 使用受支持的节点目录名称, 不直接执行前端传来的类名.
- `target` 必须由节点目录解析为允许的组件 target, 不允许任意 import path.
- `params` 只接受 JSON scalar,数组和对象, 限制深度,节点数,边数和总字节数.
- `edges` 只引用已存在节点和端口; 不再用当前 `inputs: string[]` 表达连接.
- `position` 属于 UI 元数据, 不参与训练语义.
- `execution_profile` 明确选择 `schema-v1` 或 `dag-runtime`; 未声明时拒绝而不是猜测.

## 4. 分阶段实施

### Phase 0: 冻结契约与执行语义

**目标:** 在写 UI 和后端之前固定 IR,节点目录和 profile 边界.

**产出:**

- `WorkflowSnapshot v1` JSON Schema 或等价结构化模型.
- 节点目录: kind,输入端口,输出端口,参数 schema,允许 target,编译 profile.
- 初始节点集合: dataset,transform,dataloader,model,loss,optimizer,trainer,checkpoint, 以及为 DAG 验证准备的 identity,compose,merge 等无副作用节点.
- 错误码表: `invalid_snapshot`, `unknown_node_kind`, `unknown_target`, `missing_port`, `cycle_detected`, `type_mismatch`, `unsupported_profile`.
- ADR: snapshot 版本化,Python compiler 为语义事实源,schema-v1 与 dag-runtime 双 profile.

**关键决策:**

- 标准训练图可以真实编译到现有 `config_version: 1`.
- 任意 DAG 只有在端口类型和执行 profile 支持时才可排队.
- 不引入浏览器执行 Python, 不把任意源码当作普通节点参数.

### Phase 1: 后端数据模型与持久化

**目标:** 快照成为数据库和 run 记录的一等对象.

**Files:**

- Create: `dashboard/backend/internal/workflow/model.go`
- Create: `dashboard/backend/internal/workflow/validator.go`
- Modify: `dashboard/backend/internal/store/schema.sql`
- Modify: `dashboard/backend/internal/store/store.go`
- Modify: `dashboard/backend/internal/store/store_test.go`
- Modify: `dashboard/backend/internal/queue/manager.go`
- Modify: `dashboard/backend/internal/api/server.go`
- Modify: `dashboard/backend/internal/api/design_test.go` 或新增 workflow API 测试文件

**步骤:**

1. 为 snapshot,node,port,edge,issue 定义 Go 结构体和严格 JSON 解码.
2. 实现结构校验: 版本,唯一 ID,引用完整性,节点/边数量,参数深度和 JSON 大小.
3. 实现拓扑排序和 cycle detection. 同时拒绝自环和重复 edge.
4. 添加 SQLite `workflows` 表, 保存 canonical JSON,version,name,created_at,updated_at.
5. 给 `runs` 增加 `workflow_id`,`workflow_snapshot_hash` 或等价关联字段, 保留提交时的不可变快照引用.
6. 新增 API:
   - `POST /api/workflows` 创建快照.
   - `GET /api/workflows` 列表.
   - `GET /api/workflows/:id` 读取快照.
   - `POST /api/workflows/validate` 做 Go 结构预检.
   - `POST /api/workflows/:id/runs` 以快照创建 run.
7. 保留现有 `POST /api/runs` 模板路径, 做向后兼容, 不让旧模板 run 强依赖 workflow.
8. 在 run manifest 中写入 `workflow_id`, `workflow_snapshot_path`, `workflow_snapshot_hash` 和 execution profile.

**测试:**

- 合法 DAG 可保存,重启数据库后可读取.
- 缺失节点,重复 ID,未知端口,悬空边,超限参数和 cycle 都被拒绝.
- 同一个 snapshot 创建 run 后, 后续修改 workflow 不改变 run 使用的快照.
- 旧模板提交和恢复 checkpoint 的现有测试继续通过.

### Phase 2: Python compiler 与 schema-v1 lowering

**目标:** 将标准训练 DAG 编译为 XDL schema v1 YAML, 并使用现有配置校验链.

**Files:**

- Create: `dashboard/worker/workflow_model.py`
- Create: `dashboard/worker/workflow_catalog.py`
- Create: `dashboard/worker/workflow_validator.py`
- Create: `dashboard/worker/workflow_compiler.py`
- Create: `dashboard/worker/test_workflow_validator.py`
- Create: `dashboard/worker/test_workflow_compiler.py`
- Modify: `dashboard/worker/train_worker.py`
- Modify: `dashboard/worker/config_builder.py` only for shared safe output helpers if needed
- Modify: `xdl/config/schema.py` only when schema-v1 needs a missing typed field
- Modify: `tests/config/` for any schema contract additions

**步骤:**

1. 用 dataclass 或 TypedDict 定义与 Go 契约一致的 Python IR, 读取后先做显式版本检查.
2. 加载节点目录, 把 UI label 和 target 与实际 registry/component target 分离.
3. 实现语义检查:
   - 节点 kind 与 profile 兼容.
   - 每个必需端口恰好有满足约束的输入.
   - 输入输出类型兼容.
   - 数据,模型,loss,optimizer 连接关系符合执行 profile.
   - 参数类型,范围和必需字段有效.
   - 图无环且可从输入到训练输出到达.
4. 对 `schema-v1` 实现 lowering:
   - dataset -> `train_dataset` / `val_dataset`.
   - transform -> `train_transforms` / `val_transforms`.
   - dataloader -> `train_dataloader` / `val_dataloader`.
   - model -> `model`.
   - loss -> `loss` 列表.
   - optimizer/scheduler -> `optimization`.
   - trainer/checkpoint -> `trainer` / `checkpoint`.
5. 使用 OmegaConf 生成临时 YAML, 通过 `load_config_with_schema(..., resolve=True)` 和 `setup_from_yaml()` 做最终验证.
6. 输出 compiler result: canonical snapshot,generated config path,warnings,errors,compiler version,config hash.
7. 将 `train_worker.py` 改为识别 workflow manifest: 先编译和校验, 再执行现有 `Trainer` 路径. 编译失败要产生 `run_error` 和明确字段路径, 不启动训练.
8. 增加一个 CLI/内部函数供 Go 的 validate endpoint 调用, 使用超时,临时目录和受限输入.

**测试:**

- 标准 CIFAR 示例 DAG 生成的 YAML 与现有 `config/vgg_cifar10_dashboard_demo.yaml` 在关键字段上等价.
- target 不在节点目录,参数类型错误,缺少端口,cycle 和不支持的分支都返回可定位错误.
- 生成 YAML 可被 `setup_from_yaml()` 构建.
- 编译器拒绝绝对数据路径,路径穿越,未知 target 和过大的 params.
- 旧 `build_template_config()` 测试继续通过.

### Phase 3: 前端从固定节点 UI 迁移到真正 DAG 编辑器

**目标:** 编辑器操作直接维护 Workflow IR, 而不是维护 `inputs: string[]` 的展示状态.

**Files:**

- Create: `dashboard/frontend/src/workflow/types.ts`
- Create: `dashboard/frontend/src/workflow/catalog.ts`
- Create: `dashboard/frontend/src/workflow/validation.ts`
- Create: `dashboard/frontend/src/workflow/serialization.ts`
- Modify: `dashboard/frontend/src/workflow_editor.tsx`
- Modify: `dashboard/frontend/src/api.ts`
- Modify: `dashboard/frontend/src/styles.css`
- Add frontend tests using the repository's chosen test setup, or introduce a small pure-function test target if none exists.

**步骤:**

1. 将节点状态改为 `nodes + edges + settings + version`.
2. 从节点目录渲染端口,参数表单和允许 target, 不在组件里硬编码一套独立 target 名称.
3. 支持拖拽连线,删除边,重复节点,复制节点,节点选择和键盘删除.
4. 在画布层显示 cycle,类型不匹配,缺失必需输入和不可达节点.
5. 参数输入使用 schema 类型控件, 保留数字,布尔和字符串真实类型, 不把全部输入转成字符串.
6. localStorage 只保存未提交草稿; 保存到服务端时生成 canonical snapshot.
7. `校验并排队` 改成:
   - 本地结构校验.
   - 调用 `/api/workflows/validate`.
   - 显示 errors/warnings.
   - 成功后调用 `/api/workflows/:id/runs`.
8. 在 run 详情中显示 workflow 名称,snapshot hash,compiler version 和生成配置 artifact.
9. 保留现有 monitor,run archive,SSE 和 checkpoint 控制, workflow run 进入同一队列.

**测试:**

- 添加,移动,连接,断开,复制和删除节点后 snapshot 序列化稳定.
- 页面刷新可恢复草稿, 服务端快照可重新载入.
- 画布上错误和后端错误显示一致.
- 窄屏布局不允许画布和检查器互相遮挡.
- 通过 Playwright 或等价浏览器检查桌面,平板和手机 viewport, 无 console error,无页面级横向溢出.

### Phase 4: dag-runtime

**目标:** 让不能 lowering 到固定 schema v1 的合法 DAG 有明确执行后端.

**Files:**

- Create: `xdl/workflow/ir.py`
- Create: `xdl/workflow/catalog.py`
- Create: `xdl/workflow/executor.py`
- Create: `xdl/workflow/errors.py`
- Create: `tests/workflow/`
- Modify: `dashboard/worker/workflow_compiler.py`
- Modify: `dashboard/worker/train_worker.py`
- Modify: `xdl/__init__.py` only if public export is justified

**范围:**

- 先实现有限,可验证的执行节点: dataset,transform,map,batch,model,loss,optimizer,metric,sink.
- 执行器按拓扑顺序构建组件, 对 batch/stream/state 类型做显式检查.
- 训练循环和 checkpoint 仍复用现有 Trainer 生命周期; 不在回调中塞入主训练逻辑.
- 需要多输入或分支的节点必须声明输入聚合策略, 不能依赖 Python 参数顺序猜测.
- 任何不支持的节点在 compile 阶段拒绝, 不进入半执行状态.

**测试:**

- 单输入线性图,transform 分支后 merge,双输入 loss 图的最小 CPU 测试.
- cycle,类型错接,未消费输出,重复 state writer 的错误测试.
- checkpoint,pause/resume,DashboardCallback 事件在 runtime 下仍然正确.
- 与 schema-v1 profile 的同一标准图结果保持一致的 smoke test.

### Phase 5: 端到端验收与运行安全

**目标:** 证明节点图真的控制了训练, 而不是只改变了 UI.

**验收场景:**

1. 在 dashboard 创建标准 CIFAR DAG, 修改 model target,batch size,epoch 和 optimizer lr.
2. 保存 snapshot, 刷新页面, 重新打开 snapshot.
3. 预检显示节点,端口和参数状态.
4. 排队后检查 run manifest,snapshot hash 和生成 YAML.
5. 在生成 YAML 中确认修改值存在, 并确认训练 worker 使用该文件.
6. 训练产生 metrics,stdout/stderr,checkpoint 和 SSE 状态变化.
7. pause,checkpoint-pause,resume-from-checkpoint,stop 和 kill 继续工作.
8. 提交非法 target,cycle,GPU 不存在,路径穿越和超大 snapshot, 确认不会启动 worker.
9. 重启 Go 服务, queued run,workflow snapshot 和历史记录可以恢复.

**命令:**

```bash
cd dashboard/frontend && pnpm run build
cd ../backend && go test ./...
cd ../..
python -m pytest dashboard/worker tests/config tests/callbacks -q
ruff check dashboard/worker xdl/config xdl/workflow tests
ruff format --check dashboard/worker xdl/config xdl/workflow tests
```

需要另行运行真实浏览器检查: 前端构建后启动现有 dashboard 服务, 在至少 1440,1280,768,390 px 下操作节点图并观察 network,console,SSE 和生成 artifact.

## 5. 关键 ADR 与取舍

### ADR-1: Python compiler 作为 XDL 语义事实源

**选择:** Go 只做 envelope,资源和结构校验; Python 复用 registry,OmegaConf 和 XDL config builder.

**原因:** 避免 Go 和 Python 各维护一份 target,参数和端口语义. 代价是 validate 请求需要调用 Python 或在 worker 阶段再次验证, 需要设置超时和错误映射.

### ADR-2: snapshot 不直接存成可变 YAML

**选择:** 数据库保存 canonical JSON snapshot 和 hash, 每次 run 固化一份 snapshot.

**原因:** UI 布局,端口关系和编译 profile 需要版本化; YAML 只作为编译产物. 这样可重放,可比较和可追溯.

### ADR-3: schema-v1 与 dag-runtime 并存

**选择:** 标准训练图继续走 `setup_from_yaml()`, 真正通用 DAG 走新 runtime.

**原因:** 直接把任意 DAG 塞进现有顶层 schema 会产生隐式转换和不可解释的特殊情况. 双 profile 增加实现量, 但能保持现有训练 API 和旧模板稳定.

### ADR-4: 第一阶段禁止任意 Python 节点

**选择:** 节点 target 必须来自受支持 catalog, 不允许浏览器提交 import path 或代码字符串执行.

**原因:** dashboard 通过 loopback/SSH 使用不等于输入可信; 任意代码执行会扩大权限,审计和复现风险. 后续若需要, 单独设计签名 artifact 和部署权限.

## 6. 预计完成度与里程碑

- **M1, 契约与后端持久化:** 20% -> 35%.
- **M2, schema-v1 编译闭环:** 35% -> 65%.
- **M3, 前端真实 DAG 编辑:** 65% -> 75%.
- **M4, dag-runtime 首批节点:** 75% -> 90%.
- **M5, 端到端验收与可运维性:** 90% -> 可交付.

在不实现 dag-runtime 之前, 系统可以交付"通用 DAG 编辑器 + 标准训练图真实运行"; 但必须明确标注不支持的图会被 preflight 拒绝, 不能让用户误以为所有任意图都能执行.

## 7. 完成定义

- 工作流 snapshot 有版本,hash,不可变 run 关联和服务端 API.
- 前端节点图和后端保存内容完全一致, 刷新后可恢复.
- 预检同时覆盖结构错误和 XDL 语义错误.
- 标准 schema-v1 图生成 YAML 并通过现有 XDL schema/setup 构建.
- dag-runtime 支持的图可在 CPU smoke test 和 dashboard 真实队列中运行.
- 非法图在 worker 启动前被拒绝, 不产生训练进程.
- 现有模板 run,监控,SSE,控制命令和 checkpoint 恢复没有回归.
- 所有新公开 API,配置字段和兼容边界同步更新 `dashboard/README.md` 与 `docs/md/` 事实文档.
