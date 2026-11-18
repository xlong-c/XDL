# XDL Training Observatory

`dashboard/` 是 XDL 的可选训练控制台. 它不改变 `CoreModel + Trainer + Callback` 训练链路, Go 进程负责队列,资源互斥,SQLite 历史和 SSE, Python worker 负责调用 XDL.

## Build

```bash
cd dashboard/frontend
pnpm install
pnpm run test
pnpm run build

cd ../backend
go build -o ../bin/xdl-dashboard ./cmd/xdl-dashboard
```

Windows 下把输出名写成 `../bin/xdl-dashboard.exe`.

构建产物按目录分层, 源码目录不放二进制:

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| Go 源码 | `dashboard/backend/` | `cmd/xdl-dashboard` 是入口, `internal/` 是内部包 |
| Go 最终产物 | `dashboard/bin/xdl-dashboard` (`*.exe`) | `go build` 输出, 已 gitignore |
| Go 中间产物 | 默认进 `GOCACHE`, 不入仓库 | 手工 `go test -o` / `go test -coverprofile` 产物已在 gitignore 覆盖 |
| 前端源码 | `dashboard/frontend/` | 依赖装在 `node_modules/`, 不入库 |
| 前端最终产物 | `dashboard/frontend/dist/` | 由 Go 服务托管, 已 gitignore |

前端依赖和产物都不会进入 XDL Python 基础安装.

## Run

从仓库根目录复制配置并按需修改:

```bash
cp dashboard/config.example.json dashboard/config.json
./dashboard/bin/xdl-dashboard -config ./dashboard/config.json
```

开发时也可以直接 `go run`, 不要从仓库根执行 (根目录没有 `go.mod`):

```bash
cd dashboard/backend
go run ./cmd/xdl-dashboard -config ../config.json
```

默认只监听 `127.0.0.1:8765`. 浏览器访问:

```text
http://127.0.0.1:8765/
```

配置路径会在启动时解析为绝对路径. `database_path`, `output_root`, `data_root`, `frontend_dir`, `template_catalog` 和 `workflow_catalog` 均相对于 `project_root` 解析.

`workflow_catalog` 指向 `xdl/workflow/node_catalog.json`, 是工作流节点目录的唯一实体来源; Go 只读取其中的 `limits` 并把整份 JSON 转发给前端, 不复制节点语义. `compiler_timeout` 限制语义预检子进程的等待时间, 默认 `60s`.

配置中的 `python_executable` 必须指向已安装 XDL 和 PyTorch 的解释器. 如果服务环境只有 `python3`,默认值会尝试 fallback,但它不保证包含 XDL 依赖;生产部署应填写虚拟环境或 conda 环境的绝对路径.

## SSH-only access

服务默认不实现登录系统,也不暴露公网监听. 在工作站上运行 dashboard,从客户端建立 SSH 隧道:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@training-host
```

然后访问本地 `http://127.0.0.1:8765/`. 不要把 `listen_addr` 改为 `0.0.0.0`,除非先在外层增加认证和 TLS.

## Linux systemd

复制并修改 `dashboard/deploy/xdl-dashboard.service.example`. 关键点是使用专用服务用户,固定 `WorkingDirectory`,以及只允许读取 XDL 项目和写入 dashboard output/database 的权限.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now xdl-dashboard
journalctl -u xdl-dashboard -f
```

## Windows

在 PowerShell 中从仓库根目录启动:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& .\dashboard\bin\xdl-dashboard.exe -config .\dashboard\config.json
```

Windows 多 GPU 使用与 Linux 相同的手动 GPU index 和 `CUDA_VISIBLE_DEVICES` 约束. `nvidia-smi` 必须在 `PATH` 中,或安装在 NVIDIA 默认 `NVSMI` 目录.

## Training workflow

1. 在 `New run` 中选择 XDL 模板.
2. 表单字段只允许 catalog 中的 allowlist,数据路径只能位于 configured `data_root`.
3. 选择 CPU 或实际 NVIDIA GPU index. 同一 GPU 不会被两个活动任务同时占用.
4. 单卡可以使用 single process. 多卡必须选择 `torchrun` 或 `accelerate`.
5. 运行状态,标量指标,stdout/stderr 和 host telemetry 通过 SQLite + SSE 更新. 指标和系统采样默认约 2 秒批量刷新,不会阻塞训练.
6. `Pause` 在 batch 边界生效. `Stop` 和 `Save checkpoint and pause` 等待安全边界并使用现有 `ModelCheckpoint` 保存最后 checkpoint.
7. 从 checkpoint 恢复会创建新的 run record,并保留 parent run 关系.

SQLite 历史不会按时间自动删除. 训练输出目录由每个 run 单独隔离;图片预览只允许读取 run output 内的 PNG/JPEG/WebP 文件.

## Workflow editor

`Workflow editor` 视图把画布上的有类型 DAG 变成真实训练对象, 完整契约见
[docs/md/architecture/dashboard-workflow-ir.md](../docs/md/architecture/dashboard-workflow-ir.md).

1. `装配图` 定义数据,模型和优化组件以及它们的接线. DataLoader 可单独设置批量大小, 也可接入数据拆分和批次整理节点.
2. `训练步` 定义如何取出批次张量,运行前向,计算损失和指标. 可从 `插入分类训练步` 开始, 再改端口连线和组件引用. 启用训练步即切换到 `dag-runtime`; 不启用则沿用旧计算逻辑.
3. 连线必须类型兼容; 一个输入端口只允许一个来源, 目录声明为 `multi` 的端口 (例如 `trainer.loss`) 例外.
4. `校验并排队` 依次执行: 本地结构校验 -> `POST /api/workflows/validate` 的 Python 语义预检 -> `POST /api/workflows` 固化快照 -> `POST /api/workflows/:id/runs` 入队.
5. 快照是不可变 JSON: 创建时分配 `wf-` id 并记录 `snapshot_hash`, run 目录固化一份 `workflow.json`, 后续修改 workflow 不影响历史 run.
6. 未提交的草稿只保存在浏览器 localStorage; 服务端只接收通过预检的快照.
7. 非法图在 worker 启动前被拒绝, 不会产生训练进程.

### 执行 profile

| profile | 适用图 | 执行方式 |
| --- | --- | --- |
| `schema-v1` | 能 lowering 到现有 schema v1 的标准训练图 | 生成 `config.yaml` -> `setup_from_yaml()` -> `Trainer` |
| `dag-runtime` | 含分支, 多输入或合并的图 (如 `merge` / `compose`) | `xdl.workflow.executor` 构建 `TrainSetup` -> `Trainer.from_setup()` |

两个 profile 共用同一套队列, GPU 互斥, 控制命令, checkpoint 和 SSE 通道.

### 共享示例图

以下快照是各层测试共用的输入:

- [CIFAR-10 标准图](<templates/workflows/cifar10-demo.json>): 15 个节点, 与 [YAML 示例](<../config/vgg_cifar10_dashboard_demo.yaml>) 的关键字段等价.
- [CPU 冒烟图](<templates/workflows/synthetic-cpu-demo.json>): 6 个节点, 走 `schema-v1`.
- [分支合并图](<templates/workflows/dag-runtime-demo.json>): 走 `dag-runtime` 的旧训练逻辑.
- [CPU 双层训练图](<templates/workflows/step-graph-cpu-demo.json>): 数据拆分,`DictCollate`,按名称取批次字段, 训练与验证共用同一个子图.

### CPU 端到端验证

从仓库根目录运行以下命令. `python` 应来自已安装 XDL 和 PyTorch 的环境.

```bash
(cd dashboard/backend && XDL_DASHBOARD_E2E_PYTHON="$(command -v python)" go test ./internal/api -run '^TestWorkflowTrainingE2E$' -count=1 -v)
```

测试通过 HTTP handler 创建工作流和 run, 使用临时 SQLite 数据库及真实 Python worker 验证两个 profile 的最终训练步数和快照关联. 不启动监听服务, 不需要 GPU 或下载数据. 未设置 `XDL_DASHBOARD_E2E_PYTHON` 时, 常规 Go 测试会跳过此项.

## API quick reference

- `GET /api/health`
- `GET /api/templates`
- `GET /api/system`
- `GET /api/system/stream`
- `GET /api/runs`
- `POST /api/runs`
- `GET /api/runs/:id`
- `GET /api/runs/:id/metrics`
- `GET /api/runs/:id/events`
- `GET /api/runs/:id/stream`
- `POST /api/runs/:id/actions/pause|resume|stop|checkpoint-pause|kill`
- `POST /api/runs/:id/actions/resume-from-checkpoint`
- `GET /api/runs/:id/artifacts`
- `GET /api/runs/:id/artifact/:path`
- `GET /api/workflows/catalog`
- `POST /api/workflows`
- `GET /api/workflows?limit=`
- `GET /api/workflows/:id`
- `POST /api/workflows/validate`
- `POST /api/workflows/:id/runs`

所有 API 都只应通过 loopback/SSH tunnel 使用. SSE 支持 `Last-Event-ID` 重连回放.

工作流相关端点失败时, 错误响应在 `error` 之外额外带 `issues` 数组, 每项包含 `code`, `level`,
`message`, 可选 `path` / `node_id` / `edge_id`. 只有 `level: "error"` 会阻断创建与排队.
