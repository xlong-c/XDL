# Codex 多供应商配置指南

## 工作原理

利用 Codex CLI 的 `-p` / `--profile` 参数，加载 `~/.codex/<name>.config.toml` 叠加到基础配置（`~/.codex/config.toml`）之上。切换 profile = 切换供应商 + API key。

## 当前供应商一览

| 别名 | Profile 文件 | Provider 名 | API 地址 |
|------|-------------|------------|----------|
| `codex` (默认) | `config.toml` | ccswitch | fast.sbbbbbbbbb.xyz |
| `codex-ccs` | `ccswitch.config.toml` | ccswitch | fast.sbbbbbbbbb.xyz |
| `codex-lin` | `linbot.config.toml` | OpenAI | sub.linbot.top |
| `codex-nec` | `necodex.config.toml` | necodex | fast.sbbbbbbbbb.xyz |

## 使用方式

```bash
codex "你的 prompt"           # 默认供应商 (ccswitch)
codex-lin "你的 prompt"       # linbot
codex-nec "你的 prompt"       # necodex
codex-ccs "你的 prompt"       # ccswitch (显式)
codex-with linbot "prompt"    # 通用切换
codex-list                    # 列出所有供应商
```

## 新增供应商

### 1. 创建 profile 文件

在 `~/.codex/` 下新建 `<供应商名>.config.toml`：

```toml
model_provider = "供应商代号"
model = "gpt-5.5"
model_reasoning_effort = "xhigh"

[model_providers.供应商代号]
name = "显示名称"
base_url = "https://中转地址/v1"
experimental_bearer_token = "中转给你的key"
wire_api = "responses"
```

### 2. 添加 shell 别名

编辑 `~/.bashrc`，在 `# >>> Codex 多供应商快捷切换 >>>` 区域添加：

```bash
codex-xxx()    { codex -p 供应商名 "$@"; }   # xxx 换成想要的别名
```

### 3. 生效

```bash
source ~/.bashrc
```

## 认证方式说明

profile 中两种认证方式：

| 方式 | 配置 | 适用场景 |
|------|------|---------|
| `experimental_bearer_token` | 直接在 provider 块里写 key | 中转兼容 bearer token 格式 |
| `requires_openai_auth = true` | 读 `OPENAI_API_KEY` 环境变量 | 中转只认标准 OpenAI auth 路径 |

如果 `experimental_bearer_token` 方式返回 401，改用环境变量方式：

**profile 文件**中写：
```toml
requires_openai_auth = true
# 不写 experimental_bearer_token
```

**.bashrc 包装函数**中注入 key：
```bash
codex-xxx() {
    OPENAI_API_KEY="中转的key" codex -p 供应商名 "$@"
}
```

这样 key 不会硬编码在配置文件中，也更符合 OpenAI 兼容中转型的认证规范。

## 文件位置

- Profile 文件：`~/.codex/*.config.toml`
- 基础配置：`~/.codex/config.toml`
- Shell 函数：`~/.bashrc`（搜索 `Codex 多供应商`）
- Codex 认证存储：`~/.codex/auth.json`
