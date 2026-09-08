# xdl/config - 配置构建子模块

## 目录职责

- 负责 YAML 到 `TrainSetup` 的解析,校验与组件构建
- 承担配置 schema,引用解析,builder 组装和错误定义

## 核心文件

- `setup.py`:`setup_from_yaml()` 入口
- `dataclass.py`:`TrainSetup` 数据结构
- `builder.py`:`build_model()`,`build_optimizer()` 等构建流程
- `resolver.py`:基于 `OmegaConf` 的配置引用,变量解析,合并与容器转换
- `schema.py`:配置版本与合法性约束
- `errors.py`:配置系统异常

## 修改约束

- 优先保持 `setup_from_yaml()` 的用户接口稳定
- `TrainSetup` 只保留组件和结构化配置 (`trainer / runtime / logging / checkpoint / accelerate / deepspeed`); 不新增重复的扁平训练字段, 默认值统一由 `schema.py` 的 dataclass 提供
- YAML 的 `loss` 和 `metrics` 必须是列表; 单个 dict 不再隐式归一为单元素列表
- 新字段要同步考虑 dataclass,builder,schema 和测试
- 新增或重构 YAML/层级配置解析时,优先使用 `OmegaConf.load/create/merge/resolve/to_container` 和 `DictConfig`/`ListConfig`,不要新增分散的 `yaml.safe_load` + 手写递归合并逻辑
- 轻量结构化配置加载优先遵循 `dataclass/structured config` 定义默认值和 schema, 再由 YAML 直接覆盖的顺序. 除非有明确兼容需求, 不要在 `load_config` 中额外做路径重写, 字符串 `"null"` 兼容, 旧字段迁移, clamp/奇偶修正, 或 list/tuple 强转. 需要约束时优先让 schema/OmegaConf 报错, 或在业务使用处显式校验
- 解析逻辑优先结构化处理,不靠脆弱字符串拼接
- 配置错误应抛清晰异常,便于定位字段问题

## 验证建议

- 修改后先跑 `tests/config/`
- 涉及 registry 构建时联动检查 `xdl/utils/registry.py`
