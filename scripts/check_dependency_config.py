#!/usr/bin/env python3
"""项目依赖静态扫描与配置校验工具。

本脚本会扫描仓库中的 Python import，引导你检查依赖声明是否正确。
直接在 main() 中修改配置后执行即可，不使用命令行参数解析库。

1. scan-imports-and-check-config
   作用:
   - 扫描项目内所有 Python 文件的 import / from import 语句。
   - 自动区分本地模块、标准库模块、第三方模块、无法解析的模块。
   - 自动读取依赖配置文件，例如 pyproject.toml、requirements*.txt。
   - 对照“实际引用的第三方包”和“配置中声明的包”，输出缺失、未安装、未使用等问题。

   执行后文件夹变化:
   - 会创建 output_dir 指定目录。
   - 会生成 dependency_report.json 和 dependency_report.txt。

   执行后环境变化:
   - 不会安装、卸载或修改当前 Python 环境中的任何包。
   - 只会读取源码、读取依赖配置、读取当前环境的已安装包信息。

推荐场景说明:
- 想确认 pyproject.toml / requirements.txt 是否覆盖了真实用到的依赖时，运行它。
- 想在清理依赖前先看哪些声明“可能没被静态 import 使用”时，运行它。
- 想发现仓库里哪些脚本引用了未安装或未声明的第三方包时，运行它。

注意事项:
- 这是静态扫描工具，只分析源码中的显式 import，不会执行代码。
- 对 importlib、__import__、插件注册、字符串反射加载这类动态依赖，只能做有限判断。
- “未使用声明”只是提示，不一定是错误；有些包可能只在运行时按名字动态加载。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = __import__("tomli")

from importlib import metadata as importlib_metadata


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = SCRIPT_PATH.parent.parent

DEFAULT_IGNORE_DIRS = [
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "output",
    "offline_bundle",
    ".idea",
    ".vscode",
]

IMPORT_TO_DISTRIBUTION_OVERRIDES = {
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "pil": "Pillow",
    "bs4": "beautifulsoup4",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "huggingface_hub": "huggingface-hub",
    "antlr4": "antlr4-python3-runtime",
    "IPython": "ipython",
    "pkg_resources": "setuptools",
    "tensorflow": "tensorflow",
    "ultralytics": "ultralytics",
    "pyzotero": "pyzotero",
}


@dataclass
class ScanConfig:
    """扫描配置。"""

    project_root: str = str(DEFAULT_PROJECT_ROOT)
    output_dir: str = str(DEFAULT_PROJECT_ROOT / "dependency_audit")
    scan_dirs: list[str] = field(default_factory=list)
    ignore_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_DIRS))
    config_files: list[str] = field(default_factory=list)
    fail_on_issues: bool = False


@dataclass
class ImportOccurrence:
    """单条 import 记录。"""

    file: str
    line: int
    statement: str
    top_level: str
    module: str
    is_relative: bool


@dataclass
class SyntaxIssue:
    """语法问题。"""

    file: str
    line: int
    message: str


@dataclass
class ImportResolution:
    """导入解析结果。"""

    top_level: str
    category: str
    distribution: str | None
    declared_in: list[str]
    installed: bool
    occurrences: list[ImportOccurrence]


@dataclass
class ConfigPackage:
    """依赖配置中的一条声明。"""

    name: str
    normalized_name: str
    source: str
    raw: str


class DependencyAuditError(RuntimeError):
    """统一异常。"""


def log(message: str) -> None:
    """输出信息日志。"""

    print(f"[INFO] {message}")


def warn(message: str) -> None:
    """输出警告日志。"""

    print(f"[WARN] {message}")


def normalize_name(name: str) -> str:
    """按 PEP 503 规则规整依赖名。"""

    return re.sub(r"[-_.]+", "-", name).lower()


def load_requirement_type() -> Any:
    """延迟加载 Requirement，避免静态扫描把兼容回退误认为项目依赖。"""

    try:
        module = __import__("packaging.requirements", fromlist=["Requirement"])
    except ModuleNotFoundError:
        module = __import__("pip._vendor.packaging.requirements", fromlist=["Requirement"])
    return module.Requirement


Requirement = load_requirement_type()


def path_is_relative_to(path: Path, base: Path) -> bool:
    """判断 path 是否位于 base 下."""

    return path.is_relative_to(base)


def ensure_directory(path: Path) -> None:
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)


def discover_python_files(
    project_root: Path,
    scan_dirs: list[str],
    ignore_dirs: list[str],
) -> list[Path]:
    """发现需要扫描的 Python 文件。"""

    roots = [project_root / item for item in scan_dirs] if scan_dirs else [project_root]
    ignore_set = set(ignore_dirs)
    files: list[Path] = []

    for root in roots:
        if not root.exists():
            warn(f"扫描目录不存在，已跳过: {root}")
            continue

        for current_root, dirnames, filenames in os.walk(root):
            current_path = Path(current_root)
            dirnames[:] = [
                name
                for name in dirnames
                if name not in ignore_set and not name.startswith(".tox")
            ]

            if any(part in ignore_set for part in current_path.parts):
                continue

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                file_path = current_path / filename
                files.append(file_path.resolve())

    return sorted(set(files))


def extract_imports_from_file(file_path: Path, project_root: Path) -> tuple[list[ImportOccurrence], list[SyntaxIssue]]:
    """从单个文件中提取 import。"""

    try:
        source = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = file_path.read_text(encoding="utf-8", errors="ignore")

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as error:
        issue = SyntaxIssue(
            file=str(file_path.relative_to(project_root)),
            line=error.lineno or 1,
            message=error.msg,
        )
        return [], [issue]

    occurrences: list[ImportOccurrence] = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            statement = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "import"
            for alias in node.names:
                top_level = alias.name.split(".", maxsplit=1)[0]
                occurrences.append(
                    ImportOccurrence(
                        file=str(file_path.relative_to(project_root)),
                        line=node.lineno,
                        statement=statement,
                        top_level=top_level,
                        module=alias.name,
                        is_relative=False,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            statement = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "from"
            module_name = node.module or ""
            top_level = module_name.split(".", maxsplit=1)[0] if module_name else ""
            occurrences.append(
                ImportOccurrence(
                    file=str(file_path.relative_to(project_root)),
                    line=node.lineno,
                    statement=statement,
                    top_level=top_level,
                    module=module_name,
                    is_relative=bool(node.level),
                )
            )

    return occurrences, []


def discover_config_files(project_root: Path, ignore_dirs: list[str]) -> list[Path]:
    """自动发现依赖配置文件。"""

    result: list[Path] = []
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        result.append(pyproject_path)

    ignore_set = set(ignore_dirs)
    for current_root, dirnames, filenames in os.walk(project_root):
        current_path = Path(current_root)
        dirnames[:] = [name for name in dirnames if name not in ignore_set]

        if any(part in ignore_set for part in current_path.parts):
            continue

        for filename in filenames:
            if not filename.startswith("requirements") or not filename.endswith(".txt"):
                continue
            result.append((current_path / filename).resolve())

    return sorted(set(result))


def parse_requirement_line(
    raw_line: str,
    *,
    source: str,
    project_name: str | None = None,
) -> ConfigPackage | None:
    """解析单条 requirement 声明。"""

    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith(("-r", "--requirement", "-c", "--constraint")):
        warn(f"暂不递归解析 requirement 引用: {source}: {stripped}")
        return None

    try:
        requirement = Requirement(stripped)
    except Exception:
        warn(f"无法解析 requirement，已跳过: {source}: {stripped}")
        return None

    if project_name and normalize_name(requirement.name) == normalize_name(project_name):
        return None

    return ConfigPackage(
        name=requirement.name,
        normalized_name=normalize_name(requirement.name),
        source=source,
        raw=stripped,
    )


def load_declared_packages_from_pyproject(pyproject_path: Path) -> tuple[str | None, list[ConfigPackage]]:
    """从 pyproject.toml 提取声明依赖。"""

    with pyproject_path.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)

    project = data.get("project")
    if not isinstance(project, dict):
        return None, []

    project_name = project.get("name")
    project_name_value = project_name if isinstance(project_name, str) else None

    packages: list[ConfigPackage] = []

    dependencies = project.get("dependencies", [])
    if isinstance(dependencies, list):
        for item in dependencies:
            package = parse_requirement_line(
                str(item),
                source="pyproject.toml:project.dependencies",
                project_name=project_name_value,
            )
            if package:
                packages.append(package)

    optional_dependencies = project.get("optional-dependencies", {})
    if isinstance(optional_dependencies, dict):
        for extra_name, values in optional_dependencies.items():
            if not isinstance(extra_name, str) or not isinstance(values, list):
                continue
            source = f"pyproject.toml:project.optional-dependencies.{extra_name}"
            for item in values:
                package = parse_requirement_line(
                    str(item),
                    source=source,
                    project_name=project_name_value,
                )
                if package:
                    packages.append(package)

    return project_name_value, packages


def load_declared_packages_from_requirements(requirements_path: Path, project_root: Path) -> list[ConfigPackage]:
    """从 requirements.txt 提取声明依赖。"""

    packages: list[ConfigPackage] = []
    source_prefix = str(requirements_path.relative_to(project_root))
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        package = parse_requirement_line(line, source=source_prefix)
        if package:
            packages.append(package)
    return packages


def load_declared_packages(
    project_root: Path,
    config_files: list[Path],
) -> tuple[str | None, list[ConfigPackage]]:
    """加载全部依赖配置。"""

    project_name: str | None = None
    packages: list[ConfigPackage] = []

    for config_path in config_files:
        if config_path.name == "pyproject.toml":
            discovered_project_name, discovered_packages = load_declared_packages_from_pyproject(
                config_path
            )
            if discovered_project_name:
                project_name = discovered_project_name
            packages.extend(discovered_packages)
        elif config_path.name.startswith("requirements") and config_path.suffix == ".txt":
            packages.extend(load_declared_packages_from_requirements(config_path, project_root))

    return project_name, packages


def build_declared_package_index(
    packages: list[ConfigPackage],
) -> dict[str, list[ConfigPackage]]:
    """构建按包名索引的声明表。"""

    index: dict[str, list[ConfigPackage]] = {}
    for package in packages:
        index.setdefault(package.normalized_name, []).append(package)
    return index


def build_installed_distribution_names() -> set[str]:
    """读取当前环境中已安装的 distribution 名称。"""

    names: set[str] = set()
    for distribution in importlib_metadata.distributions():
        dist_name = distribution.metadata.get("Name")
        if dist_name:
            names.add(normalize_name(dist_name))
    return names


def build_import_to_distribution_index() -> dict[str, list[str]]:
    """读取 import 名到 distribution 名的映射。"""

    try:
        mapping = importlib_metadata.packages_distributions()
    except Exception:
        return {}

    index: dict[str, list[str]] = {}
    for import_name, distributions in mapping.items():
        if not import_name:
            continue
        normalized_key = import_name.lower()
        index[normalized_key] = sorted(
            {
                normalize_name(distribution)
                for distribution in distributions
                if distribution
            }
        )
    return index


def is_probably_stdlib(top_level: str, stdlib_dir: Path | None, site_dirs: list[Path]) -> bool:
    """判断模块是否大概率属于标准库。"""

    stdlib_names = getattr(sys, "stdlib_module_names", set())
    if top_level in stdlib_names or top_level in sys.builtin_module_names:
        return True

    try:
        spec = importlib.util.find_spec(top_level)
    except Exception:
        return False

    if spec is None:
        return False

    if spec.origin in {"built-in", "frozen"}:
        return True

    if not spec.origin:
        return False

    origin_path = Path(spec.origin).resolve()
    for site_dir in site_dirs:
        if path_is_relative_to(origin_path, site_dir):
            return False

    if stdlib_dir and path_is_relative_to(origin_path, stdlib_dir):
        return True

    return False


def is_repo_local_module(top_level: str, source_file: Path, project_root: Path) -> bool:
    """按路径启发式判断是否为仓库内本地模块。"""

    if not top_level:
        return False

    current = source_file.parent.resolve()
    project_root = project_root.resolve()

    while True:
        if (current / f"{top_level}.py").exists():
            return True
        if (current / top_level).is_dir():
            return True
        if current == project_root:
            break
        if not path_is_relative_to(current, project_root):
            break
        current = current.parent

    return False


def choose_distribution_name(
    candidates: list[str],
    declared_index: dict[str, list[ConfigPackage]],
) -> str:
    """从候选 distribution 中挑一个更合理的名字。"""

    for candidate in candidates:
        if candidate in declared_index:
            return candidate
    return sorted(candidates)[0]


def resolve_distribution_name(
    top_level: str,
    *,
    import_to_distribution_index: dict[str, list[str]],
    declared_index: dict[str, list[ConfigPackage]],
    installed_distribution_names: set[str],
    site_dirs: list[Path],
) -> str | None:
    """把 import 名解析成 distribution 名。"""

    override = IMPORT_TO_DISTRIBUTION_OVERRIDES.get(top_level)
    if override:
        return normalize_name(override)

    override = IMPORT_TO_DISTRIBUTION_OVERRIDES.get(top_level.lower())
    if override:
        return normalize_name(override)

    candidates = import_to_distribution_index.get(top_level.lower(), [])
    if candidates:
        return choose_distribution_name(candidates, declared_index)

    try:
        spec = importlib.util.find_spec(top_level)
    except Exception:
        spec = None

    if spec and spec.origin and spec.origin not in {"built-in", "frozen"}:
        origin_path = Path(spec.origin).resolve()
        if any(path_is_relative_to(origin_path, site_dir) for site_dir in site_dirs):
            normalized_top_level = normalize_name(top_level)
            if normalized_top_level in installed_distribution_names:
                return normalized_top_level
            if normalized_top_level in declared_index:
                return normalized_top_level
            return normalized_top_level

    normalized_top_level = normalize_name(top_level)
    if normalized_top_level in declared_index:
        return normalized_top_level
    if normalized_top_level in installed_distribution_names:
        return normalized_top_level

    return None


def resolve_imports(
    occurrences: list[ImportOccurrence],
    *,
    project_root: Path,
    declared_index: dict[str, list[ConfigPackage]],
    installed_distribution_names: set[str],
    import_to_distribution_index: dict[str, list[str]],
) -> dict[str, ImportResolution]:
    """对全部 import 做分类和依赖映射。"""

    grouped: dict[str, list[ImportOccurrence]] = {}
    for occurrence in occurrences:
        if not occurrence.top_level and occurrence.is_relative:
            continue
        grouped.setdefault(occurrence.top_level, []).append(occurrence)

    stdlib_dir_value = sysconfig.get_paths().get("stdlib")
    stdlib_dir = Path(stdlib_dir_value).resolve() if stdlib_dir_value else None

    site_dirs: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            site_dirs.append(Path(value).resolve())

    resolutions: dict[str, ImportResolution] = {}

    for top_level, items in sorted(grouped.items()):
        first_item = items[0]
        if first_item.is_relative:
            resolutions[top_level] = ImportResolution(
                top_level=top_level,
                category="local",
                distribution=None,
                declared_in=[],
                installed=False,
                occurrences=items,
            )
            continue

        source_file = project_root / first_item.file
        if is_repo_local_module(top_level, source_file, project_root):
            resolutions[top_level] = ImportResolution(
                top_level=top_level,
                category="local",
                distribution=None,
                declared_in=[],
                installed=False,
                occurrences=items,
            )
            continue

        if is_probably_stdlib(top_level, stdlib_dir, site_dirs):
            resolutions[top_level] = ImportResolution(
                top_level=top_level,
                category="stdlib",
                distribution=None,
                declared_in=[],
                installed=False,
                occurrences=items,
            )
            continue

        distribution = resolve_distribution_name(
            top_level,
            import_to_distribution_index=import_to_distribution_index,
            declared_index=declared_index,
            installed_distribution_names=installed_distribution_names,
            site_dirs=site_dirs,
        )

        if distribution is None:
            resolutions[top_level] = ImportResolution(
                top_level=top_level,
                category="unresolved",
                distribution=None,
                declared_in=[],
                installed=False,
                occurrences=items,
            )
            continue

        declared_in = [
            package.source
            for package in declared_index.get(distribution, [])
        ]
        installed = distribution in installed_distribution_names
        resolutions[top_level] = ImportResolution(
            top_level=top_level,
            category="third_party",
            distribution=distribution,
            declared_in=sorted(set(declared_in)),
            installed=installed,
            occurrences=items,
        )

    return resolutions


def build_text_report(
    *,
    config: ScanConfig,
    scanned_files: list[Path],
    config_files: list[Path],
    declared_packages: list[ConfigPackage],
    resolutions: dict[str, ImportResolution],
    syntax_issues: list[SyntaxIssue],
    missing_declarations: list[ImportResolution],
    missing_installations: list[ImportResolution],
    unresolved_imports: list[ImportResolution],
    unused_declarations: list[ConfigPackage],
) -> str:
    """生成文本报告。"""

    lines: list[str] = []
    lines.append("项目依赖静态扫描报告")
    lines.append("====================")
    lines.append("")
    lines.append(f"项目根目录: {config.project_root}")
    lines.append(f"扫描文件数: {len(scanned_files)}")
    lines.append(f"依赖配置文件数: {len(config_files)}")
    lines.append(f"声明依赖数: {len(declared_packages)}")
    lines.append(f"第三方 import 数: {sum(1 for item in resolutions.values() if item.category == 'third_party')}")
    lines.append(f"缺失声明数: {len(missing_declarations)}")
    lines.append(f"缺失安装数: {len(missing_installations)}")
    lines.append(f"无法解析 import 数: {len(unresolved_imports)}")
    lines.append(f"语法问题文件数: {len(syntax_issues)}")
    lines.append("")

    lines.append("依赖配置文件")
    lines.append("------------")
    for path in config_files:
        lines.append(f"- {path.relative_to(Path(config.project_root))}")
    lines.append("")

    lines.append("缺失声明的第三方依赖")
    lines.append("--------------------")
    if not missing_declarations:
        lines.append("- 无")
    else:
        for item in missing_declarations:
            sample = item.occurrences[0]
            lines.append(
                f"- {item.distribution}: import `{item.top_level}`, 首次出现于 {sample.file}:{sample.line}"
            )
    lines.append("")

    lines.append("已引用但当前环境未安装的第三方依赖")
    lines.append("------------------------------")
    if not missing_installations:
        lines.append("- 无")
    else:
        for item in missing_installations:
            sample = item.occurrences[0]
            lines.append(
                f"- {item.distribution}: import `{item.top_level}`, 首次出现于 {sample.file}:{sample.line}"
            )
    lines.append("")

    lines.append("无法解析的 import")
    lines.append("----------------")
    if not unresolved_imports:
        lines.append("- 无")
    else:
        for item in unresolved_imports:
            sample = item.occurrences[0]
            lines.append(
                f"- `{item.top_level}`: 首次出现于 {sample.file}:{sample.line}"
            )
    lines.append("")

    lines.append("声明但未检测到静态 import 的依赖")
    lines.append("------------------------------")
    if not unused_declarations:
        lines.append("- 无")
    else:
        for package in unused_declarations:
            lines.append(f"- {package.name}: {package.source}")
    lines.append("")

    lines.append("语法问题")
    lines.append("--------")
    if not syntax_issues:
        lines.append("- 无")
    else:
        for issue in syntax_issues:
            lines.append(f"- {issue.file}:{issue.line}: {issue.message}")
    lines.append("")

    lines.append("第三方依赖解析明细")
    lines.append("----------------")
    third_party_items = [
        item
        for item in resolutions.values()
        if item.category == "third_party"
    ]
    if not third_party_items:
        lines.append("- 无")
    else:
        for item in sorted(third_party_items, key=lambda value: value.distribution or value.top_level):
            declared_text = ", ".join(item.declared_in) if item.declared_in else "未声明"
            sample = item.occurrences[0]
            lines.append(
                f"- {item.distribution}: import `{item.top_level}`, "
                f"declared={declared_text}, installed={item.installed}, "
                f"sample={sample.file}:{sample.line}"
            )

    lines.append("")
    return "\n".join(lines)


def run_dependency_audit(config: ScanConfig) -> int:
    """执行依赖扫描与校验。"""

    project_root = Path(config.project_root).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    ensure_directory(output_dir)

    scanned_files = discover_python_files(
        project_root,
        scan_dirs=config.scan_dirs,
        ignore_dirs=config.ignore_dirs,
    )
    log(f"发现 {len(scanned_files)} 个 Python 文件。")

    config_paths = (
        [project_root / item for item in config.config_files]
        if config.config_files
        else discover_config_files(project_root, config.ignore_dirs)
    )
    config_paths = [path.resolve() for path in config_paths if path.exists()]
    log(f"发现 {len(config_paths)} 个依赖配置文件。")

    project_name, declared_packages = load_declared_packages(project_root, config_paths)
    declared_index = build_declared_package_index(declared_packages)
    if project_name:
        log(f"检测到项目名: {project_name}")

    all_occurrences: list[ImportOccurrence] = []
    syntax_issues: list[SyntaxIssue] = []
    for file_path in scanned_files:
        occurrences, issues = extract_imports_from_file(file_path, project_root)
        all_occurrences.extend(occurrences)
        syntax_issues.extend(issues)

    log(f"提取到 {len(all_occurrences)} 条 import 记录。")

    installed_distribution_names = build_installed_distribution_names()
    import_to_distribution_index = build_import_to_distribution_index()

    resolutions = resolve_imports(
        all_occurrences,
        project_root=project_root,
        declared_index=declared_index,
        installed_distribution_names=installed_distribution_names,
        import_to_distribution_index=import_to_distribution_index,
    )

    third_party_items = [
        item
        for item in resolutions.values()
        if item.category == "third_party"
    ]
    missing_declarations = [
        item for item in third_party_items if not item.declared_in
    ]
    missing_installations = [
        item for item in third_party_items if not item.installed
    ]
    unresolved_imports = [
        item
        for item in resolutions.values()
        if item.category == "unresolved"
    ]

    used_distribution_names = {
        item.distribution
        for item in third_party_items
        if item.distribution
    }
    unused_declarations = [
        package
        for package in declared_packages
        if package.normalized_name not in used_distribution_names
    ]

    report_data = {
        "config": asdict(config),
        "project_name": project_name,
        "scanned_files": [str(path.relative_to(project_root)) for path in scanned_files],
        "config_files": [str(path.relative_to(project_root)) for path in config_paths],
        "declared_packages": [asdict(package) for package in declared_packages],
        "syntax_issues": [asdict(issue) for issue in syntax_issues],
        "resolutions": [
            {
                "top_level": item.top_level,
                "category": item.category,
                "distribution": item.distribution,
                "declared_in": item.declared_in,
                "installed": item.installed,
                "occurrences": [asdict(occurrence) for occurrence in item.occurrences],
            }
            for item in sorted(resolutions.values(), key=lambda value: value.top_level)
        ],
        "summary": {
            "python_files": len(scanned_files),
            "config_files": len(config_paths),
            "declared_packages": len(declared_packages),
            "third_party_imports": len(third_party_items),
            "missing_declarations": len(missing_declarations),
            "missing_installations": len(missing_installations),
            "unresolved_imports": len(unresolved_imports),
            "syntax_issues": len(syntax_issues),
            "unused_declarations": len(unused_declarations),
        },
        "missing_declarations": [
            {
                "distribution": item.distribution,
                "top_level": item.top_level,
                "sample_file": item.occurrences[0].file,
                "sample_line": item.occurrences[0].line,
            }
            for item in missing_declarations
        ],
        "missing_installations": [
            {
                "distribution": item.distribution,
                "top_level": item.top_level,
                "sample_file": item.occurrences[0].file,
                "sample_line": item.occurrences[0].line,
            }
            for item in missing_installations
        ],
        "unresolved_imports": [
            {
                "top_level": item.top_level,
                "sample_file": item.occurrences[0].file,
                "sample_line": item.occurrences[0].line,
            }
            for item in unresolved_imports
        ],
        "unused_declarations": [asdict(package) for package in unused_declarations],
    }

    json_report_path = output_dir / "dependency_report.json"
    text_report_path = output_dir / "dependency_report.txt"
    json_report_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    text_report_path.write_text(
        build_text_report(
            config=config,
            scanned_files=scanned_files,
            config_files=config_paths,
            declared_packages=declared_packages,
            resolutions=resolutions,
            syntax_issues=syntax_issues,
            missing_declarations=missing_declarations,
            missing_installations=missing_installations,
            unresolved_imports=unresolved_imports,
            unused_declarations=unused_declarations,
        ),
        encoding="utf-8",
    )

    log(f"JSON 报告已写入: {json_report_path}")
    log(f"文本报告已写入: {text_report_path}")

    if config.fail_on_issues and (
        missing_declarations or missing_installations or unresolved_imports or syntax_issues
    ):
        return 1

    return 0


def main() -> int:
    """程序入口，直接在这里修改参数。"""

    config = ScanConfig(
        project_root=str(DEFAULT_PROJECT_ROOT),
        output_dir=str(DEFAULT_PROJECT_ROOT / "dependency_audit"),
        scan_dirs=[],
        ignore_dirs=list(DEFAULT_IGNORE_DIRS),
        config_files=[],
        fail_on_issues=False,
    )

    try:
        return run_dependency_audit(config)
    except DependencyAuditError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[ERROR] 用户中断执行。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
